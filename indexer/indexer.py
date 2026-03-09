"""Boot-time index build + file watcher."""

import fnmatch
import logging
import threading
import time
from pathlib import Path

import watchfiles

from indexer.ast_map import ASTMap
from indexer.dep_graph import DepGraph
from indexer.embeddings import EmbeddingStore
from indexer.git_meta import GitMeta

logger = logging.getLogger("rlm.indexer")

# Extensions we index
_INDEXABLE_EXTS = {".ts", ".tsx", ".js", ".jsx", ".py", ".go"}

# Directories and suffixes to ignore in the file watcher (common noise sources)
_WATCH_IGNORE_DIRS = {
    ".next", "node_modules", ".git", "dist", "build",
    "__pycache__", ".vscode", "coverage",
}
_WATCH_IGNORE_SUFFIXES = {".log", ".tmp", ".cache"}

_DEBOUNCE_SECONDS = 2


def _watch_filter(change, path: str) -> bool:
    """Filter for watchfiles: only indexable files outside noise directories."""
    p = Path(path)
    if p.suffix in _WATCH_IGNORE_SUFFIXES:
        return False
    if p.suffix not in _INDEXABLE_EXTS:
        return False
    if _WATCH_IGNORE_DIRS & set(p.parts):
        return False
    return True


class RepoIndex:
    def __init__(self, config: dict):
        self.config = config
        self.repo_path = config.get("repo_path", "")
        idx_cfg = config.get("indexer", {})
        self.languages = idx_cfg.get("languages", ["typescript", "python", "javascript", "go"])
        self.exclude_patterns = idx_cfg.get("exclude", [])
        self.embedding_model = idx_cfg.get("embedding_model", "nomic-embed-text")
        self.embeddings_db_path = idx_cfg.get("embeddings_db_path", None)

        self.ast_map = ASTMap(self.languages)
        self.dep_graph = DepGraph()
        self.embeddings = EmbeddingStore(
            model=self.embedding_model,
            db_path=self.embeddings_db_path,
        )
        self.git_meta = GitMeta(self.repo_path)

        self.ready = False
        self.last_indexed: str | None = None
        self.file_count = 0
        self.symbol_count = 0
        self.chunk_count = 0

        self._all_files: list[str] = []
        self._file_contents: dict[str, str] = {}
        self._watcher_thread: threading.Thread | None = None
        self._stop_watcher = threading.Event()

    def build(self):
        """Full index build."""
        self.ready = False
        repo = Path(self.repo_path)
        if not repo.is_dir():
            logger.warning(f"Repo path does not exist: {self.repo_path}")
            return

        # Walk repo and collect files
        self._all_files = []
        self._file_contents = {}

        for file_path in repo.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix not in _INDEXABLE_EXTS:
                continue
            if self._is_excluded(file_path, repo):
                continue

            str_path = str(file_path)
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            self._all_files.append(str_path)
            self._file_contents[str_path] = content

        self.file_count = len(self._all_files)
        logger.info(f"Found {self.file_count} indexable files")

        # AST indexing
        for fp in self._all_files:
            self.ast_map.index_file(fp, self._file_contents[fp])
        self.symbol_count = self.ast_map.symbol_count
        logger.info(f"AST indexed: {self.symbol_count} symbols")

        # Dependency graph
        self.dep_graph.build(self.ast_map, self.repo_path, self._all_files)

        # Git metadata
        self.git_meta.init()

        # Embeddings — load from cache or build
        try:
            self.embeddings.load_or_build(self.ast_map, self._file_contents)
            self.chunk_count = self.embeddings.chunk_count
        except Exception as e:
            logger.error(f"Embedding build failed (continuing without): {e}")
            self.chunk_count = 0

        self.last_indexed = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.ready = True

    def rebuild(self):
        """Trigger a full rebuild including embeddings (called from admin endpoint)."""
        def _rebuild():
            self.ready = False
            repo = Path(self.repo_path)
            if not repo.is_dir():
                return

            self._all_files = []
            self._file_contents = {}

            for file_path in repo.rglob("*"):
                if not file_path.is_file():
                    continue
                if file_path.suffix not in _INDEXABLE_EXTS:
                    continue
                if self._is_excluded(file_path, repo):
                    continue

                str_path = str(file_path)
                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue

                self._all_files.append(str_path)
                self._file_contents[str_path] = content

            self.file_count = len(self._all_files)

            for fp in self._all_files:
                self.ast_map.index_file(fp, self._file_contents[fp])
            self.symbol_count = self.ast_map.symbol_count

            self.dep_graph.build(self.ast_map, self.repo_path, self._all_files)
            self.git_meta.init()

            try:
                self.embeddings.rebuild(self.ast_map, self._file_contents)
                self.chunk_count = self.embeddings.chunk_count
            except Exception as e:
                logger.error(f"Embedding rebuild failed: {e}")

            self.last_indexed = time.strftime("%Y-%m-%dT%H:%M:%S")
            self.ready = True
            logger.info("Full rebuild complete")

        thread = threading.Thread(target=_rebuild, daemon=True)
        thread.start()

    def _is_excluded(self, file_path: Path, repo: Path) -> bool:
        """Check if a file matches any exclude pattern."""
        try:
            rel = str(file_path.relative_to(repo))
        except ValueError:
            rel = str(file_path)

        parts = rel.split("/")
        for pattern in self.exclude_patterns:
            for part in parts:
                if fnmatch.fnmatch(part, pattern):
                    return True
            if fnmatch.fnmatch(rel, pattern):
                return True
        return False

    def _reindex_file(self, file_path: str):
        """Incrementally reindex a single file (AST + embeddings)."""
        fp = Path(file_path)
        if not fp.is_file() or fp.suffix not in _INDEXABLE_EXTS:
            return
        if self._is_excluded(fp, Path(self.repo_path)):
            return

        try:
            content = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return

        str_path = str(fp)
        self._file_contents[str_path] = content
        if str_path not in self._all_files:
            self._all_files.append(str_path)
            self.file_count = len(self._all_files)

        self.ast_map.index_file(str_path, content)
        self.symbol_count = self.ast_map.symbol_count

        # Incrementally re-embed only this file's chunks
        try:
            self.embeddings.reindex_file(str_path, self.ast_map)
            self.chunk_count = self.embeddings.chunk_count
        except Exception as e:
            logger.error(f"Incremental embedding update failed for {str_path}: {e}")

        logger.debug(f"Reindexed: {str_path}")

    def _handle_delete(self, file_path: str):
        """Handle file deletion."""
        str_path = str(file_path)
        self.ast_map._remove_file(str_path)
        self._file_contents.pop(str_path, None)
        if str_path in self._all_files:
            self._all_files.remove(str_path)
            self.file_count = len(self._all_files)

        # Remove embeddings for deleted file
        try:
            self.embeddings.remove_file(str_path)
            self.chunk_count = self.embeddings.chunk_count
        except Exception as e:
            logger.error(f"Failed to remove embeddings for {str_path}: {e}")

    def start_watcher(self):
        """Start watching repo for file changes."""
        if self._watcher_thread and self._watcher_thread.is_alive():
            return

        self._stop_watcher.clear()

        def watch():
            logger.info(f"Watching {self.repo_path} for changes")
            try:
                pending: dict[str, watchfiles.Change] = {}
                for changes in watchfiles.watch(
                    self.repo_path,
                    stop_event=self._stop_watcher,
                    watch_filter=_watch_filter,
                    rust_timeout=int(_DEBOUNCE_SECONDS * 1000),
                    yield_on_timeout=True,
                ):
                    if changes:
                        for change_type, path in changes:
                            pending[path] = change_type
                        continue

                    # Timeout with no new changes — flush pending batch
                    if not pending:
                        continue

                    for path, change_type in pending.items():
                        if change_type == watchfiles.Change.deleted:
                            self._handle_delete(path)
                        else:
                            self._reindex_file(path)
                    # Rebuild dep graph after batch of changes
                    self.dep_graph.build(self.ast_map, self.repo_path, self._all_files)
                    logger.debug(f"Debounced reindex: {len(pending)} file(s)")
                    pending.clear()
            except Exception as e:
                if not self._stop_watcher.is_set():
                    logger.error(f"Watcher error: {e}")

        self._watcher_thread = threading.Thread(target=watch, daemon=True)
        self._watcher_thread.start()

    def stop_watcher(self):
        self._stop_watcher.set()

    def get_file_content(self, file_path: str) -> str | None:
        """Get cached file content."""
        return self._file_contents.get(file_path)

    def find_file(self, partial_path: str) -> str | None:
        """Find a file by partial path match."""
        for fp in self._all_files:
            if fp.endswith(partial_path) or partial_path in fp:
                return fp
        return None

    def find_files(self, partial_path: str) -> list[str]:
        """Find all files matching a partial path."""
        return [fp for fp in self._all_files if partial_path in fp or fp.endswith(partial_path)]
