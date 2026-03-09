"""Chunk embeddings via Ollama → sqlite-vec storage (persistent on disk)."""

import logging
import sqlite3
import struct
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ollama
import sqlite_vec

from indexer.ast_map import ASTMap

logger = logging.getLogger("rlm.embeddings")

_DEFAULT_DB_PATH = Path.home() / ".rlm-gateway" / "embeddings.db"
_EMBED_WORKERS = 4
_EMBED_BATCH_SIZE = 50


def _serialize_float32(vec: list[float]) -> bytes:
    """Serialize a float32 vector to bytes for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


class EmbeddingStore:
    def __init__(self, model: str = "nomic-embed-text", db_path: str | None = None):
        self.model = model
        self._dim = 768  # nomic-embed-text dimension
        self._db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self._conn: sqlite3.Connection | None = None
        self._db_lock = threading.Lock()

    def _init_db(self):
        """Open (or create) the on-disk database."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

        self._conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks
            USING vec0(embedding float[{self._dim}])
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS chunk_meta (
                id INTEGER PRIMARY KEY,
                file_path TEXT,
                start_line INTEGER,
                end_line INTEGER,
                text TEXT
            )
        """)
        self._conn.commit()

    def _has_data(self) -> bool:
        """Check if the database already has embedded chunks."""
        if not self._conn:
            return False
        try:
            row = self._conn.execute("SELECT COUNT(*) FROM chunk_meta").fetchone()
            return row[0] > 0 if row else False
        except Exception:
            return False

    def load_or_build(self, ast_map: ASTMap, file_contents: dict[str, str]):
        """Load existing embeddings from disk, or build from scratch if empty."""
        self._init_db()

        if self._has_data():
            count = self.chunk_count
            logger.info(f"Embeddings loaded from cache: {count} chunks ({self._db_path})")
            return

        logger.info("No cached embeddings found — building from scratch")
        self._build_all(ast_map, file_contents)

    def rebuild(self, ast_map: ASTMap, file_contents: dict[str, str]):
        """Force a full rebuild, clearing existing data."""
        self._init_db()
        logger.info("Rebuilding all embeddings from scratch")
        self._clear_all()
        self._build_all(ast_map, file_contents)

    def reindex_file(self, file_path: str, ast_map: ASTMap):
        """Incrementally re-embed chunks for a single changed file."""
        if not self._conn:
            self._init_db()

        # Remove old chunks for this file
        old_ids = self._conn.execute(
            "SELECT id FROM chunk_meta WHERE file_path = ?", (file_path,)
        ).fetchall()

        for (row_id,) in old_ids:
            self._conn.execute("DELETE FROM vec_chunks WHERE rowid = ?", (row_id,))
            self._conn.execute("DELETE FROM chunk_meta WHERE id = ?", (row_id,))

        # Collect new chunks from this file's symbols
        chunks = []
        for symbol_name, defs in ast_map.symbols.items():
            for sym_def in defs:
                if sym_def.file_path == file_path and sym_def.kind in ("function", "method") and sym_def.body:
                    chunks.append({
                        "text": sym_def.body[:2000],
                        "file_path": sym_def.file_path,
                        "start_line": sym_def.start_line,
                        "end_line": sym_def.end_line,
                    })

        if not chunks:
            self._conn.commit()
            logger.debug(f"Removed embeddings for {file_path} (no new chunks)")
            return

        # Get next available rowid
        row = self._conn.execute("SELECT COALESCE(MAX(id), -1) + 1 FROM chunk_meta").fetchone()
        next_id = row[0]

        # Embed and insert
        texts = [c["text"] for c in chunks]
        try:
            response = ollama.embed(model=self.model, input=texts)
            embeddings = response.get("embeddings", [])
        except Exception as e:
            logger.error(f"Embedding failed for {file_path}: {e}")
            self._conn.commit()
            return

        for chunk, emb in zip(chunks, embeddings):
            self._conn.execute(
                "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
                (next_id, _serialize_float32(emb)),
            )
            self._conn.execute(
                "INSERT INTO chunk_meta(id, file_path, start_line, end_line, text) VALUES (?, ?, ?, ?, ?)",
                (next_id, chunk["file_path"], chunk["start_line"], chunk["end_line"], chunk["text"]),
            )
            next_id += 1

        self._conn.commit()
        logger.debug(f"Re-embedded {len(chunks)} chunks for {file_path}")

    def remove_file(self, file_path: str):
        """Remove all chunks for a deleted file."""
        if not self._conn:
            return
        old_ids = self._conn.execute(
            "SELECT id FROM chunk_meta WHERE file_path = ?", (file_path,)
        ).fetchall()
        for (row_id,) in old_ids:
            self._conn.execute("DELETE FROM vec_chunks WHERE rowid = ?", (row_id,))
            self._conn.execute("DELETE FROM chunk_meta WHERE id = ?", (row_id,))
        self._conn.commit()

    def _embed_file_chunks(self, file_path: str, chunks: list[dict]) -> list[tuple[dict, list[float]]]:
        """Embed all chunks for a single file. Called from worker threads."""
        results = []
        for i in range(0, len(chunks), _EMBED_BATCH_SIZE):
            batch = chunks[i:i + _EMBED_BATCH_SIZE]
            texts = [c["text"] for c in batch]
            try:
                response = ollama.embed(model=self.model, input=texts)
                embeddings = response.get("embeddings", [])
                for chunk, emb in zip(batch, embeddings):
                    results.append((chunk, emb))
            except Exception as e:
                logger.error(f"Embedding batch failed for {file_path}: {e}")
        return results

    def _build_all(self, ast_map: ASTMap, file_contents: dict[str, str]):
        """Embed all function-level chunks, parallelized by file."""
        # Group chunks by file
        file_chunks: dict[str, list[dict]] = defaultdict(list)
        for symbol_name, defs in ast_map.symbols.items():
            for sym_def in defs:
                if sym_def.kind in ("function", "method") and sym_def.body:
                    file_chunks[sym_def.file_path].append({
                        "text": sym_def.body[:2000],
                        "file_path": sym_def.file_path,
                        "start_line": sym_def.start_line,
                        "end_line": sym_def.end_line,
                    })

        file_list = list(file_chunks.keys())
        total_files = len(file_list)

        if total_files == 0:
            logger.info("No chunks to embed")
            return

        total_chunks = sum(len(v) for v in file_chunks.values())
        logger.info(f"Embedding {total_chunks} chunks across {total_files} files ({_EMBED_WORKERS} workers)")

        # Embed in parallel across files
        chunk_id = 0
        files_done = 0
        last_pct = -1

        with ThreadPoolExecutor(max_workers=_EMBED_WORKERS) as pool:
            futures = {
                pool.submit(self._embed_file_chunks, fp, chunks): fp
                for fp, chunks in file_chunks.items()
            }

            for future in as_completed(futures):
                fp = futures[future]
                try:
                    embedded_pairs = future.result()
                except Exception as e:
                    logger.error(f"Worker failed for {fp}: {e}")
                    embedded_pairs = []

                # Write results to DB under lock (sqlite is not thread-safe for writes)
                with self._db_lock:
                    for chunk, emb in embedded_pairs:
                        self._conn.execute(
                            "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
                            (chunk_id, _serialize_float32(emb)),
                        )
                        self._conn.execute(
                            "INSERT INTO chunk_meta(id, file_path, start_line, end_line, text) VALUES (?, ?, ?, ?, ?)",
                            (chunk_id, chunk["file_path"], chunk["start_line"], chunk["end_line"], chunk["text"]),
                        )
                        chunk_id += 1

                files_done += 1
                pct = (files_done * 100) // total_files
                if pct >= last_pct + 10 or files_done == total_files:
                    logger.info(f"Embedding files: {files_done}/{total_files} ({pct}%)...")
                    last_pct = pct

        self._conn.commit()
        logger.info(f"Embedded {chunk_id} chunks across {total_files} files")

    def _clear_all(self):
        """Wipe all embedding data."""
        self._conn.execute("DELETE FROM chunk_meta")
        self._conn.execute("DELETE FROM vec_chunks")
        self._conn.commit()

    def search(self, query: str, k: int = 8, threshold: float = 0.40) -> list[tuple[str, str, float]]:
        """Search for similar chunks. Returns (chunk_text, file_path, score)."""
        if not self._conn:
            return []

        try:
            response = ollama.embed(model=self.model, input=[query])
            query_emb = response.get("embeddings", [[]])[0]
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return []

        if not query_emb:
            return []

        query_bytes = _serialize_float32(query_emb)

        rows = self._conn.execute(
            """
            SELECT v.rowid, v.distance, m.text, m.file_path, m.start_line, m.end_line
            FROM vec_chunks v
            JOIN chunk_meta m ON v.rowid = m.id
            WHERE v.embedding MATCH ?
            AND k = ?
            ORDER BY v.distance
            """,
            (query_bytes, k),
        ).fetchall()

        results = []
        for row in rows:
            rowid, distance, text, file_path, start_line, end_line = row
            score = 1.0 / (1.0 + distance)
            if score >= threshold:
                results.append((text, file_path, score))

        return results

    @property
    def chunk_count(self) -> int:
        if not self._conn:
            return 0
        try:
            row = self._conn.execute("SELECT COUNT(*) FROM chunk_meta").fetchone()
            return row[0] if row else 0
        except Exception:
            return 0

    def close(self):
        if self._conn:
            self._conn.close()
