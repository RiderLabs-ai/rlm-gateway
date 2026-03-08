"""Module dependency graph using NetworkX."""

import logging
from pathlib import Path

import networkx as nx

from indexer.ast_map import ASTMap, ImportInfo

logger = logging.getLogger("rlm.dep_graph")


class DepGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def build(self, ast_map: ASTMap, repo_path: str, all_files: list[str]):
        """Build dependency graph from import data. Edge A → B means A imports B."""
        self.graph.clear()

        # Add all files as nodes
        for f in all_files:
            self.graph.add_node(f)

        # Map module names to file paths for resolution
        file_index = self._build_file_index(all_files, repo_path)

        # Add edges from import data
        for file_path, imports in ast_map.imports.items():
            for imp in imports:
                resolved = self._resolve_import(imp.source, file_path, file_index, repo_path)
                if resolved and resolved != file_path:
                    self.graph.add_edge(file_path, resolved)

        logger.info(f"Dep graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def _build_file_index(self, all_files: list[str], repo_path: str) -> dict[str, str]:
        """Map possible import identifiers to file paths."""
        index = {}
        repo = Path(repo_path)
        for f in all_files:
            fp = Path(f)
            # Store by relative path variants
            try:
                rel = fp.relative_to(repo)
            except ValueError:
                rel = fp
            # Without extension
            no_ext = str(rel.with_suffix(""))
            index[no_ext] = f
            index[str(rel)] = f
            # Just the stem (last component without extension)
            index[fp.stem] = f
            # Dotted module path (for Python: a/b/c.py → a.b.c)
            dotted = no_ext.replace("/", ".").replace("\\", ".")
            index[dotted] = f
        return index

    def _resolve_import(self, source: str, importer: str, file_index: dict, repo_path: str) -> str | None:
        """Try to resolve an import source to a file path."""
        # Direct match
        if source in file_index:
            return file_index[source]

        # Handle relative imports (./foo, ../foo)
        if source.startswith("."):
            importer_dir = str(Path(importer).parent)
            resolved = str(Path(importer_dir) / source)
            resolved = str(Path(resolved).resolve())
            # Try with extensions
            for ext in ("", ".ts", ".tsx", ".js", ".jsx", ".py", ".go"):
                candidate = resolved + ext
                if candidate in file_index:
                    return file_index[candidate]
                # Check values
                for v in file_index.values():
                    if v.endswith(candidate.lstrip("/")):
                        return v

        # Python dotted imports
        dotted_path = source.replace(".", "/")
        for ext in ("", ".py", ".ts", ".js", ".go"):
            candidate = dotted_path + ext
            if candidate in file_index:
                return file_index[candidate]

        # Partial match on the last component
        parts = source.replace(".", "/").split("/")
        if parts:
            last = parts[-1]
            if last in file_index:
                return file_index[last]

        return None

    def dependents(self, path: str, max_hops: int = 2) -> list[str]:
        """Files that import this file (upstream), up to max_hops."""
        if path not in self.graph:
            return []

        result = set()
        current = {path}
        for _ in range(max_hops):
            next_level = set()
            for node in current:
                for pred in self.graph.predecessors(node):
                    if pred != path:
                        result.add(pred)
                        next_level.add(pred)
            current = next_level
            if not current:
                break

        return list(result)

    def dependencies(self, path: str) -> list[str]:
        """Files this file directly imports."""
        if path not in self.graph:
            return []
        return list(self.graph.successors(path))
