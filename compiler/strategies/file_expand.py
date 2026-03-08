"""Expand seed files + 1-level imports."""

import logging

from compiler.pack import ContextPack, FileEntry
from gateway.extractor import TaskSignals

logger = logging.getLogger("rlm.strategy.file_expand")


def run(signals: TaskSignals, index, pack: ContextPack, config: dict):
    """For each file mention, add its contents and direct imports."""
    for file_mention in signals.file_mentions:
        # Resolve to actual file path
        resolved = index.find_file(file_mention)
        if not resolved:
            logger.debug(f"File not found in index: {file_mention}")
            continue

        # Add the file itself
        content = index.get_file_content(resolved)
        if content and not pack.has_file(resolved):
            pack.add_file(FileEntry(path=resolved, content=content))

        # Add direct imports (depth=1)
        deps = index.dep_graph.dependencies(resolved)
        for dep_path in deps:
            if pack.has_file(dep_path):
                continue
            dep_content = index.get_file_content(dep_path)
            if dep_content:
                pack.add_file(FileEntry(path=dep_path, content=dep_content))
