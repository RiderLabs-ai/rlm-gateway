"""What imports the target files (for refactors)."""

import logging

from compiler.pack import ContextPack, FileEntry
from gateway.extractor import TaskSignals

logger = logging.getLogger("rlm.strategy.blast_radius")


def run(signals: TaskSignals, index, pack: ContextPack, config: dict):
    """For each file in the pack, get upstream dependents (max 2 hops).
    Add a lightweight summary (path + first 3 lines) for each dependent."""
    # Snapshot current file paths to avoid modifying while iterating
    current_files = list(pack.all_file_paths())

    for file_path in current_files:
        dependents = index.dep_graph.dependents(file_path, max_hops=2)
        for dep_path in dependents:
            if pack.has_file(dep_path):
                continue
            content = index.get_file_content(dep_path)
            if content:
                # Only add first 3 lines as a summary
                summary_lines = content.split("\n")[:3]
                summary = "\n".join(summary_lines)
                pack.add_file(FileEntry(
                    path=dep_path,
                    content=f"// [blast radius - imports {file_path}]\n{summary}\n// ...",
                ))
