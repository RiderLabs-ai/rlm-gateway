"""Recent commits on relevant files."""

import logging

from compiler.pack import ContextPack, GitEntry
from gateway.extractor import TaskSignals

logger = logging.getLogger("rlm.strategy.git_ctx")


def run(signals: TaskSignals, index, pack: ContextPack, config: dict):
    """For each file in the pack, get recent commits and add as git context."""
    current_files = list(pack.all_file_paths())

    for file_path in current_files:
        commits = index.git_meta.recent_commits(file_path, n=3)
        if commits:
            pack.add_git_entry(GitEntry(
                file_path=file_path,
                commits=[
                    {
                        "hash": c.hash,
                        "message": c.message,
                        "author": c.author,
                        "date": c.date,
                    }
                    for c in commits
                ],
            ))
