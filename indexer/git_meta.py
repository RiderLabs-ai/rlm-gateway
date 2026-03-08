"""Per-file recent commits via gitpython."""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import git

logger = logging.getLogger("rlm.git_meta")


@dataclass
class CommitSummary:
    hash: str
    message: str
    author: str
    date: str


class GitMeta:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self._repo: git.Repo | None = None

    def init(self):
        try:
            self._repo = git.Repo(self.repo_path)
            logger.info(f"Git repo initialized: {self.repo_path}")
        except (git.InvalidGitRepositoryError, git.NoSuchPathError):
            logger.warning(f"Not a git repo or path not found: {self.repo_path}")
            self._repo = None

    def recent_commits(self, path: str, n: int = 5) -> list[CommitSummary]:
        """Get the last N commits that touched a file."""
        if not self._repo:
            return []

        try:
            # Get relative path
            rel_path = str(Path(path).relative_to(self.repo_path))
        except ValueError:
            rel_path = path

        try:
            commits = list(self._repo.iter_commits(paths=rel_path, max_count=n))
        except Exception as e:
            logger.debug(f"Git log failed for {rel_path}: {e}")
            return []

        results = []
        for c in commits:
            committed = datetime.fromtimestamp(c.committed_date)
            delta = datetime.now() - committed
            if delta.days > 0:
                date_str = f"{delta.days} days ago"
            elif delta.seconds > 3600:
                date_str = f"{delta.seconds // 3600} hours ago"
            else:
                date_str = f"{delta.seconds // 60} minutes ago"

            results.append(CommitSummary(
                hash=c.hexsha[:7],
                message=c.message.strip().split("\n")[0][:80],
                author=str(c.author),
                date=date_str,
            ))

        return results
