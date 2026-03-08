"""Orchestrate strategies → ContextPack."""

import hashlib
import logging
import time

from cachetools import TTLCache

from compiler import budget
from compiler.pack import ContextPack
from compiler.strategies import (
    blast_radius,
    convention,
    file_expand,
    git_ctx,
    semantic,
    symbol,
)
from gateway.extractor import TaskSignals

logger = logging.getLogger("rlm.compiler")


class Compiler:
    def __init__(self, config: dict):
        self.config = config
        cache_cfg = config.get("cache", {})
        if cache_cfg.get("enabled", True):
            self._cache = TTLCache(
                maxsize=cache_cfg.get("max_size", 256),
                ttl=cache_cfg.get("ttl_seconds", 300),
            )
        else:
            self._cache = None

    def compile(self, signals: TaskSignals, index) -> ContextPack:
        """Compile a ContextPack from TaskSignals using the repo index."""
        # Check cache
        cache_key = self._cache_key(signals)
        if self._cache is not None and cache_key in self._cache:
            logger.debug("Cache hit for context pack")
            return self._cache[cache_key]

        t0 = time.time()
        pack = ContextPack()
        max_tokens = self.config.get("compiler", {}).get("max_pack_tokens", 6000)

        # Run strategies based on signals
        if signals.symbols:
            self._run_safe("symbol", lambda: symbol.run(signals, index, pack, self.config))

        if signals.file_mentions:
            self._run_safe("file_expand", lambda: file_expand.run(signals, index, pack, self.config))

        self._run_safe("semantic", lambda: semantic.run(signals, index, pack, self.config))

        if signals.task_type == "refactor":
            self._run_safe("blast_radius", lambda: blast_radius.run(signals, index, pack, self.config))

        if signals.task_type == "add_feature":
            self._run_safe("convention", lambda: convention.run(signals, index, pack, self.config))

        if signals.task_type in ("debug", "refactor"):
            self._run_safe("git_ctx", lambda: git_ctx.run(signals, index, pack, self.config))

        # Trim to budget
        budget.trim(pack, max_tokens)

        elapsed = time.time() - t0
        logger.info(f"Compiled pack: {pack.token_count} tokens in {elapsed:.2f}s")

        # Cache result
        if self._cache is not None:
            self._cache[cache_key] = pack

        return pack

    def _run_safe(self, name: str, fn):
        """Run a strategy with error handling and timeout."""
        try:
            fn()
        except Exception as e:
            logger.error(f"Strategy '{name}' failed: {e}", exc_info=True)

    def _cache_key(self, signals: TaskSignals) -> str:
        raw = f"{signals.raw_prompt}|{signals.task_type}|{','.join(signals.symbols)}|{','.join(signals.file_mentions)}"
        return hashlib.md5(raw.encode()).hexdigest()
