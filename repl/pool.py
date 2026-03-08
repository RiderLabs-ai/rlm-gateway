"""ProcessPoolExecutor worker pool for compiler strategies."""

import logging
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger("rlm.pool")


class REPLPool:
    def __init__(self, size: int = 4, timeout_ms: int = 2000):
        self.size = size
        self.timeout_ms = timeout_ms
        self._executor: ProcessPoolExecutor | None = None

    def start(self):
        self._executor = ProcessPoolExecutor(max_workers=self.size)
        logger.info(f"REPL pool started with {self.size} workers")

    def shutdown(self):
        if self._executor:
            self._executor.shutdown(wait=False)
            logger.info("REPL pool shut down")

    @property
    def executor(self) -> ProcessPoolExecutor | None:
        return self._executor
