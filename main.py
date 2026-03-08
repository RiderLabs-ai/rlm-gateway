"""RLM Gateway — entry point."""

import logging
import sys
import threading
import time
from pathlib import Path

import uvicorn
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="[rlm] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("rlm")


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    logger.info("Loading config from config.yaml")
    config = load_config()

    # Import here to avoid circular imports
    from gateway.forwarder import Forwarder
    from gateway.server import app, init
    from indexer.indexer import RepoIndex
    from compiler.compiler import Compiler
    from repl.pool import REPLPool

    # Setup forwarder
    ds = config["downstream"]
    forwarder = Forwarder(
        base_url=ds["base_url"],
        api_key=ds["api_key"],
        model=ds["model"],
        timeout_ms=ds.get("timeout_ms", 120000),
    )

    # Setup REPL pool
    pool_cfg = config.get("repl_pool", {})
    pool = REPLPool(size=pool_cfg.get("size", 4), timeout_ms=pool_cfg.get("timeout_ms", 2000))
    pool.start()

    # Build repo index
    repo_path = config.get("repo_path", "")
    index = RepoIndex(config)

    if repo_path and Path(repo_path).is_dir():
        logger.info(f"Building repo index for {repo_path}...")
        t0 = time.time()

        def build_index():
            try:
                index.build()
                elapsed = time.time() - t0
                logger.info(
                    f"Indexed {index.file_count:,} files, "
                    f"{index.symbol_count:,} symbols, "
                    f"{index.chunk_count:,} chunks ({elapsed:.1f}s)"
                )
                if config.get("indexer", {}).get("watch", True):
                    index.start_watcher()
                    logger.info("File watcher active")
            except Exception as e:
                logger.error(f"Index build failed: {e}", exc_info=True)

        index_thread = threading.Thread(target=build_index, daemon=True)
        index_thread.start()
    else:
        logger.warning(f"repo_path not set or does not exist: {repo_path}")

    # Setup compiler
    compiler = Compiler(config)

    # Wire everything together
    init(forwarder, compiler, index, config)

    # Start server
    server_cfg = config.get("server", {})
    host = server_cfg.get("host", "127.0.0.1")
    port = server_cfg.get("port", 8787)

    logger.info(f"Gateway ready → http://{host}:{port}")
    logger.info(f"Downstream: Moonshot API → {ds['model']}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
