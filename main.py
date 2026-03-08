"""RLM Gateway — entry point with rich terminal UI."""

import logging
import os
import re
import sys
import threading
import time
from pathlib import Path

import uvicorn
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text

console = Console()

# Route all rlm logging through rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_path=False, markup=True)],
)
# Suppress noisy loggers
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger("rlm")


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _format_elapsed(seconds: float) -> str:
    """Format seconds into human readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs:02d}s"


def _check_repo_path(config: dict):
    """Validate repo_path exists and is a git repo. Exit with rich error if not."""
    repo_path = config.get("repo_path", "")

    if not repo_path or not Path(repo_path).exists():
        panel = Panel(
            f'  repo_path [bold]"{repo_path}"[/bold] does not exist.\n'
            "\n"
            "  Please update repo_path in config.yaml before starting.",
            title="[bold red]Configuration Error[/bold red]",
            border_style="red",
            padding=(1, 1),
        )
        console.print(panel)
        sys.exit(1)

    if not (Path(repo_path) / ".git").is_dir():
        panel = Panel(
            f'  repo_path [bold]"{repo_path}"[/bold] is not a git repo.\n'
            "\n"
            "  The gateway requires a git repository to build index metadata.",
            title="[bold red]Configuration Error[/bold red]",
            border_style="red",
            padding=(1, 1),
        )
        console.print(panel)
        sys.exit(1)


def _print_banner(config: dict):
    """Print startup banner."""
    ds = config["downstream"]
    server_cfg = config.get("server", {})
    host = server_cfg.get("host", "127.0.0.1")
    port = server_cfg.get("port", 9787)
    repo_path = config.get("repo_path", "")

    banner = Text()
    banner.append("\n  RLM Gateway\n", style="bold yellow")
    banner.append(f"  Downstream:  ", style="dim")
    banner.append(f"{ds['model']}", style="bold cyan")
    banner.append(f" via {ds['base_url']}\n", style="dim")
    banner.append(f"  Port:        ", style="dim")
    banner.append(f"{port}\n", style="bold cyan")
    banner.append(f"  Indexing:    ", style="dim")
    banner.append(f"{repo_path}\n", style="bold white")

    console.print(banner)


def _print_ready_panel(config: dict, index, elapsed: float):
    """Print the completion summary panel."""
    ds = config["downstream"]
    server_cfg = config.get("server", {})
    host = server_cfg.get("host", "127.0.0.1")
    port = server_cfg.get("port", 9787)

    lines = [
        f"  Files indexed:     [bold]{index.file_count:,}[/bold]",
        f"  Symbols found:     [bold]{index.symbol_count:,}[/bold]",
        f"  Chunks embedded:   [bold]{index.chunk_count:,}[/bold]",
        f"  Time taken:        [bold]{_format_elapsed(elapsed)}[/bold]",
        "",
        f"  Listening on:      [bold cyan]http://{host}:{port}[/bold cyan]",
        f"  Downstream:        [bold cyan]Moonshot API → {ds['model']}[/bold cyan]",
    ]

    panel = Panel(
        "\n".join(lines),
        title="[bold green]RLM Gateway Ready[/bold green]",
        border_style="green",
        padding=(1, 1),
    )
    console.print(panel)


class _EmbeddingProgressHandler(logging.Handler):
    """Intercept embedding progress logs to drive the rich progress bar."""

    def __init__(self):
        super().__init__()
        self.total_files: int = 0
        self.files_done: int = 0
        self.total_chunks: int = 0
        self.loaded_from_cache: bool = False
        self._progress_re = re.compile(r"Embedding files: (\d+)/(\d+)")
        self._start_re = re.compile(r"Embedding (\d+) chunks across (\d+) files")
        self._cache_re = re.compile(r"Embeddings loaded from cache: (\d+) chunks")

    def emit(self, record):
        msg = record.getMessage()

        m = self._cache_re.search(msg)
        if m:
            self.loaded_from_cache = True
            return

        m = self._start_re.search(msg)
        if m:
            self.total_chunks = int(m.group(1))
            self.total_files = int(m.group(2))
            return

        m = self._progress_re.search(msg)
        if m:
            self.files_done = int(m.group(1))
            self.total_files = int(m.group(2))


def _build_index_with_progress(index, config: dict):
    """Build the index with a rich live progress display."""
    t0 = time.time()

    # Attach a handler to capture embedding progress
    emb_handler = _EmbeddingProgressHandler()
    emb_logger = logging.getLogger("rlm.embeddings")
    emb_logger.addHandler(emb_handler)

    # Track phases via a shared state dict
    phase = {"current": "Scanning files", "done": False, "error": None}

    # Intercept indexer log messages to detect phase transitions
    class _PhaseHandler(logging.Handler):
        def emit(self, record):
            msg = record.getMessage()
            if "AST indexed" in msg:
                phase["current"] = "Building dep graph"
            elif "Found" in msg and "indexable files" in msg:
                phase["current"] = "AST indexing"

    phase_handler = _PhaseHandler()
    idx_logger = logging.getLogger("rlm.indexer")
    idx_logger.addHandler(phase_handler)

    # Run the build in a thread
    def do_build():
        try:
            index.build()
        except Exception as e:
            phase["error"] = str(e)
        finally:
            phase["done"] = True

    build_thread = threading.Thread(target=do_build, daemon=True)
    build_thread.start()

    # Drive the progress display from the main thread
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TextColumn("files"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Scanning files", total=None)

        embedding_started = False

        while not phase["done"]:
            current = phase["current"]

            if emb_handler.loaded_from_cache and not embedding_started:
                # Embeddings loaded from cache — skip embedding progress
                progress.update(task, description="Loading cached embeddings", total=1, completed=1)
                embedding_started = True
            elif emb_handler.total_files > 0 and not embedding_started:
                # Switch to embedding progress bar
                progress.update(
                    task,
                    description="Embedding chunks",
                    total=emb_handler.total_files,
                    completed=0,
                )
                embedding_started = True
            elif embedding_started and not emb_handler.loaded_from_cache:
                progress.update(
                    task,
                    completed=emb_handler.files_done,
                    total=emb_handler.total_files,
                )
            else:
                # Pre-embedding phases — indeterminate spinner
                progress.update(task, description=current)

            time.sleep(0.15)

    build_thread.join()

    # Clean up handlers
    emb_logger.removeHandler(emb_handler)
    idx_logger.removeHandler(phase_handler)

    elapsed = time.time() - t0

    if phase["error"]:
        console.print(f"  [bold red]Index build failed:[/bold red] {phase['error']}")
        return elapsed

    # Print ready panel
    _print_ready_panel(config, index, elapsed)

    if config.get("indexer", {}).get("watch", True):
        index.start_watcher()
        console.print("  [dim]File watcher active[/dim]\n")

    return elapsed


def main():
    config = load_config()

    # Safety check — validate repo_path before anything else
    _check_repo_path(config)

    _print_banner(config)

    # Import here to avoid circular imports
    from gateway.forwarder import Forwarder
    from gateway.server import app, init
    from indexer.indexer import RepoIndex
    from compiler.compiler import Compiler
    from repl.pool import REPLPool

    # Setup forwarder — env vars override config for secrets
    ds = config["downstream"]
    api_key = os.environ.get("MOONSHOT_API_KEY") or ds.get("api_key", "")
    forwarder = Forwarder(
        base_url=ds["base_url"],
        api_key=api_key,
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

    # Setup compiler
    compiler = Compiler(config)

    # Wire everything together (pass console for request logging)
    init(forwarder, compiler, index, config, console)

    # Build index with rich progress (blocks until done)
    _build_index_with_progress(index, config)

    # Start server
    server_cfg = config.get("server", {})
    host = server_cfg.get("host", "127.0.0.1")
    port = server_cfg.get("port", 9787)

    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
