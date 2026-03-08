"""FastAPI app with OpenAI-compatible endpoints."""

import logging
import time
from datetime import datetime
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from gateway.extractor import extract
from gateway.forwarder import DownstreamError, DownstreamUnavailableError, Forwarder

logger = logging.getLogger("rlm.server")

app = FastAPI(title="RLM Gateway")

# These get set by main.py during startup
_forwarder: Forwarder | None = None
_compiler = None
_index = None
_config: dict = {}
_console = None


def init(forwarder: Forwarder, compiler, index, config: dict, console=None):
    global _forwarder, _compiler, _index, _config, _console
    _forwarder = forwarder
    _compiler = compiler
    _index = index
    _config = config
    _console = console


def _log_request(method: str, path: str, task_type: str, tokens: int, status: str):
    """Print a compact per-request log line."""
    ts = datetime.now().strftime("%H:%M:%S")
    tokens_str = f"{tokens:,}" if tokens > 0 else "-"
    if _console:
        _console.print(
            f"  [dim]{ts}[/dim]  {method} {path}  "
            f"[cyan]{task_type}[/cyan]  {tokens_str} tokens  "
            f"[green]{status}[/green]"
        )
    else:
        logger.info(f"{ts}  {method} {path}  {task_type}  {tokens_str} tokens  {status}")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/admin/index/status")
async def index_status():
    if _index is None:
        return {"status": "not_initialized"}
    return {
        "file_count": _index.file_count,
        "last_indexed": _index.last_indexed,
        "status": "ready" if _index.ready else "building",
    }


@app.post("/admin/index/rebuild")
async def index_rebuild():
    if _index is None:
        return JSONResponse(status_code=503, content={"error": "Index not initialized"})
    try:
        _index.rebuild()
        return {"status": "rebuild_started"}
    except Exception as e:
        logger.error(f"Index rebuild failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/admin/preview")
async def preview(request: Request):
    """Run compiler and return the context pack as JSON without forwarding."""
    body = await request.json()
    messages = body.get("messages", [])

    signals = extract(messages)

    if _compiler is None or _index is None or not _index.ready:
        return {
            "signals": {
                "raw_prompt": signals.raw_prompt,
                "task_type": signals.task_type,
                "symbols": signals.symbols,
                "file_mentions": signals.file_mentions,
            },
            "pack": None,
            "note": "Index not ready or compiler not initialized",
        }

    try:
        pack = _compiler.compile(signals, _index)
        _log_request("POST", "/admin/preview", signals.task_type, pack.token_count, "200 OK")
        return {
            "signals": {
                "raw_prompt": signals.raw_prompt,
                "task_type": signals.task_type,
                "symbols": signals.symbols,
                "file_mentions": signals.file_mentions,
            },
            "pack_xml": pack.to_xml(),
            "pack_tokens": pack.token_count,
            "sections": pack.summary(),
        }
    except Exception as e:
        logger.error(f"Compiler error in preview: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Compiler error: {str(e)}"},
        )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Main endpoint — OpenAI-compatible request/response."""
    body = await request.json()
    messages = body.get("messages", [])

    # Step 1: Extract signals
    signals = extract(messages)

    # Step 2: Compile context pack (with fallback)
    enriched_messages = messages
    pack_tokens = 0
    if _compiler is not None and _index is not None and _index.ready:
        try:
            pack = _compiler.compile(signals, _index)
            if pack and pack.token_count > 0:
                pack_tokens = pack.token_count
                pack_xml = pack.to_xml()
                enriched_messages = _prepend_context(messages, pack_xml)
        except Exception as e:
            logger.error(f"Compiler error, forwarding unmodified: {e}", exc_info=True)
    elif _index is not None and not _index.ready:
        logger.warning("Index still building, forwarding unmodified")

    # Step 3: Forward to Moonshot API
    forward_body = {**body, "messages": enriched_messages}

    try:
        if body.get("stream", False):
            stream_iter = await _forwarder.forward(forward_body)
            _log_request("POST", "/v1/chat/completions", signals.task_type, pack_tokens, "200 OK")
            return StreamingResponse(
                stream_iter,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            result = await _forwarder.forward(forward_body)
            _log_request("POST", "/v1/chat/completions", signals.task_type, pack_tokens, "200 OK")
            return JSONResponse(content=result)

    except DownstreamUnavailableError:
        _log_request("POST", "/v1/chat/completions", signals.task_type, pack_tokens, "502 ERR")
        return JSONResponse(
            status_code=502,
            content={"error": {"message": "Moonshot API is unreachable", "type": "upstream_error"}},
        )
    except DownstreamError as e:
        _log_request("POST", "/v1/chat/completions", signals.task_type, pack_tokens, "502 ERR")
        return JSONResponse(
            status_code=502,
            content={"error": {"message": str(e), "type": "upstream_error"}},
        )


def _prepend_context(messages: list[dict], pack_xml: str) -> list[dict]:
    """Prepend context pack XML to the system message."""
    messages = [m.copy() for m in messages]

    # Find existing system message
    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            messages[i] = {
                **msg,
                "content": pack_xml + "\n\n" + msg["content"],
            }
            return messages

    # No system message — prepend one
    messages.insert(0, {"role": "system", "content": pack_xml})
    return messages
