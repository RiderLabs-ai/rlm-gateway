"""FastAPI app with OpenAI-compatible endpoints."""

import logging
import time
import uuid
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


def init(forwarder: Forwarder, compiler, index, config: dict):
    global _forwarder, _compiler, _index, _config
    _forwarder = forwarder
    _compiler = compiler
    _index = index
    _config = config


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
    if _compiler is not None and _index is not None and _index.ready:
        try:
            pack = _compiler.compile(signals, _index)
            if pack and pack.token_count > 0:
                pack_xml = pack.to_xml()
                enriched_messages = _prepend_context(messages, pack_xml)
                logger.info(
                    f"Injected context pack: {pack.token_count} tokens, "
                    f"task_type={signals.task_type}"
                )
        except Exception as e:
            logger.error(f"Compiler error, forwarding unmodified: {e}", exc_info=True)
    elif _index is not None and not _index.ready:
        logger.warning("Index still building, forwarding unmodified")

    # Step 3: Forward to Moonshot API
    forward_body = {**body, "messages": enriched_messages}

    try:
        if body.get("stream", False):
            stream_iter = await _forwarder.forward(forward_body)
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
            return JSONResponse(content=result)

    except DownstreamUnavailableError:
        return JSONResponse(
            status_code=502,
            content={"error": {"message": "Moonshot API is unreachable", "type": "upstream_error"}},
        )
    except DownstreamError as e:
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
