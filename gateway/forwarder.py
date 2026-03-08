"""Async streaming proxy to downstream Moonshot API."""

import logging
from typing import Any, AsyncIterator

import httpx

logger = logging.getLogger("rlm.forwarder")


class Forwarder:
    def __init__(self, base_url: str, api_key: str, model: str, timeout_ms: int = 120000):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        timeout_s = timeout_ms / 1000.0
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=10.0))

    async def close(self):
        await self.client.aclose()

    async def forward(
        self, request_body: dict[str, Any]
    ) -> httpx.Response | AsyncIterator[bytes]:
        """Forward request to Moonshot API. Returns response for streaming or non-streaming."""
        # Override model name with downstream config
        request_body = {**request_body, "model": self.model}

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        url = f"{self.base_url}/chat/completions"
        is_streaming = request_body.get("stream", False)

        if is_streaming:
            return self._stream(url, headers, request_body)
        else:
            return await self._non_stream(url, headers, request_body)

    async def _non_stream(
        self, url: str, headers: dict, body: dict
    ) -> dict[str, Any]:
        try:
            resp = await self.client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            return resp.json()
        except httpx.ConnectError:
            raise DownstreamUnavailableError("Moonshot API is unreachable")
        except httpx.HTTPStatusError as e:
            raise DownstreamError(f"Moonshot API returned {e.response.status_code}: {e.response.text}")

    async def _stream(
        self, url: str, headers: dict, body: dict
    ) -> AsyncIterator[bytes]:
        try:
            req = self.client.build_request("POST", url, headers=headers, json=body)
            resp = await self.client.send(req, stream=True)
            resp.raise_for_status()
            return self._iter_sse(resp)
        except httpx.ConnectError:
            raise DownstreamUnavailableError("Moonshot API is unreachable")
        except httpx.HTTPStatusError as e:
            body_text = await e.response.aread()
            raise DownstreamError(f"Moonshot API returned {e.response.status_code}: {body_text.decode()}")

    async def _iter_sse(self, resp: httpx.Response) -> AsyncIterator[bytes]:
        async for line in resp.aiter_lines():
            if line:
                yield (line + "\n").encode()
            else:
                yield b"\n"
        await resp.aclose()


class DownstreamUnavailableError(Exception):
    pass


class DownstreamError(Exception):
    pass
