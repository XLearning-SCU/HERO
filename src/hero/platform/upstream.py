"""HTTP clients for rollout and gateway services with simple retries."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class UpstreamClient:
    """Async HTTP wrapper with bounded timeouts and graceful failure."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 8.0,
        connect_timeout_seconds: float = 3.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(timeout_seconds, connect=connect_timeout_seconds)
        self._client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._lock:
            if self._client is None:
                self._client = httpx.AsyncClient(timeout=self._timeout)

    async def close(self) -> None:
        async with self._lock:
            if self._client is not None:
                await self._client.aclose()
                self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("UpstreamClient not started")
        return self._client

    async def get_json(self, path: str, **kwargs) -> Any:
        url = f"{self.base_url}{path}"
        resp = await self.client.get(url, **kwargs)
        resp.raise_for_status()
        return resp.json()

    async def safe_get_json(self, path: str, *, default: Any = None, **kwargs) -> Any:
        try:
            return await self.get_json(path, **kwargs)
        except (httpx.HTTPError, ValueError) as exc:
            logger.debug("safe_get_json(%s%s) failed: %s", self.base_url, path, exc)
            return default

    async def post_json(self, path: str, payload: dict[str, Any], **kwargs) -> Any:
        url = f"{self.base_url}{path}"
        resp = await self.client.post(url, json=payload, **kwargs)
        resp.raise_for_status()
        return resp.json()

    async def request_json(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        **kwargs,
    ) -> tuple[int, Any]:
        url = f"{self.base_url}{path}"
        if method.upper() == "GET":
            resp = await self.client.get(url, **kwargs)
        elif method.upper() == "POST":
            resp = await self.client.post(url, json=payload, **kwargs)
        elif method.upper() == "DELETE":
            resp = await self.client.delete(url, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}")
        body: Any
        try:
            body = resp.json()
        except ValueError:
            body = resp.text
        return resp.status_code, body

    @asynccontextmanager
    async def stream_sse(self, path: str):
        url = f"{self.base_url}{path}"
        async with self.client.stream(
            "GET", url, headers={"Accept": "text/event-stream"}
        ) as response:
            response.raise_for_status()
            yield response


__all__ = ["UpstreamClient"]
