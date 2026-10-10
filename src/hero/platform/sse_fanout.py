"""Subscribe to upstream SSE streams and re-publish to platform EventBus."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from hero.platform.events import EventBus

logger = logging.getLogger(__name__)


def _parse_sse_lines(raw_lines: list[str]) -> dict[str, Any] | None:
    """Convert a single SSE message (raw lines, no separators) into an event dict."""
    data_parts: list[str] = []
    for line in raw_lines:
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_parts.append(line[5:].lstrip())
    if not data_parts:
        return None
    raw = "\n".join(data_parts)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


class SseFanout:
    """Manages one upstream SSE connection that re-publishes to a local bus."""

    def __init__(
        self,
        *,
        name: str,
        url: str,
        bus: EventBus,
    ) -> None:
        self.name = name
        self.url = url.rstrip("/")
        self.bus = bus
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        self._stop = asyncio.Event()

    async def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream(
                        "GET",
                        f"{self.url}/events",
                        headers={"Accept": "text/event-stream"},
                    ) as response:
                        response.raise_for_status()
                        backoff = 1.0
                        buffer: list[str] = []
                        async for line in response.aiter_lines():
                            if self._stop.is_set():
                                return
                            if line == "":
                                event = _parse_sse_lines(buffer)
                                buffer.clear()
                                if event is None:
                                    continue
                                event_type = event.get("type") or "message"
                                data = event.get("data") if isinstance(event.get("data"), dict) else {}
                                # Add upstream source for browser-side disambiguation.
                                data.setdefault("source", self.name)
                                await self.bus.publish(event_type, data)
                            else:
                                buffer.append(line)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.debug("SSE upstream %s disconnected: %s", self.name, exc)
            # Reconnect with exponential backoff
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=backoff)
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2.0, 30.0)


__all__ = ["SseFanout"]
