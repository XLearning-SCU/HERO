"""In-process event pub/sub shared by rollout and gateway services.

This is a lightweight asyncio fanout used to drive `/events` SSE endpoints.
Subscribers receive a bounded asyncio.Queue and lose old events if they
fall behind. The same primitive is reused by rollout and gateway pubsubs.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_MAX_QUEUE_SIZE = 256


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_event(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


class EventBus:
    """Minimal asyncio-friendly publish/subscribe channel."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._lock = asyncio.Lock()

    async def publish(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        event = {
            "type": event_type,
            "ts": _utcnow_iso(),
            "data": payload or {},
        }
        async with self._lock:
            subscribers = list(self._subscribers)
        for queue in subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Drop oldest then put.
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    logger.debug("event subscriber permanently full; dropping event %s", event_type)

    def publish_threadsafe(
        self,
        loop: asyncio.AbstractEventLoop,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Schedule a publish from sync code running on another thread."""
        try:
            asyncio.run_coroutine_threadsafe(self.publish(event_type, payload), loop)
        except RuntimeError:
            # Event loop closed or not running; treat as best-effort.
            pass

    @asynccontextmanager
    async def subscribe(self) -> AsyncIterator[asyncio.Queue[dict[str, Any]]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_MAX_QUEUE_SIZE)
        async with self._lock:
            self._subscribers.add(queue)
        try:
            yield queue
        finally:
            async with self._lock:
                self._subscribers.discard(queue)

    async def stream_events(
        self,
        *,
        heartbeat_seconds: float = 15.0,
    ) -> AsyncIterator[str]:
        """Yield SSE-formatted text frames. Includes periodic pings."""
        async with self.subscribe() as queue:
            # Initial hello frame so the client knows the channel is live.
            yield _format_event({"type": "hello", "ts": _utcnow_iso(), "data": {}})
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=heartbeat_seconds)
                except asyncio.TimeoutError:
                    yield _format_event({"type": "ping", "ts": _utcnow_iso(), "data": {}})
                    continue
                yield _format_event(event)


SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


__all__ = ["EventBus", "SSE_HEADERS"]
