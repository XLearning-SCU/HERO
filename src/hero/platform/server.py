"""FastAPI app for `hero dashboard`."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI

from hero.platform.api import (
    events_router,
    sessions_router,
    tasks_router,
    topology_router,
)
from hero.platform.config import PlatformConfig
from hero.platform.events import EventBus
from hero.platform.fs_index import FsIndex
from hero.platform.sse_fanout import SseFanout
from hero.platform.upstream import UpstreamClient

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PlatformState:
    config: PlatformConfig
    rollout_client: UpstreamClient
    gateway_clients: dict[str, UpstreamClient]
    fs_index: FsIndex
    event_bus: EventBus
    fanouts: list[SseFanout] = field(default_factory=list)


def create_app(config: PlatformConfig) -> FastAPI:
    rollout_client = UpstreamClient(config.rollout_url)
    gateway_clients: dict[str, UpstreamClient] = {
        node.id: UpstreamClient(node.public_url)
        for node in config.topology.gateway.nodes
    }
    fs_index = FsIndex(config.save_dir)
    event_bus = EventBus()

    state = PlatformState(
        config=config,
        rollout_client=rollout_client,
        gateway_clients=gateway_clients,
        fs_index=fs_index,
        event_bus=event_bus,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await rollout_client.start()
        for client in gateway_clients.values():
            await client.start()
        # initial scan + start polling.
        await asyncio.to_thread(fs_index.scan)

        loop = asyncio.get_running_loop()

        def _on_change(summary):
            event_bus.publish_threadsafe(
                loop,
                "task.updated",
                {"task_id": summary.task_id, "status": summary.status},
            )

        fs_index.add_listener(_on_change)
        await fs_index.start_polling()

        # SSE fanouts: rollout + each gateway.
        fanouts: list[SseFanout] = [
            SseFanout(name="rollout", url=config.rollout_url, bus=event_bus),
        ]
        for node in config.topology.gateway.nodes:
            fanouts.append(
                SseFanout(name=f"gateway:{node.id}", url=node.public_url, bus=event_bus)
            )
        for fanout in fanouts:
            await fanout.start()
        state.fanouts = fanouts
        try:
            yield
        finally:
            for fanout in fanouts:
                await fanout.stop()
            await fs_index.stop_polling()
            await rollout_client.close()
            for client in gateway_clients.values():
                await client.close()

    app = FastAPI(
        title="HERO Platform",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.platform = state

    for router in (
        topology_router,
        tasks_router,
        sessions_router,
        events_router,
    ):
        app.include_router(router)

    _mount_service_info(app)
    return app


def _mount_service_info(app: FastAPI) -> None:
    @app.get("/")
    async def service_info() -> dict[str, Any]:
        return {
            "status": "ok",
            "message": "HERO platform API service running.",
            "api_docs": "/docs",
        }


def serve(
    topology_path: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8090,
    rollout_url: str | None = None,
    save_dir: str | None = None,
    log_level: str = "info",
) -> None:
    import uvicorn

    config = PlatformConfig.from_topology(
        topology_path,
        host=host,
        port=port,
        rollout_url=rollout_url,
        save_dir=save_dir,
    )
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port, log_level=log_level)


__all__ = ["create_app", "serve", "PlatformState"]
