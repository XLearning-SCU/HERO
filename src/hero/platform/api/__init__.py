"""API routers for the platform service."""

from hero.platform.api.events import router as events_router
from hero.platform.api.sessions import router as sessions_router
from hero.platform.api.tasks import router as tasks_router
from hero.platform.api.topology import router as topology_router

__all__ = [
    "events_router",
    "sessions_router",
    "tasks_router",
    "topology_router",
]
