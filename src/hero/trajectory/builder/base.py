"""Base interface for trajectory builders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from hero.trajectory.models import CompletionSession, Trajectory


class BaseTrajectoryBuilder(ABC):
    """Strategy plugin that converts a completion session into a trajectory.

    Builders are instantiated per-request with ``**config`` from the
    :class:`~hero.trajectory.models.StrategySpec`.
    """

    def __init__(self, **config: Any) -> None:  # noqa: B027
        ...

    @abstractmethod
    async def build(self, session: CompletionSession) -> Trajectory:
        """Build a trajectory from a session with ordered completions."""
