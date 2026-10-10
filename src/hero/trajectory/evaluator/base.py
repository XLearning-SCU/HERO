"""Base interface for trajectory evaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from hero.trajectory.models import EvalResult, Trajectory


class BaseTrajectoryEvaluator(ABC):
    """Strategy plugin that scores a built trajectory.

    Evaluators are instantiated per-request with ``**config`` from the
    :class:`~hero.trajectory.models.StrategySpec` and receive runtime context
    (session_id, task_id, session_dir, artifacts_dir, agent_result, runtime,
    runtime_spec) as keyword arguments in :meth:`evaluate`.
    """

    def __init__(self, **config: Any) -> None:  # noqa: B027
        ...

    @abstractmethod
    async def evaluate(
        self,
        trajectory: Trajectory,
        **runtime: Any,
    ) -> EvalResult:
        ...
