"""Built-in evaluator that scores based on whether the session completed."""

from __future__ import annotations

from typing import Any

from hero.trajectory.evaluator.base import BaseTrajectoryEvaluator
from hero.trajectory.models import EvalResult, Trajectory


class SessionCompletedEvaluator(BaseTrajectoryEvaluator):
    """Return reward 1.0 for COMPLETED trajectories, 0.0 otherwise."""

    async def evaluate(self, trajectory: Trajectory, **runtime: Any) -> EvalResult:
        reward = 1.0 if trajectory.status == "COMPLETED" else 0.0
        return EvalResult(outcome_reward=reward)
