"""Trajectory-level rollout metrics for the HERO Slime bridge."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def log_rollout_data(
    rollout_id: int,
    args: Any,
    samples: list[Any],
    extra_metrics: dict[str, float],
    rollout_time: float,
) -> bool:
    """Add trajectory-equal raw-reward metrics and keep Slime's default log.

    HERO expands one agent session (one rollout trajectory) into one or more
    trace samples.  Slime's built-in ``rollout/raw_reward`` averages that flat
    sample list, so a trajectory with many traces is over-represented.  Group
    samples by their stable ``(group_index, rollout_id)`` trajectory key first.

    For SWE-Gym, the evaluator broadcasts its binary resolved reward to every
    trace.  Consequently both metrics below equal the fraction of trajectories
    that resolved:

    * ``rollout/raw_reward_trajectory_mean``: mean raw outcome reward;
    * ``rollout/resolved_trajectory_rate``: explicit binary success rate.

    The first name intentionally matches Dressage.  For non-binary evaluators,
    the raw-reward metric remains meaningful while ``resolved_trajectory_rate``
    is the fraction with a trajectory reward of at least one.
    """
    del rollout_id, rollout_time
    by_trajectory: dict[tuple[object, object], list[float]] = defaultdict(list)
    for position, sample in enumerate(samples):
        by_trajectory[_trajectory_key(sample, position)].append(
            float(sample.get_reward_value(args))
        )

    if not by_trajectory:
        return False

    trajectory_rewards = [sum(values) / len(values) for values in by_trajectory.values()]
    extra_metrics["rollout/raw_reward_trajectory_mean"] = (
        sum(trajectory_rewards) / len(trajectory_rewards)
    )
    extra_metrics["rollout/resolved_trajectory_rate"] = (
        sum(reward >= 1.0 for reward in trajectory_rewards) / len(trajectory_rewards)
    )
    return False


def _trajectory_key(sample: Any, position: int) -> tuple[object, object]:
    """Return the key assigned by ``session_result_to_samples`` for one session."""
    group_index = _hashable(getattr(sample, "group_index", None), "group", position)
    trajectory_id = getattr(sample, "trajectory_id", None)
    if trajectory_id is None:
        trajectory_id = getattr(sample, "rollout_id", None)
    if trajectory_id is None:
        trajectory_id = getattr(sample, "group_id", None)
    if trajectory_id is None:
        trajectory_id = getattr(sample, "index", None)
    return group_index, _hashable(trajectory_id, "sample", position)


def _hashable(value: Any, prefix: str, fallback: int) -> object:
    try:
        hash(value)
    except TypeError:
        return prefix, repr(value)
    return value if value is not None else (prefix, fallback)
