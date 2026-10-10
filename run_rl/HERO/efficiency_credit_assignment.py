"""Multi-granularity Efficiency Credit Assignment (draft_v2.tex, Section 3.3).

Coarse-grained trajectory advantage uses resolution-calibrated token budgets;
fine-grained turn advantage penalizes low-entropy turns.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from .constants import (
    ENTROPY_PERCENTILE,
    EPSILON,
    MIN_COMPLETED_PER_GROUP,
    MIN_OUTCOME_SUBGROUP,
)
from .group_statistics import _linear_quantile, _standardize
from .rollout_metadata import TrajectoryInfo


def coarse_grained_trajectory_advantages(
    completed: list[TrajectoryInfo],
    *,
    success_only: bool = False,
    unified: bool = False,
) -> dict[tuple[Any, Any], float]:
    if success_only and unified:
        raise ValueError("success_only and unified are mutually exclusive")

    if unified:
        if len(completed) < MIN_COMPLETED_PER_GROUP:
            return {}
        budget = float(statistics.median(item.total_tokens for item in completed))
        if budget <= 0:
            return {}
        utilities = {
            item.key: 1.0 - math.tanh(max(0.0, float(item.total_tokens) / budget - 1.0))
            for item in completed
        }
        return _standardize(utilities)

    utilities: dict[tuple[Any, Any], float] = {}
    outcomes = (True,) if success_only else (False, True)
    for resolved in outcomes:
        subgroup = [item for item in completed if item.resolved is resolved]
        if len(subgroup) < MIN_OUTCOME_SUBGROUP:
            continue
        budget = float(statistics.median(item.total_tokens for item in subgroup))
        if budget <= 0:
            continue
        for item in subgroup:
            excess = max(0.0, float(item.total_tokens) / budget - 1.0)
            utilities[item.key] = 1.0 - math.tanh(excess)
    return _standardize(utilities)


def fine_grained_turn_advantages(
    completed: list[TrajectoryInfo],
) -> dict[tuple[tuple[Any, Any], str], float]:
    all_entropies = [
        entropy for trajectory in completed for entropy in trajectory.turn_entropy.values()
    ]
    if not all_entropies:
        return {}
    threshold = _linear_quantile(all_entropies, ENTROPY_PERCENTILE)
    if threshold <= EPSILON:
        return {}

    penalties: dict[tuple[tuple[Any, Any], str], float] = {}
    for item in completed:
        for completion_id, entropy in item.turn_entropy.items():
            deficit = max(0.0, 1.0 - entropy / threshold)
            penalty = -math.tanh(deficit)
            # A sparse mapping makes zero-penalty turns explicit in behavior:
            # they create no span and cannot receive a positive Fine signal.
            if penalty < 0.0:
                penalties[(item.key, completion_id)] = penalty
    return penalties
