"""Hierarchical Policy Optimization: fixed capability threshold."""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import torch

from .constants import (
    MIN_COMPLETED_PER_GROUP,
    NONNEGATIVE_BOUND,
    NONPOSITIVE_BOUND,
    OUTCOME_ADVANTAGE_BOUND_KEY,
)
from .efficiency_credit_assignment import (
    coarse_grained_trajectory_advantages,
    fine_grained_turn_advantages,
)
from .efficiency_gating import fixed_capability_gate as efficiency_gate
from .group_statistics import _standardize
from .resolution_first_clipping import resolution_first_clipping
from .rollout_metadata import (
    _apply_efficiency_spans,
    _assign_trajectory_ids,
    _attach_fine_spans,
    _clear_efficiency_spans,
    _collect_trajectories,
    _context_parallel_slice,
    _train_metadata,
)

logger = logging.getLogger(__name__)

DEFAULT_CAPABILITY_THRESHOLD = 0.6
CAPABILITY_THRESHOLD = float(os.environ.get("GATE_THRESHOLD", str(DEFAULT_CAPABILITY_THRESHOLD)))
if not math.isfinite(CAPABILITY_THRESHOLD) or not 0.0 <= CAPABILITY_THRESHOLD <= 1.0:
    raise ValueError("GATE_THRESHOLD must be a finite number in [0, 1]")
EFFICIENCY_COEFFICIENT = float(os.environ.get("EFFICIENCY_COEFFICIENT", "0.5"))
if not math.isfinite(EFFICIENCY_COEFFICIENT) or EFFICIENCY_COEFFICIENT < 0:
    raise ValueError("EFFICIENCY_COEFFICIENT must be a finite non-negative number")
COARSE_SUCCESS_ONLY = os.environ.get("COARSE_SUCCESS_ONLY", "0") == "1"


def compute_hierarchical_advantages(
    args: Any,
    samples: list[Any],
    *,
    experiment: str,
    use_gate: bool,
    use_coarse: bool,
    use_fine: bool,
    coarse_unified: bool = False,
) -> tuple[list[float], list[float]]:
    """Compute binary raw rewards and configured total advantages."""
    if not samples:
        return [], []
    estimator = str(getattr(args, "advantage_estimator", "grpo"))
    if estimator not in {"grpo", "gspo", "cispo"}:
        raise ValueError(f"{experiment} requires GRPO/GSPO/CISPO, got {estimator!r}")

    trajectories, group_keys = _collect_trajectories(samples, args=args)
    _assign_trajectory_ids(samples, trajectories)
    raw_rewards = [0.0] * len(samples)
    scalar_advantages = [0.0] * len(samples)
    for trajectory in trajectories.values():
        reward = trajectory.task_reward
        for sample_index in trajectory.sample_indices:
            raw_rewards[sample_index] = reward
            _clear_efficiency_spans(samples[sample_index])

    efficiency_enabled = use_gate or use_coarse or use_fine
    efficiency_step = [
        trajectory.resolved
        for trajectory in trajectories.values()
        if efficiency_enabled and trajectory.efficiency_eligible
    ]
    step_success_rate = (
        sum(efficiency_step) / len(efficiency_step) if efficiency_step else math.nan
    )

    active_groups = 0
    coarse_trajectories = 0
    fine_turns = 0
    outcome_eligible_step = 0
    efficiency_eligible_step = 0
    coarse_eligible_step = 0
    fine_eligible_step = 0
    for group_idx, keys in group_keys.items():
        group = [trajectories[key] for key in keys]
        outcome_eligible = [trajectory for trajectory in group if trajectory.outcome_eligible]
        coarse_eligible = [trajectory for trajectory in group if trajectory.coarse_eligible]
        fine_eligible = [trajectory for trajectory in group if trajectory.fine_eligible]
        completed = [
            trajectory
            for trajectory in group
            if efficiency_enabled and trajectory.efficiency_eligible
        ]
        outcome_eligible_step += len(outcome_eligible)
        efficiency_eligible_step += len(completed)
        coarse_eligible_step += len(coarse_eligible)
        fine_eligible_step += len(fine_eligible)
        has_success = any(trajectory.resolved for trajectory in completed)
        has_failure = any(not trajectory.resolved for trajectory in completed)
        if has_success and has_failure:
            for trajectory in completed:
                bound = NONNEGATIVE_BOUND if trajectory.resolved else NONPOSITIVE_BOUND
                for sample_index in trajectory.sample_indices:
                    _train_metadata(samples[sample_index])[OUTCOME_ADVANTAGE_BOUND_KEY] = bound
        task_eligible = [trajectory for trajectory in group if trajectory.task_eligible]
        resolution_advantages = _standardize(
            {trajectory.key: trajectory.task_reward for trajectory in task_eligible}
        )
        coarse: dict[tuple[Any, Any], float] = {}
        fine: dict[tuple[tuple[Any, Any], str], float] = {}
        gate_weight = 0.0

        if len(completed) >= MIN_COMPLETED_PER_GROUP:
            group_success_rate = sum(trajectory.resolved for trajectory in completed) / len(
                completed
            )
            gate_weight = (
                efficiency_gate(group_success_rate, threshold=CAPABILITY_THRESHOLD)
                if use_gate
                else 1.0
            )
            if gate_weight > 0:
                if use_coarse:
                    coarse = coarse_grained_trajectory_advantages(
                        completed,
                        success_only=COARSE_SUCCESS_ONLY,
                        unified=coarse_unified,
                    )
                if use_fine:
                    fine = fine_grained_turn_advantages(completed)
                active_groups += int(bool(coarse or fine))
                coarse_trajectories += len(coarse)
                fine_turns += len(fine)

        for trajectory in group:
            total = resolution_advantages.get(trajectory.key, 0.0)
            if use_coarse:
                total += EFFICIENCY_COEFFICIENT * gate_weight * coarse.get(trajectory.key, 0.0)
            for sample_index in trajectory.sample_indices:
                scalar_advantages[sample_index] = total
                if use_fine:
                    _attach_fine_spans(
                        samples[sample_index],
                        trajectory.key,
                        fine,
                        scale=EFFICIENCY_COEFFICIENT * gate_weight,
                    )

        logger.debug(
            "%s group=%s trajectories=%d completed=%d outcome_eligible=%d "
            "coarse_eligible=%d fine_eligible=%d step_success_rate=%.6f "
            "gate_threshold=%.6f",
            experiment,
            group_idx,
            len(group),
            len(completed),
            len(outcome_eligible),
            len(coarse_eligible),
            len(fine_eligible),
            step_success_rate,
            CAPABILITY_THRESHOLD,
        )

    logger.info(
        "%s: trajectories=%d completed=%d outcome_eligible=%d coarse_eligible=%d "
        "fine_eligible=%d step_success_rate=%.6f gate_threshold=%.6f "
        "active_groups=%d "
        "coarse_trajectories=%d fine_turns=%d",
        experiment,
        len(trajectories),
        efficiency_eligible_step,
        outcome_eligible_step,
        coarse_eligible_step,
        fine_eligible_step,
        step_success_rate,
        CAPABILITY_THRESHOLD,
        active_groups,
        coarse_trajectories,
        fine_turns,
    )
    return raw_rewards, scalar_advantages


def build_token_level_advantages(args: Any, rollout_data: dict[str, Any]) -> None:
    """Expand scalar totals and add Fine advantage on assistant-turn spans."""
    estimator = str(getattr(args, "advantage_estimator", "grpo"))
    if estimator not in {"grpo", "gspo", "cispo"}:
        raise ValueError("token-efficiency advantage requires GRPO/GSPO/CISPO")

    rewards = rollout_data.get("rewards") or []
    kl = rollout_data.get("kl") or []
    response_lengths = rollout_data.get("response_lengths") or []
    total_lengths = rollout_data.get("total_lengths") or []
    metadata = rollout_data.get("metadata") or [None] * len(rewards)
    if not (
        len(rewards) == len(kl) == len(response_lengths) == len(total_lengths) == len(metadata)
    ):
        raise ValueError("token-efficiency advantage inputs have inconsistent lengths")

    advantages: list[torch.Tensor] = []
    for reward, template, response_length, total_length, item_metadata in zip(
        rewards,
        kl,
        response_lengths,
        total_lengths,
        metadata,
        strict=True,
    ):
        response_length = int(response_length)
        full = torch.full(
            (response_length,),
            float(reward),
            dtype=torch.float32,
            device=template.device,
        )
        full = _apply_efficiency_spans(full, item_metadata)
        full = resolution_first_clipping(full, item_metadata)
        advantages.append(
            _context_parallel_slice(
                full,
                total_length=int(total_length),
                response_length=response_length,
                expected_local_length=int(template.numel()),
            )
        )
    rollout_data["advantages"] = advantages
    rollout_data["returns"] = [advantage.clone() for advantage in advantages]
