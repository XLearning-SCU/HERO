"""Dynamic-trace reward post-processor for Slime.

Registered via Slime's ``--custom-reward-post-process-path`` hook.  A HERO
trajectory can fan out into a variable number of trace samples, and each trace
keeps its own reward.  The exchangeable unit is still the trajectory, so this
processor computes per-trace advantages against a leave-one-trajectory-out
baseline built from other trajectories in the same prompt group.

Adapter contract:
    All Slime samples produced from the same HERO ``SessionResult`` share
    the same ``Sample.rollout_id`` and ``Sample.trajectory_id`` partition.
    Current Slime uses the former for step grouping and the latter to average
    all trace contributions as one gradient unit, reproducing v0.3 group_id.
"""

from __future__ import annotations

import logging
import statistics
from typing import Any

logger = logging.getLogger(__name__)


def post_process_rewards(
    args: Any,
    samples: list[Any],
) -> tuple[list[float], list[float]]:
    """Slime reward-post-process hook. Returns (raw_rewards, rewards)."""
    raw_rewards = [float(sample.get_reward_value(args)) for sample in samples]
    turn_level = _turn_level_failure_efficiency_enabled(args)
    if not turn_level:
        raw_rewards = _shape_failure_token_efficiency_rewards(
            args, samples, raw_rewards
        )

    if not getattr(args, "rewards_normalization", True):
        rewards = list(raw_rewards)
        _attach_turn_level_failure_efficiency(args, samples, rewards)
        return raw_rewards, rewards

    estimator = getattr(args, "advantage_estimator", None)
    if estimator not in ("grpo", "gspo", "reinforce_plus_plus_baseline"):
        rewards = list(raw_rewards)
        _attach_turn_level_failure_efficiency(args, samples, rewards)
        return raw_rewards, rewards

    std_norm = estimator in ("grpo", "gspo") and bool(
        getattr(args, "grpo_std_normalization", False)
    )

    traj_sample_indices: dict[tuple[Any, Any], list[int]] = {}
    traj_valid_rewards: dict[tuple[Any, Any], list[float]] = {}
    traj_failed: dict[tuple[Any, Any], bool] = {}
    group_keys: dict[Any, list[tuple[Any, Any]]] = {}
    key_by_sample: list[tuple[Any, Any]] = []

    for i, sample in enumerate(samples):
        group_idx, key = _trajectory_key(sample, i)
        key_by_sample.append(key)
        if key not in traj_sample_indices:
            traj_sample_indices[key] = []
            traj_valid_rewards[key] = []
            traj_failed[key] = False
            group_keys.setdefault(group_idx, []).append(key)
        traj_sample_indices[key].append(i)
        if _is_failed_trajectory(sample):
            traj_failed[key] = True
        elif _has_trainable_tokens(sample):
            traj_valid_rewards[key].append(raw_rewards[i])

    normalized_by_sample = [0.0] * len(samples)
    for keys in group_keys.values():
        valid_keys = [
            key for key in keys
            if not traj_failed[key] and traj_valid_rewards[key]
        ]
        traj_mean = {
            key: sum(traj_valid_rewards[key]) / len(traj_valid_rewards[key])
            for key in valid_keys
        }

        for key in keys:
            if key not in traj_mean:
                continue
            other_means = [
                traj_mean[other_key]
                for other_key in valid_keys
                if other_key != key
            ]
            baseline = sum(other_means) / len(other_means) if other_means else 0.0
            scale = _loo_scale(other_means) if std_norm else 1.0
            for sample_index in traj_sample_indices[key]:
                normalized_by_sample[sample_index] = (
                    raw_rewards[sample_index] - baseline
                ) / scale

    _attach_turn_level_failure_efficiency(args, samples, normalized_by_sample)
    return raw_rewards, normalized_by_sample


def _failure_efficiency_config(args: Any) -> dict[str, Any] | None:
    config = getattr(args, "hero_failure_token_efficiency_reward", None)
    if config is True:
        config = {"enabled": True}
    if not isinstance(config, dict) or not bool(config.get("enabled", False)):
        return None
    return config


def _turn_level_failure_efficiency_enabled(args: Any) -> bool:
    config = _failure_efficiency_config(args)
    return bool(config and str(config.get("credit_assignment", "trajectory")) == "turn")


def _shape_failure_token_efficiency_rewards(
    args: Any,
    samples: list[Any],
    raw_rewards: list[float],
) -> list[float]:
    """Optionally penalize outcome-gated, token-inefficient failures.

    The budget is the median input-token cost of successful trajectories in
    the same prompt group. A configured fallback may be used for groups with
    no successes; the safe default is to leave those groups unchanged.

    Two independently configurable branches are supported:

    * no-patch failures above ``no_patch_budget_ratio * budget``;
    * all failures above ``extreme_failure_budget_ratio * budget``.

    The second branch is disabled by default.  It is intended for the extreme
    tail of failures that did produce a usable patch, not ordinary patch
    failures.  Overlapping branches use the stronger penalty rather than
    double-counting the same trajectory.
    """
    config = _failure_efficiency_config(args)
    if config is None:
        return list(raw_rewards)

    fallback = float(config.get("fallback_success_input_tokens", 0) or 0)
    minimum = float(config.get("minimum_input_tokens", 0) or 0)
    no_patch_ratio = float(config.get("no_patch_budget_ratio", 1.0))
    extreme_ratio = float(config.get("extreme_failure_budget_ratio", 0) or 0)
    max_abs_penalty = float(config.get("max_abs_penalty", 1.0))
    if (
        fallback < 0
        or minimum < 0
        or no_patch_ratio <= 0
        or extreme_ratio < 0
        or not 0 < max_abs_penalty <= 1
    ):
        raise ValueError(
            "invalid failure token-efficiency reward budget ratio or penalty cap"
        )

    trajectory_indices: dict[tuple[Any, Any], list[int]] = {}
    group_trajectories: dict[Any, list[tuple[Any, Any]]] = {}
    info_by_trajectory: dict[tuple[Any, Any], dict[str, Any]] = {}
    for i, sample in enumerate(samples):
        group_idx, key = _trajectory_key(sample, i)
        trajectory_indices.setdefault(key, []).append(i)
        if key not in info_by_trajectory:
            info_by_trajectory[key] = _token_efficiency_info(sample, raw_rewards[i])
            group_trajectories.setdefault(group_idx, []).append(key)

    shaped = list(raw_rewards)
    for keys in group_trajectories.values():
        success_costs = [
            float(info_by_trajectory[key]["input_tokens"])
            for key in keys
            if info_by_trajectory[key]["resolved"]
            and info_by_trajectory[key]["input_tokens"] is not None
        ]
        if success_costs:
            budget = max(float(statistics.median(success_costs)), minimum)
        elif fallback > 0:
            budget = max(fallback, minimum)
        else:
            continue

        for key in keys:
            info = info_by_trajectory[key]
            cost = info["input_tokens"]
            if info["resolved"] or cost is None or float(cost) <= 0:
                continue
            cost = float(cost)
            candidates: list[float] = []
            if not info["patch_usable"]:
                threshold = no_patch_ratio * budget
                if cost > threshold:
                    candidates.append((cost - threshold) / cost)
            if extreme_ratio > 0:
                threshold = extreme_ratio * budget
                if cost > threshold:
                    candidates.append((cost - threshold) / cost)
            if not candidates:
                continue
            penalty = -min(max_abs_penalty, max(candidates))
            for sample_index in trajectory_indices[key]:
                shaped[sample_index] = penalty
    return shaped


def _attach_turn_level_failure_efficiency(
    args: Any,
    samples: list[Any],
    solve_advantages: list[float],
) -> None:
    """Attach sparse response-token penalty spans for a custom advantage hook.

    This path deliberately leaves scalar rewards untouched.  The companion
    ``slime_bridge.turn_advantage`` hook adds these penalties only to assistant
    tokens in the expensive suffix. Missing or incomplete Git observations
    suppress the no-patch branch; the outcome-gated extreme-turn branch does
    not depend on Git.
    """
    config = _failure_efficiency_config(args)
    if config is None or str(config.get("credit_assignment", "trajectory")) != "turn":
        return
    no_patch_ratio = float(config.get("no_patch_budget_ratio", 0.5))
    extreme_ratio = float(config.get("extreme_failure_budget_ratio", 2.0))
    max_abs_penalty = float(config.get("max_abs_penalty", 0.5))
    fallback_turns = float(config.get("fallback_success_turns", 0) or 0)
    if (
        no_patch_ratio <= 0
        or extreme_ratio < 0
        or not 0 < max_abs_penalty <= 1
        or fallback_turns < 0
    ):
        raise ValueError("invalid turn-level failure token-efficiency configuration")

    trajectory_samples: dict[tuple[Any, Any], list[int]] = {}
    group_trajectories: dict[Any, list[tuple[Any, Any]]] = {}
    info_by_trajectory: dict[tuple[Any, Any], dict[str, Any]] = {}
    for i, sample in enumerate(samples):
        group_idx, key = _trajectory_key(sample, i)
        trajectory_samples.setdefault(key, []).append(i)
        if key not in info_by_trajectory:
            info_by_trajectory[key] = _token_efficiency_info(sample, solve_advantages[i])
            group_trajectories.setdefault(group_idx, []).append(key)

    global_success_turns = [
        float(info["turn_count"])
        for info in info_by_trajectory.values()
        if info["resolved"] and info["turn_count"] is not None
    ]
    global_success_budget = (
        float(statistics.median(global_success_turns))
        if global_success_turns
        else fallback_turns
    )

    for keys in group_trajectories.values():
        success_turns = [
            float(info_by_trajectory[key]["turn_count"])
            for key in keys
            if info_by_trajectory[key]["resolved"]
            and info_by_trajectory[key]["turn_count"] is not None
        ]
        if success_turns:
            budget = float(statistics.median(success_turns))
        elif global_success_budget > 0:
            budget = global_success_budget
        else:
            continue
        if budget <= 0:
            continue
        for key in keys:
            info = info_by_trajectory[key]
            if info["resolved"]:
                continue
            for sample_index in trajectory_samples[key]:
                sample = samples[sample_index]
                solve_cap = abs(float(solve_advantages[sample_index]))
                cap = min(max_abs_penalty, solve_cap)
                spans = _turn_penalty_spans(
                    sample,
                    budget=budget,
                    no_patch_ratio=no_patch_ratio,
                    extreme_ratio=extreme_ratio,
                    cap=cap,
                )
                metadata = getattr(sample, "train_metadata", None)
                if not isinstance(metadata, dict):
                    metadata = {}
                    sample.train_metadata = metadata
                metadata["hero_efficiency_advantage_spans"] = spans


def _turn_penalty_spans(
    sample: Any,
    *,
    budget: float,
    no_patch_ratio: float,
    extreme_ratio: float,
    cap: float,
) -> list[dict[str, float | int | str]]:
    if cap <= 0:
        return []
    metadata = getattr(sample, "metadata", None)
    hero = metadata.get("hero", {}) if isinstance(metadata, dict) else {}
    trajectory = hero.get("trajectory_metadata", {}) if isinstance(hero, dict) else {}
    evaluation = trajectory.get("evaluation", {}) if isinstance(trajectory, dict) else {}
    usage = trajectory.get("token_usage", {}) if isinstance(trajectory, dict) else {}
    progress = evaluation.get("git_progress", {}) if isinstance(evaluation, dict) else {}
    checkpoints = [
        point
        for point in progress.get("turn_checkpoints", [])
        if isinstance(point, dict) and point.get("turn")
    ] if isinstance(progress, dict) else []
    checkpoint_by_turn = {int(point["turn"]): point for point in checkpoints}
    valid_by_turn = _valid_git_prefixes(sample, checkpoint_by_turn)
    turn_usage = [
        item for item in usage.get("turn_usage", []) if isinstance(item, dict)
    ] if isinstance(usage, dict) else []
    train_metadata = getattr(sample, "train_metadata", None)
    turn_spans = (
        train_metadata.get("hero_turn_token_spans", [])
        if isinstance(train_metadata, dict)
        else []
    )
    output: list[dict[str, float | int | str]] = []
    checkpoint_turns = sorted(checkpoint_by_turn)
    for span in turn_spans:
        if not isinstance(span, dict):
            continue
        turn = int(span.get("turn") or 0)
        start = int(span.get("start") or 0)
        end = int(span.get("end") or 0)
        if turn < 1 or end <= start or turn > len(turn_usage):
            continue
        progress_ratio = turn / budget
        candidates: list[tuple[float, str]] = []
        # A checkpoint at T certifies the block ending at T. Do not forward-fill
        # an old no-patch state into later turns: the agent may have edited the
        # repo after that observation. Post-hoc credit can safely back-fill the
        # next checkpoint over its preceding collection interval instead.
        observed_at = min((value for value in checkpoint_turns if value >= turn), default=None)
        if observed_at is not None and valid_by_turn.get(observed_at, False):
            point = checkpoint_by_turn[observed_at]
            if not bool(point.get("changed_from_head")) and progress_ratio > no_patch_ratio:
                candidates.append(
                    ((progress_ratio - no_patch_ratio) / progress_ratio, "no_patch_stall")
                )
        if extreme_ratio > 0 and progress_ratio > extreme_ratio:
            candidates.append(
                ((progress_ratio - extreme_ratio) / progress_ratio, "extreme_failure_tail")
            )
        if not candidates:
            continue
        severity, reason = max(candidates, key=lambda item: item[0])
        output.append(
            {
                "start": start,
                "end": end,
                "advantage": -min(cap, severity),
                "turn": turn,
                "reason": reason,
            }
        )
    return output


def _valid_git_prefixes(
    sample: Any,
    checkpoint_by_turn: dict[int, dict[str, Any]],
) -> dict[int, bool]:
    metadata = getattr(sample, "metadata", None)
    hero = metadata.get("hero", {}) if isinstance(metadata, dict) else {}
    trajectory = hero.get("trajectory_metadata", {}) if isinstance(hero, dict) else {}
    evaluation = trajectory.get("evaluation", {}) if isinstance(trajectory, dict) else {}
    progress = evaluation.get("git_progress", {}) if isinstance(evaluation, dict) else {}
    if not isinstance(progress, dict) or not bool(progress.get("baseline_fixed", False)):
        return {checkpoint_turn: False for checkpoint_turn in checkpoint_by_turn}
    records = progress.get("records", []) if isinstance(progress, dict) else []
    recorded = {
        str(record.get("tool_use_id"))
        for record in records
        if isinstance(record, dict) and record.get("tool_use_id")
    }
    trace_debug = hero.get("trace_debug", {}) if isinstance(hero, dict) else {}
    messages = trace_debug.get("response_messages", []) if isinstance(trace_debug, dict) else []
    results = {
        str(message.get("tool_call_id")): str(message.get("content") or "")
        for message in messages
        if isinstance(message, dict)
        and message.get("role") == "tool"
        and message.get("tool_call_id")
    }
    successful_calls: list[tuple[int, str]] = []
    turn = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        turn += 1
        for call in message.get("tool_calls") or []:
            tool_id = str(call.get("id") or "")
            if tool_id and not results.get(tool_id, "").startswith("[Tool Error]"):
                successful_calls.append((turn, tool_id))
    return {
        checkpoint_turn: all(
            tool_id in recorded
            for call_turn, tool_id in successful_calls
            if call_turn <= checkpoint_turn
        )
        for checkpoint_turn in checkpoint_by_turn
    }


def _token_efficiency_info(sample: Any, raw_reward: float) -> dict[str, Any]:
    metadata = getattr(sample, "metadata", None)
    hero = metadata.get("hero", {}) if isinstance(metadata, dict) else {}
    trajectory = (
        hero.get("trajectory_metadata", {}) if isinstance(hero, dict) else {}
    )
    if not isinstance(trajectory, dict):
        trajectory = {}
    evaluation = trajectory.get("evaluation", {})
    if not isinstance(evaluation, dict):
        evaluation = {}
    report = evaluation.get("report", {})
    if not isinstance(report, dict):
        report = {}
    usage = trajectory.get("token_usage", {})
    if not isinstance(usage, dict):
        usage = {}

    outcome = evaluation.get("outcome_reward", raw_reward)
    resolved = bool(report.get("resolved", False)) or bool(
        isinstance(outcome, (int, float)) and float(outcome) >= 1.0
    )
    if "patch_usable" in evaluation:
        patch_usable = bool(evaluation.get("patch_usable"))
    else:
        model_patch = evaluation.get("model_patch")
        patch_usable = bool(
            isinstance(model_patch, str)
            and model_patch.strip()
            and not report.get("failed_apply_patch", False)
        )
    try:
        input_tokens = float(usage["total_input_tokens"])
    except (KeyError, TypeError, ValueError):
        input_tokens = None
    try:
        turn_count = int(usage["turn_count"])
    except (KeyError, TypeError, ValueError):
        turn_usage = usage.get("turn_usage")
        turn_count = len(turn_usage) if isinstance(turn_usage, list) and turn_usage else None
    return {
        "resolved": resolved,
        "patch_usable": patch_usable,
        "input_tokens": input_tokens,
        "turn_count": turn_count,
    }


def _trajectory_key(sample: Any, sample_position: int) -> tuple[Any, tuple[Any, Any]]:
    group_idx = _key_value(getattr(sample, "group_index", None), -1)
    traj_idx = getattr(sample, "trajectory_id", None)
    if traj_idx is None:
        traj_idx = getattr(sample, "rollout_id", None)
    if traj_idx is None:
        # Compatibility with HERO checkpoints/samples produced for old Slime.
        traj_idx = getattr(sample, "group_id", None)
    if traj_idx is None:
        traj_idx = getattr(sample, "index", None)
    return group_idx, (group_idx, _key_value(traj_idx, sample_position))


def _key_value(value: Any, default: Any) -> Any:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _loo_scale(other_means: list[float]) -> float:
    if len(other_means) <= 1:
        return 1.0
    scale = statistics.stdev(other_means)
    if scale <= 1e-6:
        return 1.0
    return scale + 1e-6


def _has_trainable_tokens(sample: Any) -> bool:
    if bool(getattr(sample, "remove_sample", False)):
        return False
    loss_mask = getattr(sample, "loss_mask", None)
    if loss_mask is None:
        return int(getattr(sample, "response_length", 0) or 0) > 0
    return any(int(value) != 0 for value in loss_mask)


def _is_failed_trajectory(sample: Any) -> bool:
    """True if the sample's status marks it as agent ERROR or TIMEOUT."""
    status = getattr(sample, "status", None)
    name = getattr(status, "name", None) or str(status).rsplit(".", 1)[-1]
    return name.upper() in ("FAILED", "ABORTED")
