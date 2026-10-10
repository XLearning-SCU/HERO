"""Trajectory eligibility, entropy reconstruction, trace identities and token spans."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from run_rl._shared.eligibility import (
    sample_has_trainable_tokens,
    sample_is_failed,
    sample_reward,
)

from .constants import (
    EXPECTED_ENTROPY_TOP_K,
    OUTCOME_ADVANTAGE_BOUND_KEY,
)


@dataclass
class TrajectoryInfo:
    key: tuple[Any, Any]
    sample_indices: list[int] = field(default_factory=list)
    valid_rewards: list[float] = field(default_factory=list)
    failed: bool = False
    resolved: bool = False
    resolved_label_present: bool = False
    normal_completion: bool = False
    token_metrics_complete: bool = False
    total_tokens: float | None = None
    # A trajectory may fan out into several dynamic-history traces (main agent
    # and sub-agents). Completion IDs are trajectory-global; local turn numbers
    # restart in each trace and therefore cannot be used for deduplication.
    turn_entropy: dict[str, float] = field(default_factory=dict)
    expected_turn_count: int | None = None

    @property
    def task_eligible(self) -> bool:
        return not self.failed and bool(self.valid_rewards)

    @property
    def task_reward(self) -> float:
        return sum(self.valid_rewards) / len(self.valid_rewards) if self.task_eligible else 0.0

    @property
    def outcome_eligible(self) -> bool:
        """Whether normal completion and an outcome label are available."""
        return bool(self.task_eligible and self.normal_completion and self.resolved_label_present)

    @property
    def coarse_eligible(self) -> bool:
        """Whether complete token-usage data is available for Coarse."""
        return bool(
            self.outcome_eligible and self.token_metrics_complete and self.total_tokens is not None
        )

    @property
    def fine_eligible(self) -> bool:
        """Whether complete per-turn entropy is available for Fine."""
        return bool(
            self.outcome_eligible
            and self.expected_turn_count is not None
            and self.expected_turn_count > 0
            and len(self.turn_entropy) == self.expected_turn_count
        )

    @property
    def efficiency_eligible(self) -> bool:
        """Whether this trajectory may enter the shared efficiency pool."""
        return self.coarse_eligible and self.fine_eligible

    @property
    def completed(self) -> bool:
        """Method-spec alias for a complete, efficiency-eligible trajectory."""
        return self.efficiency_eligible


def _collect_trajectories(
    samples: list[Any],
    *,
    args: Any = None,
) -> tuple[dict[tuple[Any, Any], TrajectoryInfo], dict[Any, list[tuple[Any, Any]]]]:
    trajectories: dict[tuple[Any, Any], TrajectoryInfo] = {}
    group_keys: dict[Any, list[tuple[Any, Any]]] = {}
    for sample_index, sample in enumerate(samples):
        group_idx, key = _trajectory_key(sample, sample_index)
        if key not in trajectories:
            info = _base_trajectory_info(sample, key)
            trajectories[key] = info
            group_keys.setdefault(group_idx, []).append(key)
        info = trajectories[key]
        info.sample_indices.append(sample_index)
        if sample_is_failed(sample):
            info.failed = True
        elif sample_has_trainable_tokens(sample):
            info.valid_rewards.append(sample_reward(args, sample))
        info.turn_entropy.update(_turn_entropies(sample))
    return trajectories, group_keys


def _assign_trajectory_ids(
    samples: list[Any], trajectories: dict[tuple[Any, Any], TrajectoryInfo]
) -> None:
    """Assign one v0.3-compatible partition for scheduling and loss."""
    for trajectory_id, trajectory in enumerate(trajectories.values()):
        for sample_index in trajectory.sample_indices:
            samples[sample_index].rollout_id = trajectory_id
            samples[sample_index].trajectory_id = trajectory_id


def _base_trajectory_info(sample: Any, key: tuple[Any, Any]) -> TrajectoryInfo:
    hero = _hero_metadata(sample)
    trajectory = hero.get("trajectory_metadata", {})
    trajectory = trajectory if isinstance(trajectory, dict) else {}
    evaluation = trajectory.get("evaluation", {})
    evaluation = evaluation if isinstance(evaluation, dict) else {}
    report = evaluation.get("report", {})
    report = report if isinstance(report, dict) else {}

    report_resolved = report.get("resolved")
    outcome_reward = evaluation.get("outcome_reward")
    if isinstance(report_resolved, bool):
        resolved = report_resolved
        label_present = True
    elif (
        isinstance(outcome_reward, (int, float))
        and math.isfinite(float(outcome_reward))
        and float(outcome_reward) in {0.0, 1.0}
    ):
        resolved = float(outcome_reward) == 1.0
        label_present = True
    else:
        resolved = False
        label_present = False

    usage = trajectory.get("token_usage", {})
    usage = usage if isinstance(usage, dict) else {}
    total_tokens = _finite_nonnegative(usage.get("total_tokens"))
    expected_turn_count = _positive_int(usage.get("turn_count"))
    trajectory_status = str(hero.get("trajectory_status") or "").upper()
    normal_completion = trajectory_status == "COMPLETED"
    if not normal_completion:
        # Incomplete trajectories may still receive the separately defined Task
        # term, but never enter outcome-dependent efficiency statistics.
        resolved = False
    return TrajectoryInfo(
        key=key,
        resolved=resolved,
        resolved_label_present=label_present,
        normal_completion=normal_completion,
        token_metrics_complete=usage.get("metrics_missing") is False,
        total_tokens=total_tokens,
        expected_turn_count=expected_turn_count,
    )


def _turn_entropies(sample: Any) -> dict[str, float]:
    """Return per-turn mean entropy of renormalized top-10 probabilities."""
    trace = _hero_metadata(sample).get("trace_metadata", {})
    trace = trace if isinstance(trace, dict) else {}
    entropy = trace.get("generation_entropy", {})
    entropy = entropy if isinstance(entropy, dict) else {}
    if (
        entropy.get("complete") is not True
        or int(entropy.get("requested_top_k") or 0) != EXPECTED_ENTROPY_TOP_K
    ):
        return {}

    contributions = entropy.get("topk_entropy_contribution")
    tail_masses = entropy.get("tail_probability_mass")
    available_counts = entropy.get("available_top_k_counts")
    if (
        not isinstance(contributions, list)
        or not isinstance(tail_masses, list)
        or not isinstance(available_counts, list)
    ):
        return {}
    if (
        len(contributions) != len(tail_masses)
        or len(contributions) != len(available_counts)
        or not contributions
        or any(_positive_int(count) != EXPECTED_ENTROPY_TOP_K for count in available_counts)
    ):
        return {}

    token_entropies: list[float | None] = []
    for contribution, tail_mass in zip(contributions, tail_masses, strict=True):
        try:
            contribution_value = float(contribution)
            top10_mass = 1.0 - float(tail_mass)
        except (TypeError, ValueError):
            token_entropies.append(None)
            continue
        if (
            not math.isfinite(contribution_value)
            or not math.isfinite(top10_mass)
            or top10_mass <= 0
        ):
            token_entropies.append(None)
            continue
        # If q_i=p_i/S, then H(q)=(-sum p_i log p_i)/S + log(S).
        normalized = contribution_value / top10_mass + math.log(top10_mass)
        token_entropies.append(max(0.0, normalized))

    counts = entropy.get("turn_generated_token_counts")
    if not isinstance(counts, list):
        counts = [len(token_entropies)]
    spans = _train_metadata(sample).get("hero_turn_token_spans", [])
    if not isinstance(spans, list) or len(spans) != len(counts):
        return {}

    output: dict[str, float] = {}
    offset = 0
    for span, raw_count in zip(spans, counts, strict=True):
        if not isinstance(span, dict) or not span.get("completion_id"):
            return {}
        count = _positive_int(raw_count)
        if count is None or offset + count > len(token_entropies):
            return {}
        values = token_entropies[offset : offset + count]
        if any(value is None for value in values):
            return {}
        output[str(span["completion_id"])] = sum(float(value) for value in values) / count
        offset += count
    if offset != len(token_entropies):
        return {}
    return output


def _attach_fine_spans(
    sample: Any,
    trajectory_key: tuple[Any, Any],
    fine: dict[tuple[tuple[Any, Any], str], float],
    *,
    scale: float,
) -> None:
    train_metadata = _train_metadata(sample)
    spans = train_metadata.get("hero_turn_token_spans", [])
    if not isinstance(spans, list):
        return
    attached: list[dict[str, float | int | str]] = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        try:
            turn = int(span["turn"])
            start = int(span["start"])
            end = int(span["end"])
        except (KeyError, TypeError, ValueError):
            continue
        completion_id = str(span.get("completion_id") or "")
        value = scale * fine.get((trajectory_key, completion_id), 0.0)
        if value == 0.0:
            continue
        attached.append(
            {
                "turn": turn,
                "completion_id": completion_id,
                "start": start,
                "end": end,
                "advantage": value,
                "source": "fine_entropy",
            }
        )
    train_metadata["hero_efficiency_advantage_spans"] = attached


def _clear_efficiency_spans(sample: Any) -> None:
    train_metadata = _train_metadata(sample)
    train_metadata["hero_efficiency_advantage_spans"] = []
    train_metadata.pop(OUTCOME_ADVANTAGE_BOUND_KEY, None)


def _train_metadata(sample: Any) -> dict[str, Any]:
    metadata = getattr(sample, "train_metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        sample.train_metadata = metadata
    return metadata


def _apply_efficiency_spans(advantage: torch.Tensor, metadata: Any) -> torch.Tensor:
    output = advantage.clone()
    if not isinstance(metadata, dict):
        return output
    spans = metadata.get("hero_efficiency_advantage_spans", [])
    if not isinstance(spans, list):
        return output
    for span in spans:
        if not isinstance(span, dict):
            continue
        try:
            start = int(span["start"])
            end = int(span["end"])
            value = float(span["advantage"])
        except (KeyError, TypeError, ValueError):
            continue
        if start < 0 or end <= start or end > output.numel():
            raise ValueError(
                f"efficiency span [{start}, {end}) outside response length {output.numel()}"
            )
        output[start:end] += value
    return output


def _context_parallel_slice(
    full_response: torch.Tensor,
    *,
    total_length: int,
    response_length: int,
    expected_local_length: int,
) -> torch.Tensor:
    if expected_local_length == response_length:
        return full_response
    from slime.backends.megatron_utils.cp_utils import (  # type: ignore[import-not-found]
        get_logits_and_tokens_offset_with_cp,
    )

    prompt_length = total_length - response_length
    _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(total_length, response_length)
    parts: list[torch.Tensor] = []
    for start, end in token_offsets:
        response_start = max(0, start - prompt_length)
        response_end = max(0, end - prompt_length)
        if response_end > response_start:
            parts.append(full_response[response_start:response_end])
    local = torch.cat(parts) if parts else full_response.new_empty((0,))
    if local.numel() != expected_local_length:
        raise ValueError(
            "context-parallel advantage length mismatch: "
            f"expected {expected_local_length}, got {local.numel()}"
        )
    return local


def _hero_metadata(sample: Any) -> dict[str, Any]:
    metadata = getattr(sample, "metadata", None)
    hero = metadata.get("hero", {}) if isinstance(metadata, dict) else {}
    return hero if isinstance(hero, dict) else {}


def _trajectory_key(sample: Any, position: int) -> tuple[Any, tuple[Any, Any]]:
    group_idx = _key_value(getattr(sample, "group_index", None), -1)
    trajectory_idx = getattr(sample, "trajectory_id", None)
    if trajectory_idx is None:
        trajectory_idx = getattr(sample, "group_id", None)
    if trajectory_idx is None:
        trajectory_idx = getattr(sample, "index", None)
    return group_idx, (group_idx, _key_value(trajectory_idx, position))


def _key_value(value: Any, default: Any) -> Any:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _finite_nonnegative(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
