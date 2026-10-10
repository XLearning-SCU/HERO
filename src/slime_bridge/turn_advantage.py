"""Token-local GRPO advantages for HERO failure-efficiency credit.

The reward post-processor keeps the ordinary scalar solve advantage and stores
sparse response-token penalty spans in ``Sample.train_metadata``.  This custom
Slime advantage hook reconstructs the normal GRPO broadcast, then adds those
penalties only to the indicated assistant-token suffix.
"""

from __future__ import annotations

from typing import Any

import torch


def apply_efficiency_spans(
    advantage: torch.Tensor,
    metadata: Any,
) -> torch.Tensor:
    """Return a full-response advantage with validated sparse spans applied."""
    output = advantage.clone()
    if not isinstance(metadata, dict):
        return output
    spans = metadata.get("hero_efficiency_advantage_spans", [])
    if not isinstance(spans, list):
        return output
    length = output.numel()
    for span in spans:
        if not isinstance(span, dict):
            continue
        try:
            start = int(span["start"])
            end = int(span["end"])
            value = float(span["advantage"])
        except (KeyError, TypeError, ValueError):
            continue
        if start < 0 or end <= start or end > length:
            raise ValueError(
                f"efficiency advantage span [{start}, {end}) outside response length {length}"
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
    _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(
        total_length, response_length
    )
    parts: list[torch.Tensor] = []
    for start, end in token_offsets:
        response_start = max(0, start - prompt_length)
        response_end = max(0, end - prompt_length)
        if response_end > response_start:
            parts.append(full_response[response_start:response_end])
    local = torch.cat(parts) if parts else full_response.new_empty((0,))
    if local.numel() != expected_local_length:
        raise ValueError(
            "context-parallel efficiency advantage length mismatch: "
            f"expected {expected_local_length}, got {local.numel()}"
        )
    return local


def compute_turn_level_advantages(args: Any, rollout_data: dict[str, Any]) -> None:
    """Slime ``--custom-advantage-function-path`` implementation.

    This matches GRPO's ordinary scalar reward broadcast when no efficiency
    spans are present.  It therefore remains safe to install for mixed batches
    as long as the selected estimator is GRPO/GSPO/CISPO.
    """
    estimator = str(getattr(args, "advantage_estimator", "grpo"))
    if estimator not in {"grpo", "gspo", "cispo"}:
        raise ValueError(
            "turn-level HERO efficiency credit currently requires GRPO/GSPO/CISPO"
        )
    rewards = rollout_data.get("rewards") or []
    kl = rollout_data.get("kl") or []
    response_lengths = rollout_data.get("response_lengths") or []
    total_lengths = rollout_data.get("total_lengths") or []
    metadata = rollout_data.get("metadata") or [None] * len(rewards)
    if not (
        len(rewards)
        == len(kl)
        == len(response_lengths)
        == len(total_lengths)
        == len(metadata)
    ):
        raise ValueError("turn-level advantage inputs have inconsistent batch lengths")

    advantages: list[torch.Tensor] = []
    for reward, local_template, response_length, total_length, item_metadata in zip(
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
            device=local_template.device,
        )
        full = apply_efficiency_spans(full, item_metadata)
        local = _context_parallel_slice(
            full,
            total_length=int(total_length),
            response_length=response_length,
            expected_local_length=int(local_template.numel()),
        )
        advantages.append(local)
    rollout_data["advantages"] = advantages
    rollout_data["returns"] = [advantage.clone() for advantage in advantages]

