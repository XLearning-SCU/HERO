"""Training eligibility matching ``slime_bridge.reward_post_process``.

Method experiments may change how eligible rewards are normalized or shaped,
but must not change which HERO/Slime samples enter the Task reward baseline.
"""

from __future__ import annotations

from typing import Any


def sample_is_failed(sample: Any) -> bool:
    """Match the baseline's FAILED/ABORTED trajectory exclusion."""
    status = getattr(sample, "status", None)
    name = getattr(status, "name", None) or str(status).rsplit(".", 1)[-1]
    return name.upper() in {"FAILED", "ABORTED"}


def sample_has_trainable_tokens(sample: Any) -> bool:
    """Match the baseline's remove_sample/loss-mask eligibility check."""
    if bool(getattr(sample, "remove_sample", False)):
        return False
    loss_mask = getattr(sample, "loss_mask", None)
    if loss_mask is None:
        return int(getattr(sample, "response_length", 0) or 0) > 0
    return any(int(value) != 0 for value in loss_mask)


def sample_reward(args: Any, sample: Any) -> float:
    """Read the same reward value consumed by the baseline post-processor."""
    return float(sample.get_reward_value(args))
