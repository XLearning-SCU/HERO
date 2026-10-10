"""HERO hierarchical advantages followed by per-trace identities."""

from __future__ import annotations

import logging
from typing import Any

from run_rl.HERO.reward import (
    post_process_rewards as _base_post_process_rewards,
)

logger = logging.getLogger(__name__)


def post_process_rewards(
    args: Any,
    samples: list[Any],
) -> tuple[list[float], list[float]]:
    """Compute the original method, then schedule and weight by trace."""
    raw_rewards, advantages = _base_post_process_rewards(args, samples)
    for trace_id, sample in enumerate(samples):
        sample.rollout_id = trace_id
        sample.trajectory_id = trace_id
    logger.info(
        "HERO: base reward -> trace scheduling/loss (%d traces)",
        len(samples),
    )
    return raw_rewards, advantages
