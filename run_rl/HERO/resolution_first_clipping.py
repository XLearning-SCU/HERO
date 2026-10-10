"""Resolution-first clipping Gamma(A), applied after turn-level credit expansion."""

from __future__ import annotations

from typing import Any

import torch

from .constants import (
    NONNEGATIVE_BOUND,
    NONPOSITIVE_BOUND,
    OUTCOME_ADVANTAGE_BOUND_KEY,
)


def resolution_first_clipping(advantage: torch.Tensor, metadata: Any) -> torch.Tensor:
    """Preserve success-over-failure ordering in mixed-outcome groups."""
    if not isinstance(metadata, dict):
        return advantage
    bound = metadata.get(OUTCOME_ADVANTAGE_BOUND_KEY)
    if bound == NONNEGATIVE_BOUND:
        return torch.clamp(advantage, min=0.0)
    if bound == NONPOSITIVE_BOUND:
        return torch.clamp(advantage, max=0.0)
    return advantage
