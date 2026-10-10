"""Population normalization and linear percentile interpolation."""

from __future__ import annotations

import math
from typing import Any

from .constants import (
    EPSILON,
)


def _standardize(values: dict[Any, float]) -> dict[Any, float]:
    if not values:
        return {}
    mean = sum(values.values()) / len(values)
    variance = sum((value - mean) ** 2 for value in values.values()) / len(values)
    if variance <= 0:
        return {key: 0.0 for key in values}
    denominator = math.sqrt(variance) + EPSILON
    return {key: (value - mean) / denominator for key, value in values.items()}


def _linear_quantile(values: list[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
