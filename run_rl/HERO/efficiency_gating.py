"""Efficiency gate for a fixed capability threshold."""

def fixed_capability_gate(group_rate: float, *, threshold: float) -> float:
    """Return the stateless Gate weight for an explicit fixed threshold."""
    if threshold == 1.0:
        return 1.0 if group_rate >= 1.0 else 0.0
    return max(0.0, (group_rate - threshold) / (1.0 - threshold))


