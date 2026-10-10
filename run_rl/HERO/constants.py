"""Numerical guards and serialized metadata keys; preserve checkpoint compatibility."""

EPSILON = 1e-6
MIN_COMPLETED_PER_GROUP = 4
MIN_OUTCOME_SUBGROUP = 3
ENTROPY_PERCENTILE = 0.2
EXPECTED_ENTROPY_TOP_K = 10
OUTCOME_ADVANTAGE_BOUND_KEY = "hero_outcome_advantage_bound"
NONNEGATIVE_BOUND = "nonnegative"
NONPOSITIVE_BOUND = "nonpositive"
