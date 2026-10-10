"""HERO multi-granularity efficiency credit hook."""

from typing import Any

from run_rl.HERO.hierarchical_policy_optimization import (
    build_token_level_advantages,
    compute_hierarchical_advantages,
)

EXPERIMENT = "HERO"


def post_process_rewards(args: Any, samples: list[Any]) -> tuple[list[float], list[float]]:
    return compute_hierarchical_advantages(
        args,
        samples,
        experiment=EXPERIMENT,
        use_gate=True,
        use_coarse=True,
        use_fine=True,
    )


def compute_advantages(args: Any, rollout_data: dict[str, Any]) -> None:
    build_token_level_advantages(args, rollout_data)
