"""Reward adapter for Slime custom reward model hooks."""

from __future__ import annotations

from typing import Any


async def reward_func(args: Any, sample_or_samples: Any, **kwargs: Any) -> Any:
    """Read the reward already embedded in HERO-converted Slime samples."""
    del kwargs
    reward_key = str(getattr(args, "hero_reward_key", getattr(args, "reward_key", "score")))
    if isinstance(sample_or_samples, list):
        return [{reward_key: _extract_reward(sample, reward_key)} for sample in sample_or_samples]
    return {reward_key: _extract_reward(sample_or_samples, reward_key)}


def _extract_reward(sample: Any, reward_key: str) -> float:
    reward = getattr(sample, "reward", None)
    if isinstance(reward, dict):
        if reward_key in reward:
            return float(reward[reward_key])
        if "score" in reward:
            return float(reward["score"])
        for value in reward.values():
            if isinstance(value, (int, float)):
                return float(value)
        return 0.0
    if isinstance(reward, (int, float)):
        return float(reward)
    return 0.0
