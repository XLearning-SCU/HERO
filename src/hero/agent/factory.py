"""Harness factory with built-in name map and import_path support."""

from __future__ import annotations

from hero.agent.base import BaseHarness
from hero.agent.models import AgentSpec
from hero._imports import import_subclass


def _builtin_harness_map() -> dict[str, type[BaseHarness]]:
    """Lazy import to avoid circular imports at module level."""
    from hero.agent.presets.claude_code import ClaudeCodeHarness
    from hero.agent.presets.codex import CodexHarness
    from hero.agent.presets.gemini_cli import GeminiCliHarness
    from hero.agent.presets.hermes import HermesHarness
    from hero.agent.presets.mini_swe_agent import MiniSweAgentHarness
    from hero.agent.presets.openclaw import OpenClawHarness
    from hero.agent.presets.openhands_sdk import OpenHandsSdkHarness
    from hero.agent.presets.opencode import OpenCodeHarness
    from hero.agent.presets.pi import PiHarness
    from hero.agent.presets.qwen_code import QwenCodeHarness
    from hero.agent.presets.shell import ShellHarness

    return {
        "claude_code": ClaudeCodeHarness,
        "codex": CodexHarness,
        "gemini_cli": GeminiCliHarness,
        "hermes": HermesHarness,
        "mini_swe_agent": MiniSweAgentHarness,
        "openclaw": OpenClawHarness,
        "openhands_sdk": OpenHandsSdkHarness,
        "opencode": OpenCodeHarness,
        "pi": PiHarness,
        "qwen_code": QwenCodeHarness,
        "shell": ShellHarness,
    }


def create_harness(agent_spec: AgentSpec) -> BaseHarness:
    """Resolve and instantiate a harness from an AgentSpec."""
    if agent_spec.import_path is not None:
        cls = _import_harness_class(agent_spec.import_path)
        return cls(agent_spec)

    if agent_spec.harness is not None:
        harness_map = _builtin_harness_map()
        cls = harness_map.get(agent_spec.harness)
        if cls is None:
            raise ValueError(f"Unknown harness: {agent_spec.harness!r}")
        return cls(agent_spec)

    raise ValueError("AgentSpec must specify harness or import_path")


def _import_harness_class(import_path: str) -> type[BaseHarness]:
    return import_subclass(import_path, BaseHarness, kind="harness import path")
