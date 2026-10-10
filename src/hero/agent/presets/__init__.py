"""Preset harnesses — ready-made launchers for popular agents.

These are *conveniences*, not integrations. HERO runs any agent unmodified:
a harness only starts the agent process and lets its LLM calls flow through
the gateway proxy (which serves the model and captures the trajectory). Each
preset here is a thin ``BaseHarness`` that writes the agent's config and emits
its run command — typically a few dozen lines.

To run an agent that isn't listed here, you do **not** add code to HERO:
- use the ``shell`` preset to wrap any command, or
- point ``agent.import_path`` at your own ``BaseHarness`` subclass.

See ``hero/agent/README.md`` for the contract and a "bring your own" guide.
"""

from hero.agent.presets.claude_code import ClaudeCodeHarness
from hero.agent.presets.codex import CodexHarness
from hero.agent.presets.gemini_cli import GeminiCliHarness
from hero.agent.presets.hermes import HermesHarness
from hero.agent.presets.mini_swe_agent import MiniSweAgentHarness
from hero.agent.presets.openclaw import OpenClawHarness
from hero.agent.presets.opencode import OpenCodeHarness
from hero.agent.presets.openhands_sdk import OpenHandsSdkHarness
from hero.agent.presets.pi import PiHarness
from hero.agent.presets.qwen_code import QwenCodeHarness
from hero.agent.presets.shell import ShellHarness

__all__ = [
    "ClaudeCodeHarness",
    "CodexHarness",
    "GeminiCliHarness",
    "HermesHarness",
    "MiniSweAgentHarness",
    "OpenClawHarness",
    "OpenCodeHarness",
    "OpenHandsSdkHarness",
    "PiHarness",
    "QwenCodeHarness",
    "ShellHarness",
]
