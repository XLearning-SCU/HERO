"""Base harness contract for agent runners."""

from __future__ import annotations

from abc import ABC, abstractmethod

from hero.agent.models import AgentRunResult, AgentSpec
from hero.runtime.base import BaseRuntime
from hero.runtime.models import ExecInput


class BaseHarness(ABC):
    """Abstract base for all agent harnesses.

    Each harness converts structured AgentSpec configuration into a sequence
    of ExecInput commands that the gateway node runs inside a runtime.
    """

    def __init__(
        self,
        agent_spec: AgentSpec,
    ) -> None:
        self.agent_spec = agent_spec
        self.model_name = agent_spec.model_name
        self.settings = agent_spec.settings
        self.env = agent_spec.env
        self.mcp_servers = agent_spec.mcp_servers
        self.skills_path = agent_spec.skills_path

    async def setup(self, runtime: BaseRuntime) -> None:
        """Optional setup step run before the agent.

        Override to install packages, write config files, etc.
        """

    @abstractmethod
    def run_steps(self, instruction: str) -> list[ExecInput]:
        """Return the ordered list of commands to execute the agent task."""

    def postrun_steps(self) -> list[ExecInput]:
        """Return best-effort commands to run during post-run teardown.

        This hook runs after trajectory building/evaluation has finished but
        before the runtime is stopped.
        """
        return []

    async def postprocess(
        self, runtime: BaseRuntime, result: AgentRunResult
    ) -> None:
        """Optional artifact collection after the run-step exec loop completes.

        ``result`` may represent a successful run, a command failure, or a
        step-level timeout.
        """
