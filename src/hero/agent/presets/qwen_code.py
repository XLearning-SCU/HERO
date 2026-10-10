"""Qwen Code harness — https://github.com/QwenLM/qwen-code"""

from __future__ import annotations

import json
import shlex

from hero.agent.base import BaseHarness
from hero.agent.models import AgentSpec
from hero.runtime.base import BaseRuntime, RUNTIME_AGENT_LOG_DIR
from hero.runtime.models import ExecInput


class QwenCodeHarness(BaseHarness):
    """Run Qwen Code CLI in non-interactive mode."""

    def __init__(self, agent_spec: AgentSpec) -> None:
        super().__init__(agent_spec)
        self._qwen_dir = "$HOME/.qwen"

    async def setup(self, runtime: BaseRuntime) -> None:
        await runtime.exec(f"mkdir -p {self._qwen_dir}")

        # Register MCP servers
        if self.mcp_servers:
            servers_config: list[dict] = []
            for server in self.mcp_servers:
                entry: dict = {"name": server.name, "transport": server.transport}
                if server.transport == "stdio":
                    entry["command"] = server.command
                    if server.args:
                        entry["args"] = server.args
                else:
                    entry["url"] = server.url
                servers_config.append(entry)
            config = {"mcpServers": servers_config}
            config_json = json.dumps(config)
            await runtime.exec(
                f"cat > {self._qwen_dir}/settings.json << 'HEROCFG'\n{config_json}\nHEROCFG"
            )

        # Copy skills
        if self.skills_path:
            await runtime.exec(
                f"mkdir -p {self._qwen_dir}/skills && "
                f"cp -r {shlex.quote(self.skills_path)}/* {self._qwen_dir}/skills/ 2>/dev/null || true"
            )

    def run_steps(self, instruction: str) -> list[ExecInput]:
        escaped = shlex.quote(instruction)
        env: dict[str, str] = {**self.env}
        # qwen-code reads the model from OPENAI_MODEL; passing both an env var
        # and a --model CLI flag created conflicts on proxied backends, so only
        # the env var form is used
        if self.model_name:
            env["OPENAI_MODEL"] = self.model_name

        return [
            ExecInput(
                command=(
                    f"qwen --yolo --prompt={escaped} "
                    f"2>&1 | tee {RUNTIME_AGENT_LOG_DIR}/qwen-code.txt"
                ),
                env=env,
            )
        ]
