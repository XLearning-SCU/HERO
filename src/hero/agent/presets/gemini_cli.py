"""Gemini CLI harness — https://github.com/google/gemini-cli"""

from __future__ import annotations

import json
import shlex

from hero.agent.base import BaseHarness
from hero.agent.models import AgentSpec
from hero.runtime.base import BaseRuntime, RUNTIME_AGENT_LOG_DIR
from hero.runtime.models import ExecInput


class GeminiCliHarness(BaseHarness):
    """Run Google Gemini CLI in non-interactive mode."""

    def __init__(self, agent_spec: AgentSpec) -> None:
        super().__init__(agent_spec)
        self._gemini_dir = "$HOME/.gemini"

    async def setup(self, runtime: BaseRuntime) -> None:
        await runtime.exec(f"mkdir -p {self._gemini_dir}")

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
                f"cat > {self._gemini_dir}/settings.json << 'HEROCFG'\n{config_json}\nHEROCFG"
            )

        # Copy skills
        if self.skills_path:
            await runtime.exec(
                f"mkdir -p {self._gemini_dir}/skills && "
                f"cp -r {shlex.quote(self.skills_path)}/* {self._gemini_dir}/skills/ 2>/dev/null || true"
            )

    def run_steps(self, instruction: str) -> list[ExecInput]:
        escaped = shlex.quote(instruction)
        env: dict[str, str] = {
            "GEMINI_CLI_TRUST_WORKSPACE": "true",
            **self.env,
        }

        # --approval-mode=yolo is the documented replacement for the legacy
        # --yolo flag (both currently work but --yolo is listed as deprecated
        # in the latest CLI reference). Sandbox is off by default; we only
        # pass --sandbox when the caller explicitly opts in.
        flags: list[str] = ["--approval-mode=yolo"]
        if self.model_name:
            flags.append(f"--model={shlex.quote(self.model_name)}")
        if self.settings.get("sandbox") is True:
            flags.append("--sandbox")

        flags_str = " ".join(flags)
        return [
            ExecInput(
                command=(
                    # The gateway injects GOOGLE_API_KEY / GOOGLE_API_URL; the
                    # Gemini CLI reads GEMINI_API_KEY / GOOGLE_GEMINI_BASE_URL,
                    # so map one onto the other to route calls at the proxy.
                    'export GEMINI_API_KEY="$GOOGLE_API_KEY" '
                    'GOOGLE_GEMINI_BASE_URL="$GOOGLE_API_URL" && '
                    f"gemini {flags_str} --prompt={escaped} "
                    f"2>&1 </dev/null | tee {RUNTIME_AGENT_LOG_DIR}/gemini-cli.txt"
                ),
                env=env,
            )
        ]
