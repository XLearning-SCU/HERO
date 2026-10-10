"""OpenClaw harness — https://github.com/openclaw/openclaw (Node 22.19+)"""

from __future__ import annotations

import json
import shlex

from hero.agent.base import BaseHarness
from hero.runtime.base import BaseRuntime, RUNTIME_AGENT_LOG_DIR
from hero.runtime.models import ExecInput


class OpenClawHarness(BaseHarness):
    """Run OpenClaw's embedded agent in headless mode (``openclaw agent --local``).

    OpenClaw reads its config from ``~/.openclaw/openclaw.json`` and never picks
    up ``OPENAI_BASE_URL`` from the environment, so the gateway endpoint must be
    written into ``models.providers.openai.baseUrl``. The provider's API key is
    read from ``OPENAI_API_KEY`` (injected by the gateway as the session id).

    ``model_name`` is ``openai/<model>``; everything after the first ``/`` is the
    model id registered under the ``openai`` provider.
    """

    _CONFIG_PATH = "$HOME/.openclaw/openclaw.json"
    # Placeholder substituted with $OPENAI_BASE_URL at exec time (the gateway
    # env is only present during run steps, not setup), keeping the JSON static.
    _BASE_URL_PLACEHOLDER = "__HERO_GATEWAY_BASE_URL__"

    async def setup(self, runtime: BaseRuntime) -> None:
        # `setup` creates the baseline config, workspace, and per-agent session
        # folders ("main" agent dir) that the embedded `--local` run expects.
        await runtime.exec("openclaw setup --workspace . </dev/null")

        if self.skills_path:
            await runtime.exec(
                "mkdir -p $HOME/.openclaw/skills && "
                f"cp -r {shlex.quote(self.skills_path)}/* "
                "$HOME/.openclaw/skills/ 2>/dev/null || true"
            )

    def run_steps(self, instruction: str) -> list[ExecInput]:
        model_id = (self.model_name or "openai/gpt-5.4").split("/", 1)[-1]
        config = self._build_config(model_id)
        config_json = json.dumps(config)

        # `openclaw agent --local` requires a session target; `setup` registers
        # the "main" agent, so default to it.
        agent_id = str(self.settings.get("openclaw_agent_id", "main"))
        flags = [
            "--local",
            "--json",
            f"--agent {shlex.quote(agent_id)}",
            f"--model openai/{shlex.quote(model_id)}",
        ]
        thinking = self.settings.get("thinking")
        if thinking is not None:
            flags.append(f"--thinking {shlex.quote(str(thinking))}")
        flags_str = " ".join(flags)
        escaped = shlex.quote(instruction)

        return [
            ExecInput(
                command=(
                    "mkdir -p $HOME/.openclaw && "
                    # baseUrl placeholder -> $OPENAI_BASE_URL at exec time.
                    f"printf '%s' {shlex.quote(config_json)} "
                    f'| sed "s|{self._BASE_URL_PLACEHOLDER}|$OPENAI_BASE_URL|g" '
                    f"> {self._CONFIG_PATH} && "
                    f"openclaw agent {flags_str} --message {escaped} "
                    f"2>&1 </dev/null | tee {RUNTIME_AGENT_LOG_DIR}/openclaw.txt"
                ),
                env={**self.env},
            )
        ]

    def _build_config(self, model_id: str) -> dict:
        config: dict = {
            "agents": {"defaults": {"workspace": "."}},
            "gateway": {"mode": "local"},
            "models": {
                "providers": {
                    "openai": {
                        "baseUrl": self._BASE_URL_PLACEHOLDER,
                        "api": "openai-completions",
                        "models": [{"id": model_id, "name": model_id}],
                    }
                }
            },
        }

        if self.mcp_servers:
            servers: dict[str, dict] = {}
            for server in self.mcp_servers:
                if server.transport == "stdio":
                    entry: dict = {"command": server.command}
                    if server.args:
                        entry["args"] = server.args
                else:
                    entry = {"url": server.url, "transport": server.transport}
                servers[server.name] = entry
            config["mcp"] = {"servers": servers}

        return config
