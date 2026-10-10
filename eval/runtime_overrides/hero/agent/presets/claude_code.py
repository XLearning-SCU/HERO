"""Claude Code harness — https://docs.anthropic.com/en/docs/claude-code"""

from __future__ import annotations

import json
from pathlib import Path
import shlex

from hero.agent.base import BaseHarness
from hero.agent.models import AgentSpec
from hero.runtime.base import BaseRuntime, RUNTIME_AGENT_LOG_DIR, RUNTIME_SESSION_DIR
from hero.runtime.models import ExecInput


class ClaudeCodeHarness(BaseHarness):
    """Run Claude Code CLI in non-interactive mode."""

    def __init__(self, agent_spec: AgentSpec) -> None:
        super().__init__(agent_spec)
        # Absolute path outside the workspace — $HOME won't expand in docker
        # exec -e, and a literal "$HOME" dir would get swept into git add -A.
        self._config_dir = f"{RUNTIME_SESSION_DIR}/.claude"

    async def setup(self, runtime: BaseRuntime) -> None:
        await runtime.exec(f"mkdir -p {self._config_dir}")

        # ``CLAUDE_CODE_AUTO_COMPACT_WINDOW`` only supplies a window override.
        # Persist the enable flag as a Claude Code setting as well, so batch
        # sessions do not depend on the CLI's per-install default.  Do not put
        # the window in settings.json: recent Claude Code builds reject setting
        # values below 100K, while the environment override intentionally
        # supports the smaller windows used by local rollout models.
        claude_settings: dict[str, object] = {}
        if "auto_compact_enabled" in self.settings:
            claude_settings["autoCompactEnabled"] = bool(
                self.settings["auto_compact_enabled"]
            )
        git_progress = self.settings.get("git_progress")
        if isinstance(git_progress, dict) and bool(git_progress.get("enabled", False)):
            interval = max(1, int(git_progress.get("checkpoint_interval", 10)))
            repo_dir = getattr(getattr(runtime, "spec", None), "workdir", None)
            repo_dir = str(repo_dir or "/hero/session/workspace")
            progress_state_dir = f"{RUNTIME_AGENT_LOG_DIR}/.git-progress"
            # Capture the immutable comparison base before the agent can run a
            # Bash `git commit`. Failure only disables trusted Git credit; it
            # must never abort the rollout.
            await runtime.exec(
                f"mkdir -p {shlex.quote(progress_state_dir)} && "
                f"git -C {shlex.quote(repo_dir)} rev-parse --verify HEAD "
                f"> {shlex.quote(progress_state_dir + '/baseline-commit')} "
                "2>/dev/null || true"
            )
            hook_source = Path(__file__).with_name("git_progress_hook.py")
            hook_target = f"{self._config_dir}/git_progress_hook.py"
            await runtime.upload_file(str(hook_source), hook_target)
            hook_command = (
                f"HERO_GIT_PROGRESS_INTERVAL={interval} "
                f"HERO_GIT_PROGRESS_REPO={shlex.quote(repo_dir)} "
                # Hard wall-clock bound + low CPU priority: observability must
                # never stall or fail the agent when Git/FUSE is unhealthy.
                f"nice -n 10 timeout 3s python3 {shlex.quote(hook_target)} "
                f">/dev/null 2>&1 || true"
            )
            claude_settings["hooks"] = {
                "PostToolUse": [
                    {
                        "matcher": "",
                        "hooks": [{"type": "command", "command": hook_command}],
                    }
                ]
            }
        if claude_settings:
            settings_json = json.dumps(claude_settings)
            await runtime.exec(
                f"cat > {self._config_dir}/settings.json << 'HEROCFG'\n"
                f"{settings_json}\nHEROCFG"
            )

        # Register MCP servers
        if self.mcp_servers:
            mcp_config: dict[str, dict] = {}
            for server in self.mcp_servers:
                entry: dict = {}
                if server.transport == "stdio":
                    entry["command"] = server.command
                    if server.args:
                        entry["args"] = server.args
                    entry["type"] = "stdio"
                else:
                    entry["url"] = server.url
                    entry["type"] = server.transport
                mcp_config[server.name] = entry
            config = {"mcpServers": mcp_config}
            config_json = json.dumps(config)
            await runtime.exec(
                f"cat > {self._config_dir}/.claude.json << 'HEROCFG'\n{config_json}\nHEROCFG"
            )

        # Copy skills
        if self.skills_path:
            await runtime.exec(
                f"mkdir -p {self._config_dir}/skills && "
                f"cp -r {shlex.quote(self.skills_path)}/* {self._config_dir}/skills/ 2>/dev/null || true"
            )

    def run_steps(self, instruction: str) -> list[ExecInput]:
        escaped = shlex.quote(instruction)

        # Docker runtimes intentionally use a login shell so language-specific
        # toolchains from the image's /etc/profile remain available.  Some
        # images replace PATH there, however, which can hide a host-mounted
        # Claude CLI.  Allow launchers to pin an absolute executable and
        # restore its directory after login-shell initialization.  This also
        # keeps /usr/bin/env node in the Claude wrapper's shebang reliable.
        executable = str(self.settings.get("executable") or "claude")
        executable_command = shlex.quote(executable)
        path_prefix = ""
        executable_path = Path(executable)
        if executable_path.is_absolute():
            executable_dir = shlex.quote(str(executable_path.parent))
            path_prefix = f'export PATH={executable_dir}:"$PATH"; '

        flags: list[str] = [
            "--verbose",
            "--output-format=stream-json",
            "--dangerously-skip-permissions",
        ]
        for key, cli in [
            ("max_turns", "--max-turns"),
            ("reasoning_effort", "--effort"),
            ("max_budget_usd", "--max-budget-usd"),
            ("fallback_model", "--fallback-model"),
            ("append_system_prompt", "--append-system-prompt"),
            ("tools", "--tools"),
            ("allowed_tools", "--allowedTools"),
            ("disallowed_tools", "--disallowedTools"),
        ]:
            value = self.settings.get(key)
            # Empty template values mean "do not pass this optional flag".
            if value is not None and str(value):
                flags.append(f"{cli} {shlex.quote(str(value))}")

        flags_str = " ".join(flags)
        env: dict[str, str] = {
            **self.env,
            "CLAUDE_CONFIG_DIR": self._config_dir,
            # Allow --dangerously-skip-permissions / bypassPermissions inside
            "IS_SANDBOX": "1",
            # Suppress Statsig / telemetry calls that the CLI otherwise makes
            # to api.anthropic.com even when ANTHROPIC_BASE_URL points elsewhere.
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }
        if self.settings.get("max_thinking_tokens"):
            env["MAX_THINKING_TOKENS"] = str(self.settings["max_thinking_tokens"])
        if self.settings.get("auto_compact_enabled") is False:
            # Pass the CLI's actual disable switches into docker exec;
            # launcher variables alone do not reach the agent process.
            env["DISABLE_AUTO_COMPACT"] = "1"
            env["DISABLE_COMPACT"] = "1"
            env.pop("CLAUDE_CODE_AUTO_COMPACT_WINDOW", None)
        elif self.settings.get("auto_compact_window") is not None:
            env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(
                int(self.settings["auto_compact_window"])
            )

        # Model config: if model_name is set, use --model flag and pin all tier
        # aliases to the same model so claude-code doesn't try to route a
        # sub-agent / fallback request back to api.anthropic.com.
        model_flag = ""
        if self.model_name:
            model_flag = f" --model {shlex.quote(self.model_name)}"
            for alias in (
                "ANTHROPIC_MODEL",
                "ANTHROPIC_DEFAULT_SONNET_MODEL",
                "ANTHROPIC_DEFAULT_OPUS_MODEL",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL",
                "CLAUDE_CODE_SUBAGENT_MODEL",
            ):
                env[alias] = self.model_name

        return [
            ExecInput(
                command=(
                    f"set -o pipefail; {path_prefix}"
                    f"{executable_command} {flags_str}{model_flag} -p {escaped} "
                    f"2>&1 | tee {RUNTIME_AGENT_LOG_DIR}/claude-code.txt"
                ),
                env=env,
            )
        ]
