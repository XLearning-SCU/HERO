"""Data models for agent harness configuration and execution."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from hero.runtime.models import ExecInput


class MCPServerSpec(BaseModel):
    """MCP server definition passed to harnesses that support tool servers."""

    name: str
    transport: Literal["stdio", "sse", "streamable-http"]
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None

    @model_validator(mode="after")
    def _validate_transport(self) -> MCPServerSpec:
        if self.transport == "stdio":
            if not self.command:
                raise ValueError("stdio transport requires command")
        elif self.transport in ("sse", "streamable-http"):
            if not self.url:
                raise ValueError(f"{self.transport} transport requires url")
        return self


class AgentSpec(BaseModel):
    """Agent execution specification."""

    harness: str | None = None
    import_path: str | None = None
    model_name: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    mcp_servers: list[MCPServerSpec] = Field(default_factory=list)
    skills_path: str | None = None
    custom_shell: ExecInput | None = None

    @model_validator(mode="after")
    def _validate_agent_source(self) -> AgentSpec:
        sources = sum(1 for source in (self.harness, self.import_path) if source is not None)
        if sources != 1:
            raise ValueError(
                "exactly one of harness or import_path must be provided"
            )
        if self.custom_shell is not None:
            if self.harness != "shell":
                raise ValueError(
                    "custom_shell is only valid when agent.harness is 'shell'"
                )
            if self.mcp_servers:
                raise ValueError(
                    "mcp_servers must be omitted when custom_shell is used"
                )
            if self.skills_path is not None:
                raise ValueError(
                    "skills_path must be omitted when custom_shell is used"
                )
        elif self.harness == "shell":
            raise ValueError("agent.harness='shell' requires custom_shell")
        return self


class AgentRunResult(BaseModel):
    """Terminal result for the agent execution."""

    status: Literal["completed", "failed", "timeout"]
    return_code: int
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
