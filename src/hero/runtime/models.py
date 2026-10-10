"""Data models for runtime configuration and execution."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ExecInput(BaseModel):
    """Command specification for runtime execution."""

    command: str
    cwd: str | None = None
    env: dict[str, str] | None = None


class PrepareAction(BaseModel):
    """One ordered step in the runtime preparation recipe.

    Interleaves uploads and shell commands in exact order needed by the task.
    """

    type: Literal["upload_file", "upload_dir", "exec"]
    source: str | None = None
    target: str | None = None
    command: str | None = None
    cwd: str | None = None
    env: dict[str, str] | None = None

    @model_validator(mode="after")
    def _validate_fields(self) -> PrepareAction:
        if self.type in ("upload_file", "upload_dir"):
            if not self.source or not self.target:
                raise ValueError(f"{self.type} requires source and target")
            for field_name in ("command", "cwd", "env"):
                if getattr(self, field_name) is not None:
                    raise ValueError(f"{self.type} must not set {field_name}")
        elif self.type == "exec":
            if not self.command:
                raise ValueError("exec requires command")
            for field_name in ("source", "target"):
                if getattr(self, field_name) is not None:
                    raise ValueError(f"exec must not set {field_name}")
        return self


class ExecResult(BaseModel):
    """Result of a command executed inside a runtime."""

    stdout: str | None = None
    stderr: str | None = None
    return_code: int


class RuntimeSpec(BaseModel):
    """Container runtime configuration for one rollout session."""

    backend: Literal["docker", "apptainer"] = "docker"
    image: str
    prepare: list[PrepareAction] = Field(default_factory=list)
    eval_prepare: list[PrepareAction] | None = None
    env: dict[str, str] = Field(default_factory=dict)
    network: str | None = "host"
    workdir: str | None = None
    cpus: int | None = None
    memory_mb: int | None = None
    storage_mb: int | None = None
    gpus: int = 0
    allow_internet: bool = True
    import_path: str | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("image")
    @classmethod
    def _validate_image(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("runtime image must be non-empty")
        return normalized

    @field_validator("workdir")
    @classmethod
    def _validate_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("workdir must be non-empty when provided")
        return normalized
