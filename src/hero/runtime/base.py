"""Runtime abstraction for container-backed rollout execution."""

from __future__ import annotations

import asyncio
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Final

from hero.runtime.models import ExecResult, RuntimeSpec

RUNTIME_SESSION_DIR: Final[str] = "/hero/session"
RUNTIME_ARTIFACTS_DIR: Final[str] = f"{RUNTIME_SESSION_DIR}/artifacts"
RUNTIME_LOGS_DIR: Final[str] = f"{RUNTIME_SESSION_DIR}/logs"
RUNTIME_AGENT_LOG_DIR: Final[str] = f"{RUNTIME_LOGS_DIR}/agent"
RUNTIME_EVAL_LOG_DIR: Final[str] = f"{RUNTIME_LOGS_DIR}/eval"
RUNTIME_EVAL_ARTIFACT_DIR: Final[str] = f"{RUNTIME_SESSION_DIR}/eval_artifacts"


class BaseRuntime(ABC):
    """Base class for long-lived per-session execution runtimes."""

    def __init__(self, spec: RuntimeSpec, session_id: str, session_dir: Path) -> None:
        self.spec = spec
        self.session_id = session_id
        self.session_dir = session_dir
        self.artifacts_dir = session_dir / "artifacts"
        self.runtime_session_dir = RUNTIME_SESSION_DIR
        self.runtime_artifacts_dir = RUNTIME_ARTIFACTS_DIR
        self.runtime_logs_dir = RUNTIME_LOGS_DIR
        self.runtime_agent_log_dir = RUNTIME_AGENT_LOG_DIR
        self._active_process: asyncio.subprocess.Process | None = None
        self._destroyed = False

    @property
    @abstractmethod
    def runtime_id(self) -> str:
        """Identifier for the live runtime instance."""

    @property
    def supports_gpus(self) -> bool:
        return False

    @property
    def can_disable_internet(self) -> bool:
        return False

    @property
    def supports_cpu_limits(self) -> bool:
        return False

    @property
    def supports_memory_limits(self) -> bool:
        return False

    @property
    def supports_storage_limits(self) -> bool:
        return False

    @abstractmethod
    async def start(self) -> None:
        """Create and start the runtime instance."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop and remove the runtime instance."""

    async def cancel(self) -> None:
        """Stop any in-flight command and tear the runtime down."""
        process = self._active_process
        if process is not None and process.returncode is None:
            process.kill()
            try:
                await process.wait()
            except ProcessLookupError:
                pass
        await self.stop()

    @abstractmethod
    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: float | None = None,
    ) -> ExecResult:
        """Execute one command inside the runtime and return captured output."""

    @abstractmethod
    async def upload_file(self, local_path: str, remote_path: str) -> None:
        """Copy a single file from the host into the runtime."""

    @abstractmethod
    async def upload_dir(self, local_path: str, remote_path: str) -> None:
        """Copy a directory tree from the host into the runtime."""

    @abstractmethod
    async def download_file(self, remote_path: str, local_path: str) -> None:
        """Copy a single file from inside the runtime to the host."""

    @abstractmethod
    async def download_dir(self, remote_path: str, local_path: str) -> None:
        """Copy a directory tree from inside the runtime to the host."""

    def resolve_host_path(self, runtime_path: str) -> Path | None:
        """Map a runtime path back to a host path via the session bind mount."""
        normalized = Path(runtime_path)
        runtime_root = Path(RUNTIME_SESSION_DIR)
        try:
            relative = normalized.relative_to(runtime_root)
        except ValueError:
            return None
        return self.session_dir / relative

    def _copy_from_bind_mount(self, runtime_path: str, local_path: Path) -> bool:
        host_path = self.resolve_host_path(runtime_path)
        if host_path is None or not host_path.exists():
            return False
        if host_path.is_dir():
            if local_path.exists():
                shutil.rmtree(local_path)
            shutil.copytree(host_path, local_path)
        else:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(host_path, local_path)
        return True

    def _copy_to_bind_mount(self, local_path: str, runtime_path: str) -> bool:
        host_path = self.resolve_host_path(runtime_path)
        if host_path is None:
            return False
        source = Path(local_path)
        if not source.exists():
            raise FileNotFoundError(f"source path does not exist: {local_path}")
        if source.is_dir():
            if host_path.exists():
                shutil.rmtree(host_path)
            shutil.copytree(source, host_path)
        else:
            host_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, host_path)
        return True

    async def _run_local_command(
        self,
        *args: str,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        capture: bool = False,
    ) -> tuple[int, str | None, str | None]:
        """Run a local subprocess, optionally capturing stdout/stderr."""
        process_env = None if env is None else {**os.environ, **env}
        if capture:
            stdout_target = asyncio.subprocess.PIPE
            stderr_target = asyncio.subprocess.PIPE
        else:
            stdout_target = asyncio.subprocess.DEVNULL
            stderr_target = asyncio.subprocess.DEVNULL

        process = await asyncio.create_subprocess_exec(
            *args,
            env=process_env,
            stdout=stdout_target,
            stderr=stderr_target,
        )
        self._active_process = process
        try:
            if timeout is None:
                stdout_bytes, stderr_bytes = await process.communicate()
            else:
                try:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        process.communicate(), timeout=timeout
                    )
                except asyncio.TimeoutError:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        # The subprocess may exit between wait_for timing out
                        # and delivery of the kill signal.
                        pass
                    try:
                        await process.wait()
                    except ProcessLookupError:
                        pass
                    return -1, None, None
        finally:
            self._active_process = None

        rc = process.returncode or 0
        stdout_str = stdout_bytes.decode(errors="replace") if stdout_bytes else None
        stderr_str = stderr_bytes.decode(errors="replace") if stderr_bytes else None
        return rc, stdout_str, stderr_str
