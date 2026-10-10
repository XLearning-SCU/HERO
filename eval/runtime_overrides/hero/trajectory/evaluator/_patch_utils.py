"""Shared skeleton for git-diff patch evaluators.

Both :mod:`hero.trajectory.evaluator.swebench_harness` and
:mod:`hero.trajectory.evaluator.test_on_output` grade a trajectory by:

1. Pulling a git-diff patch out of the source runtime.
2. Filtering out noise (``__pycache__`` etc.).
3. Optionally applying the patch on a fresh replay runtime.
4. Running a test command and interpreting its output.

Only step 4 differs between the two strategies. This module hosts the shared
base class :class:`BasePatchEvaluator` plus the patch-wrangling helpers.
Subclasses implement :meth:`BasePatchEvaluator._grade` and set ``MODE`` to
tag themselves in the returned metadata.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from hero.runtime.base import BaseRuntime
from hero.runtime.models import RuntimeSpec
from hero.trajectory.evaluator.base import BaseTrajectoryEvaluator
from hero.trajectory.models import EvalResult, Trajectory

APPLY_PATCH_PASS = "__HERO_APPLY_PATCH_PASS__"
APPLY_PATCH_FAIL = "__HERO_APPLY_PATCH_FAIL__"
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m|\r")
GIT_PROGRESS_RELATIVE_PATH = Path("logs/agent/git-progress.jsonl")
GIT_PROGRESS_MAX_SNAPSHOT_BYTES = 8 * 1024 * 1024
GIT_PROGRESS_MAX_TOTAL_SNAPSHOT_BYTES = 32 * 1024 * 1024

_DEFAULT_EXCLUDE_PATTERNS = (
    "__pycache__/**",
    "**/__pycache__/**",
    "*.pyc",
    "**/*.pyc",
    "*.pyo",
    "**/*.pyo",
    ".pytest_cache/**",
    "**/.pytest_cache/**",
)


def shell_quote(value: str) -> str:
    """Single-quote *value* so it survives one round of bash interpolation."""
    return "'" + value.replace("'", "'\"'\"'") + "'"


def bounded_timeout(base_timeout: float, timeout_cap: float | None) -> float:
    """Clamp *base_timeout* to the session-wide *timeout_cap* when provided."""
    if timeout_cap is None:
        return base_timeout
    return min(base_timeout, timeout_cap)


class PatchExtractionError(RuntimeError):
    """Patch collection failed; this is not an empty model patch."""


class BasePatchEvaluator(BaseTrajectoryEvaluator):
    """Extract-filter-apply-test skeleton for git-diff-based grading.

    Subclasses pick how the test output is interpreted by overriding
    :meth:`_grade` and set a string ``MODE`` class attribute (stamped into
    ``metadata['mode']``).
    """

    MODE: str = "patch"

    def __init__(
        self,
        *,
        repo_dir: str = "/testbed",
        patch_command: str | None = None,
        apply_timeout: float = 60.0,
        test_timeout: float = 1200.0,
        exclude_patterns: list[str] | None = None,
    ) -> None:
        self.repo_dir = repo_dir.strip()
        if not self.repo_dir:
            raise ValueError("repo_dir must be non-empty")
        self.patch_command = (
            patch_command.strip()
            if patch_command is not None
            else f"cd {shell_quote(self.repo_dir)} && git diff --binary --submodule=diff"
        )
        if not self.patch_command:
            raise ValueError("patch_command must be non-empty")
        self.apply_timeout = float(apply_timeout)
        self.test_timeout = float(test_timeout)
        if self.apply_timeout <= 0 or self.test_timeout <= 0:
            raise ValueError("timeouts must be greater than 0")
        self.exclude_patterns = list(
            dict.fromkeys([*_DEFAULT_EXCLUDE_PATTERNS, *(exclude_patterns or [])])
        )

    # ------------------------------------------------------------------
    # Top-level flow
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        trajectory: Trajectory,
        **runtime: Any,
    ) -> EvalResult:
        source_runtime = runtime.get("runtime")
        if not isinstance(source_runtime, BaseRuntime):
            raise RuntimeError(f"{self.MODE} evaluator requires a live runtime")

        runtime_spec = runtime.get("runtime_spec")
        if runtime_spec is not None and not isinstance(runtime_spec, RuntimeSpec):
            raise TypeError("runtime_spec must be a RuntimeSpec when provided")

        session_dir = Path(runtime["session_dir"])
        artifacts_dir = Path(runtime["artifacts_dir"])
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        eval_env = runtime.get("env")
        if not isinstance(eval_env, dict):
            eval_env = {}
        timeout_cap = runtime.get("timeout_seconds")
        timeout_cap = float(timeout_cap) if timeout_cap is not None else None
        refresh_runtime = bool(runtime.get("refresh_runtime", False))

        patch_path = artifacts_dir / "patch.diff"
        patch = await self._extract_patch(
            source_runtime,
            patch_path,
            env=eval_env,
            timeout_cap=timeout_cap,
        )
        patch = self._filter_patch(patch)
        patch_path.write_text(patch)
        metadata: dict[str, Any] = {
            "mode": self.MODE,
            "patch_path": str(patch_path),
            # Expose the filtered model patch text so out-of-band graders (e.g.
            # the standalone eval harness that re-grades via the official
            # SWE-bench runner) can fetch it from the task/session result
            # without re-reading the (often cleaned-up) session artifacts_dir.
            "model_patch": patch,
            "patch_usable": False,
            "git_progress": load_git_progress(session_dir),
            "report": {
                "empty_generation": len(patch.strip()) == 0,
                "resolved": False,
                "failed_apply_patch": False,
                "error_eval": False,
                "test_timeout": False,
            },
        }
        if not patch.strip():
            return EvalResult(outcome_reward=0.0, metadata=metadata)

        eval_runtime: BaseRuntime = source_runtime
        eval_session_dir = session_dir
        eval_artifacts_dir = artifacts_dir
        apply_patch_output = ""
        try:
            fresh_runtime = runtime.get("fresh_eval_runtime")
            if isinstance(fresh_runtime, BaseRuntime):
                eval_runtime = fresh_runtime
                eval_session_dir = fresh_runtime.session_dir
                eval_artifacts_dir = fresh_runtime.artifacts_dir
                eval_artifacts_dir.mkdir(parents=True, exist_ok=True)

                apply_patch_output = await self._apply_patch(
                    eval_runtime,
                    patch,
                    host_session_dir=eval_session_dir,
                    log_dir=eval_artifacts_dir,
                    env=eval_env,
                    timeout_cap=timeout_cap,
                )
                metadata["apply_patch_output_path"] = str(
                    eval_artifacts_dir / "apply_patch.stdout.log"
                )
                if (
                    APPLY_PATCH_FAIL in apply_patch_output
                    or APPLY_PATCH_PASS not in apply_patch_output
                ):
                    metadata["report"]["failed_apply_patch"] = True
                    # Surface the tail of the apply output into the result so
                    # we can diagnose failures after session_dir is cleaned up.
                    metadata["apply_patch_output"] = apply_patch_output[-8000:]
                    metadata["apply_patch_patch_head"] = patch[:4000]
                    return EvalResult(outcome_reward=0.0, metadata=metadata)
            elif refresh_runtime:
                raise RuntimeError(
                    f"refresh_runtime=true requires fresh_eval_runtime for {self.MODE}"
                )

            # A non-empty filtered patch is usable once it has either applied
            # to the fresh evaluator checkout or entered the legacy in-place
            # grading path (where no replay apply step exists).
            metadata["patch_usable"] = True

            report, test_output_path = await self._grade(
                runtime=eval_runtime,
                patch=patch,
                host_session_dir=eval_session_dir,
                log_dir=eval_artifacts_dir,
                env=eval_env,
                timeout_cap=timeout_cap,
            )
            metadata["report"] = report
            metadata["test_output_path"] = str(test_output_path)
            if apply_patch_output:
                metadata["apply_patch_output"] = apply_patch_output
            return EvalResult(
                outcome_reward=1.0 if report.get("resolved", False) else 0.0,
                metadata=metadata,
            )
        except TimeoutError:
            metadata["report"]["test_timeout"] = True
            raise
        except Exception:
            metadata["report"]["error_eval"] = True
            raise

    async def _grade(
        self,
        *,
        runtime: BaseRuntime,
        patch: str,
        host_session_dir: Path,
        log_dir: Path,
        env: dict[str, str],
        timeout_cap: float | None,
    ) -> tuple[dict[str, Any], Path]:
        """Strategy-specific test execution + output interpretation."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Patch extraction / application
    # ------------------------------------------------------------------

    async def _extract_patch(
        self,
        runtime: BaseRuntime,
        patch_path: Path,
        *,
        env: dict[str, str],
        timeout_cap: float | None,
    ) -> str:
        result = await runtime.exec(
            self.patch_command,
            env=env,
            timeout_sec=bounded_timeout(self.apply_timeout, timeout_cap),
        )
        if result.stdout:
            patch_path.write_text(result.stdout)
        if result.stderr:
            patch_path.with_suffix(".stderr.log").write_text(result.stderr)
        if result.return_code == -1:
            raise PatchExtractionError("patch_extraction_timeout: timed out while collecting git diff patch")
        if result.return_code != 0:
            raise PatchExtractionError(
                f"patch_extraction_failed: git diff command failed with exit code {result.return_code}: {result.stderr}"
            )
        return result.stdout or ""

    async def _apply_patch(
        self,
        runtime: BaseRuntime,
        patch: str,
        *,
        host_session_dir: Path,
        log_dir: Path,
        env: dict[str, str],
        timeout_cap: float | None,
    ) -> str:
        patch_path = host_session_dir / "patch.diff"
        patch_path.write_text(patch)
        runtime_patch_path = f"{runtime.runtime_session_dir}/patch.diff"
        apply_cmd = (
            f"cd {shell_quote(self.repo_dir)} && "
            f"(git apply -v {shell_quote(runtime_patch_path)} && echo '{APPLY_PATCH_PASS}' || "
            f"(echo 'Failed to apply patch with git apply, trying with patch command...' && "
            f"(patch --batch --fuzz=5 -p1 -i {shell_quote(runtime_patch_path)} && "
            f"echo '{APPLY_PATCH_PASS}' || echo '{APPLY_PATCH_FAIL}')))"
        )
        # Pin cwd to repo_dir: the eval runtime is fresh and its spec.workdir
        # (e.g. /hero/session/workspace) won't exist there, so falling through
        # to spec.workdir would make `docker exec -w ...` fail at OCI chdir.
        result = await runtime.exec(
            apply_cmd,
            cwd=self.repo_dir,
            env=env,
            timeout_sec=bounded_timeout(self.apply_timeout, timeout_cap),
        )
        stdout_path = log_dir / "apply_patch.stdout.log"
        stderr_path = log_dir / "apply_patch.stderr.log"
        if result.stdout:
            stdout_path.write_text(result.stdout)
        if result.stderr:
            stderr_path.write_text(result.stderr)
        output = (result.stdout or "") + (result.stderr or "")
        if result.return_code == -1:
            raise TimeoutError("timed out while applying git diff patch")
        return output

    # ------------------------------------------------------------------
    # Patch filtering
    # ------------------------------------------------------------------

    def _filter_patch(self, patch: str) -> str:
        if not patch.strip():
            return patch
        kept_sections: list[str] = []
        current_lines: list[str] = []
        for line in patch.splitlines(keepends=True):
            if line.startswith("diff --git "):
                if current_lines and not self._exclude_patch_section(current_lines):
                    kept_sections.extend(current_lines)
                current_lines = [line]
            elif current_lines:
                current_lines.append(line)
            else:
                kept_sections.append(line)
        if current_lines and not self._exclude_patch_section(current_lines):
            kept_sections.extend(current_lines)
        return "".join(kept_sections)

    def _exclude_patch_section(self, lines: list[str]) -> bool:
        # Test/build commands frequently leave large untracked executables in
        # SWE images (for example Go's ``foo.test`` binaries). ``git add -A``
        # stages those pre-existing artifacts as newly-created binary files,
        # producing multi-megabyte prediction patches that cannot be replayed
        # in the fresh official grading checkout. Do not drop modifications to
        # tracked binaries; only discard a newly-added Git binary section.
        if (
            any(line.startswith("new file mode ") for line in lines)
            and any(line.rstrip("\n") == "GIT binary patch" for line in lines)
        ):
            return True
        for line in lines:
            if line.startswith("diff --git "):
                # Git does not necessarily quote spaces in diff paths, so a
                # plain split() mis-parses paths such as
                # ``.venv/.../launcher manifest.xml``. The destination path is
                # unambiguous after the final `` b/`` separator.
                _, separator, destination = line.rstrip("\n").rpartition(" b/")
                if separator:
                    path = destination
                    return self._matches_exclude(path)
            if line.startswith("+++ "):
                path = line[4:].strip()
                if path != "/dev/null":
                    normalized = path[2:] if path.startswith("b/") else path
                    return self._matches_exclude(normalized)
        return False

    def _matches_exclude(self, path: str) -> bool:
        normalized = path.strip()
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(normalized, pattern):
                return True
        return False


def load_git_progress(session_dir: Path) -> dict[str, Any]:
    """Load compact PostToolUse Git observations from the session bind mount."""
    path = session_dir / GIT_PROGRESS_RELATIVE_PATH
    records: list[dict[str, Any]] = []
    parse_errors = 0
    if path.is_file():
        try:
            progress_text = path.read_text(errors="replace")
        except OSError:
            progress_text = ""
            parse_errors += 1
        for line in progress_text.splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue
            if isinstance(record, dict):
                records.append(record)
            else:
                parse_errors += 1
    state_changes = sum(bool(record.get("state_changed")) for record in records)
    attempts_path = path.parent / "git-progress-attempts.jsonl"
    attempts: list[dict[str, Any]] = []
    attempt_parse_errors = 0
    if attempts_path.is_file():
        try:
            attempt_text = attempts_path.read_text(errors="replace")
        except OSError:
            attempt_text = ""
            attempt_parse_errors += 1
        for line in attempt_text.splitlines():
            try:
                attempt = json.loads(line)
            except json.JSONDecodeError:
                attempt_parse_errors += 1
                continue
            if isinstance(attempt, dict):
                attempts.append(attempt)
            else:
                attempt_parse_errors += 1
    started_ids = {
        str(item.get("tool_use_id"))
        for item in attempts
        if item.get("status") == "started" and item.get("tool_use_id")
    }
    completed_ids = {
        str(item.get("tool_use_id"))
        for item in attempts
        if item.get("status") == "completed" and item.get("tool_use_id")
    }
    incomplete_attempt_ids = sorted(started_ids - completed_ids)
    baseline_commits = sorted(
        {
            str(record.get("baseline_commit"))
            for record in records
            if record.get("baseline_commit")
        }
    )
    baseline_fixed = bool(
        records
        and len(baseline_commits) == 1
        and all(record.get("baseline_source") == "setup" for record in records)
    )
    state_dir = path.parent / ".git-progress"
    snapshots: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    snapshot_errors = 0
    snapshot_bytes = 0
    for record in records:
        names = [record.get("state_snapshot"), record.get("checkpoint")]
        for name in names:
            if not isinstance(name, str) or Path(name).name != name:
                continue
            snapshot_path = state_dir / name
            if not snapshot_path.is_file():
                continue
            try:
                size = snapshot_path.stat().st_size
                if (
                    size > GIT_PROGRESS_MAX_SNAPSHOT_BYTES
                    or snapshot_bytes + size > GIT_PROGRESS_MAX_TOTAL_SNAPSHOT_BYTES
                ):
                    snapshot_errors += 1
                    continue
                diff = snapshot_path.read_text(errors="replace")
            except OSError:
                snapshot_errors += 1
                continue
            snapshot_bytes += size
            digest = hashlib.sha256(diff.encode()).hexdigest()
            # Identical repo states need only one persisted full diff.
            state_hash = str(record.get("state_hash") or "")
            dedupe_key = state_hash or digest
            if dedupe_key in seen_hashes:
                continue
            seen_hashes.add(dedupe_key)
            snapshots.append(
                {
                    "snapshot_id": len(snapshots),
                    "action_index": record.get("action_index"),
                    "state_hash": state_hash,
                    "patch_bytes": record.get("patch_bytes"),
                    "changed_files": record.get("changed_files"),
                    "changed_file_paths": record.get("changed_file_paths") or [],
                    "sha256": digest,
                    "diff": diff,
                }
            )
    interval = next(
        (record.get("checkpoint_interval") for record in records
         if isinstance(record.get("checkpoint_interval"), int)),
        10,
    )
    return {
        "schema_version": 1,
        "enabled": path.is_file(),
        "checkpoint_interval": interval,
        "record_count": len(records),
        "parse_errors": parse_errors,
        "attempt_parse_errors": attempt_parse_errors,
        "attempt_count": len(attempts),
        "incomplete_attempt_count": len(incomplete_attempt_ids),
        "incomplete_attempt_tool_use_ids": incomplete_attempt_ids,
        "baseline_commits": baseline_commits,
        "baseline_fixed": baseline_fixed,
        "snapshot_errors": snapshot_errors,
        "snapshot_bytes": snapshot_bytes,
        "state_change_count": state_changes,
        "records": records,
        "diff_snapshots": snapshots,
    }
