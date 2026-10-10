"""``test_on_output`` evaluator — grade by parsing pytest-style output.

Use this strategy for custom unit-test tasks where you own the test command
and can express the expected per-test outcomes as a JSON mapping. This is the
strategy for calculator-scale / toy RL tasks that do not need the SWE-Bench
harness machinery.

Config schema
-------------

Passed through :class:`hero.trajectory.models.EvaluatorSpec.config`:

- ``test_command`` *(str, required)* — shell command executed inside the
  (optionally patched) eval runtime. It should emit lines of the form
  ``PASSED <nodeid>``, ``FAILED <nodeid>``, ``ERROR <nodeid>``, or
  ``SKIPPED <nodeid>`` — the same prefix vocabulary as ``pytest -rA``.
- ``expected_output_json`` *(dict or JSON string, required)* — mapping from
  ``nodeid`` to the expected status. The task is considered **resolved only
  if the parsed mapping equals the expected mapping exactly**.
- ``repo_dir`` *(str, default ``/testbed``)* — repository root inside the
  runtime; only used when collecting a git diff (see ``patch_command``).
- ``patch_command`` *(str, default ``git diff --binary --submodule=diff``)* —
  command run in the source runtime to emit the generated patch.
- ``apply_timeout`` *(float, default 60)* — seconds allowed for patch
  extraction and ``git apply`` on the fresh eval runtime.
- ``test_timeout`` *(float, default 1200)* — seconds allowed for
  ``test_command``.
- ``exclude_patterns`` *(list[str])* — extra globs appended to the default
  skip list when filtering the extracted diff.

Node-ID normalization
---------------------

Both expected keys and parsed lines are passed through the same normalizer
(:func:`_normalize_expected_nodeid`): the trailing ``Class.method`` or
``function`` is preserved and everything upstream (``tests/`` prefix, module
path) is dropped. This makes the expected JSON terse and lets the parser
tolerate formatting differences across pytest versions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hero.runtime.base import BaseRuntime
from hero.trajectory.evaluator._patch_utils import (
    ANSI_ESCAPE_RE,
    BasePatchEvaluator,
    bounded_timeout,
)


class TestOnOutputEvaluator(BasePatchEvaluator):
    """Grades generated patches by diffing pytest output against an expected map."""

    __test__ = False  # tell pytest this "Test*" class isn't a test class
    MODE = "test_on_output"

    def __init__(
        self,
        *,
        test_command: str,
        expected_output_json: dict[str, str] | str,
        repo_dir: str = "/testbed",
        patch_command: str | None = None,
        apply_timeout: float = 60.0,
        test_timeout: float = 1200.0,
        exclude_patterns: list[str] | None = None,
    ) -> None:
        super().__init__(
            repo_dir=repo_dir,
            patch_command=patch_command,
            apply_timeout=apply_timeout,
            test_timeout=test_timeout,
            exclude_patterns=exclude_patterns,
        )
        cleaned = test_command.strip()
        if not cleaned:
            raise ValueError("test_on_output requires a non-empty 'test_command'")
        self.test_command = cleaned
        if expected_output_json is None:
            raise ValueError(
                "test_on_output requires 'expected_output_json' in evaluator config"
            )
        self.expected_output_json = expected_output_json

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
        combined_path = log_dir / "expected_output.test_output.log"
        result = await runtime.exec(
            self.test_command,
            cwd=self.repo_dir,
            env=env,
            timeout_sec=bounded_timeout(self.test_timeout, timeout_cap),
        )
        if result.return_code == -1:
            raise TimeoutError("test_on_output evaluation timed out")
        output = (result.stdout or "") + (result.stderr or "")
        combined_path.write_text(output)
        expected = self._coerce_expected_output_json()
        parsed = _parse_expected_output(output)
        report = {
            "empty_generation": False,
            "resolved": bool(parsed) and parsed == expected,
            "failed_apply_patch": False,
            "error_eval": False,
            "test_timeout": False,
            "exit_code": result.return_code,
            "parsed_tests": parsed,
            "expected_tests": expected,
        }
        return report, combined_path

    def _coerce_expected_output_json(self) -> dict[str, str]:
        raw = self.expected_output_json
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(parsed, dict):
            raise ValueError("expected_output_json must decode to a JSON object")
        return {
            _normalize_expected_nodeid(str(key)): str(value)
            for key, value in parsed.items()
        }


def _parse_expected_output(output: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in output.splitlines():
        line = ANSI_ESCAPE_RE.sub("", line).strip()
        if not line.startswith(("PASSED", "FAILED", "ERROR", "SKIPPED")):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        status, nodeid = parts
        normalized = _normalize_expected_nodeid(nodeid)
        if normalized:
            parsed[normalized] = status
    return parsed


def _normalize_expected_nodeid(nodeid: str) -> str:
    nodeid = nodeid.split(" - ")[0]
    parts = nodeid.split("::")
    if len(parts) >= 3:
        return ".".join(parts[-2:])
    if len(parts) == 2:
        return parts[-1]
    return nodeid
