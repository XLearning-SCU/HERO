"""``swebench_harness`` evaluator — grade via the SWE-Bench / SWE-Gym harness.

Use this strategy when you are reproducing SWE-Bench-style benchmarks and
want per-instance grading that matches what the harness reports. The harness
owns the test spec, eval script, and resolved/unresolved classification.

Config schema
-------------

Passed through :class:`hero.trajectory.models.EvaluatorSpec.config`:

- ``instance`` *(dict, required)* — the raw SWE-Bench / SWE-Gym instance dict
  (at minimum ``instance_id``; ``base_commit``/``version`` are derived when
  missing). The grader uses it to build the test spec and eval script.
- ``repo_dir`` *(str, default ``/testbed``)* — repository root inside the
  runtime. Must contain the checked-out instance.
- ``patch_command`` *(str, default ``git diff --binary --submodule=diff``)* —
  command run *in the source runtime* to emit the generated patch.
- ``apply_timeout`` *(float, default 60)* — seconds allowed for patch
  extraction and ``git apply`` on the fresh eval runtime.
- ``test_timeout`` *(float, default 1200)* — seconds allowed for the
  harness-generated ``eval.sh`` run.
- ``exclude_patterns`` *(list[str])* — extra globs appended to the default
  skip list (``__pycache__``, ``*.pyc``, ``.pytest_cache`` etc.) when
  filtering the extracted diff.

The generated patch is captured, filtered, applied on the fresh eval runtime,
and then the harness-produced ``eval.sh`` is executed. Grading is delegated
to ``swegym.harness.grading.get_eval_report`` (or ``swebench`` when swegym is
not installed); ``resolved=True`` → ``outcome_reward=1.0``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hero.runtime.base import BaseRuntime
from hero.trajectory.evaluator._patch_utils import (
    BasePatchEvaluator,
    bounded_timeout,
)
from hero.trajectory.evaluator.swe_grading import (
    SweBackend,
    resolve_backend,
    resolved_and_reward,
    run_harness_on_applied_patch,
    run_prebuilt_eval_script,
)


class SwebenchHarnessEvaluator(BasePatchEvaluator):
    """Grades generated patches using the SWE-Bench / SWE-Gym harness."""

    MODE = "swebench_harness"

    def __init__(
        self,
        *,
        instance: dict[str, Any],
        repo_dir: str = "/testbed",
        patch_command: str | None = None,
        apply_timeout: float = 60.0,
        test_timeout: float = 1200.0,
        exclude_patterns: list[str] | None = None,
        backend: str = "auto",
        source_dataset: str = "",
    ) -> None:
        super().__init__(
            repo_dir=repo_dir,
            patch_command=patch_command,
            apply_timeout=apply_timeout,
            test_timeout=test_timeout,
            exclude_patterns=exclude_patterns,
        )
        if not instance:
            raise ValueError("swebench_harness requires a non-empty 'instance' config")
        self.instance = instance
        routing_instance = dict(instance, source_dataset=source_dataset)
        self.backend = resolve_backend(backend, routing_instance)

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
        instance = dict(self.instance)
        instance_id = str(instance["instance_id"])
        if "version" not in instance and "base_commit" in instance:
            instance["version"] = instance["base_commit"]

        timeout_sec = bounded_timeout(self.test_timeout, timeout_cap)
        if self.backend in {SweBackend.MULTI_SWE, SweBackend.SWE_REBENCH}:
            report, combined_path, exit_code = await run_prebuilt_eval_script(
                runtime=runtime,
                instance=instance,
                backend=self.backend,
                host_session_dir=host_session_dir,
                log_dir=log_dir,
                repo_dir=self.repo_dir,
                env=env,
                timeout_sec=timeout_sec,
            )
        else:
            report, combined_path, exit_code = await run_harness_on_applied_patch(
                runtime=runtime,
                instance=instance,
                backend=self.backend,
                patch=patch,
                host_session_dir=host_session_dir,
                log_dir=log_dir,
                repo_dir=self.repo_dir,
                env=env,
                timeout_sec=timeout_sec,
            )
        resolved, reward = resolved_and_reward(report)
        return (
            {
                "empty_generation": False,
                "resolved": resolved,
                "reward": reward,
                "grading_backend": self.backend.value,
                "failed_apply_patch": False,
                "error_eval": False,
                "test_timeout": False,
                "exit_code": exit_code,
                "grading_report": report,
            },
            combined_path,
        )
