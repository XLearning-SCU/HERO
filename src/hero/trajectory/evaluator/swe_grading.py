"""Single SWE grading boundary shared by HERO training and standalone eval.

The two benchmark families intentionally use different upstream packages:
official ``swebench`` for Verified/Multilingual and ``swegym`` for SWE-Gym.
Routing must be explicit; selecting whichever package happens to import first
silently grades one family with the other family's repository/test maps.
"""

from __future__ import annotations

from enum import Enum
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Callable

from hero.runtime.base import BaseRuntime


class SweBackend(str, Enum):
    SWEBENCH = "swebench"
    SWEGYM = "swegym"
    MULTI_SWE = "multi_swe"
    SWE_REBENCH = "swe_rebench"


SOURCE_BACKENDS = {
    "swe_gym": SweBackend.SWEGYM,
    "multi_swe": SweBackend.MULTI_SWE,
    "swe_rebench": SweBackend.SWE_REBENCH,
}


def backend_marker_for_source(source: str) -> str:
    backend = SOURCE_BACKENDS[str(source).strip().lower()]
    return "swebench_fork" if backend is SweBackend.SWEGYM else backend.value


def resolve_backend(value: str | SweBackend, instance: dict[str, Any]) -> SweBackend:
    if isinstance(value, SweBackend):
        return value
    normalized = str(value or "auto").strip().lower().replace("-", "_")
    aliases = {
        "swebench": SweBackend.SWEBENCH,
        "official": SweBackend.SWEBENCH,
        "verified": SweBackend.SWEBENCH,
        "multilingual": SweBackend.SWEBENCH,
        "swegym": SweBackend.SWEGYM,
        "swe_gym": SweBackend.SWEGYM,
        "swebench_fork": SweBackend.SWEGYM,
        "multi_swe": SweBackend.MULTI_SWE,
        "swe_rebench": SweBackend.SWE_REBENCH,
    }
    if normalized in aliases:
        return aliases[normalized]
    if normalized != "auto":
        raise ValueError(f"unsupported SWE grading backend: {value!r}")

    source = str(instance.get("source_dataset") or "").strip().lower()
    if source in SOURCE_BACKENDS:
        return SOURCE_BACKENDS[source]

    # Backward compatibility for older prepared files.
    declared = str(instance.get("eval_backend") or "").strip().lower().replace("-", "_")
    if declared in aliases:
        return aliases[declared]
    raise ValueError(
        "SWE grading backend is ambiguous; configure backend='swebench' for "
        "Verified/Multilingual or backend='swegym' for SWE-Gym"
    )


def load_harness(backend: SweBackend) -> tuple[Callable[..., Any], Callable[..., Any]]:
    if backend is SweBackend.SWEGYM:
        from swegym.harness.grading import get_eval_report
        from swegym.harness.test_spec import make_test_spec

        return make_test_spec, get_eval_report

    from swebench.harness.grading import get_eval_report
    try:
        from swebench.harness.test_spec.test_spec import make_test_spec
    except ModuleNotFoundError:
        from swebench.harness.test_spec import make_test_spec
    return make_test_spec, get_eval_report


def make_test_spec(instance: dict[str, Any], backend: SweBackend) -> Any:
    if backend in {SweBackend.MULTI_SWE, SweBackend.SWE_REBENCH}:
        eval_script = str(instance.get("eval_script") or "")
        if not eval_script:
            raise ValueError(f"{backend.value} instance is missing eval_script")
        return SimpleNamespace(eval_script=eval_script)
    if backend is SweBackend.SWEBENCH:
        return _make_swebench_spec_in_official_checkout(instance)
    factory, _ = load_harness(backend)
    row = dict(instance)
    return factory(row)


def _official_checkout() -> str:
    return os.environ.get(
        "HERO_SWEBENCH_PATH",
        str(Path(__file__).resolve().parents[5] / "harnesses/SWE-bench"),
    )


def _official_subprocess(program: str, payload: dict[str, Any]) -> str:
    checkout = _official_checkout()
    env = dict(os.environ)
    env["PYTHONPATH"] = checkout + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=env,
        cwd=checkout,
        timeout=float(os.environ.get("HERO_SWEBENCH_PARSE_TIMEOUT_SEC", "120")),
        check=False,
    )
    marker = "__HERO_RESULT__"
    position = completed.stdout.rfind(marker)
    if completed.returncode != 0 or position < 0:
        raise RuntimeError(
            "official SWE-bench helper failed: "
            f"rc={completed.returncode} stdout={completed.stdout[-1000:]!r} "
            f"stderr={completed.stderr[-1000:]!r}"
        )
    return completed.stdout[position + len(marker) :].splitlines()[0]


def _make_swebench_spec_in_official_checkout(instance: dict[str, Any]) -> Any:
    program = r'''
import json, sys
from swebench.harness.test_spec.test_spec import make_test_spec
p = json.load(sys.stdin)
s = make_test_spec(p["instance"], namespace="swebench")
print("__HERO_RESULT__" + json.dumps({"eval_script": s.eval_script, "instance_image_key": s.instance_image_key}))
'''
    data = json.loads(_official_subprocess(program, {"instance": instance}))
    return SimpleNamespace(**data)


def grade_test_log(
    *,
    instance: dict[str, Any],
    backend: SweBackend,
    patch: str,
    log_path: str | Path,
) -> dict[str, Any]:
    """Return the upstream per-instance report without reinterpreting tests."""

    if backend is SweBackend.SWEBENCH:
        return _grade_swebench_in_official_checkout(
            instance=instance, patch=patch, log_path=log_path
        )

    test_spec = make_test_spec(instance, backend)
    _, grader = load_harness(backend)
    instance_id = str(instance["instance_id"])
    prediction = {
        "model_patch": patch,
        "model_name_or_path": "hero",
        "instance_id": instance_id,
    }
    kwargs = {
        "test_spec": test_spec,
        "prediction": prediction,
        "include_tests_status": True,
    }
    try:
        reports = grader(log_path=str(log_path), **kwargs)
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        reports = grader(test_log_path=str(log_path), **kwargs)
    report = reports.get(instance_id)
    if report is None:
        # Some legacy SWE-Gym parsers normalize the report key. Do not mutate
        # the instance before test-spec creation; only tolerate that output.
        report = reports.get(instance_id.lower())
    if not isinstance(report, dict):
        raise KeyError(f"upstream grader returned no report for {instance_id!r}")
    return report


def _grade_swebench_in_official_checkout(
    *, instance: dict[str, Any], patch: str, log_path: str | Path
) -> dict[str, Any]:
    """Parse with the pinned official checkout in an isolated interpreter.

    SWE-Gym's fork also installs a top-level ``swebench`` package. Keeping the
    official parser out of this process prevents import-order-dependent scores.
    """

    program = r'''
import json, sys
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import make_test_spec
p = json.load(sys.stdin)
i = p["instance"]
s = make_test_spec(i, namespace="swebench")
prediction = {"instance_id": i["instance_id"], "model_patch": p["patch"], "model_name_or_path": "hero"}
try:
    reports = get_eval_report(test_spec=s, prediction=prediction, test_log_path=p["log_path"], include_tests_status=True)
except TypeError:
    reports = get_eval_report(test_spec=s, prediction=prediction, log_path=p["log_path"], include_tests_status=True)
print("__HERO_RESULT__" + json.dumps(reports[i["instance_id"]]))
'''
    return json.loads(
        _official_subprocess(
            program,
            {
                "instance": instance,
                "patch": patch,
                "log_path": str(Path(log_path).resolve()),
            },
        )
    )


def resolved_and_reward(report: dict[str, Any]) -> tuple[bool, float]:
    """The sole HERO mapping from upstream classification to RL reward."""

    resolved = bool(report.get("resolved", False))
    return resolved, 1.0 if resolved else 0.0


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            value = [value] if value else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _expected_status(log_text: str, names: list[str]) -> dict[str, str]:
    status: dict[str, str] = {}
    for name in (str(item).strip() for item in names):
        if not name:
            continue
        token = rf"(?<![A-Za-z0-9_./:-]){re.escape(name)}(?![A-Za-z0-9_./:-])"
        failed = (
            rf"--- FAIL: {token}(?:\s|$|\()",
            rf"test {token} \.\.\. FAILED",
            rf"\bFAIL(?:ED)?\b.*{token}",
            rf"{token}.*\bFAIL(?:ED)?\b",
        )
        passed = (
            rf"--- PASS: {token}(?:\s|$|\()",
            rf"test {token} \.\.\. ok",
            rf"\bPASS\b.*{token}",
            rf"{token}.*\bPASS(?:ED)?\b",
        )
        if any(re.search(pattern, log_text, re.MULTILINE) for pattern in failed):
            status[name] = "FAILED"
        elif any(re.search(pattern, log_text, re.MULTILINE) for pattern in passed):
            status[name] = "PASSED"
    return status


def prebuilt_report_from_log(
    instance: dict[str, Any], log_text: str, exit_code: int
) -> dict[str, Any]:
    """Match the combined golden evaluator's Multi-SWE/ReBench verdict."""

    f2p = _as_str_list(instance.get("FAIL_TO_PASS"))
    p2p = _as_str_list(instance.get("PASS_TO_PASS"))
    f2p_status = _expected_status(log_text, f2p)
    p2p_status = _expected_status(log_text, p2p)
    f2p_success = [name for name in f2p if f2p_status.get(name) == "PASSED"]
    f2p_failure = [name for name in f2p if f2p_status.get(name) != "PASSED"]
    p2p_success = [name for name in p2p if p2p_status.get(name) == "PASSED"]
    p2p_failure = [name for name in p2p if p2p_status.get(name) == "FAILED"]
    parsed_expected = bool(f2p_status or p2p_status)
    resolved = not f2p_failure and not p2p_failure if f2p else exit_code == 0
    if not parsed_expected:
        resolved = exit_code == 0
    return {
        "resolved": resolved,
        "tests_status": {
            "FAIL_TO_PASS": {"success": f2p_success, "failure": f2p_failure},
            "PASS_TO_PASS": {"success": p2p_success, "failure": p2p_failure},
        },
        "exit_code": exit_code,
        "parsed_expected_tests": parsed_expected,
    }


async def run_prebuilt_eval_script(
    *,
    runtime: BaseRuntime,
    instance: dict[str, Any],
    backend: SweBackend,
    host_session_dir: Path,
    log_dir: Path,
    repo_dir: str,
    env: dict[str, str],
    timeout_sec: float,
) -> tuple[dict[str, Any], Path, int]:
    """Grade Multi-SWE/ReBench in HERO's already-refreshed runtime."""

    if backend is SweBackend.SWE_REBENCH:
        test_patch = str(instance.get("test_patch") or "")
        if not test_patch:
            raise ValueError("swe_rebench instance is missing test_patch")
        host_test_patch = host_session_dir / "test.patch"
        host_test_patch.write_text(test_patch)
        runtime_test_patch = f"{runtime.runtime_session_dir}/test.patch"
        applied = await runtime.exec(
            f"git apply --whitespace=nowarn '{runtime_test_patch}'",
            cwd=repo_dir,
            env=env,
            timeout_sec=min(timeout_sec, 120.0),
        )
        if applied.return_code != 0:
            raise RuntimeError(f"failed to apply ReBench test_patch: {applied.stderr}")

    eval_script = str(instance.get("eval_script") or "")
    if not eval_script:
        raise ValueError(f"{backend.value} instance is missing eval_script")
    host_eval = host_session_dir / "eval.sh"
    host_eval.write_text(eval_script)
    runtime_eval = f"{runtime.runtime_session_dir}/eval.sh"
    result = await runtime.exec(
        f"bash '{runtime_eval}' 2>&1",
        cwd=repo_dir,
        env=env,
        timeout_sec=timeout_sec,
    )
    if result.return_code == -1:
        raise TimeoutError(f"{backend.value} evaluation timed out")
    log_path = log_dir / str(instance["instance_id"]) / "test_output.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_text = (result.stdout or "") + (result.stderr or "")
    log_path.write_text(log_text)
    return prebuilt_report_from_log(instance, log_text, result.return_code), log_path, result.return_code


async def run_harness_on_applied_patch(
    *,
    runtime: BaseRuntime,
    instance: dict[str, Any],
    backend: SweBackend,
    patch: str,
    host_session_dir: Path,
    log_dir: Path,
    repo_dir: str,
    env: dict[str, str],
    timeout_sec: float,
) -> tuple[dict[str, Any], Path, int]:
    """Run and grade an already-applied patch; shared by train and eval."""

    test_spec = make_test_spec(instance, backend)
    eval_script_host = host_session_dir / "eval.sh"
    eval_script_host.write_text(test_spec.eval_script)
    instance_id = str(instance["instance_id"])
    instance_log_dir = log_dir / instance_id
    instance_log_dir.mkdir(parents=True, exist_ok=True)
    combined_path = instance_log_dir / "test_output.txt"
    result = await runtime.exec(
        # The official runner captures one merged stream. Shell tracing and
        # test-output markers are written to stderr while test results are
        # commonly written to stdout; concatenating the two streams after the
        # process exits moves results outside the marker region and makes the
        # official parser report every expected test as missing.
        f"/bin/bash '{runtime.runtime_session_dir}/eval.sh' 2>&1",
        cwd=repo_dir,
        env=env,
        timeout_sec=timeout_sec,
    )
    if result.return_code == -1:
        raise TimeoutError("SWE harness evaluation timed out")
    combined_path.write_text((result.stdout or "") + (result.stderr or ""))
    report = grade_test_log(
        instance=instance,
        backend=backend,
        patch=patch,
        log_path=combined_path,
    )
    return report, combined_path, result.return_code
