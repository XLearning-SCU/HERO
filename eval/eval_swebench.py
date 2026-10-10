#!/usr/bin/env python3
"""Submit local-image SWE tasks to HERO, then re-grade with the official harness.

Pipeline:
- Model rollouts run on the HERO rollout server (``/rollout/task/submit``) and
  the resulting model patch is exposed on the session result via
  ``trajectory.metadata.evaluation.model_patch`` (written by the
  ``swebench_harness`` evaluator's ``_patch_utils`` base).
- This script NEVER trusts a rollout-returned reward. It re-grades the patch
  through ProRL's own training-time SWE boundary: the pinned official
  SWE-bench checkout for Verified/Multilingual and ``swegym`` for SWE-Gym.

Resume: rows already present in ``--work-dir/results.jsonl`` are skipped unless
``--no-resume`` is given.
"""

from __future__ import annotations

# This explicit bootstrap applies only to this evaluation process.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from eval_runtime import install as _install_eval_runtime
_install_eval_runtime()

import argparse
import asyncio
import collections
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import httpx

import logging
logger = logging.getLogger("hero.eval_swebench")


def _patch_unavailable_result(detail: str, sample_index: int, evaluation: dict | None = None) -> dict:
    """Preserve collection/session errors instead of claiming an empty patch."""
    report = (evaluation or {}).get("report") or {}
    if (
        isinstance(evaluation, dict)
        and evaluation.get("model_patch") == ""
        and report.get("empty_generation") is True
        and not report.get("error_eval", False)
        and not report.get("test_timeout", False)
    ):
        return {
            "resolved": False, "reward": 0.0, "applied": False,
            "error": "empty_patch", "grading_report": {}, "sample_index": sample_index,
        }
    return {
        "resolved": False, "reward": 0.0, "applied": False,
        "error": f"grading_error:patch_unavailable: {detail}",
        "grading_report": {}, "sample_index": sample_index,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--rollout-url", default="http://127.0.0.1:8080")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--instance-id")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument(
        "--grade-concurrency",
        type=int,
        default=16,
        help=(
            "Maximum independent official graders. This is separate from "
            "rollout concurrency so slow tests cannot starve model requests."
        ),
    )
    parser.add_argument("--timeout", type=float, default=2400.0)
    parser.add_argument("--test-timeout", type=float, default=1200.0)
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--max-context-tokens", type=int, default=131072)
    parser.add_argument("--max-response-tokens", type=int, default=16000)
    parser.add_argument(
        "--auto-compact-enabled",
        type=int,
        choices=(0,),
        default=0,
        help="Evaluation uses no compaction; only 0 is supported.",
    )
    parser.add_argument("--auto-compact-window", type=int, default=80000)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument(
        "--agent-tools",
        help="Optional comma-separated Claude Code built-in tool allowlist.",
    )
    parser.add_argument(
        "--agent-disallowed-tools",
        help="Optional comma-separated Claude Code built-in tool denylist.",
    )
    parser.add_argument(
        "--git-progress-enabled",
        type=int,
        choices=(0, 1),
        default=1,
        help="Record out-of-band Git progress after Claude Code tool calls.",
    )
    parser.add_argument(
        "--git-progress-checkpoint-interval",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--gold-patch-dir",
        type=Path,
        help="Optional directory containing <instance_id>/patch.diff golden patches.",
    )
    parser.add_argument(
        "--agent-cli-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2] / "cli/claude_code"
        ),
    )
    parser.add_argument(
        "--golden",
        action="store_true",
        help="Golden-patch mode: skip HERO rollout; grade the dataset's own gold "
        "patch through ProRL's shared training/eval grader.",
    )
    parser.add_argument(
        "--health-gate", choices=("off", "warn", "strict"), default="warn",
        help="off=best-effort; warn=log a warning on unhealthy runs; strict=non-zero exit.",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--retry-failed-from", type=Path,
        help="Re-grade only instance_ids unresolved in a previous results.jsonl.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def instance_from_row(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    nested = metadata.get("instance")
    if isinstance(nested, dict):
        return nested
    mswe = metadata.get("mswe_row")
    if isinstance(mswe, dict):
        # Multilingual rows also have top-level instance_id/problem_statement,
        # but their canonical SWE specification is under mswe_row.  Check it
        # before treating the top-level wrapper as a flat SWE-bench row.
        instance = dict(mswe)
        for key in ("instance_id", "repo", "base_commit", "version", "patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS"):
            if key in metadata:
                instance.setdefault(key, metadata[key])
        instance.setdefault("problem_statement", row.get("problem_statement", ""))
        return instance
    if "instance_id" in row and "problem_statement" in row:
        return dict(row)
    instance = {
        key: metadata[key]
        for key in ("instance_id", "repo", "base_commit", "version", "patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS")
        if key in metadata
    }
    instance["problem_statement"] = row.get("problem_statement") or _prompt_text(row)
    if "instance_id" not in instance:
        raise ValueError("row has no usable SWE instance metadata")
    return instance


def _prompt_text(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if isinstance(prompt, list) and prompt:
        return str(prompt[-1].get("content", ""))
    return str(row.get("problem_statement", prompt or ""))


def image_for_row(
    row: dict[str, Any], instance: dict[str, Any], image_dir: Path
) -> tuple[str, Path]:
    metadata = row.get("metadata") or {}
    image = metadata.get("sandbox_image") or metadata.get("image")
    image_tar = metadata.get("image_tar")
    if image and image_tar:
        path = Path(str(image_tar))
        if not path.is_absolute():
            # Bundled rows use ../docker/<dataset>/<archive> from HERO.
            # IMAGE_DIR remains an override for the selected dataset directory.
            path = image_dir / (path.name if path.parts[:2] == ("..", "docker") else path)
        return str(image), path

    instance_id = str(instance["instance_id"])
    owner, repo_issue = instance_id.split("__", 1)
    repo, issue = repo_issue.rsplit("-", 1)
    filename = f"sweb.eval.x86_64.{owner}_1776_{repo}-{issue}__latest.tar"
    path = image_dir / filename
    if not path.is_file():
        suffix = instance_id.replace("__", "_s_").lower()
        filename = f"sweb.eval.x86_64.{suffix}__latest.tar"
        path = image_dir / filename
        image = f"xingyaoww/sweb.eval.x86_64.{suffix}:latest"
    else:
        image = f"swebench/sweb.eval.x86_64.{owner}_1776_{repo}-{issue}:latest"
    return str(image), path


def safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value)


def build_payload(args: argparse.Namespace, row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    instance = instance_from_row(row)
    instance_id = str(instance["instance_id"])
    image, image_tar = image_for_row(row, instance, args.image_dir)
    grading_backend = (
        "swegym" if str(image).startswith("xingyaoww/sweb.eval") else "swebench"
    )
    # Carry the backend in the instance as well as evaluator.config.  The
    # instance is the durable grading payload and resolve_backend() explicitly
    # supports this marker; keeping it here prevents an evaluator config lost
    # by an older rollout/gateway wire schema from failing before model-patch
    # extraction (which previously turned every row into ``empty_patch``).
    instance["eval_backend"] = grading_backend
    if not image_tar.is_file():
        raise FileNotFoundError(f"local image tar missing for {instance_id}: {image_tar}")
    instruction = str(instance.get("problem_statement") or _prompt_text(row)).strip()
    payload = {
        "task_id": f"hero-eval-{safe_id(instance_id)}-{int(time.time())}",
        "instruction": instruction,
        "num_samples": args.num_samples,
        "timeout_seconds": args.timeout,
        "runtime": {
            "backend": "docker",
            "import_path": "run_rl._shared.runtime.local_docker_runtime:ManagedDockerRuntime",
            "image": image,
            "network": "host",
            "workdir": "/hero/session/workspace",
            "env": {
                "HOME": "/hero/session/home",
                "PATH": "/hero/session/home/.venv/bin:/opt/node/bin:/opt/miniconda3/envs/testbed/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            },
            "kwargs": {
                "image_tar": str(image_tar.resolve()),
                "volumes": [f"{Path(args.agent_cli_dir).resolve()}:/opt/node:ro"],
                # Run the agent command via a non-login shell (``bash -c``).  The
                # default ``login_shell=True`` makes some SWE-bench images' bash
                # profile reset PATH and drop ``/opt/node/bin`` (where the
                # Claude Code CLI is mounted), so ``claude`` is then "command
                # not found" (exit 127) and every step-0 turns into
                # ``no completions``.  The golden grader path already uses
                # ``login_shell=False`` for the same reason; mirroring it here
                # fixes e.g. every astral-sh__ruff instance in multilingual.
                "login_shell": False,
            },
            # The image's /testbed must alias the agent workspace: tools and
            # installed editable packages may use that absolute path. Keep
            # refresh-runtime grading on its separate, pristine /testbed.
            "prepare": [{
                "type": "exec",
                "command": "rm -rf /hero/session/workspace && mkdir -p /hero/session/logs/agent /hero/session/workspace $HOME/.venv/bin && cp -a /testbed/. /hero/session/workspace/ && ln -sf /opt/miniconda3/envs/testbed/bin/python $HOME/.venv/bin/python && ln -sf /opt/miniconda3/envs/testbed/bin/python $HOME/.venv/bin/python3 && git config --global --add safe.directory /hero/session/workspace && cd /hero/session/workspace && (git reset --hard || true) && rm -rf /testbed && ln -s /hero/session/workspace /testbed",
            }],
            "eval_prepare": [{
                "type": "exec",
                "command": "mkdir -p /hero/session/logs/agent $HOME/.venv/bin && ln -sf /opt/miniconda3/envs/testbed/bin/python $HOME/.venv/bin/python && ln -sf /opt/miniconda3/envs/testbed/bin/python $HOME/.venv/bin/python3 && git config --global --add safe.directory /testbed",
            }],
        },
        "agent": {
            "harness": "claude_code",
            "model_name": args.model_name,
            # Evaluation protocol: disable compaction with CLI environment
            # flags as well as task-local settings.
            "settings": {
                "max_turns": args.max_turns,
                "auto_compact_enabled": False,
                "git_progress": {
                    "enabled": bool(args.git_progress_enabled),
                    "checkpoint_interval": args.git_progress_checkpoint_interval,
                },
            },
            "env": {
                "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(args.max_context_tokens),
                "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(args.max_response_tokens),
                "DISABLE_AUTO_COMPACT": "1",
                "DISABLE_COMPACT": "1",
            },
        },
        "builder": {"strategy": "prefix_merging"},
        "evaluator": {
            "strategy": "swebench_harness",
            "refresh_runtime": True,
            "config": {
                "repo_dir": "/testbed",
                "patch_command": "cd /hero/session/workspace && git add -A && git diff --cached --binary",
                "instance": instance,
                # Auto-detection is intentionally rejected when a row does not
                # carry an unambiguous dataset marker.  The image selection
                # above is the authoritative distinction used by this driver.
                "backend": grading_backend,
                "test_timeout": args.test_timeout,
                "exclude_patterns": [".claude/**", "**/.claude/**", ".venv/**", "**/.venv/**", "node_modules/**", "**/node_modules/**"],
            },
        },
    }
    if args.agent_tools:
        payload["agent"]["settings"]["tools"] = args.agent_tools
    if args.agent_disallowed_tools:
        payload["agent"]["settings"]["disallowed_tools"] = args.agent_disallowed_tools
    return instance_id, payload


def validate_payload(args: argparse.Namespace, row: dict[str, Any]) -> None:
    """Fail at launch if the model-path evaluator cannot be constructed."""
    _, payload = build_payload(args, row)
    evaluator = payload["evaluator"]
    from hero.trajectory.models import StrategySpec
    from hero.trajectory.registry import default_evaluator_registry

    registry = default_evaluator_registry()
    registry.create(StrategySpec.model_validate(evaluator))


# ---------------------------------------------------------------------------
# Grading: ProRL-owned boundary shared with training.
# ---------------------------------------------------------------------------


async def grade_patch(
    row: dict[str, Any],
    instance: dict[str, Any],
    image: str,
    image_tar: str,
    patch: str,
    *,
    eval_timeout: float,
    eval_log_dst: str | None,
) -> dict[str, Any]:
    """Grade ``patch`` through ProRL's training-time SWE grading boundary.

    Returns a result dict with ``resolved``/``reward``/``applied``/``error``/
    ``grading_report`` plus per-instance f2p/p2p ratios when available.
    """
    if not patch or "diff --git" not in patch:
        return {
            "resolved": False,
            "reward": 0.0,
            "applied": True,
            "error": "empty_patch",
            "grading_report": {},
        }

    from hero.runtime.models import RuntimeSpec
    from hero.trajectory.evaluator.swe_grading import (
        SweBackend,
        resolved_and_reward,
        run_harness_on_applied_patch,
    )
    from run_rl._shared.runtime.local_docker_runtime import ManagedDockerRuntime

    backend = (
        SweBackend.SWEGYM
        if str(image).startswith("xingyaoww/sweb.eval")
        else SweBackend.SWEBENCH
    )
    workdir = os.environ.get("HERO_EVAL_WORKDIR", "/testbed")
    runtime_root = Path(eval_log_dst).parent / "grader_runtime" if eval_log_dst else Path("/tmp") / f"hero-grade-{safe_id(str(instance['instance_id']))}"
    runtime_root.mkdir(parents=True, exist_ok=True)
    runtime_root = runtime_root.resolve()
    runtime = ManagedDockerRuntime(
        RuntimeSpec(
            image=image,
            workdir=workdir,
            network="host",
            kwargs={"image_tar": str(image_tar), "login_shell": False},
        ),
        session_id=f"hero-grade-{safe_id(str(instance['instance_id']))}-{int(time.time() * 1000)}",
        session_dir=runtime_root,
    )
    applied = False
    try:
        await runtime.start()
        patch_host = runtime_root / "model.patch"
        patch_host.write_text(patch)
        apply_result = await runtime.exec(
            "git apply --verbose /hero/session/model.patch || "
            "git apply --verbose --reject /hero/session/model.patch || "
            "patch --batch --fuzz=5 -p1 -i /hero/session/model.patch",
            cwd=workdir,
            timeout_sec=min(120.0, eval_timeout),
        )
        applied = apply_result.return_code == 0
        if not applied:
            return {
                "resolved": False, "reward": 0.0, "applied": False,
                "error": "apply_failed", "grading_report": {},
            }
        report, log_path, _ = await run_harness_on_applied_patch(
            runtime=runtime,
            instance=instance,
            backend=backend,
            patch=patch,
            host_session_dir=runtime_root,
            log_dir=runtime_root / "artifacts",
            repo_dir=workdir,
            env={},
            timeout_sec=eval_timeout,
        )
        if eval_log_dst and log_path != Path(eval_log_dst):
            Path(eval_log_dst).write_text(log_path.read_text())
    except TimeoutError:
        return {
            "resolved": False, "reward": 0.0, "applied": applied,
            "error": "grading_timeout", "grading_report": {},
        }
    except Exception as exc:  # noqa: BLE001 - surface any grader crash
        return {
            "resolved": False, "reward": 0.0, "applied": applied,
            "error": f"grading_error:{type(exc).__name__}:{exc}",
            "grading_report": {},
        }
    finally:
        await runtime.stop()

    resolved, reward = resolved_and_reward(report)
    error = "" if resolved else str(report.get("failure_reason") or "tests_failed")
    result = {
        "resolved": resolved,
        "reward": reward,
        "applied": applied,
        "error": error,
        "grading_report": report,
    }
    for key in ("f2p_ratio", "p2p_ratio", "f2p_passed", "f2p_total", "p2p_passed", "p2p_total"):
        if key in report:
            result[key] = report[key]
    return result


def _session_usage(session: dict[str, Any]) -> dict[str, Any]:
    trajectory = session.get("trajectory") if isinstance(session.get("trajectory"), dict) else {}
    usage = trajectory.get("metadata", {}).get("token_usage") if isinstance(trajectory.get("metadata"), dict) else None
    if isinstance(usage, dict):
        return usage
    return {
        "turn_count": None,
        "total_input_tokens": None,
        "total_output_tokens": None,
        "total_tokens": None,
        "total_reasoning_tokens": None,
        "total_content_tokens": None,
        "metrics_missing": True,
        "turn_usage": [],
    }


def _result_usage(result: dict[str, Any]) -> dict[str, Any]:
    sessions = result.get("results") if isinstance(result.get("results"), list) else []
    per_session = [
        {"session_id": item.get("session_id"), "status": item.get("status"), "token_usage": _session_usage(item)}
        for item in sessions if isinstance(item, dict)
    ]

    def total(field: str) -> int:
        return sum(int((item["token_usage"].get(field) or 0)) for item in per_session)

    def optional_total(field: str) -> int | None:
        values = [item["token_usage"].get(field) for item in per_session]
        return (
            sum(int(value) for value in values)
            if values and all(value is not None for value in values)
            else None
        )
    return {
        "schema_version": 1,
        "trajectory_count": len(per_session),
        "turn_count": total("turn_count"),
        "total_input_tokens": total("total_input_tokens"),
        "total_output_tokens": total("total_output_tokens"),
        "total_tokens": total("total_tokens"),
        "total_reasoning_tokens": optional_total("total_reasoning_tokens"),
        "total_content_tokens": optional_total("total_content_tokens"),
        "metrics_missing": any(item["token_usage"].get("metrics_missing", True) for item in per_session),
        "sessions": per_session,
    }


def _write_usage_artifacts(instance_dir: Path, usage: dict[str, Any]) -> None:
    (instance_dir / "usage.json").write_text(json.dumps(usage, ensure_ascii=False, indent=2) + "\n")
    with (instance_dir / "token_usage.jsonl").open("w") as output:
        for session in usage["sessions"]:
            for turn in session["token_usage"].get("turn_usage", []):
                output.write(json.dumps({"session_id": session["session_id"], **turn}, ensure_ascii=False) + "\n")


def _extract_session_patches(result: dict[str, Any]) -> list[str]:
    """Pull the model patch (one per session) the gateway evaluator wrote back."""
    patches: list[str] = []
    for session in result.get("results") or []:
        if not isinstance(session, dict):
            patches.append("")
            continue
        trajectory = session.get("trajectory") if isinstance(session.get("trajectory"), dict) else {}
        metadata = trajectory.get("metadata") if isinstance(trajectory, dict) else None
        evaluation = metadata.get("evaluation") if isinstance(metadata, dict) else None
        patch = evaluation.get("model_patch") if isinstance(evaluation, dict) else None
        patches.append(str(patch or ""))
    return patches


async def _grade_one(
    args: argparse.Namespace,
    row: dict[str, Any],
    instance: dict[str, Any],
    image: str,
    image_tar: str,
    patch: str,
    instance_dir: Path,
    *,
    eval_timeout: float,
) -> dict[str, Any]:
    eval_log_dst = str(instance_dir / "eval.log")
    graded = await grade_patch(
        row, instance, image, image_tar, patch,
        eval_timeout=eval_timeout, eval_log_dst=eval_log_dst,
    )
    (instance_dir / "model.patch").write_text(patch, encoding="utf-8")
    return graded


async def _golden_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Grade the dataset's gold patch directly (no HERO rollout)."""
    instance = instance_from_row(row)
    instance_id = str(instance["instance_id"])
    image, image_tar = image_for_row(row, instance, args.image_dir)
    if not image_tar.is_file():
        raise FileNotFoundError(f"local image tar missing for {instance_id}: {image_tar}")
    gold_patch = str(instance.get("patch") or (row.get("metadata") or {}).get("patch") or "")
    if args.gold_patch_dir is not None:
        patch_path = args.gold_patch_dir / instance_id / "patch.diff"
        if not patch_path.is_file():
            raise FileNotFoundError(f"gold patch missing: {patch_path}")
        gold_patch = patch_path.read_text(encoding="utf-8")
    instance_dir = args.work_dir / "instances" / safe_id(instance_id)
    instance_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    graded = await _grade_one(
        args, row, instance, image, image_tar, gold_patch, instance_dir,
        eval_timeout=args.test_timeout,
    )
    return {
        "instance_id": instance_id,
        "resolved": graded["resolved"],
        "reward": graded["reward"],
        "applied": graded["applied"],
        "golden": True,
        "patch": gold_patch,
        "error": graded["error"],
        "grading_report": graded["grading_report"],
        "elapsed_sec": round(time.monotonic() - started, 3),
    }


_ROLLOUT_RETRY_EXC = (
    httpx.TransportError,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.PoolTimeout,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.ConnectError,
)
_ROLLOUT_MAX_RETRIES = 6
_ROLLOUT_BACKOFF = 1.5  # seconds, doubled each retry


def _rollout_request(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    json: dict[str, Any] | None = None,
    tag: str = "",
) -> httpx.Response:
    """Submit/poll to HERO with retries for transient transport failures.

    HERO is a long-lived local server and reuses keepalive connections across
    thousands of polls. The server may close an idle connection at any moment,
    which httpx surfaces as ``RemoteProtocolError`` / ``TransportError``. Without
    retries a single such race kills the whole eval (observed: ckpt1 died at
    466/500). We also retry on 5xx since HERO can briefly return them while
    recycling a worker. Permanent errors (4xx) are returned to the caller, which
    calls ``raise_for_status()`` as before.
    """
    delay = _ROLLOUT_BACKOFF
    last_exc: Exception | None = None
    for attempt in range(_ROLLOUT_MAX_RETRIES + 1):
        try:
            response = client.request(method, url, json=json)
        except _ROLLOUT_RETRY_EXC as exc:
            last_exc = exc
            if attempt < _ROLLOUT_MAX_RETRIES:
                logger.warning(
                    "hero %s transient %s: %s (attempt %d/%d), retrying in %.1fs",
                    tag or url, type(exc).__name__, exc,
                    attempt + 1, _ROLLOUT_MAX_RETRIES, delay,
                )
                time.sleep(delay)
                delay *= 2
                continue
            raise
        # Retry only on server errors; 4xx is a real failure for the caller.
        if response.status_code >= 500 and attempt < _ROLLOUT_MAX_RETRIES:
            logger.warning(
                "hero %s returned %d (attempt %d/%d), retrying in %.1fs",
                tag or url, response.status_code,
                attempt + 1, _ROLLOUT_MAX_RETRIES, delay,
            )
            time.sleep(delay)
            delay *= 2
            continue
        return response
    # Unreachable: loop either returns or raises. Defensive for type-checkers.
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("unreachable rollout retry loop")


async def main_async(args: argparse.Namespace) -> int:
    args.work_dir.mkdir(parents=True, exist_ok=True)
    instances_dir = args.work_dir / "instances"
    instances_dir.mkdir(exist_ok=True)
    results_path = args.work_dir / "results.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if results_path.exists() and not args.no_resume:
        for line in results_path.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                completed[str(item["instance_id"])] = item
    elif args.no_resume:
        results_path.write_text("")

    selected: list[dict[str, Any]] = []
    retry_ids: set[str] | None = None
    if args.retry_failed_from is not None:
        retry_ids = set()
        with args.retry_failed_from.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                previous = json.loads(line)
                if not bool(previous.get("resolved")):
                    retry_ids.add(str(previous["instance_id"]))

    for row in load_rows(args.input):
        instance = instance_from_row(row)
        instance_id = str(instance["instance_id"])
        if retry_ids is not None and instance_id not in retry_ids:
            continue
        if args.instance_id and instance_id != args.instance_id:
            continue
        if instance_id in completed:
            continue
        selected.append(row)
        if args.limit > 0 and len(selected) >= args.limit:
            break

    if selected and not args.golden:
        validate_payload(args, selected[0])

    expected_ids = {str(instance_from_row(row)["instance_id"]) for row in selected}

    # ---- Golden path: grade gold patches directly, no HERO server. ----
    if args.golden:
        # Golden mode only grades, so use the dedicated Docker grader ceiling.
        semaphore = asyncio.Semaphore(max(1, args.grade_concurrency))
        write_lock = asyncio.Lock()
        new_records: list[dict[str, Any]] = []

        async def _golden_one(row: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                try:
                    return await _golden_row(row, args)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("golden failed for %s: %s", instance_from_row(row)["instance_id"], exc)
                    return {
                        "instance_id": str(instance_from_row(row)["instance_id"]),
                        "resolved": False, "reward": 0.0, "applied": False,
                        "golden": True, "error": f"golden_exception:{type(exc).__name__}:{exc}",
                        "patch": "", "grading_report": {},
                    }

        for coro in asyncio.as_completed([_golden_one(row) for row in selected]):
            record = await coro
            async with write_lock:
                with results_path.open("a") as output:
                    output.write(json.dumps(record, ensure_ascii=True) + "\n")
                completed[record["instance_id"]] = record
                new_records.append(record)
            print(f"golden {record['instance_id']}: resolved={record['resolved']} error={record.get('error') or '-'}", flush=True)
        return _summarize(args, completed, expected_ids)

    # ---- Model path: keep rollout and independent grading as two pipelines. ----
    # Previously this loop awaited each official re-grade while it held the
    # submission loop. A single long test therefore stopped task replacement,
    # left every model worker with an empty request queue, and serialized all
    # completed patches. Separate semaphores retain the independent official
    # grade while bounding both Docker-heavy stages.
    timeout = httpx.Timeout(connect=30.0, read=120.0, write=60.0, pool=60.0)
    limits = httpx.Limits(
        max_connections=max(100, args.concurrency * 2),
        max_keepalive_connections=max(20, args.concurrency),
    )
    rollout_semaphore = asyncio.Semaphore(args.concurrency)
    grade_semaphore = asyncio.Semaphore(args.grade_concurrency)
    write_lock = asyncio.Lock()

    async def _request(
        client: httpx.Client,
        method: str,
        url: str,
        *,
        payload: dict[str, Any] | None = None,
        tag: str,
    ) -> httpx.Response:
        # _rollout_request deliberately remains the single retry policy. The
        # blocking local HTTP call runs in the default executor so 48 polling
        # coroutines never block Docker graders or one another.
        return await asyncio.to_thread(
            _rollout_request,
            client,
            method,
            url,
            json=payload,
            tag=tag,
        )

    async def _persist(record: dict[str, Any], instance_dir: Path) -> None:
        async with write_lock:
            with results_path.open("a") as output:
                output.write(json.dumps(record, ensure_ascii=True) + "\n")
            if "token_usage" in record:
                _write_usage_artifacts(instance_dir, record["token_usage"])
            completed[str(record["instance_id"])] = record

    async def _run_one(client: httpx.Client, row: dict[str, Any]) -> None:
        instance = instance_from_row(row)
        instance_id = str(instance["instance_id"])
        instance_dir = instances_dir / safe_id(instance_id)
        instance_dir.mkdir(exist_ok=True)
        task_id = ""
        try:
            # Release this slot as soon as HERO has finished the rollout. The
            # next row can then start while this patch waits for a grader.
            async with rollout_semaphore:
                _, payload = build_payload(args, row)
                response = await _request(
                    client, "POST", "/rollout/task/submit",
                    payload=payload, tag=f"submit:{instance_id}",
                )
                response.raise_for_status()
                task_id = str(response.json()["task_id"])
                print(f"submitted {instance_id}: {task_id}", flush=True)
                while True:
                    await asyncio.sleep(args.poll_interval)
                    response = await _request(
                        client, "GET", f"/rollout/task/{task_id}",
                        tag=f"poll:{instance_id}",
                    )
                    response.raise_for_status()
                    result = response.json()
                    if result.get("status") != "running":
                        break

            image, image_tar = image_for_row(row, instance, args.image_dir)
            patches = _extract_session_patches(result)
            sessions = result.get("results") or []
            num_samples = len(sessions)
            sample_records: list[dict[str, Any]] = []
            for idx, patch in enumerate(patches if patches else [""]):
                session = sessions[idx] if idx < len(sessions) and isinstance(sessions[idx], dict) else {}
                trajectory = session.get("trajectory") if isinstance(session.get("trajectory"), dict) else {}
                metadata = trajectory.get("metadata") if isinstance(trajectory, dict) else {}
                evaluation = metadata.get("evaluation") if isinstance(metadata, dict) else None
                if session.get("status") == "ERROR" or not isinstance(evaluation, dict):
                    detail = str(
                        session.get("error")
                        or trajectory.get("error")
                        or "rollout completed without evaluator metadata"
                    )
                    report = evaluation.get("report") if isinstance(evaluation, dict) else None
                    report_ok = (
                        isinstance(report, dict)
                        and not report.get("error_eval", False)
                        and isinstance(report.get("grading_report"), dict)
                    )
                    if report_ok:
                        model_patch = evaluation.get("model_patch") or ""
                        (instance_dir / "model.patch").write_text(model_patch, encoding="utf-8")
                        graded = {
                            "resolved": bool(report.get("resolved")),
                            "reward": float(report.get("reward", 0.0) or 0.0),
                            "applied": bool(report.get("grading_report", {}).get("patch_successfully_applied")),
                            "error": (
                                f"session_exit_anomaly:{detail[:200]}"
                                if session.get("status") == "ERROR" else ""
                            ),
                            "grading_report": report.get("grading_report", {}),
                        }
                        sample_records.append({**graded, "sample_index": idx})
                        continue
                    if patch and "diff --git" in patch:
                        async with grade_semaphore:
                            graded = await _grade_one(
                                args, row, instance, image, image_tar, patch,
                                instance_dir, eval_timeout=args.test_timeout,
                            )
                        sample_records.append({**graded, "sample_index": idx})
                        continue
                    if trajectory.get("traces"):
                        (instance_dir / "model.patch").write_text("", encoding="utf-8")
                        sample_records.append(_patch_unavailable_result(detail, idx, evaluation))
                        continue
                    graded = {
                        "resolved": False, "reward": 0.0, "applied": False,
                        "error": f"rollout_error:{detail}", "grading_report": {},
                    }
                    (instance_dir / "model.patch").write_text("", encoding="utf-8")
                    sample_records.append({**graded, "sample_index": idx})
                    continue

                async with grade_semaphore:
                    graded = await _grade_one(
                        args, row, instance, image, image_tar, patch, instance_dir,
                        eval_timeout=args.test_timeout,
                    )
                sample_records.append({**graded, "sample_index": idx})

            resolved_any = any(item["resolved"] for item in sample_records)
            record = {
                "instance_id": instance_id,
                "resolved": resolved_any,
                "resolved_samples": sum(1 for item in sample_records if item["resolved"]),
                "num_samples": num_samples or len(sample_records),
                "task_id": task_id,
                "samples": sample_records,
                "token_usage": _result_usage(result),
                "result_paths": list(result.get("result_paths") or []),
            }
            await _persist(record, instance_dir)
            print(
                f"finished {instance_id}: resolved={resolved_any} "
                f"({record['resolved_samples']}/{record['num_samples']})",
                flush=True,
            )
        except Exception as exc:  # keep one bad instance from aborting 500 rows
            logger.exception("instance pipeline failed for %s", instance_id)
            detail = f"rollout_error:driver_pipeline:{type(exc).__name__}:{exc}"
            record = {
                "instance_id": instance_id,
                "resolved": False,
                "resolved_samples": 0,
                "num_samples": 1,
                "task_id": task_id,
                "samples": [{
                    "resolved": False, "reward": 0.0, "applied": False,
                    "error": detail, "grading_report": {}, "sample_index": 0,
                }],
                "token_usage": _result_usage({}),
                "result_paths": [],
            }
            await _persist(record, instance_dir)
            print(f"failed {instance_id}: {detail}", flush=True)

    with httpx.Client(base_url=args.rollout_url, timeout=timeout, limits=limits) as client:
        tasks = [asyncio.create_task(_run_one(client, row)) for row in selected]
        for task in asyncio.as_completed(tasks):
            await task

    return _summarize(args, completed, expected_ids)


def _summarize(args: argparse.Namespace, completed: dict[str, dict[str, Any]], expected_ids: set[str]) -> int:
    total = len(completed)
    resolved = sum(bool(item.get("resolved")) for item in completed.values())
    infra_failed = 0
    error_counts: collections.Counter = collections.Counter()
    for item in completed.values():
        error = item.get("error")
        samples = item.get("samples")
        if samples and isinstance(samples, list):
            for s in samples:
                err = s.get("error") or ""
                if err:
                    error_counts[err] += 1
                    if err.startswith(("grading_error", "rollout_error")) or err in {"grading_timeout", "evaluability_check"}:
                        infra_failed += 1
        elif error:
            error_counts[error] += 1
            if str(error).startswith(("grading_", "evaluability_check", "golden_exception")):
                infra_failed += 1
    result_ids = [str(i) for i in completed]
    result_id_set = set(result_ids)
    missing_count = len(expected_ids - result_id_set)
    extra_count = len(result_id_set - expected_ids)
    duplicate_count = len(result_ids) - len(result_id_set)
    summary = {
        "total": total,
        "resolved": resolved,
        "resolved_rate": resolved / total if total else 0.0,
        "golden": bool(args.golden),
        "infra_failed": infra_failed,
        "missing_count": missing_count,
        "extra_count": extra_count,
        "duplicate_count": duplicate_count,
        "error_counts": dict(sorted(error_counts.items())),
    }
    (args.work_dir / "eval_result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    unhealthy = bool(missing_count or extra_count or duplicate_count or infra_failed)
    if args.health_gate != "off" and unhealthy:
        print(f"[health] unhealthy run: {summary}", file=sys.stderr)
    return 1 if args.health_gate == "strict" and unhealthy else 0


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = parse_args()
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be positive")
    if args.grade_concurrency <= 0:
        raise SystemExit("--grade-concurrency must be positive")
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
