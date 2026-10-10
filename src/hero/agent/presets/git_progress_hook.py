#!/usr/bin/env python3
"""Claude Code PostToolUse hook for lightweight Git progress snapshots.

The hook is copied into the per-session bind mount by ``ClaudeCodeHarness``.
It never mutates the Git index and always exits successfully so observability
cannot fail an agent rollout.
"""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


DEFAULT_EXCLUDES = (
    "__pycache__/**",
    "**/__pycache__/**",
    "*.pyc",
    "**/*.pyc",
    ".pytest_cache/**",
    "**/.pytest_cache/**",
    ".cache/**",
    "**/.cache/**",
    ".venv/**",
    "**/.venv/**",
    "node_modules/**",
    "**/node_modules/**",
    "trajectories/**",
    "**/trajectories/**",
    ".claude/**",
    "**/.claude/**",
    ".codex/**",
    "**/.codex/**",
)
GIT_TIMEOUT_SECONDS = float(os.environ.get("HERO_GIT_PROGRESS_GIT_TIMEOUT", "1"))
MAX_UNTRACKED_HASH_BYTES = int(
    os.environ.get("HERO_GIT_PROGRESS_MAX_HASH_BYTES", str(8 * 1024 * 1024))
)


def _excluded(path: str) -> bool:
    normalized = path.replace(os.sep, "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return any(fnmatch.fnmatch(normalized, pattern) for pattern in DEFAULT_EXCLUDES)


def _git(repo: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=GIT_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.decode(errors="replace")[-1000:])
    return completed.stdout


def _tracked_diff(repo: Path, baseline: str) -> bytes:
    pathspecs = [".", *(f":(exclude){pattern}" for pattern in DEFAULT_EXCLUDES)]
    return _git(repo, "diff", "--binary", "--no-ext-diff", baseline, "--", *pathspecs)


def _complete_diff(repo: Path, tracked_diff: bytes) -> bytes:
    """Compose tracked and untracked patches without scanning via ``git add``.

    ``git diff BASE`` already includes staged and agent-committed changes. Only
    genuinely untracked files need synthetic ``--no-index`` patches. This is
    substantially cheaper on large repos than building a temporary full index
    after every state change.
    """
    parts = [tracked_diff]
    for relative in _untracked_files(repo):
        completed = subprocess.run(
            [
                "git", "-C", str(repo), "diff", "--no-index", "--binary",
                "--", "/dev/null", relative,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=GIT_TIMEOUT_SECONDS,
        )
        # `git diff --no-index` returns 1 when differences were found.
        if completed.returncode not in (0, 1):
            raise RuntimeError(completed.stderr.decode(errors="replace")[-1000:])
        parts.append(completed.stdout)
    return b"".join(parts)


def _changed_tracked_files(repo: Path, baseline: str) -> list[str]:
    raw = _git(repo, "diff", "--name-only", "-z", baseline, "--", ".")
    return [
        value.decode(errors="replace")
        for value in raw.split(b"\0")
        if value and not _excluded(value.decode(errors="replace"))
    ]


def _untracked_files(repo: Path) -> list[str]:
    raw = _git(repo, "ls-files", "--others", "--exclude-standard", "-z")
    return [
        value.decode(errors="replace")
        for value in raw.split(b"\0")
        if value and not _excluded(value.decode(errors="replace"))
    ]


def _hash_file(path: Path, digest: Any) -> int:
    if path.is_symlink():
        value = os.readlink(path).encode(errors="surrogateescape")
        digest.update(b"symlink\0" + value)
        return len(value)
    full_size = path.stat().st_size
    size = 0
    with path.open("rb") as source:
        while size < MAX_UNTRACKED_HASH_BYTES and (
            chunk := source.read(min(1024 * 1024, MAX_UNTRACKED_HASH_BYTES - size))
        ):
            digest.update(chunk)
            size += len(chunk)
    # Size keeps large-file tail changes observable without hashing unlimited
    # generated content on the agent's synchronous PostToolUse path.
    digest.update(f"full-size:{full_size}".encode())
    return full_size


def capture_git_state(
    repo: Path,
    baseline: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    """Return a compact state summary and the tracked binary diff."""
    baseline = baseline or _git(repo, "rev-parse", "--verify", "HEAD").decode().strip()
    tracked_diff = _tracked_diff(repo, baseline)
    tracked = _changed_tracked_files(repo, baseline)
    untracked = _untracked_files(repo)
    digest = hashlib.sha256()
    digest.update(b"tracked-diff\0")
    digest.update(tracked_diff)
    untracked_bytes = 0
    for relative in sorted(untracked):
        digest.update(b"untracked\0" + relative.encode(errors="surrogateescape") + b"\0")
        path = repo / relative
        try:
            untracked_bytes += _hash_file(path, digest)
        except OSError as exc:
            digest.update(f"unreadable:{exc}".encode(errors="replace"))
    changed = sorted(set(tracked + untracked))
    return (
        {
            "state_hash": digest.hexdigest(),
            "patch_bytes": len(tracked_diff) + untracked_bytes,
            "tracked_diff_bytes": len(tracked_diff),
            "untracked_bytes": untracked_bytes,
            "changed_files": len(changed),
            "changed_file_paths": changed[:20],
            "changed_file_paths_truncated": len(changed) > 20,
        },
        tracked_diff,
    )


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def record_progress(payload: dict[str, Any]) -> dict[str, Any]:
    log_path = Path(
        os.environ.get(
            "HERO_GIT_PROGRESS_LOG",
            "/hero/session/logs/agent/git-progress.jsonl",
        )
    )
    state_dir = log_path.parent / ".git-progress"
    state_dir.mkdir(parents=True, exist_ok=True)
    interval = max(1, int(os.environ.get("HERO_GIT_PROGRESS_INTERVAL", "10")))
    repo = Path(
        os.environ.get("HERO_GIT_PROGRESS_REPO")
        or str(payload.get("cwd") or "")
        or os.getcwd()
    ).resolve()

    lock_path = state_dir / "lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        counter_path = state_dir / "counter"
        try:
            action_index = int(counter_path.read_text()) + 1
        except (OSError, ValueError):
            action_index = 1

        baseline_path = state_dir / "baseline-commit"
        try:
            baseline = baseline_path.read_text().strip()
            baseline_source = "setup"
        except OSError:
            baseline = ""
        if not baseline:
            baseline = _git(repo, "rev-parse", "--verify", "HEAD").decode().strip()
            baseline_source = "hook_fallback"
            baseline_path.write_text(baseline + "\n")

        # This side log is written before any potentially slow Git operation.
        # If the outer hard timeout kills the hook, the unmatched start record
        # makes the missing observation explicit during evaluation.
        attempts_path = log_path.parent / "git-progress-attempts.jsonl"
        attempt = {
            "schema_version": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action_index": action_index,
            "tool_use_id": payload.get("tool_use_id"),
            "tool": payload.get("tool_name"),
            "status": "started",
        }
        _append_jsonl(attempts_path, attempt)
        counter_path.write_text(str(action_index))

        summary, tracked_diff = capture_git_state(repo, baseline)
        last_hash_path = state_dir / "last-state-hash"
        try:
            previous_hash = last_hash_path.read_text().strip() or None
        except OSError:
            previous_hash = None
        state_changed = (
            previous_hash != summary["state_hash"]
            if previous_hash is not None
            else bool(summary["changed_files"])
        )

        checkpoint_name = None
        state_snapshot_name = None
        if state_changed or action_index % interval == 0:
            complete_diff = _complete_diff(repo, tracked_diff)
        if action_index % interval == 0:
            checkpoint_name = f"git-progress-{action_index:04d}.diff"
            (state_dir / checkpoint_name).write_bytes(complete_diff)
        if state_changed:
            state_snapshot_name = f"git-progress-state-{action_index:04d}.diff"
            (state_dir / state_snapshot_name).write_bytes(complete_diff)

        record = {
            "schema_version": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action_index": action_index,
            "hook_event_name": payload.get("hook_event_name"),
            "tool_use_id": payload.get("tool_use_id"),
            "tool": payload.get("tool_name"),
            "state_changed": state_changed,
            "checkpoint": checkpoint_name,
            "state_snapshot": state_snapshot_name,
            "checkpoint_interval": interval,
            "baseline_commit": baseline,
            "baseline_source": baseline_source,
            **summary,
        }
        _append_jsonl(log_path, record)
        last_hash_path.write_text(str(summary["state_hash"]))
        _append_jsonl(
            attempts_path,
            {
                **attempt,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "completed",
            },
        )
        return record


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            payload = {}
        record_progress(payload)
    except Exception as exc:  # Observability must never abort the agent.
        error_log = Path(
            os.environ.get(
                "HERO_GIT_PROGRESS_ERROR_LOG",
                "/hero/session/logs/agent/git-progress-errors.log",
            )
        )
        try:
            error_log.parent.mkdir(parents=True, exist_ok=True)
            with error_log.open("a", encoding="utf-8") as output:
                output.write(f"{datetime.now(timezone.utc).isoformat()} {type(exc).__name__}: {exc}\n")
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
