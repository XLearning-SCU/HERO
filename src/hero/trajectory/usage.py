"""Exact, portable token accounting for gateway completion sessions.

The gateway stores the upstream OpenAI-compatible response for every model
request.  This module turns the several usage dialects seen there into a
single per-turn and per-trajectory schema.  Counts are deliberately request
counts: repeated context on later agent turns is included, which reflects the
actual inference work.
"""

from __future__ import annotations

from typing import Any

from hero.trajectory.models import CompletionRecord, CompletionSession


def _nonnegative_int(value: Any) -> int | None:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _first_int(*values: Any) -> int | None:
    for value in values:
        parsed = _nonnegative_int(value)
        if parsed is not None:
            return parsed
    return None


def completion_token_usage(completion: CompletionRecord, *, turn: int) -> dict[str, Any]:
    """Normalize one stored model call, including optional reasoning/content."""
    response = completion.response if isinstance(completion.response, dict) else {}
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    choices = response.get("choices") if isinstance(response.get("choices"), list) else []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    details = usage.get("completion_tokens_details")
    details = details if isinstance(details, dict) else {}

    prompt_ids = choice.get(
        "input_token_ids",
        choice.get("prompt_token_ids", response.get("prompt_token_ids")),
    )
    response_ids = choice.get("token_ids", response.get("token_ids"))
    input_tokens = _first_int(usage.get("prompt_tokens"), usage.get("input_tokens"))
    output_tokens = _first_int(usage.get("completion_tokens"), usage.get("output_tokens"))
    source = "upstream_usage"
    if input_tokens is None:
        input_tokens = len(prompt_ids) if isinstance(prompt_ids, list) else 0
        source = "token_ids" if isinstance(prompt_ids, list) else "missing"
    if output_tokens is None:
        output_tokens = len(response_ids) if isinstance(response_ids, list) else 0
        source = "token_ids" if isinstance(response_ids, list) and source != "missing" else source
    total_tokens = _first_int(usage.get("total_tokens"))
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens
    reasoning_tokens = _first_int(
        details.get("reasoning_tokens"),
        usage.get("reasoning_tokens"),
        choice.get("reasoning_tokens"),
    )
    content_tokens = _first_int(details.get("content_tokens"), usage.get("content_tokens"))
    if content_tokens is None and reasoning_tokens is not None:
        content_tokens = max(0, output_tokens - reasoning_tokens)
    return {
        "turn": turn,
        "completion_id": completion.completion_id,
        "timestamp": completion.timestamp,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens,
        "content_tokens": content_tokens,
        "finish_reason": choice.get("finish_reason"),
        "source": source,
        "metrics_missing": source == "missing",
    }


def trajectory_token_usage(session: CompletionSession) -> dict[str, Any]:
    """Return a lossless summary suitable for metadata, artifacts and W&B."""
    turns = [
        completion_token_usage(record, turn=index)
        for index, record in enumerate(session.completions)
    ]
    def total(field: str) -> int:
        return sum(int(item.get(field) or 0) for item in turns)
    known_reasoning = [
        item["reasoning_tokens"] for item in turns if item["reasoning_tokens"] is not None
    ]
    known_content = [
        item["content_tokens"] for item in turns if item["content_tokens"] is not None
    ]
    return {
        "schema_version": 1,
        "turn_count": len(turns),
        "total_input_tokens": total("input_tokens"),
        "total_output_tokens": total("output_tokens"),
        "total_tokens": total("total_tokens"),
        # A partial sum would look exact downstream.  Only expose a total
        # when every turn supplied the relevant breakdown.
        "total_reasoning_tokens": (
            sum(known_reasoning) if len(known_reasoning) == len(turns) else None
        ),
        "total_content_tokens": (
            sum(known_content) if len(known_content) == len(turns) else None
        ),
        "metrics_missing": any(item["metrics_missing"] for item in turns),
        "turn_usage": turns,
    }


def git_progress_turn_checkpoints(
    token_usage: dict[str, Any],
    progress: dict[str, Any],
) -> list[dict[str, Any]]:
    """Map action-level Git observations to exact 10-turn boundaries."""
    turns = [item for item in token_usage.get("turn_usage", []) if isinstance(item, dict)]
    records = [item for item in progress.get("records", []) if isinstance(item, dict)]
    snapshots = [
        item for item in progress.get("diff_snapshots", []) if isinstance(item, dict)
    ]
    interval = max(1, int(progress.get("checkpoint_interval") or 10))
    if not turns or not records:
        return []

    checkpoints: list[dict[str, Any]] = []
    previous_hash: str | None = None
    for turn_number in range(interval, len(turns) + 1, interval):
        # Completion timestamps mark model-response completion. PostToolUse
        # follows it and precedes the next completion, so the next turn is the
        # exclusive boundary for state after this turn.
        boundary = (
            str(turns[turn_number].get("timestamp") or "")
            if turn_number < len(turns)
            else None
        )
        eligible = [
            record for record in records
            if record.get("timestamp")
            and (boundary is None or str(record["timestamp"]) < boundary)
        ]
        if not eligible:
            continue
        current = eligible[-1]
        state_hash = str(current.get("state_hash") or "")
        matching = [
            snapshot for snapshot in snapshots
            if str(snapshot.get("state_hash") or "") == state_hash
            and int(snapshot.get("action_index") or 0)
            <= int(current.get("action_index") or 0)
        ]
        snapshot = matching[-1] if matching else None
        changed_from_head = int(current.get("patch_bytes") or 0) > 0
        checkpoints.append(
            {
                "turn": turn_number,
                "action_index": current.get("action_index"),
                "state_hash": state_hash,
                "changed_from_head": changed_from_head,
                "changed_since_previous_checkpoint": (
                    changed_from_head if previous_hash is None else state_hash != previous_hash
                ),
                "patch_bytes": current.get("patch_bytes"),
                "changed_files": current.get("changed_files"),
                "changed_file_paths": current.get("changed_file_paths") or [],
                "diff_snapshot_id": snapshot.get("snapshot_id") if snapshot else None,
            }
        )
        previous_hash = state_hash
    return checkpoints
