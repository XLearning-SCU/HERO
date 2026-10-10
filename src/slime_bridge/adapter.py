"""Convert HERO rollout results into Slime samples.

Every trace in ``Trajectory.traces`` becomes one Slime ``Sample``. All samples
produced from the same session share both ``Sample.rollout_id`` and
``Sample.trajectory_id``. This preserves Slime v0.3 ``group_id`` semantics:
siblings stay in one optimizer step and their loss counts as one trajectory.
Builders own trace curation and per-token loss masks
— the adapter does not infer trainable positions from bridge details. Traces
that lack training tokens are dropped and represented as fully masked samples
so callers can keep the rest of the group trainable.
"""

from __future__ import annotations

from copy import deepcopy
import logging
import os
from typing import Any, TYPE_CHECKING

from slime_bridge._messages import messages_to_text

if TYPE_CHECKING:
    from hero.rollout.models import SessionResult
    from hero.trajectory.models import Trace

logger = logging.getLogger(__name__)

_TRAIN_POLICY_EARLY_STOPS_ENV = "HERO_TRAIN_POLICY_EARLY_STOP_TRAJECTORIES"
_MAX_STEP_ERROR_MARKERS = (
    "error_max_turns",
    "error_max_completions",
    "reached maximum number of turns",
    "max_steps_exceeded",
    "turn exceeded max_steps",
)
_CONTEXT_OVERFLOW_ERROR_MARKERS = (
    "context_overflow",
    "context_length_exceeded",
    "longer than the model's context length",
    "longer than the model’s context length",
)


class RolloutLogprobError(ValueError):
    """Raised when a trainable HERO trace lacks aligned rollout logprobs."""


def session_result_to_samples(
    result: "SessionResult",
    group_index: int,
    *,
    trajectory_index: int,
    reward_key: str = "score",
    max_tokens: int | None = None,
) -> list[Any]:
    """Convert one HERO session result into Slime samples — one per trace.

    Every usable trace becomes a Sample sharing the same scheduling
    ``rollout_id`` and loss ``trajectory_id``. Slime's reducer then averages
    all trace contributions as one trajectory, while the reward post-processor
    can still assign each trace its own advantage.

    Traces with empty tokens or exceeding ``max_tokens`` are dropped
    (logged). If *all* traces are dropped we emit a single zero-gradient
    placeholder so Slime's flattener doesn't crash on an empty list and
    the rest of the group can still train.
    """
    Sample = _load_sample_type()
    traces = result.trajectory.traces
    samples: list[Any] = []
    for trace_index, trace in enumerate(traces):
        sample = _build_sample(
            Sample=Sample,
            result=result,
            trace=trace,
            trace_index=trace_index,
            group_index=group_index,
            index=trajectory_index,
            reward_key=reward_key,
            max_tokens=max_tokens,
        )
        if sample is not None:
            samples.append(sample)

    if samples:
        return samples

    logger.warning(
        "Session %s: no usable trace (traces=%d, max_tokens=%s); emitting dummy placeholder",
        result.session_id, len(traces), max_tokens,
    )
    return [_build_dummy_sample(
        Sample=Sample,
        result=result,
        group_index=group_index,
        index=trajectory_index,
        reward_key=reward_key,
    )]


def _build_sample(
    *,
    Sample: Any,
    result: "SessionResult",
    trace: "Trace",
    trace_index: int,
    group_index: int,
    index: int,
    reward_key: str,
    max_tokens: int | None = None,
) -> Any | None:
    prompt_ids = list(trace.prompt_ids)
    response_ids = list(trace.response_ids)

    if not prompt_ids or not response_ids:
        logger.warning(
            "Dropping trace %d from session %s: missing tokens (prompt=%d, response=%d)",
            trace_index, result.session_id, len(prompt_ids), len(response_ids),
        )
        return None

    total_len = len(prompt_ids) + len(response_ids)
    if max_tokens is not None and total_len > max_tokens:
        logger.warning(
            "Dropping trace %d from session %s: total_len=%d > max_tokens=%d",
            trace_index, result.session_id, total_len, max_tokens,
        )
        return None

    prompt_messages = deepcopy(trace.prompt_messages)
    response_messages = deepcopy(trace.response_messages)
    response_text = messages_to_text(response_messages)

    termination_kind = _policy_termination_kind(result)
    early_stop_kind = (
        termination_kind
        if termination_kind in {"max_steps_exceeded", "context_overflow"}
        else None
    )
    harvest_early_stop = (
        early_stop_kind is not None
        and _train_policy_early_stops_enabled()
        and _trace_has_complete_training_data(trace, len(response_ids))
    )
    status = _sample_status(
        Sample,
        result,
        trace,
        harvest_policy_early_stop=harvest_early_stop,
        context_overflow=termination_kind == "context_overflow",
    )
    # Harvesting controls trainability, not task outcome. Preserve the
    # evaluator's reward: an agent may hit a policy limit after already
    # producing a patch that resolves the task, matching Dressage semantics.
    reward_value = _reward_value(trace)

    trainable = status not in (Sample.Status.ABORTED, Sample.Status.FAILED)
    loss_mask = _loss_mask_from_trace(
        trace,
        len(response_ids),
        require_loss_mask=trainable,
        session_id=result.session_id,
        trace_index=trace_index,
    )
    if status in (Sample.Status.ABORTED, Sample.Status.FAILED):
        loss_mask = [0] * len(response_ids)
    response_log_probs = _extract_rollout_log_probs(
        trace,
        response_len=len(response_ids),
        loss_mask=loss_mask,
        require_trainable_logprobs=trainable,
        session_id=result.session_id,
        trace_index=trace_index,
    )

    prompt_value = prompt_messages if prompt_messages else ""

    hero_metadata: dict[str, Any] = {
        "node_id": result.node_id,
        "result_metadata": deepcopy(getattr(result, "metadata", {}) or {}),
        "result_error": result.error,
        "session_id": result.session_id,
        "session_status": result.status,
        "task_id": result.task_id,
        "timing": result.timing.model_dump(mode="python"),
        "trace_index": trace_index,
        "trace_metadata": deepcopy(getattr(trace, "metadata", {}) or {}),
        "trajectory_error": result.trajectory.error,
        "trajectory_metadata": deepcopy(result.trajectory.metadata),
        "trajectory_status": result.trajectory.status,
        # Preserved for the longest-trace wandb artifact dump; training reads
        # tokens+logprobs, not these.
        "trace_debug": {
            "finish_reason": trace.finish_reason,
            # response_messages is already a private deep copy made above.
            "response_messages": response_messages,
        },
    }
    if harvest_early_stop:
        hero_metadata["harvested_policy_early_stop"] = early_stop_kind
    hero_metadata.update(_scheduler_metadata(result, trace))
    turn_token_spans = (
        deepcopy(trace.metadata.get("turn_token_spans"))
        if isinstance(trace.metadata, dict)
        else None
    )
    train_metadata = (
        {"hero_turn_token_spans": turn_token_spans}
        if isinstance(turn_token_spans, list)
        else None
    )

    return Sample(
        group_index=group_index,
        index=index,
        prompt=prompt_value,
        tokens=prompt_ids + response_ids,
        response=response_text,
        response_length=len(response_ids),
        # Slime v0.3 used one group_id for both DP-step grouping and the loss
        # denominator. Current Slime splits those roles across two fields, so
        # stamp the same trajectory key onto both to preserve that behavior.
        rollout_id=index,
        trajectory_id=index,
        reward={reward_key: reward_value},
        loss_mask=loss_mask,
        rollout_log_probs=response_log_probs,
        status=status,
        session_id=result.session_id,
        metadata={"hero": hero_metadata},
        remove_sample=not trainable,
        train_metadata=train_metadata,
    )


def _build_dummy_sample(
    *,
    Sample: Any,
    result: "SessionResult",
    group_index: int,
    index: int,
    reward_key: str,
) -> Any:
    """Fully masked placeholder for a session with no usable trace.

    This carries no policy, TIS, or KL contribution. It lets the scheduler
    accept a partially usable group while still surfacing empty sessions in
    HERO metrics.
    """
    hero_metadata: dict[str, Any] = {
        "node_id": result.node_id,
        "result_metadata": deepcopy(getattr(result, "metadata", {}) or {}),
        "result_error": result.error,
        "session_id": result.session_id,
        "session_status": result.status,
        "task_id": result.task_id,
        "timing": result.timing.model_dump(mode="python"),
        "trace_index": -1,
        "trajectory_error": result.trajectory.error,
        "trajectory_metadata": deepcopy(result.trajectory.metadata),
        "trajectory_status": result.trajectory.status,
        "placeholder": True,
    }
    hero_metadata.update(_scheduler_metadata(result, None))
    return Sample(
        group_index=group_index,
        index=index,
        prompt="",
        tokens=[0, 0],
        response="",
        response_length=1,
        rollout_id=index,
        trajectory_id=index,
        reward={reward_key: 0.0},
        loss_mask=[0],
        rollout_log_probs=[0.0],
        status=Sample.Status.ABORTED,
        remove_sample=True,
        session_id=result.session_id,
        metadata={"hero": hero_metadata},
    )


def _reward_value(trace: "Trace") -> float:
    """Read the reward the evaluator already placed on the trace.

    Reward assignment is the evaluator's job (including any broadcasting
    from session-level outcomes). slime_bridge just consumes what's there.
    """
    return float(trace.reward) if trace.reward is not None else 0.0


def _scheduler_metadata(result: "SessionResult", trace: "Trace | None") -> dict[str, Any]:
    keys = {"group_id", "policy_version", "rollout_step"}
    merged: dict[str, Any] = {}
    for source in (
        getattr(result, "metadata", None),
        getattr(result.trajectory, "metadata", None),
        getattr(trace, "metadata", None) if trace is not None else None,
    ):
        if not isinstance(source, dict):
            continue
        for key in keys:
            if key in source:
                merged[key] = source[key]
    return merged


def _sample_status(
    Sample: Any,
    result: "SessionResult",
    trace: "Trace",
    *,
    harvest_policy_early_stop: bool = False,
    context_overflow: bool = False,
) -> Any:
    trajectory_status = result.trajectory.status
    if trajectory_status == "TIMEOUT" or result.status == "TIMEOUT":
        return Sample.Status.ABORTED
    # A context overflow is trainable only through the explicit harvesting
    # path above, which first verifies that every captured response token has
    # an aligned loss-mask entry and rollout logprob.  Otherwise fail closed.
    if context_overflow and not harvest_policy_early_stop:
        return Sample.Status.ABORTED
    if trajectory_status == "ERROR" or result.status == "ERROR" or result.error or result.trajectory.error:
        if harvest_policy_early_stop:
            return Sample.Status.TRUNCATED
        return Sample.Status.FAILED
    if trace.finish_reason == "length":
        return Sample.Status.TRUNCATED
    return Sample.Status.COMPLETED


def _train_policy_early_stops_enabled() -> bool:
    return os.environ.get(_TRAIN_POLICY_EARLY_STOPS_ENV, "0").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _policy_termination_kind(result: "SessionResult") -> str | None:
    """Classify policy limits while keeping upstream overflow exceptional.

    Max-step markers take precedence because a long error transcript may
    contain an earlier, recovered context warning before the harness finally
    terminates at its turn limit.
    """
    error_text = "\n".join(
        str(value)
        for value in (result.error, result.trajectory.error)
        if value
    ).lower()
    if any(marker in error_text for marker in _MAX_STEP_ERROR_MARKERS):
        return "max_steps_exceeded"
    if any(marker in error_text for marker in _CONTEXT_OVERFLOW_ERROR_MARKERS):
        return "context_overflow"
    return None


def _trace_has_complete_training_data(trace: "Trace", response_len: int) -> bool:
    """Fail closed unless all captured training arrays are present and aligned."""
    loss_mask = trace.loss_mask
    logprobs = trace.response_logprobs
    return bool(
        response_len > 0
        and loss_mask
        and len(loss_mask) == response_len
        and any(int(value) != 0 for value in loss_mask)
        and logprobs
        and len(logprobs) == response_len
    )


def _extract_rollout_log_probs(
    trace: "Trace",
    *,
    response_len: int,
    loss_mask: list[int],
    require_trainable_logprobs: bool,
    session_id: str,
    trace_index: int,
) -> list[float]:
    logprobs = trace.response_logprobs
    if not logprobs:
        if require_trainable_logprobs and any(loss_mask):
            raise RolloutLogprobError(
                f"Session {session_id} trace {trace_index}: missing rollout_log_probs "
                "for trainable response tokens"
            )
        return [0.0] * response_len

    if len(logprobs) != response_len:
        raise RolloutLogprobError(
            f"Session {session_id} trace {trace_index}: rollout_log_probs length "
            f"{len(logprobs)} != response length {response_len}"
        )

    # response_logprobs is one float per response token (interstitials are 0.0,
    # masked out by loss_mask); the builder guarantees trainable tokens carry
    # their real sampled logprob.
    return [float(value) for value in logprobs]


def _loss_mask_from_trace(
    trace: "Trace",
    response_len: int,
    *,
    require_loss_mask: bool,
    session_id: str,
    trace_index: int,
) -> list[int]:
    """Read and validate the builder-assigned per-response-token loss mask."""
    mask = list(trace.loss_mask)
    if not mask:
        if require_loss_mask:
            raise RolloutLogprobError(
                f"Session {session_id} trace {trace_index}: missing loss_mask"
            )
        return [0] * response_len
    if len(mask) != response_len:
        raise RolloutLogprobError(
            f"Session {session_id} trace {trace_index}: loss_mask length "
            f"{len(mask)} != response length {response_len}"
        )
    return [1 if int(value) else 0 for value in mask]


def _load_sample_type() -> Any:
    try:
        from slime.utils.types import Sample
    except ImportError as exc:
        raise ImportError(
            "Slime is required to convert HERO rollouts into training samples. "
            "Ensure the Slime package is installed in the current environment."
        ) from exc
    return Sample
