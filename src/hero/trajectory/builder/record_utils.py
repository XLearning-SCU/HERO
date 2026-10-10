"""Helpers for converting completion records into trajectory traces."""

from __future__ import annotations

from copy import deepcopy
import math
import os
from typing import Any

from hero.trajectory.models import CompletionRecord, Trace


def _coerce_int_list(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    extracted: list[int] = []
    for item in value:
        try:
            extracted.append(int(item))
        except (TypeError, ValueError):
            return None
    return extracted


def _paired_tokens_from_logprobs_content(
    choice: dict[str, Any],
) -> tuple[list[int], list[float]] | None:
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        return None
    content = logprobs.get("content")
    if not isinstance(content, list) or not content:
        return None

    token_ids: list[int] = []
    token_logprobs: list[float] = []
    for item in content:
        if not isinstance(item, dict):
            return None
        token_id = item.get("token_id")
        logprob = item.get("logprob")
        if token_id is None or logprob is None:
            return None
        try:
            token_ids.append(int(token_id))
            token_logprobs.append(float(logprob))
        except (TypeError, ValueError):
            return None
    return token_ids, token_logprobs


def _paired_tokens_from_sglang_meta(
    choice: dict[str, Any],
) -> tuple[list[int], list[float]] | None:
    meta_info = choice.get("meta_info")
    if not isinstance(meta_info, dict):
        return None
    output_logprobs = meta_info.get("output_token_logprobs")
    if not isinstance(output_logprobs, list) or not output_logprobs:
        return None

    token_ids: list[int] = []
    token_logprobs: list[float] = []
    for item in output_logprobs:
        token_id = None
        logprob = None
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            logprob = item[0]
            token_id = item[1]
        elif isinstance(item, dict):
            token_id = item.get("token_id")
            logprob = item.get("logprob", item.get("token_logprob"))

        if token_id is None or logprob is None:
            return None
        try:
            token_ids.append(int(token_id))
            token_logprobs.append(float(logprob))
        except (TypeError, ValueError):
            return None
    return token_ids, token_logprobs


def _extract_response_tokens(
    response: dict[str, Any],
    choice: dict[str, Any],
) -> tuple[list[int], list[float] | None]:
    """Extract response ids and logprobs without mixing unaligned sources."""
    token_ids = choice.get("token_ids", response.get("token_ids"))
    response_ids = _coerce_int_list(token_ids)
    paired_candidates = (
        _paired_tokens_from_logprobs_content(choice),
        _paired_tokens_from_sglang_meta(choice),
    )

    if response_ids is not None:
        for paired in paired_candidates:
            if paired is None:
                continue
            paired_ids, paired_logprobs = paired
            if paired_ids == response_ids:
                return response_ids, paired_logprobs
        return response_ids, None

    for paired in paired_candidates:
        if paired is not None:
            return paired

    return [], None


def _generation_entropy_metadata(
    choice: dict[str, Any],
    *,
    generated_token_count: int,
) -> dict[str, Any] | None:
    """Compute opt-in entropy diagnostics for generated tokens only.

    OpenAI-compatible top_logprobs do not contain the whole vocabulary.  We
    therefore store both the observed top-k entropy contribution and a useful
    lower bound obtained by treating all omitted probability mass as one
    additional bucket.  No prompt tokens participate in these arrays.
    """
    configured_top_k = os.environ.get("HERO_GENERATION_ENTROPY_TOP_K")
    if not configured_top_k:
        return None
    try:
        requested_top_k = int(configured_top_k)
    except ValueError as exc:
        raise ValueError("HERO_GENERATION_ENTROPY_TOP_K must be an integer") from exc
    if requested_top_k <= 0:
        raise ValueError("HERO_GENERATION_ENTROPY_TOP_K must be positive")
    logprobs = choice.get("logprobs")
    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    source_content = content if isinstance(content, list) else []

    topk_contribution: list[float | None] = []
    tail_mass: list[float | None] = []
    bucketed_lower_bound: list[float | None] = []
    available_counts: list[int] = []
    for token_index in range(generated_token_count):
        generated = (
            source_content[token_index]
            if token_index < len(source_content)
            else None
        )
        alternatives = generated.get("top_logprobs") if isinstance(generated, dict) else None
        if not isinstance(alternatives, list) or not alternatives:
            topk_contribution.append(None)
            tail_mass.append(None)
            bucketed_lower_bound.append(None)
            available_counts.append(0)
            continue
        # Prefer token IDs for deduplication. Without IDs, retain rank as part
        # of the key because distinct byte-level IDs can decode to equal text.
        probabilities: dict[object, float] = {}
        for rank, item in enumerate(alternatives):
            if not isinstance(item, dict):
                continue
            value = item.get("logprob")
            try:
                probability = math.exp(float(value))
            except (TypeError, ValueError, OverflowError):
                continue
            if (
                not math.isfinite(probability)
                or probability <= 0
                or probability > 1.0 + 1e-6
            ):
                continue
            probability = min(1.0, probability)
            key = item.get("token_id")
            if key is None:
                key = (item.get("token"), rank)
            probabilities[key] = max(probabilities.get(key, 0.0), probability)
        probs = sorted(probabilities.values(), reverse=True)[:requested_top_k]
        raw_mass = sum(probs)
        if raw_mass > 1.0 + 1e-5:
            topk_contribution.append(None)
            tail_mass.append(None)
            bucketed_lower_bound.append(None)
            available_counts.append(len(probs))
            continue
        mass = min(1.0, raw_mass)
        contribution = -sum(p * math.log(p) for p in probs)
        omitted = max(0.0, 1.0 - mass)
        lower_bound = contribution - (omitted * math.log(omitted) if omitted > 0 else 0.0)
        topk_contribution.append(contribution)
        tail_mass.append(omitted)
        bucketed_lower_bound.append(lower_bound)
        available_counts.append(len(probs))

    source_aligned = len(source_content) == generated_token_count
    complete = (
        source_aligned
        and all(
            count == requested_top_k
            and topk_contribution[i] is not None
            and bucketed_lower_bound[i] is not None
            for i, count in enumerate(available_counts)
        )
    )
    return {
        "method": "top_k_plus_tail_bucket_lower_bound",
        "units": "nats",
        "requested_top_k": requested_top_k,
        "generated_token_count": generated_token_count,
        "source_logprob_token_count": len(source_content),
        "source_aligned": source_aligned,
        "complete": complete,
        "missing_token_indices": [
            i
            for i, count in enumerate(available_counts)
            if count != requested_top_k
            or topk_contribution[i] is None
            or bucketed_lower_bound[i] is None
        ],
        "available_top_k_counts": available_counts,
        "topk_entropy_contribution": topk_contribution,
        "tail_probability_mass": tail_mass,
        "entropy_lower_bound": bucketed_lower_bound,
    }


def _extract_prompt_messages(request: dict[str, Any]) -> list[dict[str, Any]]:
    messages = request.get("messages")
    if not isinstance(messages, list):
        return []
    return [deepcopy(message) for message in messages if isinstance(message, dict)]


def _extract_tools(request: dict[str, Any]) -> list[dict[str, Any]] | None:
    tools = request.get("tools")
    if not isinstance(tools, list) or not tools:
        return None
    extracted = [deepcopy(tool) for tool in tools if isinstance(tool, dict)]
    return extracted or None


def build_trace_from_completion(completion: CompletionRecord) -> Trace:
    """Normalize one stored completion record into a trajectory trace."""

    request = completion.request if isinstance(completion.request, dict) else {}
    response = completion.response if isinstance(completion.response, dict) else {}
    choices = response.get("choices")
    first_choice = (
        choices[0]
        if isinstance(choices, list) and choices and isinstance(choices[0], dict)
        else {}
    )
    prompt_ids = (
        first_choice.get("input_token_ids")
        or first_choice.get("prompt_token_ids")
        or response.get("prompt_token_ids")
    )
    response_message = first_choice.get("message")
    finish_reason = first_choice.get("finish_reason")

    response_ids, response_logprobs = _extract_response_tokens(response, first_choice)

    trace_metadata = deepcopy(completion.metadata)
    entropy_metadata = _generation_entropy_metadata(
        first_choice,
        generated_token_count=len(response_ids),
    )
    if entropy_metadata is not None:
        trace_metadata["generation_entropy"] = entropy_metadata

    return Trace(
        prompt_ids=list(prompt_ids) if isinstance(prompt_ids, list) else [],
        response_ids=response_ids,
        loss_mask=[1] * len(response_ids),
        prompt_messages=_extract_prompt_messages(request),
        response_messages=[deepcopy(response_message)] if isinstance(response_message, dict) else [],
        tools=_extract_tools(request),
        finish_reason=str(finish_reason) if finish_reason is not None else None,
        response_logprobs=response_logprobs,
        metadata=trace_metadata,
    )
