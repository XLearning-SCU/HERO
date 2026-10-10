"""Inference backend strategies for the HERO gateway.

The gateway speaks the OpenAI Chat Completions API to a local inference server.
Two backends are supported, and they differ only in:

  1. the request params that make them emit the token ids + per-token logprobs
     HERO needs for training, and
  2. the exact shape of those fields in the response.

The base implements the canonical contract -- request ``logprobs`` (the one
training param every backend needs) and a pass-through response. A backend
overrides only what it does differently, via two hooks: ``prepare_request``
(extra request params) and ``normalize_response`` (response canonicalization).
Everything downstream -- storage, trace builder, transforms, slime adapter --
then sees one shape. The canonical shape is HERO's training output:

  - prompt token ids:   ``choice.input_token_ids`` (or ``response.prompt_token_ids``)
  - response token ids: ``choice.token_ids``       (or ``logprobs.content[].token_id``)
  - per-token logprobs: ``choice.logprobs.content[]`` with ``{token, token_id, logprob, ...}``

SGLang reaches this shape via source-supported prompt-token/meta-info extensions
plus a light response normalization. vLLM reaches it natively via the
``return_token_ids`` request flag plus a light response rename.
"""

from __future__ import annotations

from abc import ABC
import os
from typing import Any


class InferenceEngine(ABC):
    """Strategy for one OpenAI-compatible inference backend.

    The base encodes the canonical contract: request ``logprobs`` (the one
    training param every backend needs) and pass the response through
    unchanged. A backend overrides only what it does differently.
    """

    name: str

    def prepare_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Inject the request params this backend needs to emit training signals.

        ``logprobs`` is universal; subclasses add backend-specific params (e.g.
        token-id flags) on top via ``super().prepare_request(...)``.
        """
        request["logprobs"] = True
        return request

    def normalize_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Canonicalize the backend's response (in place) and return it.

        The default is a pass-through; a backend that differs from HERO's
        training shape overrides this.
        """
        return response

    @staticmethod
    def _stamp_token_ids_onto_logprobs(choice: dict[str, Any]) -> None:
        """Copy sampled token IDs onto OpenAI chat logprob entries when aligned."""

        token_ids = choice.get("token_ids")
        logprobs = choice.get("logprobs")
        if not isinstance(token_ids, list) or not isinstance(logprobs, dict):
            return
        content = logprobs.get("content")
        if not isinstance(content, list) or len(content) != len(token_ids):
            return
        for entry, token_id in zip(content, token_ids):
            if isinstance(entry, dict):
                entry.setdefault("token_id", token_id)


class SGLangEngine(InferenceEngine):
    """SGLang via its source-supported OpenAI-compatible extensions."""

    name = "sglang"

    def prepare_request(self, request: dict[str, Any]) -> dict[str, Any]:
        request = super().prepare_request(request)  # logprobs=True
        request["return_prompt_token_ids"] = True
        request["return_meta_info"] = True
        # Opt-in entropy observability. Method launchers enable it only when
        # the reward/analysis path consumes token-level entropy.
        entropy_top_k = os.environ.get("HERO_GENERATION_ENTROPY_TOP_K")
        if entropy_top_k:
            top_k = int(entropy_top_k)
            if top_k <= 0:
                raise ValueError("HERO_GENERATION_ENTROPY_TOP_K must be positive")
            request["top_logprobs"] = top_k
        return request

    def normalize_response(self, response: dict[str, Any]) -> dict[str, Any]:
        choices = response.get("choices")
        if not isinstance(choices, list):
            return response
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            self._canonicalize_prompt_token_ids(choice)
            self._canonicalize_response_token_ids(choice)
            self._canonicalize_logprobs_from_meta(choice)
            self._stamp_token_ids_onto_logprobs(choice)
            # return_meta_info is requested as an internal bridge to recover
            # token IDs and lossless top-k alternatives from source SGLang; do
            # not expose or store the duplicate raw meta payload downstream.
            choice.pop("meta_info", None)
        return response

    @staticmethod
    def _canonicalize_prompt_token_ids(choice: dict[str, Any]) -> None:
        if choice.get("input_token_ids") is not None:
            return
        prompt_token_ids = choice.get("prompt_token_ids")
        if isinstance(prompt_token_ids, list):
            choice["input_token_ids"] = list(prompt_token_ids)

    @staticmethod
    def _canonicalize_response_token_ids(choice: dict[str, Any]) -> None:
        if isinstance(choice.get("token_ids"), list):
            return
        meta_info = choice.get("meta_info")
        if not isinstance(meta_info, dict):
            return
        output_logprobs = meta_info.get("output_token_logprobs")
        if not isinstance(output_logprobs, list):
            return

        token_ids: list[int] = []
        for item in output_logprobs:
            token_id = None
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                token_id = item[1]
            elif isinstance(item, dict):
                token_id = item.get("token_id")
            if token_id is None:
                continue
            try:
                token_ids.append(int(token_id))
            except (TypeError, ValueError):
                continue

        if token_ids:
            choice["token_ids"] = token_ids

    @staticmethod
    def _source_logprob_entry(item: Any) -> dict[str, Any] | None:
        """Convert one native SGLang logprob tuple without losing token IDs."""
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            raw_logprob = item[0]
            raw_token_id = item[1]
            raw_token = item[2] if len(item) >= 3 else ""
        elif isinstance(item, dict):
            raw_logprob = item.get("logprob", item.get("token_logprob"))
            raw_token_id = item.get("token_id")
            raw_token = item.get("token", "")
        else:
            return None

        try:
            logprob = float(raw_logprob)
            token_id = int(raw_token_id)
        except (TypeError, ValueError):
            return None
        token = raw_token if isinstance(raw_token, str) else ""
        return {
            "token": token,
            "token_id": token_id,
            "logprob": logprob,
            "bytes": list(token.encode("utf-8", errors="replace")),
        }

    @classmethod
    def _canonicalize_logprobs_from_meta(cls, choice: dict[str, Any]) -> None:
        """Restore SGLang's token-ID-bearing logprobs before dropping meta_info.

        SGLang's OpenAI adapter keys top-logprobs by decoded token text, which
        collapses distinct byte-level token IDs that decode to the same string.
        ``return_meta_info`` retains the original ``(logprob, token_id, text)``
        tuples, so use those as the canonical entropy source.
        """
        meta_info = choice.get("meta_info")
        if not isinstance(meta_info, dict):
            return

        logprobs = choice.get("logprobs")
        if not isinstance(logprobs, dict):
            logprobs = {}
            choice["logprobs"] = logprobs
        content = logprobs.get("content")

        output_logprobs = meta_info.get("output_token_logprobs")
        if isinstance(output_logprobs, list):
            restored_content = [
                cls._source_logprob_entry(item) for item in output_logprobs
            ]
            if restored_content and all(entry is not None for entry in restored_content):
                if not isinstance(content, list) or len(content) != len(restored_content):
                    content = restored_content
                    logprobs["content"] = content

        if not isinstance(content, list):
            return
        output_top_logprobs = meta_info.get("output_top_logprobs")
        if (
            not isinstance(output_top_logprobs, list)
            or len(output_top_logprobs) != len(content)
        ):
            return

        for generated, raw_candidates in zip(
            content, output_top_logprobs, strict=True
        ):
            if not isinstance(generated, dict) or not isinstance(raw_candidates, list):
                continue
            restored_candidates = [
                cls._source_logprob_entry(item) for item in raw_candidates
            ]
            if restored_candidates and all(
                candidate is not None for candidate in restored_candidates
            ):
                generated["top_logprobs"] = restored_candidates


class VLLMEngine(InferenceEngine):
    """vLLM via its native OpenAI-compatible server.

    ``return_token_ids`` makes vLLM emit ``response.prompt_token_ids`` and
    ``choice.token_ids``. `top_logprobs`` must be set (not None) for vLLM
    to populate ``logprobs.content[]`` given ``logprobs=True``; 0 returns just
    the sampled token's logprob, which is all training needs.
    """

    name = "vllm"

    def prepare_request(self, request: dict[str, Any]) -> dict[str, Any]:
        request = super().prepare_request(request)  # logprobs=True
        request["return_token_ids"] = True
        entropy_top_k = os.environ.get("HERO_GENERATION_ENTROPY_TOP_K")
        if entropy_top_k:
            try:
                top_k = int(entropy_top_k)
            except ValueError as exc:
                raise ValueError("HERO_GENERATION_ENTROPY_TOP_K must be an integer") from exc
            if top_k <= 0:
                raise ValueError("HERO_GENERATION_ENTROPY_TOP_K must be positive")
            request["top_logprobs"] = top_k
        else:
            request.setdefault("top_logprobs", 0)
        # vLLM reads input reasoning from `reasoning`, not HERO's canonical
        # `reasoning_content`; rename it so prior turns' interleaved thinking
        # survives templating (else they render an empty `<think></think>`).
        for message in request.get("messages") or []:
            if isinstance(message, dict) and message.get("reasoning_content") is not None:
                message["reasoning"] = message.pop("reasoning_content")
        return request

    def normalize_response(self, response: dict[str, Any]) -> dict[str, Any]:
        choices = response.get("choices")
        if not isinstance(choices, list):
            return response
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            self._canonicalize_reasoning(choice.get("message"))
            self._stamp_token_ids_onto_logprobs(choice)
        return response

    @staticmethod
    def _canonicalize_reasoning(message: Any) -> None:
        """vLLM names the field ``reasoning``; HERO's canonical field is ``reasoning_content``."""
        if not isinstance(message, dict):
            return
        if message.get("reasoning_content") is None and message.get("reasoning") is not None:
            message["reasoning_content"] = message.pop("reasoning")

    @staticmethod
    def _stamp_token_ids_onto_logprobs(choice: dict[str, Any]) -> None:
        """Parity with SGLang: copy token_id onto each logprob entry.

        vLLM builds ``logprobs.content`` and ``choice.token_ids`` from the same
        ``output.token_ids``, so they align; guard on equal length regardless.
        Not load-bearing for training (which reads ``choice.token_ids`` and the
        per-entry ``logprob``) -- it keeps stored traces one shape across engines.
        """
        InferenceEngine._stamp_token_ids_onto_logprobs(choice)


_ENGINES: dict[str, type[InferenceEngine]] = {
    SGLangEngine.name: SGLangEngine,
    VLLMEngine.name: VLLMEngine,
}


def get_engine(name: str) -> InferenceEngine:
    """Return the inference engine strategy for ``name`` (``sglang`` | ``vllm``)."""
    try:
        engine_cls = _ENGINES[name]
    except KeyError:
        supported = ", ".join(sorted(_ENGINES))
        raise ValueError(
            f"Unknown inference engine {name!r}; supported: {supported}"
        ) from None
    return engine_cls()
