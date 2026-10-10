"""HTTP client for forwarding requests to an OpenAI-compatible inference server.

Backend differences (request params, response shape) are isolated in the
``InferenceEngine`` strategy this client holds; the HTTP/streaming/pause logic
here is backend-agnostic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import httpx

from hero.gateway.engine import InferenceEngine

logger = logging.getLogger(__name__)


class UpstreamError(RuntimeError):
    """Base class for upstream gateway failures."""


class UpstreamHTTPError(UpstreamError):
    """Raised when the upstream returns a non-2xx status."""

    def __init__(self, status_code: int, body: dict[str, Any] | str | None = None):
        self.status_code = status_code
        self.body = body
        super().__init__(self._build_message(status_code, body))

    @staticmethod
    def _build_message(status_code: int, body: dict[str, Any] | str | None) -> str:
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str) and message:
                    return message
            message = body.get("message")
            if isinstance(message, str) and message:
                return message
        if isinstance(body, str) and body:
            return body
        return f"Upstream request failed with status {status_code}"


class UpstreamTimeoutError(UpstreamError):
    """Raised when the upstream times out."""


class UpstreamTransportError(UpstreamError):
    """Raised for connection and transport failures."""


class InferenceClient:
    """Direct httpx client to an inference server's OpenAI-compatible API.

    Per-call bound comes from the session's remaining-timeout budget
    (`_await_with_budget` at the gateway node). The internal httpx timeout
    is a high liveness ceiling so that a stuck engine can't pin a request
    past the session deadline. The ``engine`` strategy injects backend-specific
    request params and canonicalizes responses.
    """

    _LIVENESS_TIMEOUT_SECONDS = 900.0
    _CONTEXT_LIMIT_RE = re.compile(
        r"maximum context length of (?P<context>\d+) tokens.*?"
        r"(?P<input>\d+) tokens from the input messages.*?"
        r"(?P<completion>\d+) tokens for the completion",
        re.IGNORECASE,
    )

    def __init__(
        self,
        base_url: str,
        engine: InferenceEngine,
        *,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.engine = engine
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._client: httpx.AsyncClient | None = None
        self._generation_paused = False
        self._inflight_generations = 0
        self._generation_condition = asyncio.Condition()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self._LIVENESS_TIMEOUT_SECONDS, connect=30),
            )
        return self._client

    async def _read_error_body(self, response: httpx.Response) -> dict[str, Any] | str | None:
        content = await response.aread()
        if not content:
            return None

        text = content.decode("utf-8", errors="replace").strip()
        if not text:
            return None

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    async def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return

        body = await self._read_error_body(response)
        await response.aclose()
        raise UpstreamHTTPError(response.status_code, body)

    @classmethod
    def _remaining_context_budget(
        cls, status_code: int, body: dict[str, Any] | str | None
    ) -> int | None:
        """Return a safe completion budget for an SGLang context-limit error.

        Agent CLIs can request a fixed completion length (Claude Code currently
        requests 32k) while the supplied SWE issue consumes a variable portion
        of the model context.  SGLang rejects such an otherwise valid request
        before generation.  Its error reports the exact token accounting, so
        retrying once with the remaining budget preserves every launch setting
        and only makes an impossible request executable.
        """
        if status_code != 400:
            return None
        if isinstance(body, dict):
            error = body.get("error")
            message = error.get("message") if isinstance(error, dict) else body.get("message")
        else:
            message = body
        if not isinstance(message, str):
            return None
        match = cls._CONTEXT_LIMIT_RE.search(message)
        if match is None:
            return None
        remaining = int(match.group("context")) - int(match.group("input"))
        return remaining if remaining > 0 else None

    @staticmethod
    def _translate_transport_error(exc: httpx.RequestError) -> UpstreamError:
        if isinstance(exc, httpx.TimeoutException):
            return UpstreamTimeoutError("Upstream request timed out")
        return UpstreamTransportError(f"Upstream request failed: {exc}")

    async def completion(self, request: dict[str, Any]) -> dict[str, Any]:
        """Non-streaming chat completion. Returns the full JSON response."""
        await self._acquire_generation_slot()
        client = await self._get_client()
        from copy import deepcopy

        request_copy = deepcopy(request)
        request_copy.pop("stream", None)
        request_copy["stream"] = False
        if self.max_output_tokens is not None:
            request_copy["max_tokens"] = min(
                int(request_copy.get("max_tokens", self.max_output_tokens)),
                self.max_output_tokens,
            )
        if self.temperature is not None:
            request_copy["temperature"] = self.temperature
        if self.top_p is not None:
            request_copy["top_p"] = self.top_p
        request_copy = self.engine.prepare_request(request_copy)
        try:
            resp = await client.post(
                "/v1/chat/completions",
                json=request_copy,
                headers={"Content-Type": "application/json"},
            )
            if not resp.is_success:
                error_body = await self._read_error_body(resp)
                remaining = self._remaining_context_budget(resp.status_code, error_body)
                requested = request_copy.get("max_tokens")
                if remaining is not None and isinstance(requested, int) and requested > remaining:
                    logger.info(
                        "Retrying context-limited inference request with max_tokens=%d (requested=%d)",
                        remaining,
                        requested,
                    )
                    await resp.aclose()
                    request_copy["max_tokens"] = remaining
                    resp = await client.post(
                        "/v1/chat/completions",
                        json=request_copy,
                        headers={"Content-Type": "application/json"},
                    )
                else:
                    # _raise_for_status needs the parsed body.  Rebuild a
                    # response so it can consume it normally below.
                    await resp.aclose()
                    resp = httpx.Response(resp.status_code, json=error_body)
        except httpx.RequestError as exc:
            raise self._translate_transport_error(exc) from exc
        finally:
            await self._release_generation_slot()

        await self._raise_for_status(resp)
        return self.engine.normalize_response(resp.json())

    async def _acquire_generation_slot(self) -> None:
        async with self._generation_condition:
            await self._generation_condition.wait_for(lambda: not self._generation_paused)
            self._inflight_generations += 1

    async def _release_generation_slot(self) -> None:
        async with self._generation_condition:
            self._inflight_generations -= 1
            self._generation_condition.notify_all()

    async def pause_generation(self, *, timeout_seconds: float = 300.0) -> dict[str, Any]:
        """Block new generation requests and wait for current inference calls to drain."""
        async with self._generation_condition:
            self._generation_paused = True
            self._generation_condition.notify_all()
            await asyncio.wait_for(
                self._generation_condition.wait_for(lambda: self._inflight_generations == 0),
                timeout=timeout_seconds,
            )
            return self.generation_status()

    async def resume_generation(self) -> dict[str, Any]:
        async with self._generation_condition:
            self._generation_paused = False
            self._generation_condition.notify_all()
            return self.generation_status()

    def generation_status(self) -> dict[str, Any]:
        return {
            "paused": self._generation_paused,
            "inflight": self._inflight_generations,
            "base_url": self.base_url,
            "engine": self.engine.name,
        }

    async def list_models(self) -> dict[str, Any]:
        """Passthrough GET /v1/models."""
        client = await self._get_client()
        try:
            resp = await client.get("/v1/models")
        except httpx.RequestError as exc:
            raise self._translate_transport_error(exc) from exc
        await self._raise_for_status(resp)
        return resp.json()

    async def health(self) -> dict[str, Any]:
        """Passthrough GET /health."""
        client = await self._get_client()
        try:
            resp = await client.get("/health")
        except httpx.RequestError as exc:
            raise self._translate_transport_error(exc) from exc
        await self._raise_for_status(resp)
        content = await resp.aread()
        if not content:
            return {"status": "ok"}

        text = content.decode("utf-8", errors="replace").strip()
        if not text:
            return {"status": "ok"}

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"status": "ok", "body": text}

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
