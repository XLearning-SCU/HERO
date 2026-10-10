"""Detect incoming API type from request path, headers, and body."""

from __future__ import annotations

from enum import Enum
from typing import Any


class APIType(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI_CHAT = "openai_chat"
    OPENAI_RESPONSES = "openai_responses"
    GOOGLE = "google"


def detect(path: str, headers: dict[str, str], body: dict[str, Any]) -> APIType:
    """Detect API type. Priority: path → headers → body structure."""

    # Path-based detection (most reliable)
    if "/v1/messages" in path:
        return APIType.ANTHROPIC
    if "/v1/chat/completions" in path:
        return APIType.OPENAI_CHAT
    if "/v1/responses" in path:
        return APIType.OPENAI_RESPONSES
    if "generateContent" in path:
        return APIType.GOOGLE

    # Header-based detection
    lower_headers = {k.lower(): v for k, v in headers.items()}
    if "anthropic-version" in lower_headers:
        return APIType.ANTHROPIC

    # Body-based detection (fallback)
    if "contents" in body:
        return APIType.GOOGLE
    if "input" in body and "instructions" in body:
        return APIType.OPENAI_RESPONSES

    # Default to OpenAI Chat
    return APIType.OPENAI_CHAT


def extract_model(api_type: APIType, body: dict[str, Any]) -> str:
    """Extract the model name from the request body based on API type."""
    if api_type == APIType.GOOGLE:
        # Google model is usually in the URL path, not body
        return body.get("model", "gemini-pro")
    return body.get("model", "unknown")
