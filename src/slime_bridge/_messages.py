"""Shared message/content flattening helpers.

Both :mod:`slime_bridge.rollout` (when building instruction text from a
dataset prompt) and :mod:`slime_bridge.adapter` (when rendering assistant
response messages into a human-readable string) need the same content
flattening. Keep one copy here so they never drift.
"""

from __future__ import annotations

from typing import Any


def flatten_content(content: Any) -> str:
    """Render OpenAI-style message content into a plain string.

    Accepts the three shapes the Chat Completions API uses:

    - ``None`` → ``""``
    - ``str`` → returned as-is
    - ``list[dict]`` → each dict's ``"text"`` field is concatenated
      (``{"type": "text", "text": ...}`` and bare ``{"text": ...}`` are
      both handled). Non-dict items are skipped.

    Anything else is coerced via ``str(content)``.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif "text" in item:
                parts.append(str(item.get("text", "")))
        return "".join(parts).strip()
    if content is None:
        return ""
    return str(content)


def prompt_to_instruction_text(prompt: Any) -> str:
    """Flatten a dataset prompt (str or chat-message list) into instruction text.

    Single-role lists (e.g. just ``[{"role": "user", "content": ...}]``, which
    is how we shape prompts for VLM checkpoints that require list form) render
    as the bare content so the instruction template sees the same text as the
    string-prompt path. Multi-role lists fall back to ``[role] content`` blocks
    joined by blank lines for a symmetric view of conversation data.
    """
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        messages = [m for m in prompt if isinstance(m, dict)]
        contents = [flatten_content(m.get("content")) for m in messages]
        roles = {str(m.get("role", "user")) for m in messages}
        if len(roles) <= 1:
            return "\n\n".join(c for c in contents if c)
        parts: list[str] = []
        for message, content in zip(messages, contents):
            if content:
                role = str(message.get("role", "user"))
                parts.append(f"[{role}] {content}")
        return "\n\n".join(parts)
    if prompt is None:
        return ""
    return str(prompt)


def messages_to_text(messages: list[dict[str, Any]]) -> str:
    """Render a list of chat messages into a ``[role] content`` block string.

    Known limitation: drops assistant ``tool_calls`` structure into the plain
    text view. Training consumes tokens + logprobs, so this is only a
    degraded human-readable representation of ``Sample.response``.
    """
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "assistant"))
        content = flatten_content(message.get("content"))
        if content:
            parts.append(f"[{role}] {content}")
    return "\n\n".join(parts)
