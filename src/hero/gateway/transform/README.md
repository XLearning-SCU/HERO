# API Transforms

`hero.gateway.transform` is the adapter layer inside the proxy. It converts each
intercepted request from its native API (Anthropic / OpenAI Chat / OpenAI
Responses / Google) into **OpenAI Chat Completions** for the served model, then
converts the response back — so every agent sees the API shape it expects while
one backend serves them all.

## Mental model

- **One transformer per API type**, selected by `TransformManager` from the
  detected `APIType`. OpenAI Chat is near-passthrough; Anthropic / Responses /
  Google fully restructure messages, tools, and system prompts.
- The canonical internal format is **OpenAI Chat Completions**.
- Shared request normalization lives in `base.py` (`_normalize_request`): merge
  `developer`→`system` roles, drop internal keys, and for Qwen3.5 models force
  `enable_thinking=False`. Training-signal params (`logprobs`, token ids) and all
  backend-specific request/response handling live in the gateway's `engine.py`,
  not here.
- The model swap to the served model happens in the proxy (`server.py`);
  transformers carry the requested model through and **restore it on the
  response**, so clients still see the name they asked for.

## Main files

- `base.py`: the transformer interface + shared training enhancement (role
  merge, logprobs, Qwen3.5 thinking-off).
- `openai_chat.py`: near-passthrough (e.g. `max_completion_tokens`→`max_tokens`).
- `openai_responses.py`: OpenAI Responses ↔ Chat, including reasoning items and
  shell/function tools (used by Codex).
- `anthropic.py`: Anthropic Messages ↔ Chat — tool_use/tool_result and
  Claude-Code header handling.
- `google.py`: Gemini `generateContent` ↔ Chat — functionDeclarations / Call /
  Response.
- `images.py`: cross-API image and document content normalization.
- `reasoning.py`: round-trips reasoning content across the per-API thinking /
  signature shapes.
- `__init__.py`: the `APIType` → transformer registry.

## Streaming

Each non-OpenAI transformer carries stream state so it can emit a correctly
ordered SSE sequence (open block → deltas → close). In practice the gateway
calls the backend once and drives the transformer with a single chunk +
finalize (see the synthetic-streaming note in the
[gateway README](../README.md)), so this machinery turns one complete response
into the event stream a client SDK expects.
