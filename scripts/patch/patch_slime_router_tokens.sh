#!/usr/bin/env bash
# Preserve exact SGLang token metadata in Slime's OpenAI-compatible agent
# adapter responses. HERO consumes these fields to build training
# trajectories without local retokenization.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/slime}"

if [ ! -d "${SLIME_DIR}/.git" ]; then
    echo "ERROR: Slime git checkout not found at ${SLIME_DIR}" >&2
    exit 1
fi

PATCH_FILE="$(mktemp)"
cleanup() {
    rm -f "${PATCH_FILE}"
}
trap cleanup EXIT

cat > "${PATCH_FILE}" <<'PATCH'
diff --git a/slime/agent/adapters/common.py b/slime/agent/adapters/common.py
index ed5d2e06..cb915f2b 100644
--- a/slime/agent/adapters/common.py
+++ b/slime/agent/adapters/common.py
@@ -237,6 +237,7 @@ async def call_sglang_generate(
         output_ids=output_ids,
         finish_reason=finish,
         output_log_probs=output_log_probs,
+        meta_info=dict(meta),
     )
 
 
diff --git a/slime/agent/adapters/openai.py b/slime/agent/adapters/openai.py
index f9d91d8f..e11a7c06 100644
--- a/slime/agent/adapters/openai.py
+++ b/slime/agent/adapters/openai.py
@@ -351,6 +351,93 @@ def _usage(in_tok: int, out_tok: int) -> dict[str, int]:
     }
 
 
+def _decode_token(tok, token_id: int) -> str:
+    try:
+        return tok.decode([token_id], skip_special_tokens=False)
+    except TypeError:
+        return tok.decode([token_id])
+
+
+def _output_token_logprobs(turn: TurnRecord, tok) -> list[Any]:
+    raw = turn.meta_info.get("output_token_logprobs")
+    if isinstance(raw, list) and len(raw) == len(turn.output_ids):
+        return [list(item) if isinstance(item, (list, tuple)) else item for item in raw]
+
+    if len(turn.output_log_probs) != len(turn.output_ids):
+        return []
+
+    return [
+        [float(logprob), int(token_id), _decode_token(tok, int(token_id))]
+        for logprob, token_id in zip(turn.output_log_probs, turn.output_ids, strict=True)
+    ]
+
+
+def _token_text_from_logprob_item(item: Any, tok, token_id: int) -> str:
+    if isinstance(item, (list, tuple)) and len(item) >= 3 and isinstance(item[2], str):
+        return item[2]
+    if isinstance(item, dict):
+        token = item.get("token")
+        if isinstance(token, str):
+            return token
+        text = item.get("text")
+        if isinstance(text, str):
+            return text
+    return _decode_token(tok, token_id)
+
+
+def _logprob_from_item(item: Any, fallback: float) -> float:
+    if isinstance(item, (list, tuple)) and item:
+        return float(item[0])
+    if isinstance(item, dict) and item.get("logprob") is not None:
+        return float(item["logprob"])
+    return float(fallback)
+
+
+def _chat_logprobs(turn: TurnRecord, tok) -> dict[str, list[dict[str, Any]]] | None:
+    output_token_logprobs = _output_token_logprobs(turn, tok)
+    if len(output_token_logprobs) != len(turn.output_ids):
+        return None
+
+    content: list[dict[str, Any]] = []
+    for i, (token_id, item) in enumerate(zip(turn.output_ids, output_token_logprobs, strict=True)):
+        token = _token_text_from_logprob_item(item, tok, int(token_id))
+        logprob = _logprob_from_item(
+            item,
+            turn.output_log_probs[i] if i < len(turn.output_log_probs) else 0.0,
+        )
+        content.append(
+            {
+                "token": token,
+                "bytes": list(token.encode("utf-8")),
+                "logprob": logprob,
+                "token_id": int(token_id),
+                "top_logprobs": [],
+            }
+        )
+    return {"content": content}
+
+
+def _chat_meta_info(turn: TurnRecord, tok) -> dict[str, Any]:
+    meta = dict(turn.meta_info)
+    output_token_logprobs = _output_token_logprobs(turn, tok)
+    if output_token_logprobs:
+        meta["output_token_logprobs"] = output_token_logprobs
+    meta.setdefault("finish_reason", {"type": turn.finish_reason})
+    return meta
+
+
+def _attach_training_token_fields(choice: dict[str, Any], turn: TurnRecord, tok) -> None:
+    prompt_ids = [int(token_id) for token_id in turn.prompt_ids]
+    output_ids = [int(token_id) for token_id in turn.output_ids]
+    choice["input_token_ids"] = prompt_ids
+    choice["prompt_token_ids"] = list(prompt_ids)
+    choice["token_ids"] = output_ids
+    logprobs = _chat_logprobs(turn, tok)
+    if logprobs is not None:
+        choice["logprobs"] = logprobs
+    choice["meta_info"] = _chat_meta_info(turn, tok)
+
+
 def _responses_usage(in_tok: int, out_tok: int) -> dict[str, int]:
     return {
         "input_tokens": in_tok,
@@ -396,28 +483,29 @@ async def _handle_chat_completions(request: web.Request) -> web.StreamResponse:
     turn, parsed, in_tok, out_tok = await _run_turn(request, body, messages)
     if body.get("stream"):
         return await _stream_chat_completion(request, body, parsed, turn.finish_reason, in_tok, out_tok)
-    return web.json_response(_chat_completion_response(body, parsed, turn.finish_reason, in_tok, out_tok))
+    return web.json_response(_chat_completion_response(body, parsed, turn, request.app[TOKENIZER_KEY], in_tok, out_tok))
 
 
 def _chat_completion_response(
     body: dict,
     parsed: ParsedModelOutput,
-    finish: str,
+    turn: TurnRecord,
+    tok,
     in_tok: int,
     out_tok: int,
 ) -> dict[str, Any]:
+    choice = {
+        "index": 0,
+        "message": _chat_message(parsed),
+        "finish_reason": _finish_reason(parsed, turn.finish_reason),
+    }
+    _attach_training_token_fields(choice, turn, tok)
     return {
         "id": f"chatcmpl_{secrets.token_hex(12)}",
         "object": "chat.completion",
         "created": int(time.time()),
         "model": body.get("model", "slime-actor"),
-        "choices": [
-            {
-                "index": 0,
-                "message": _chat_message(parsed),
-                "finish_reason": _finish_reason(parsed, finish),
-            }
-        ],
+        "choices": [choice],
         "usage": _usage(in_tok, out_tok),
     }
 
diff --git a/slime/agent/trajectory.py b/slime/agent/trajectory.py
index 51db112f..8d102e90 100644
--- a/slime/agent/trajectory.py
+++ b/slime/agent/trajectory.py
@@ -27,6 +27,7 @@ class TurnRecord:
     output_ids: list[int]
     finish_reason: str
     output_log_probs: list[float] = dataclasses.field(default_factory=list)
+    meta_info: dict[str, Any] = dataclasses.field(default_factory=dict)
 
 
 @dataclasses.dataclass(frozen=True)
PATCH

if git -C "${SLIME_DIR}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
    echo "Slime router token patch already applied."
    exit 0
fi

git -C "${SLIME_DIR}" apply --check "${PATCH_FILE}"
git -C "${SLIME_DIR}" apply "${PATCH_FILE}"
echo "Applied Slime router token patch to ${SLIME_DIR}."
