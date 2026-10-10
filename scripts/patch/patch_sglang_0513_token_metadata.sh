#!/usr/bin/env bash
#
# Patch SGLang 0.5.13 to expose the token metadata HERO needs through the
# normal OpenAI chat-completions response when `logprobs=true`.
#

set -euo pipefail

SERVER_URL="${1:-${SGLANG_SERVER_URL:-}}"

if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_BIN=("${VIRTUAL_ENV}/bin/python")
elif [[ -x ".venv/bin/python3" ]]; then
  PYTHON_BIN=(".venv/bin/python3")
elif [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=(".venv/bin/python")
elif command -v uv >/dev/null 2>&1; then
  PYTHON_BIN=(uv run python)
else
  PYTHON_BIN=(python3)
fi

"${PYTHON_BIN[@]}" - <<'PY'
from __future__ import annotations

from pathlib import Path
import importlib.util


SUPPORTED_SGLANG_VERSION = "0.5.13"


def fail(message: str) -> None:
    raise SystemExit(message)


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        fail(f"Failed to patch {label}: expected snippet not found")
    return text.replace(old, new, 1)


import sglang  # noqa: E402

if sglang.__version__ != SUPPORTED_SGLANG_VERSION:
    fail(
        f"patch_sglang_0513_token_metadata.sh is pinned to "
        f"sglang=={SUPPORTED_SGLANG_VERSION} but found {sglang.__version__}."
    )

spec = importlib.util.find_spec("sglang")
if spec is None or spec.origin is None:
    fail("sglang is not installed in the active Python environment")

root = Path(spec.origin).resolve().parent
protocol_path = root / "srt/entrypoints/openai/protocol.py"
utils_path = root / "srt/entrypoints/openai/utils.py"
serving_chat_path = root / "srt/entrypoints/openai/serving_chat.py"

for path in (protocol_path, utils_path, serving_chat_path):
    if not path.exists():
        fail(f"Expected SGLang file is missing: {path}")

# ---------------------------------------------------------------------------
# protocol.py - make token ids serializable in OpenAI logprob/choice payloads.
# ---------------------------------------------------------------------------
protocol_text = protocol_path.read_text()
protocol_text = replace_once(
    protocol_text,
    "class LogProbs(BaseModel):\n"
    "    text_offset: List[int] = Field(default_factory=list)\n"
    "    token_logprobs: List[Optional[float]] = Field(default_factory=list)\n"
    "    tokens: List[str] = Field(default_factory=list)\n"
    "    top_logprobs: List[Optional[Dict[str, float]]] = Field(default_factory=list)\n",
    "class LogProbs(BaseModel):\n"
    "    text_offset: List[int] = Field(default_factory=list)\n"
    "    token_logprobs: List[Optional[float]] = Field(default_factory=list)\n"
    "    token_ids: List[int] = Field(default_factory=list)\n"
    "    tokens: List[str] = Field(default_factory=list)\n"
    "    top_logprobs: List[Optional[Dict[str, float]]] = Field(default_factory=list)\n",
    label=str(protocol_path),
)
protocol_text = replace_once(
    protocol_text,
    "class ChatCompletionTokenLogprob(BaseModel):\n"
    "    token: str\n"
    "    bytes: List[int]\n"
    "    logprob: float\n"
    "    top_logprobs: List[TopLogprob]\n",
    "class ChatCompletionTokenLogprob(BaseModel):\n"
    "    token: str\n"
    "    token_id: int = 0\n"
    "    bytes: List[int]\n"
    "    logprob: float\n"
    "    top_logprobs: List[TopLogprob]\n",
    label=str(protocol_path),
)
protocol_text = replace_once(
    protocol_text,
    "class ChatCompletionResponseChoice(BaseModel):\n"
    "    index: int\n"
    "    message: ChatMessage\n"
    "    logprobs: Optional[Union[LogProbs, ChoiceLogprobs]] = None\n"
    "    finish_reason: Optional[\n"
    "        Literal[\n"
    "            \"stop\", \"length\", \"tool_calls\", \"content_filter\", \"function_call\", \"abort\"\n"
    "        ]\n"
    "    ] = None\n"
    "    matched_stop: Union[None, int, str] = None\n"
    "    hidden_states: Optional[object] = None\n"
    "    prompt_token_ids: Optional[List[int]] = None\n"
    "    meta_info: Optional[Dict[str, Any]] = None\n",
    "class ChatCompletionResponseChoice(BaseModel):\n"
    "    index: int\n"
    "    message: ChatMessage\n"
    "    logprobs: Optional[Union[LogProbs, ChoiceLogprobs]] = None\n"
    "    finish_reason: Optional[\n"
    "        Literal[\n"
    "            \"stop\", \"length\", \"tool_calls\", \"content_filter\", \"function_call\", \"abort\"\n"
    "        ]\n"
    "    ] = None\n"
    "    matched_stop: Union[None, int, str] = None\n"
    "    hidden_states: Optional[object] = None\n"
    "    input_token_ids: Optional[List[int]] = None\n"
    "    prompt_token_ids: Optional[List[int]] = None\n"
    "    token_ids: Optional[List[int]] = None\n"
    "    meta_info: Optional[Dict[str, Any]] = None\n",
    label=str(protocol_path),
)
protocol_text = replace_once(
    protocol_text,
    "        if self.hidden_states is None:\n"
    "            data.pop(\"hidden_states\", None)\n"
    "        if self.prompt_token_ids is None:\n"
    "            data.pop(\"prompt_token_ids\", None)\n"
    "        if self.meta_info is None:\n"
    "            data.pop(\"meta_info\", None)\n",
    "        if self.hidden_states is None:\n"
    "            data.pop(\"hidden_states\", None)\n"
    "        if self.input_token_ids is None:\n"
    "            data.pop(\"input_token_ids\", None)\n"
    "        if self.prompt_token_ids is None:\n"
    "            data.pop(\"prompt_token_ids\", None)\n"
    "        if self.token_ids is None:\n"
    "            data.pop(\"token_ids\", None)\n"
    "        if self.meta_info is None:\n"
    "            data.pop(\"meta_info\", None)\n",
    label=str(protocol_path),
)
protocol_path.write_text(protocol_text)

# ---------------------------------------------------------------------------
# utils.py - preserve token ids while converting internal logprobs to OpenAI.
# ---------------------------------------------------------------------------
utils_text = utils_path.read_text()
utils_text = replace_once(
    utils_text,
    "    def append_token_logprobs(token_logprobs):\n"
    "        for logprob, _, token_text in token_logprobs:\n"
    "            ret_logprobs.tokens.append(token_text)\n"
    "            ret_logprobs.token_logprobs.append(logprob)\n"
    "\n"
    "            # Not supported yet\n"
    "            ret_logprobs.text_offset.append(-1)\n",
    "    def append_token_logprobs(token_logprobs):\n"
    "        for logprob, token_id, token_text in token_logprobs:\n"
    "            ret_logprobs.tokens.append(token_text)\n"
    "            ret_logprobs.token_logprobs.append(logprob)\n"
    "            ret_logprobs.token_ids.append(token_id)\n"
    "\n"
    "            # Not supported yet\n"
    "            ret_logprobs.text_offset.append(-1)\n",
    label=str(utils_path),
)
utils_path.write_text(utils_text)

# ---------------------------------------------------------------------------
# serving_chat.py - when logprobs are requested, emit the same training
# metadata that 0.5.13 otherwise gates behind return_prompt_token_ids /
# return_meta_info request extensions.
# ---------------------------------------------------------------------------
serving_chat_text = serving_chat_path.read_text()
serving_chat_text = replace_once(
    serving_chat_text,
    "            return_prompt_token_ids=request.return_prompt_token_ids,\n",
    "            return_prompt_token_ids=request.return_prompt_token_ids or request.logprobs,\n",
    label=str(serving_chat_path),
)
serving_chat_text = replace_once(
    serving_chat_text,
    "            # Extract prompt_token_ids if requested\n"
    "            choice_prompt_token_ids = (\n"
    "                ret_item.get(\"prompt_token_ids\")\n"
    "                if request.return_prompt_token_ids\n"
    "                else None\n"
    "            )\n"
    "\n"
    "            choice_meta_info = (\n"
    "                ret_item[\"meta_info\"] if request.return_meta_info else None\n"
    "            )\n",
    "            emit_prompt_token_ids = request.return_prompt_token_ids or request.logprobs\n"
    "            emit_meta_info = request.return_meta_info or request.logprobs\n"
    "            choice_prompt_token_ids = (\n"
    "                ret_item.get(\"prompt_token_ids\") if emit_prompt_token_ids else None\n"
    "            )\n"
    "            choice_output_token_ids = ret_item.get(\"output_ids\") if request.logprobs else None\n"
    "            choice_meta_info = ret_item[\"meta_info\"] if emit_meta_info else None\n",
    label=str(serving_chat_path),
)
serving_chat_text = replace_once(
    serving_chat_text,
    "                hidden_states=hidden_states,\n"
    "                prompt_token_ids=choice_prompt_token_ids,\n"
    "                meta_info=choice_meta_info,\n"
    "            )\n",
    "                hidden_states=hidden_states,\n"
    "                input_token_ids=choice_prompt_token_ids,\n"
    "                prompt_token_ids=choice_prompt_token_ids,\n"
    "                token_ids=choice_output_token_ids,\n"
    "                meta_info=choice_meta_info,\n"
    "            )\n",
    label=str(serving_chat_path),
)
serving_chat_text = replace_once(
    serving_chat_text,
    "        for token_idx, (token, logprob) in enumerate(\n"
    "            zip(logprobs.tokens, logprobs.token_logprobs)\n"
    "        ):\n"
    "            token_bytes = list(token.encode(\"utf-8\"))\n",
    "        for token_idx, (token, logprob) in enumerate(\n"
    "            zip(logprobs.tokens, logprobs.token_logprobs)\n"
    "        ):\n"
    "            token_id = logprobs.token_ids[token_idx] if token_idx < len(logprobs.token_ids) else 0\n"
    "            token_bytes = list(token.encode(\"utf-8\"))\n",
    label=str(serving_chat_path),
)
serving_chat_text = replace_once(
    serving_chat_text,
    "                ChatCompletionTokenLogprob(\n"
    "                    token=token,\n"
    "                    bytes=token_bytes,\n"
    "                    logprob=logprob,\n"
    "                    top_logprobs=top_logprobs,\n"
    "                )\n",
    "                ChatCompletionTokenLogprob(\n"
    "                    token=token,\n"
    "                    token_id=token_id,\n"
    "                    bytes=token_bytes,\n"
    "                    logprob=logprob,\n"
    "                    top_logprobs=top_logprobs,\n"
    "                )\n",
    label=str(serving_chat_path),
)
serving_chat_path.write_text(serving_chat_text)

print(f"Patched SGLang {SUPPORTED_SGLANG_VERSION} token metadata in {root}")
PY

if [[ -n "${SERVER_URL}" ]]; then
  "${PYTHON_BIN[@]}" - "${SERVER_URL}" <<'PY'
from __future__ import annotations

import json
import sys
from urllib.request import Request, urlopen


server_url = sys.argv[1].rstrip("/")


def fetch_json(url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


models = fetch_json(f"{server_url}/v1/models")
model_id = models["data"][0]["id"]
response = fetch_json(
    f"{server_url}/v1/chat/completions",
    {
        "model": model_id,
        "messages": [{"role": "user", "content": "Reply with the single word ok."}],
        "max_tokens": 8,
        "temperature": 0,
        "logprobs": True,
    },
)

choice = response["choices"][0]
content = (choice.get("logprobs") or {}).get("content") or []
if not isinstance(choice.get("prompt_token_ids"), list):
    raise SystemExit("Patch smoke test failed: missing choice.prompt_token_ids")
if not isinstance(choice.get("input_token_ids"), list):
    raise SystemExit("Patch smoke test failed: missing choice.input_token_ids")
if not isinstance(choice.get("token_ids"), list):
    raise SystemExit("Patch smoke test failed: missing choice.token_ids")
if not isinstance(choice.get("meta_info"), dict):
    raise SystemExit("Patch smoke test failed: missing choice.meta_info")
if not content or content[0].get("token_id") is None:
    raise SystemExit("Patch smoke test failed: missing logprobs.content[].token_id")

print("SGLang 0.5.13 token metadata patch smoke test passed")
PY
fi
