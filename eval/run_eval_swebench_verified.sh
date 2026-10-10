#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python3}"
export DATASET="${DATASET:-swe-bench-verified}"
export EXP_TAG="${EXP_TAG:-$(basename "${BASH_SOURCE[0]}" .sh)}"
export MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-131072}"
export CLAUDE_CODE_AUTO_COMPACT_WINDOW="${CLAUDE_CODE_AUTO_COMPACT_WINDOW:-80000}"
export MAX_RESPONSE_TOKENS="${MAX_RESPONSE_TOKENS:-16000}"
export CLAUDE_CODE_MAX_TURNS="${CLAUDE_CODE_MAX_TURNS:-200}"

exec bash "${SCRIPT_DIR}/run_eval_swe_bench.sh" "$@"
