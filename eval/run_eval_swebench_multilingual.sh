#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python3}"
export DATASET="${DATASET:-swe-bench-multilingual}"
export EXP_TAG="${EXP_TAG:-$(basename "${BASH_SOURCE[0]}" .sh)}"

exec bash "${SCRIPT_DIR}/run_eval_swe_bench.sh" "$@"
