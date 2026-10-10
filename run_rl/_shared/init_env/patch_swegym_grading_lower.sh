#!/usr/bin/env bash
# Reapply the swegym get_logs_eval repo-case fix so MONAI (and any other
# uppercase-owner/repo) instances grade correctly instead of raising
# KeyError on MAP_REPO_TO_PARSER. Idempotent: re-running is a no-op once
# applied. Run after every swegym reinstall (called from the training wrapper
# alongside the other patch_* scripts, and from install_env.sh).
#
# Background: swegym 2.0.13 harness/grading.py::get_logs_eval derives the repo
# key from the test-log path parent stem (the instance_id, e.g.
# Project-MONAI__MONAI-6924) and indexes MAP_REPO_TO_PARSER[repo] without
# lowercasing. MAP_REPO_TO_PARSER keys are lowercase, so Project-MONAI/MONAI
# (53/293 = 18% of swe-gym-293 prompts) raises KeyError and the episode is
# graded failed instead of resolved. Training rollout grading and the eval
# run_harness_on_applied_patch path share this code, so the fix benefits both.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python3}"

GRADING_PY="${GRADING_PY:-${PROJECT_ROOT}/.venv/lib/python3.12/site-packages/swegym/harness/grading.py}"
[[ -f "${GRADING_PY}" ]] || {
  echo "ERROR: swegym grading.py not found: ${GRADING_PY}" >&2
  echo "(is swegym installed in the venv?)" >&2
  exit 1
}

SWEGYM_GRADING_PY="${GRADING_PY}" "${PYTHON_BIN}" "${SCRIPT_DIR}/patch_swegym_grading_lower.py" "${GRADING_PY}"