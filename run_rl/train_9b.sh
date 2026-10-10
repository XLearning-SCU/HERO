#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/HERO/launch.sh"
SCALING_MODEL_NAME=qwen35_9b
SCALING_LAUNCHER=run_rl/_shared/run_9b.sh
SCALING_METHOD=hero_fixed_gate
SCALING_EFFICIENCY_COEFFICIENT="${EFFICIENCY_COEFFICIENT:-0.2}"
SCALING_GATE_THRESHOLD="${SCALING_GATE_THRESHOLD:-0.6}"
SCALING_EXP_TAG="HERO_context64k_lambda${SCALING_EFFICIENCY_COEFFICIENT}_tau${SCALING_GATE_THRESHOLD}"
scaling_launch "$@"
