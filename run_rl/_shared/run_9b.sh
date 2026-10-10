#!/usr/bin/env bash
# Qwen3.5-9B variant of run_4b.sh for one 8-GPU node.
# The maintained 4B launcher owns the common HERO/Docker/Slime setup; this
# wrapper only supplies model- and memory-specific overrides.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

export EXP_TAG="${EXP_TAG:-run_multi_9b}"
export HERO_MODEL_NAME="${HERO_MODEL_NAME:-Qwen/Qwen3.5-9B}"

MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/../models}"
export HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_ROOT}/Qwen3.5-9B}"
export REF_LOAD="${REF_LOAD:-${MODEL_ROOT}/Qwen3.5-9B_torch_dist}"
export TRAINING_MODEL_SIZE=9b
export TRAINING_MODEL_TYPE=dense
export MODEL_ARGS_FILE="${MODEL_ARGS_FILE:-${PROJECT_ROOT}/examples/swegym_slime_grpo/model_args.sh}"

# Four actor GPUs: TP=2 shards the larger model and CP=2 distributes long
# sequences. TP*CP=4, so this remains a single actor replica.
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
export CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-2}"
# The Qwen3.5 SGLang path is only valid with rollout TP=1.
export ROLLOUT_NUM_GPUS_PER_ENGINE=1

# Keep rollout generation and Megatron's trainable-trace admission limit at
# the same 64K contract. The adapter admits at most max_tokens_per_gpu * CP;
# 16K with CP=2 silently reduced 9B training to 32K and replaced longer,
# otherwise valid trajectories with zero-gradient placeholders.
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-65536}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}"
for _context_value in CONTEXT_PARALLEL_SIZE MAX_TOKENS_PER_GPU SGLANG_CONTEXT_LENGTH; do
  [[ "${!_context_value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "${_context_value} must be a positive integer" >&2
    exit 2
  }
done
if (( SGLANG_CONTEXT_LENGTH != 65536 )); then
  echo "9B scaling is fixed at SGLANG_CONTEXT_LENGTH=65536 (64K)" >&2
  exit 2
fi
if (( MAX_TOKENS_PER_GPU * CONTEXT_PARALLEL_SIZE != 65536 )); then
  echo "9B requires MAX_TOKENS_PER_GPU * CONTEXT_PARALLEL_SIZE == SGLANG_CONTEXT_LENGTH (64K)" >&2
  exit 2
fi

exec bash "${SCRIPT_DIR}/run_4b.sh" "$@"
