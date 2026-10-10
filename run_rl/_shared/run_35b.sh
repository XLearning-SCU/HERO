#!/usr/bin/env bash
# Qwen3.5-35B-A3B ProRL async training on 8 nodes x 8 GPUs.
# The official Slime recipe colocates 64-GPU actor training and rollout so the
# same allocation is reused instead of requiring 128 GPUs.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

export EXP_TAG="${EXP_TAG:-run_multi_35b_a3b}"
export HERO_MODEL_NAME="${HERO_MODEL_NAME:-Qwen/Qwen3.5-35B-A3B}"
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/../models}"
export HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_ROOT}/Qwen3.5-35B-A3B}"
export REF_LOAD="${REF_LOAD:-${MODEL_ROOT}/Qwen3.5-35B-A3B_torch_dist}"
export TRAINING_MODEL_SIZE=35b
export TRAINING_MODEL_TYPE=moe
export MODEL_ARGS_FILE="${MODEL_ARGS_FILE:-${PROJECT_ROOT}/examples/swegym_slime_grpo/model_args.sh}"

export RAY_NUM_NODES="${RAY_NUM_NODES:-8}"
export RAY_NUM_GPUS_PER_NODE="${RAY_NUM_GPUS_PER_NODE:-8}"
export RAY_HEAD_IP="${RAY_HEAD_IP:-${MASTER_ADDR:-${MLP_WORKER_0_HOST:-$(hostname -I | awk '{print $1}')}}}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-8}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-64}"
export SLIME_COLOCATE="${SLIME_COLOCATE:-1}"

# BEGIN VFS_MULTINODE_PROFILE
# TASK_RECORD_DIR is set by the cluster worker bootstrap, including queued
# submissions. Local IDE and overlay2 runs keep the historical comm policy.
export SLIME_VFS_MULTINODE=0
if [[ -n "${TASK_RECORD_DIR:-}" && "${HERO_LOCAL_DOCKER_STORAGE_DRIVER:-}" == "vfs" && "${ACTOR_NUM_NODES}" =~ ^[0-9]+$ ]] && (( ACTOR_NUM_NODES > 1 )); then
  export SLIME_VFS_MULTINODE=1
  export SLIME_TRAIN_PHASE_TRACE_DIR="${TASK_RECORD_DIR}/train_phase_trace"
  echo "[35b] cluster VFS multi-node: disabling CUDA memory probes/cleanup inside collective calls"
  rdma_devices=(/sys/class/infiniband/*)
  if [[ -e "${rdma_devices[0]}" && -d /dev/infiniband ]]; then
    echo "[35b] RDMA devices visible; verify NCCL NET/IB selection in actor logs"
  else
    echo "[35b] WARNING: RDMA devices not visible; NCCL may fall back to TCP Socket. NCCL_IB_* variables alone do not allocate RDMA devices." >&2
  fi
  unset rdma_devices
fi
# END VFS_MULTINODE_PROFILE

# Official Qwen3.5-35B-A3B 64-GPU training topology.
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
export CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-8}"
export PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
export EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-8}"
export EXPERT_TENSOR_PARALLEL_SIZE="${EXPERT_TENSOR_PARALLEL_SIZE:-1}"
export OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-1}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"

# Keep the ProRL 64K rollout contract while matching the official Slime
# Qwen3.5-35B-A3B per-engine MoE topology. With 64 rollout GPUs this launches
# eight replicas of the official 8-GPU engine shape.
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-65536}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-8}"
# The shared 4B launcher normally forces dense Qwen3.5 rollout engines to one
# GPU. Preserve this model-specific, officially validated TP8/EP8 MoE shape.
export RUN_MULTI_PRESERVE_ROLLOUT_ENGINE_GPUS=1
export SGLANG_DP_SIZE="${SGLANG_DP_SIZE:-8}"
export SGLANG_EP_SIZE="${SGLANG_EP_SIZE:-8}"
export SGLANG_MOE_DENSE_TP_SIZE="${SGLANG_MOE_DENSE_TP_SIZE:-1}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.75}"
# Keep agent history below the 64K SGLang limit.  Without compaction, a smoke
# rollout reached 65,566 tokens and received an avoidable HTTP 400 before the
# agent recovered on its next request.
export CLAUDE_CODE_AUTO_COMPACT="${CLAUDE_CODE_AUTO_COMPACT:-1}"
export CLAUDE_CODE_AUTO_COMPACT_WINDOW="${CLAUDE_CODE_AUTO_COMPACT_WINDOW:-50000}"

# The shared runtime uses Torch cu130 while upstream Slime defaults colocated
# memory-saver actors to the cu12 preload hook. Select the matching hook only
# for this MoE launcher; dense launchers never export this opt-in.
_moe_python_bin="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python3}"
[[ -x "${_moe_python_bin}" ]] || {
  echo "Missing Python needed to select the MoE TMS preload hook: ${_moe_python_bin}" >&2
  exit 1
}
_torch_cuda_major="$("${_moe_python_bin}" -c 'import torch; print(str(torch.version.cuda).split(".", 1)[0])')"
_tms_package_root="$("${_moe_python_bin}" -c 'import os, torch_memory_saver; print(os.path.dirname(os.path.dirname(torch_memory_saver.__file__)))')"
export SLIME_TMS_PRELOAD_LIBRARY="${SLIME_TMS_PRELOAD_LIBRARY:-${_tms_package_root}/torch_memory_saver_hook_mode_preload_cu${_torch_cuda_major}.abi3.so}"
[[ -f "${SLIME_TMS_PRELOAD_LIBRARY}" ]] || {
  echo "Missing CUDA ${_torch_cuda_major} TMS preload hook: ${SLIME_TMS_PRELOAD_LIBRARY}" >&2
  exit 1
}
export SLIME_COLOCATED_WEIGHT_SYNC_CPU="${SLIME_COLOCATED_WEIGHT_SYNC_CPU:-1}"
export UPDATE_WEIGHT_BUFFER_SIZE="${UPDATE_WEIGHT_BUFFER_SIZE:-536870912}"
# The colocated MoE trainer/reference pair consumes substantially more host
# memory than the dense 4B job before rollout containers are admitted.  Keep a
# 150-GiB floor for this launcher: with the 32-GiB per-container cgroup limit it
# still leaves ample emergency headroom, while avoiding the dense launcher's
# 300-GiB default permanently blocking every MoE rollout container.
export HERO_MIN_AVAILABLE_MEMORY_DURING_RUN_GB="${HERO_MIN_AVAILABLE_MEMORY_DURING_RUN_GB:-150}"
# Megatron actors receive the preload through actor_group.py. SGLang remains
# on its Python hook because preloading TMS before its DP scheduler allocates
# memory makes cuMemCreate reject the allocation on CUDA 13.
export SGLANG_MEMORY_SAVER_CUDA_GRAPH=false
export SLIME_TRAIN_ENTRY="${SLIME_TRAIN_ENTRY:-${PROJECT_ROOT}/examples/swegym_slime_grpo/moe_compat.py}"
for _topology_value in RAY_NUM_NODES RAY_NUM_GPUS_PER_NODE ACTOR_NUM_NODES \
  ACTOR_NUM_GPUS_PER_NODE ROLLOUT_NUM_GPUS ROLLOUT_NUM_GPUS_PER_ENGINE \
  TENSOR_MODEL_PARALLEL_SIZE PIPELINE_MODEL_PARALLEL_SIZE \
  EXPERT_MODEL_PARALLEL_SIZE SGLANG_DP_SIZE SGLANG_EP_SIZE; do
  [[ "${!_topology_value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "${_topology_value} must be a positive integer" >&2
    exit 2
  }
done
for _context_value in CONTEXT_PARALLEL_SIZE MAX_TOKENS_PER_GPU SGLANG_CONTEXT_LENGTH; do
  [[ "${!_context_value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "${_context_value} must be a positive integer" >&2
    exit 2
  }
done
if (( ACTOR_NUM_NODES > RAY_NUM_NODES )); then
  echo "ACTOR_NUM_NODES cannot exceed RAY_NUM_NODES" >&2
  exit 2
fi
_actor_world_size=$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))
_model_parallel_size=$((TENSOR_MODEL_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE))
if (( _actor_world_size % _model_parallel_size != 0 )); then
  echo "Actor world size must be divisible by TP*CP*PP" >&2
  exit 2
fi
if (( _actor_world_size % EXPERT_MODEL_PARALLEL_SIZE != 0 )); then
  echo "Actor world size must be divisible by EXPERT_MODEL_PARALLEL_SIZE" >&2
  exit 2
fi
if (( ROLLOUT_NUM_GPUS % ROLLOUT_NUM_GPUS_PER_ENGINE != 0 )); then
  echo "ROLLOUT_NUM_GPUS must be divisible by ROLLOUT_NUM_GPUS_PER_ENGINE" >&2
  exit 2
fi
if (( SGLANG_DP_SIZE > ROLLOUT_NUM_GPUS_PER_ENGINE || ROLLOUT_NUM_GPUS_PER_ENGINE % SGLANG_DP_SIZE != 0 )); then
  echo "SGLANG_DP_SIZE must divide ROLLOUT_NUM_GPUS_PER_ENGINE" >&2
  exit 2
fi
if (( SGLANG_EP_SIZE > SGLANG_DP_SIZE || SGLANG_DP_SIZE % SGLANG_EP_SIZE != 0 )); then
  echo "SGLANG_EP_SIZE must divide SGLANG_DP_SIZE" >&2
  exit 2
fi
if (( SGLANG_CONTEXT_LENGTH != 65536 )); then
  echo "35B-A3B scaling is fixed at SGLANG_CONTEXT_LENGTH=65536 (64K)" >&2
  exit 2
fi
if (( MAX_TOKENS_PER_GPU * CONTEXT_PARALLEL_SIZE != 65536 )); then
  echo "35B-A3B requires MAX_TOKENS_PER_GPU * CONTEXT_PARALLEL_SIZE == SGLANG_CONTEXT_LENGTH (64K)" >&2
  exit 2
fi

export UPSTREAM_RUN="${UPSTREAM_RUN:-${PROJECT_ROOT}/examples/swegym_slime_grpo/run.sh}"
exec bash "${SCRIPT_DIR}/run_4b.sh" "$@"
