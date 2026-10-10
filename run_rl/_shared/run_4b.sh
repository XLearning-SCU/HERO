#!/usr/bin/env bash
# Local-Docker deployment wrapper for combined SWE-Gym / Multi-SWE /
# SWE-ReBench Slime GRPO training.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export TRAINING_MODEL_SIZE="${TRAINING_MODEL_SIZE:-4b}"
export TRAINING_MODEL_TYPE="${TRAINING_MODEL_TYPE:-dense}"
UPSTREAM_RUN="${UPSTREAM_RUN:-${PROJECT_ROOT}/examples/swegym_slime_grpo/run.sh}"
source "${SCRIPT_DIR}/runtime/docker_storage_guard.sh"

# Supply credentials through the environment or the W&B login configuration.
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_KEY="${WANDB_KEY:-${WANDB_API_KEY}}"
export WANDB_PROJECT="${WANDB_PROJECT:-hero}"
# Upload training runs by default and tolerate slow shared-run initialization.
# Callers can still explicitly select offline/disabled or another timeout.
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
EXP_TAG="${EXP_TAG:-$(basename "${BASH_SOURCE[0]}" .sh)}"
# Name the wandb run after this script (e.g. run_prorl_swegym293_qwen3.5_4b vs
# ..._compact) so the two experiments are distinguishable in the wandb UI.
# slime's wandb_utils uses --wandb-group as the run name; upstream run.sh reads
# $WANDB_GROUP and appends a random suffix + "-RANK_<rank>" by default.
export WANDB_GROUP="${WANDB_GROUP:-${EXP_TAG}}"
# Logical model name exposed by HERO/Claude Code. Model-specific wrappers
# override this so routing metadata and logs match the checkpoint actually
# served; the maintained 4B launcher retains the baseline default.
export HERO_MODEL_NAME="${HERO_MODEL_NAME:-Qwen/Qwen3.5-4B}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)_${BASHPID}}"
RUN_DATE="${RUN_DATE:-$(date +%m%d)}"
export RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/../results}"
export RUN_ROOT="${RUN_ROOT:-${RESULTS_ROOT}/train/${TRAINING_METHOD:-HERO}/qwen35_${TRAINING_MODEL_SIZE}/${STAMP}_${EXP_TAG}}"
[[ "${RUN_ROOT}" == /* ]] || RUN_ROOT="${PWD}/${RUN_ROOT}"
RUN_ID="${RUN_ID:-${EXP_TAG}_${STAMP}}"
export HERO_RUN_ID="${HERO_RUN_ID:-${RUN_ID}}"
# Triton and TorchInductor compile Qwen3.5's FLA kernels on demand.  Their
# cache cannot live on the CPFS project mount: concurrent workers can observe
# a stale FUSE file handle while reading a compiled artifact.  Keep this
# per-run cache on the node-local ext4 volume; callers can still override it.
export COMPILER_CACHE_ROOT="${COMPILER_CACHE_ROOT:-${XDG_CACHE_HOME:-${HOME}/.cache}/hero/compiler/${RUN_ID}}"

# Ray CLI/status helpers import NumPy even though they do not perform
# throughput-sensitive BLAS work.  Keep OpenBLAS from creating one thread per
# host CPU for every helper process; under high Docker concurrency those
# short-lived thread bursts can exhaust the job's PID budget.  This does not
# reduce rollout/container concurrency or GPU training parallelism, and
# callers that genuinely need threaded host BLAS can still override it.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

export DOCKER_HOST="${HERO_LOCAL_DOCKER_HOST:-${DOCKER_HOST:-tcp://127.0.0.1:2377}}"
export HERO_LOCAL_DOCKER_HOST="${DOCKER_HOST}"
export HERO_LOCAL_DOCKER_DATA_ROOT="${HERO_LOCAL_DOCKER_DATA_ROOT:-${XDG_CACHE_HOME:-${HOME}/.cache}/hero/docker-data}"
export HERO_LOCAL_DOCKER_LOAD_CONCURRENCY="${HERO_LOCAL_DOCKER_LOAD_CONCURRENCY:-6}"
export HERO_LOCAL_DOCKER_RUN_CONCURRENCY="${HERO_LOCAL_DOCKER_RUN_CONCURRENCY:-16}"
export HERO_LOCAL_DOCKER_ACTIVE_CONCURRENCY="${HERO_LOCAL_DOCKER_ACTIVE_CONCURRENCY:-64}"
export HERO_LOCAL_DOCKER_IMAGE_CACHE_SIZE="${HERO_LOCAL_DOCKER_IMAGE_CACHE_SIZE:-16}"
export HERO_LOCAL_DOCKER_MIN_FREE_GB="${HERO_LOCAL_DOCKER_MIN_FREE_GB:-500}"
export HERO_LOCAL_DOCKER_RMI="${HERO_LOCAL_DOCKER_RMI:-1}"
# A VFS container can spend minutes removing its copied writable tree. Return
# the completed rollout/reward before teardown while retaining its active slot
# until Docker confirms deletion. This also applies to already-queued cluster
# jobs because they execute this launcher from the live /ring worktree.
if [[ "${HERO_LOCAL_DOCKER_STORAGE_DRIVER:-}" == "vfs" ]]; then
  export HERO_GATEWAY_PUBLISH_BEFORE_TEARDOWN="${HERO_GATEWAY_PUBLISH_BEFORE_TEARDOWN:-1}"
fi
export HERO_LOCAL_DOCKER_TRANSIENT_RETRIES="${HERO_LOCAL_DOCKER_TRANSIENT_RETRIES:-4}"
export HERO_LOCAL_DOCKER_MEMORY_LIMIT_GB="${HERO_LOCAL_DOCKER_MEMORY_LIMIT_GB:-32}"
export HERO_LOCAL_DOCKER_MEMORY_CHECK_INTERVAL_SEC="${HERO_LOCAL_DOCKER_MEMORY_CHECK_INTERVAL_SEC:-30}"
export SLIME_MIN_AVAILABLE_MEMORY_BEFORE_TRAIN_GB="${SLIME_MIN_AVAILABLE_MEMORY_BEFORE_TRAIN_GB:-${HERO_MIN_AVAILABLE_MEMORY_DURING_RUN_GB:-300}}"
export SLIME_HOST_MEMORY_CHECK_INTERVAL_SEC="${SLIME_HOST_MEMORY_CHECK_INTERVAL_SEC:-15}"
# Runtime-only Ray cache budget.  This does not change any model, batch,
# rollout, CPU, or GPU setting.  It keeps headroom for long-trace Megatron
# peaks; surplus Ray objects spill to the local Ray session directory.
export RAY_OBJECT_STORE_MEMORY_BYTES="${RAY_OBJECT_STORE_MEMORY_BYTES:-103079215104}"
export FLASHINFER_USE_CUDA_NORM="${FLASHINFER_USE_CUDA_NORM:-1}"
# Use the verified 64K training/rollout context: four CP ranks each
# carry a 16K token budget. This single-node launcher has four actor GPUs, so
# TP must be 1 for TP*CP to fit the actor world size.
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
export CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-4}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-65536}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.8}"
# Qwen3.5 dense-model rollout has a known multi-GPU-engine bug. Keep 4B/9B
# SGLang engines strictly TP=1 unless a model-specific wrapper explicitly
# preserves its validated engine topology (the 35B-A3B MoE launcher does so).
case "${RUN_MULTI_PRESERVE_ROLLOUT_ENGINE_GPUS:-0}" in
  0)
    export ROLLOUT_NUM_GPUS_PER_ENGINE=1
    ;;
  1)
    [[ "${ROLLOUT_NUM_GPUS_PER_ENGINE:-}" =~ ^[1-9][0-9]*$ ]] || {
      echo "ROLLOUT_NUM_GPUS_PER_ENGINE must be a positive integer when RUN_MULTI_PRESERVE_ROLLOUT_ENGINE_GPUS=1" >&2
      exit 2
    }
    ;;
  *)
    echo "RUN_MULTI_PRESERVE_ROLLOUT_ENGINE_GPUS must be 0 or 1" >&2
    exit 2
    ;;
esac
export HERO_MIN_COMPLETE_ACCEPT_FRACTION="${HERO_MIN_COMPLETE_ACCEPT_FRACTION:-0}"
export HERO_TRAIN_POLICY_EARLY_STOP_TRAJECTORIES=0
# Avoid Transformer Engine FusedAttention NaNs observed during CP training.
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash}"

# Follow the verified Claude Code preset defaults: do not pass --max-turns
# unless a method wrapper explicitly opts in to bound length growth.
if [[ "${RUN_MULTI_PRESERVE_MAX_TURNS:-0}" == "1" ]]; then
  [[ "${CLAUDE_CODE_MAX_TURNS:-}" =~ ^[1-9][0-9]*$ ]] || {
    echo "CLAUDE_CODE_MAX_TURNS must be a positive integer when RUN_MULTI_PRESERVE_MAX_TURNS=1" >&2
    exit 2
  }
else
  unset CLAUDE_CODE_MAX_TURNS
fi
# Do not impose a gateway completion cap. Bound each Claude Code response to
# 8192 tokens by default, while allowing callers to override the value.
unset HERO_MAX_COMPLETIONS
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-8192}"
[[ "${CLAUDE_CODE_MAX_OUTPUT_TOKENS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "CLAUDE_CODE_MAX_OUTPUT_TOKENS must be a positive integer" >&2
  exit 2
}
# Let Claude expose its default built-in tool set. Only its two dedicated
# web-access tools are denied.
unset CLAUDE_CODE_TOOLS
export CLAUDE_CODE_DISALLOWED_TOOLS="WebFetch,WebSearch,Agent"

# Keep optional out-of-band Git progress disabled for this official baseline.
export HERO_GIT_PROGRESS_ENABLED="${HERO_GIT_PROGRESS_ENABLED:-0}"
export HERO_GIT_PROGRESS_CHECKPOINT_INTERVAL="${HERO_GIT_PROGRESS_CHECKPOINT_INTERVAL:-10}"

# Keep the rollout group shape explicit in this launcher.  These are passed
# unchanged to the upstream run.sh / train_async.py command.
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
export CUSTOM_ROLLOUT_LOG_FUNCTION_PATH="${CUSTOM_ROLLOUT_LOG_FUNCTION_PATH:-slime_bridge.rollout_log.log_rollout_data}"

# Run exactly this many sampling→training rollout steps.  This is independent
# of the 4 prompts × 16 samples collected in each rollout step.
export NUM_ROLLOUT="${NUM_ROLLOUT:-80}"

[[ "${HERO_GIT_PROGRESS_ENABLED}" == "0" || "${HERO_GIT_PROGRESS_ENABLED}" == "1" ]] || {
  echo "HERO_GIT_PROGRESS_ENABLED must be 0 or 1" >&2
  exit 2
}
[[ "${HERO_GIT_PROGRESS_CHECKPOINT_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || {
  echo "HERO_GIT_PROGRESS_CHECKPOINT_INTERVAL must be a positive integer" >&2
  exit 2
}
[[ "${NUM_ROLLOUT}" =~ ^[1-9][0-9]*$ ]] || {
  echo "NUM_ROLLOUT must be a positive integer" >&2
  exit 2
}
[[ "${ROLLOUT_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ROLLOUT_BATCH_SIZE must be a positive integer" >&2
  exit 2
}
[[ "${N_SAMPLES_PER_PROMPT}" =~ ^[1-9][0-9]*$ ]] || {
  echo "N_SAMPLES_PER_PROMPT must be a positive integer" >&2
  exit 2
}

# Auto-compaction is opt-in.  When enabled, render persistent Claude Code
# settings into each task's CLAUDE_CONFIG_DIR; the CLI env variable alone only
# overrides the window and does not enable the feature.
CLAUDE_CODE_AUTO_COMPACT="${CLAUDE_CODE_AUTO_COMPACT:-0}"
CLAUDE_CODE_AUTO_COMPACT_WINDOW="${CLAUDE_CODE_AUTO_COMPACT_WINDOW:-50000}"
case "${CLAUDE_CODE_AUTO_COMPACT}" in
  0|false|False|FALSE|no|No|NO)
    export CLAUDE_CODE_AUTO_COMPACT_SETTINGS=""
    ;;
  1|true|True|TRUE|yes|Yes|YES)
    [[ "${CLAUDE_CODE_AUTO_COMPACT_WINDOW}" =~ ^[0-9]+$ ]] || {
      echo "CLAUDE_CODE_AUTO_COMPACT_WINDOW must be an integer" >&2
      exit 2
    }
    export CLAUDE_CODE_AUTO_COMPACT_SETTINGS=$'auto_compact_enabled: true\n      auto_compact_window: '"${CLAUDE_CODE_AUTO_COMPACT_WINDOW}"
    ;;
  *)
    echo "CLAUDE_CODE_AUTO_COMPACT must be 0/1 (or false/true)" >&2
    exit 2
    ;;
esac

# Claude Code tool limits are set in hero_config_combine_local_docker.yaml under
# agent.settings.  That is the execution path used by HERO's Claude preset.
# Transformer Engine's CUDA-13 common library needs an explicit CUDA include
# root on this host.  Otherwise the system TE namespace package cannot locate
# cuda_runtime after the CUDA-12 wheels have been removed.
export NVTE_CUDA_INCLUDE_DIR="${NVTE_CUDA_INCLUDE_DIR:-/usr/local/cuda/include}"

SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/framework/slime}"
MEGATRON_DIR="${MEGATRON_DIR:-${PROJECT_ROOT}/framework/Megatron-LM}"
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/../models}"
HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_ROOT}/Qwen3.5-4B}"
REF_LOAD="${REF_LOAD:-${MODEL_ROOT}/Qwen3.5-4B_torch_dist}"
PROMPT_DATA="${PROMPT_DATA:-${PROJECT_ROOT}/../data/swe_train_640.jsonl}"
# The pinned Claude Code runtime is prepared beside HERO under cli/.
AGENT_CLI_DIR="${AGENT_CLI_DIR:-${PROJECT_ROOT}/../cli/claude_code}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python3}"


# Torch 2.11 cu130 loads NVSHMEM/NCCL dynamically. venv has
# include-system-site-packages=false, so torch/sglang need the system nvidia.*
# CUDA-13 library dirs on LD_LIBRARY_PATH:
#   - nvshmem: libnvshmem_host.so.3 (torch import-time dlopen)
#   - nccl:    libnccl.so.2 with ncclDevCommDestroy (cu130 symbol; the distro
#              /lib/x86_64-linux-gnu/libnccl.so.2 lacks it and shadows it)
#   - cudnn + cu13: TE/torch CUDA runtime libs
# Probe each via its python package; prepend before everything else so distro
# libs do not shadow the cu130 runtime.
#
# CRITICAL: probe with $PYTHON_BIN (venv) first, but the venv ships the
# nvidia.{nvshmem,nccl,cudnn} packages as *empty namespace stubs* whose lib/
# directory does not exist. In that case fall back to the system interpreter
# (/usr/bin/python3 -> /usr/local/lib/python3.12/dist-packages/nvidia/*), which
# carries the real .so files. Without this fallback torch import-time dlopen
# of libnvshmem_host.so.3 fails with "cannot open shared object file".
_nvidia_lib() {
  local _name="$1"
  local _d
  _d="$("${PYTHON_BIN}" -c "import nvidia.${_name},os;print(os.path.join(list(nvidia.${_name}.__path__)[0],'lib'))" 2>/dev/null)"
  if [ -z "${_d}" ] || [ ! -d "${_d}" ]; then
    _d="$(/usr/bin/python3 -c "import nvidia.${_name},os;print(os.path.join(list(nvidia.${_name}.__path__)[0],'lib'))" 2>/dev/null)"
  fi
  printf '%s' "${_d}"
}
_PREPEND=""
for _lib in nvshmem nccl cudnn cu13; do
  _d="$(_nvidia_lib "${_lib}")"
  [ -n "${_d}" ] && [ -d "${_d}" ] && _PREPEND="${_d}:${_PREPEND}"
done
[ -n "${_PREPEND}" ] && export LD_LIBRARY_PATH="${_PREPEND}${LD_LIBRARY_PATH:-}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/settings" "${RUN_ROOT}/runtime" "${RUN_ROOT}/ckpt"
[[ -f "${PROMPT_DATA}" ]] || { echo "Missing training data: ${PROMPT_DATA}" >&2; exit 1; }
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_training_data.py" \
  --input "${PROMPT_DATA}"
[[ -x "${AGENT_CLI_DIR}/bin/claude" ]] || {
  echo "Missing ProRL Claude Code CLI: ${AGENT_CLI_DIR}/bin/claude" >&2
  echo "Run: bash run_rl/_shared/init_env/install_env.sh" >&2
  exit 1
}
[[ -f "${REF_LOAD}/latest_checkpointed_iteration.txt" ]] || {
  echo "Missing torch_dist checkpoint: ${REF_LOAD}" >&2
  exit 1
}
if [[ "${ATTENTION_BACKEND}" == "flash" ]]; then
  # Match the Ray worker import path.  The interactive shell may prepend
  # system site-packages and otherwise hide a missing venv FlashAttention.
  PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${MEGATRON_DIR}" "${PYTHON_BIN}" - <<'PY'
from importlib.metadata import PackageNotFoundError, version
from packaging.version import Version
import sys

try:
    installed = Version(version("flash-attn"))
except PackageNotFoundError:
    sys.exit(
        "FlashAttention is required for CP training but is not installed. "
        "Install flash-attn==2.7.4.post1 first."
    )

# Transformer Engine 2.5.0 only discovers FlashAttention in this range.
if not (Version("2.1.1") <= installed <= Version("2.7.4.post1")):
    sys.exit(
        f"flash-attn {installed} is incompatible with Transformer Engine 2.5.0. "
        "Install flash-attn==2.7.4.post1; refusing the unsafe FusedAttention fallback."
    )
PY
fi

hero_check_docker_storage
hero_check_available_memory
hero_cleanup_stale_containers
trap hero_cleanup_run_containers EXIT INT TERM

# Re-apply the Slime numpy-assert patch before every launch in case Slime was
# reinstalled/re-cloned since install_env. Idempotent no-op when already applied.
SLIME_DIR="${SLIME_DIR}" PYTHON_BIN="${PYTHON_BIN}" \
  bash "${SCRIPT_DIR}/init_env/patch_slime_numpy_assert.sh"
# Re-apply the slime_bridge adapter v0.3 group-id compatibility migration in
# case hero was reinstalled/re-cloned. It aligns scheduling and loss IDs.
PYTHON_BIN="${PYTHON_BIN}" \
  bash "${SCRIPT_DIR}/init_env/patch_slime_bridge_group_id.sh"
# Re-apply the SGLang 0.5.13 token-metadata patch (text-only rollout on the
# Qwen3.5 VLM checkpoint needs it; without it agent chat requests get HTTP 400
# and every rollout is dropped as "zero trainable tokens"). Run with the venv
# Python on PATH so the version check sees venv sglang==0.5.13, not a system
# sglang. Idempotent.
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
# The caller's PYTHONPATH contains the system site-packages dir, which makes
# the patch script's `import sglang` resolve to a system sglang (dev build,
# version mismatch) and silently skip the patch. Scrub PYTHONPATH so it sees
# the venv sglang==0.5.13.
unset PYTHONPATH
bash "${PROJECT_ROOT}/scripts/patch/patch_sglang_0513_token_metadata.sh"

cp "${BASH_SOURCE[0]}" "${SCRIPT_DIR}/config/hero_config_combine_local_docker.yaml" \
  "${SCRIPT_DIR}/config/topology_local_docker_train.yaml" "${RUN_ROOT}/settings/"
env | sort | awk -F= '$1 ~ /(KEY|TOKEN|SECRET|PASSWORD)$/ {print $1"=<redacted>"; next} {print}' \
  >"${RUN_ROOT}/settings/environment.txt"

# Never inherit the caller's PYTHONPATH.  The launcher environment on these
# hosts contains the system site-packages directory, which otherwise makes
# Ray's head process use system Ray while workers use the project venv Ray.
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${MEGATRON_DIR}"
export RUN_DIR="${RUN_ROOT}/runtime"
# Training intentionally uses the upstream topology without eval-only sampling
# overrides. The local-Docker backend lives solely in the HERO task config.
export TOPOLOGY_TEMPLATE="${SCRIPT_DIR}/config/topology_local_docker_train.yaml"
export HERO_CONFIG_TEMPLATE="${SCRIPT_DIR}/config/hero_config_combine_local_docker.yaml"
export TOPOLOGY_PATH="${RUN_ROOT}/settings/topology.rendered.yaml"
export CUSTOM_CONFIG_PATH="${RUN_ROOT}/settings/hero_config.rendered.yaml"
export SAVE_ROOT="${RUN_ROOT}/ckpt"
export SAVE_DIR="${RUN_ROOT}/ckpt"
# Save a Megatron torch_dist dict AND a HuggingFace checkpoint every
# SAVE_INTERVAL rollouts (both fire on the same tick in slime's save_model).
# Mirrors slime/run_rl/run_qwen3.5_4b_swe_gym_293.sh's CKPT_ARGS shape.  The
# {rollout_id:06d} placeholder is expanded per-iteration by slime; set
# SAVE_HF to override the path, or SAVE_HF_DISABLE=1 to skip HF saving.
export SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
# NB: build the {rollout_id:06d} default into a temp var before feeding it
# through ":-" — embedding the literal placeholder directly inside
# "${SAVE_HF:-...}" lets bash brace expansion mangle it into
# "{rollout_id:06d_hf}" on bash 5.x.
_SAVE_HF_DEFAULT="${SAVE_DIR}/iter_{rollout_id:06d}_hf"
export SAVE_HF="${SAVE_HF:-${_SAVE_HF_DEFAULT}}"

echo "[run_rl] RUN_ROOT=${RUN_ROOT}"
echo "[run_rl] HERO model name=${HERO_MODEL_NAME}"
echo "[run_rl] data_mode=combined prompt_data=${PROMPT_DATA} hero_config=${HERO_CONFIG_TEMPLATE}"
echo "[run_rl] Docker=${DOCKER_HOST} data_root=${HERO_LOCAL_DOCKER_DATA_ROOT}"
echo "[run_rl] Claude Code cli=${AGENT_CLI_DIR}"
echo "[run_rl] Claude Code auto-compact=${CLAUDE_CODE_AUTO_COMPACT}"
echo "[run_rl] Git progress enabled=${HERO_GIT_PROGRESS_ENABLED} checkpoint_interval=${HERO_GIT_PROGRESS_CHECKPOINT_INTERVAL}"
echo "[run_rl] Claude Code max_turns=${CLAUDE_CODE_MAX_TURNS:-<unset>} max_completions=<unset> max_output_tokens=${CLAUDE_CODE_MAX_OUTPUT_TOKENS}"
echo "[run_rl] Claude Code tools=<default> disallowed_tools=${CLAUDE_CODE_DISALLOWED_TOOLS}"
echo "[run_rl] num_rollout=${NUM_ROLLOUT} rollout_batch_size=${ROLLOUT_BATCH_SIZE} n_samples_per_prompt=${N_SAMPLES_PER_PROMPT}"
echo "[run_rl] save_interval=${SAVE_INTERVAL} save_dir=${SAVE_DIR}"
echo "[run_rl] save_hf=${SAVE_HF:-<disabled>}"
echo "[run_rl] compiler cache=${COMPILER_CACHE_ROOT}"
echo "[run_rl] Ray object-store cap=${RAY_OBJECT_STORE_MEMORY_BYTES} bytes"
echo "[run_rl] attention backend=${ATTENTION_BACKEND}"
echo "[run_rl] context TP=${TENSOR_MODEL_PARALLEL_SIZE} CP=${CONTEXT_PARALLEL_SIZE} max_tokens_per_gpu=${MAX_TOKENS_PER_GPU} sglang=${SGLANG_CONTEXT_LENGTH} mem_fraction=${SGLANG_MEM_FRACTION_STATIC} min_complete_fraction=${HERO_MIN_COMPLETE_ACCEPT_FRACTION}"
echo "[run_rl] ProRL upstream training entry=${UPSTREAM_RUN}"

cd "${RUN_ROOT}"
HF_CHECKPOINT="${HF_CHECKPOINT}" REF_LOAD="${REF_LOAD}" PROMPT_DATA="${PROMPT_DATA}" \
AGENT_CLI_DIR="${AGENT_CLI_DIR}" SLIME_DIR="${SLIME_DIR}" MEGATRON_DIR="${MEGATRON_DIR}" \
PYTHON_BIN="${PYTHON_BIN}" RUN_ID="${RUN_ID}" \
  bash "${UPSTREAM_RUN}" 2>&1 | tee "${RUN_ROOT}/run.log"
