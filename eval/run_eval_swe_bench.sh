#!/usr/bin/env bash
# Evaluate an HF checkpoint through ProRL + Claude Code + local Docker.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
source "${PROJECT_ROOT}/run_rl/_shared/runtime/docker_storage_guard.sh"

DATASET="${DATASET:-swe-bench-verified}"
case "${DATASET}" in
  swe-bench-verified|swebench-verified|verified)
    INPUT_DEFAULT="${PROJECT_ROOT}/../data/swe_bench_verified.jsonl"
    IMAGE_DIR_DEFAULT="${PROJECT_ROOT}/../docker/swe-bench-verified"
    ;;
  swe-bench-multilingual|swebench-multilingual|multilingual)
    INPUT_DEFAULT="${PROJECT_ROOT}/../data/swe_bench_multilingual.jsonl"
    IMAGE_DIR_DEFAULT="${PROJECT_ROOT}/../docker/swe-bench-multilingual"
    ;;
  *) echo "Unknown DATASET=${DATASET}" >&2; exit 1 ;;
esac

INPUT="${INPUT:-${INPUT_DEFAULT}}"
IMAGE_DIR="${IMAGE_DIR:-${IMAGE_DIR_DEFAULT}}"
MODEL="${MODEL:-${PROJECT_ROOT}/../models/Qwen3.5-4B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/$(basename "${MODEL}")}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python3}"
# Evaluation-only runtime: do not load or modify the shared training gateway.
HERO_BIN="${SCRIPT_DIR}/hero_eval.sh"
export PYTHONDONTWRITEBYTECODE=1
# Align evaluation with run_rl/_shared/run_4b.sh's verified
# Claude Code 2.1.111 runtime. AGENT_CLI_DIR remains an explicit override.
AGENT_CLI_DIR="${AGENT_CLI_DIR:-${PROJECT_ROOT}/../cli/claude_code}"
EXP_TAG="${EXP_TAG:-$(basename "${BASH_SOURCE[0]}" .sh)}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)_${BASHPID}}"
RUN_DATE="${RUN_DATE:-$(date +%m%d)}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/../results}"
EVAL_MODEL_NAME="${EVAL_MODEL_NAME:-$(basename "${MODEL%/}")}"
EVAL_MODEL_NAME="${EVAL_MODEL_NAME//[^a-zA-Z0-9_.-]/_}"
OUT="${OUT:-${RESULTS_ROOT}/eval/${DATASET}/${EVAL_MODEL_NAME}/${STAMP}_${EXP_TAG}}"
[[ "${OUT}" == /* ]] || OUT="${PWD}/${OUT}"
GOLDEN="${GOLDEN:-0}"
# Eight-card L20X host profile: one complete model replica per GPU. Keep six
# agent tasks per replica in flight.  The previous 48-way setting completed the
# comparable Qwen3.5-35B-A3B Verified entropy run with 4 infra failures; raising
# this to 56 coincided with 44 task timeouts, so retain the proven headroom.
NUM_SHARDS="${NUM_SHARDS:-8}"
SGLANG_TP_SIZE="${SGLANG_TP_SIZE:-1}"
SGLANG_GPU_IDS="${SGLANG_GPU_IDS:-}"
CONCURRENCY="${CONCURRENCY:-48}"
GRADE_CONCURRENCY="${GRADE_CONCURRENCY:-16}"
LIMIT="${LIMIT:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
TASK_EXECUTION_TIMEOUT="${TASK_TIMEOUT:-3000}"
EVAL_TEST_TIMEOUT="${EVAL_TEST_TIMEOUT:-1500}"
# Keep the evaluator, Claude Code task environment, and SGLang server on the
# same 128K context ceiling.
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-131072}"
MAX_RESPONSE_TOKENS="${MAX_RESPONSE_TOKENS:-16000}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-10}"
HERO_GATEWAY_WORKERS="${HERO_GATEWAY_WORKERS:-48}"
# VFS image extraction is already capped by the 12-way Docker load gate.  Do
# not move another 36 sessions into INITIALIZING (and start their timeout
# clocks) while they are only waiting for that gate.  Keep overlay2's existing
# 48/48/48 topology. Both storage drivers reserve a bounded grading window.
if [[ "${HERO_LOCAL_DOCKER_STORAGE_DRIVER:-}" == "vfs" ]]; then
  HERO_GATEWAY_INIT_WORKERS="${HERO_GATEWAY_INIT_WORKERS:-12}"
  HERO_GATEWAY_POSTRUN_RESERVE_SECONDS="${HERO_GATEWAY_POSTRUN_RESERVE_SECONDS:-$((EVAL_TEST_TIMEOUT + 600))}"
  HERO_GATEWAY_PUBLISH_BEFORE_TEARDOWN="${HERO_GATEWAY_PUBLISH_BEFORE_TEARDOWN:-1}"
  # TASK_TIMEOUT remains the INIT+RUN budget for callers.  Extend only the
  # gateway/rollout wall clock so build, grading and callback have their own
  # bounded window instead of inheriting an already exhausted deadline.
  TASK_TIMEOUT=$((TASK_EXECUTION_TIMEOUT + HERO_GATEWAY_POSTRUN_RESERVE_SECONDS))
else
  HERO_GATEWAY_INIT_WORKERS="${HERO_GATEWAY_INIT_WORKERS:-${HERO_GATEWAY_WORKERS}}"
  HERO_GATEWAY_POSTRUN_RESERVE_SECONDS="${HERO_GATEWAY_POSTRUN_RESERVE_SECONDS:-$((EVAL_TEST_TIMEOUT + 600))}"
  HERO_GATEWAY_PUBLISH_BEFORE_TEARDOWN=0
  TASK_TIMEOUT=$((TASK_EXECUTION_TIMEOUT + HERO_GATEWAY_POSTRUN_RESERVE_SECONDS))
fi
export HERO_GATEWAY_POSTRUN_RESERVE_SECONDS
export HERO_GATEWAY_PUBLISH_BEFORE_TEARDOWN
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
MAX_TURNS="${MAX_TURNS:-${CLAUDE_CODE_MAX_TURNS:-200}}"
CLAUDE_CODE_MAX_TURNS="${MAX_TURNS}"
HERO_MAX_COMPLETIONS="${MAX_TURNS}"
# Evaluation protocol: no compaction, including callers with stale overrides.
CLAUDE_CODE_AUTO_COMPACT=0
CLAUDE_CODE_AUTO_COMPACT_WINDOW="${CLAUDE_CODE_AUTO_COMPACT_WINDOW:-80000}"
# Match run_rl/_shared/run_4b.sh: expose Claude Code's default built-in tools
# and deny only its web-access tools plus subagents.
# Previous setting (kept for provenance; intentionally disabled):
# CLAUDE_CODE_TOOLS="${CLAUDE_CODE_TOOLS-Bash,Read,Edit,Write,Glob,Grep,NotebookRead,NotebookEdit}"
unset CLAUDE_CODE_TOOLS
# Use the default denylist only when the variable is unset.  An explicitly
# empty value is meaningful: expose Claude Code's complete default tool set.
CLAUDE_CODE_DISALLOWED_TOOLS="${CLAUDE_CODE_DISALLOWED_TOOLS-WebFetch,WebSearch,Agent}"
HERO_GIT_PROGRESS_ENABLED="${HERO_GIT_PROGRESS_ENABLED:-1}"
HERO_GIT_PROGRESS_CHECKPOINT_INTERVAL="${HERO_GIT_PROGRESS_CHECKPOINT_INTERVAL:-10}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.8}"
# CUDA graph works on the current L20X stack once FlashInfer's stable CUDA
# RMSNorm backend is selected. Keep disabling it as an explicit fallback only.
SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH:-0}"
SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH="${SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH:-0}"
SGLANG_ENABLE_HTTP2="${SGLANG_ENABLE_HTTP2:-1}"
SGLANG_ROUTER_DISABLE_HEALTH_CHECK="${SGLANG_ROUTER_DISABLE_HEALTH_CHECK:-1}"
# cache_aware collapses SWE agent traffic onto the replica holding the common
# Claude Code prefix.  power_of_two favors the less-loaded of two replicas and
# keeps all eight single-GPU model servers doing useful work.
SGLANG_ROUTER_POLICY="${SGLANG_ROUTER_POLICY:-power_of_two}"
SGLANG_DIRECT_WORKER="${SGLANG_DIRECT_WORKER:-0}"
SGLANG_TOOL_CALL_PARSER="${SGLANG_TOOL_CALL_PARSER:-qwen3_coder}"
SGLANG_REASONING_PARSER="${SGLANG_REASONING_PARSER:-qwen3}"
ROUTER_HOST="${ROUTER_HOST:-127.0.0.1}"
ROUTER_PORT="${ROUTER_PORT:-30080}"
WORKER_PORT_START="${WORKER_PORT_START:-30088}"
WORKER_NCCL_PORT_START="${WORKER_NCCL_PORT_START:-29100}"
HERO_ROLLOUT_PORT="${HERO_ROLLOUT_PORT:-8080}"
HERO_NODE_PORT="${HERO_NODE_PORT:-8100}"

# The official Slime Docker grader imports Torch through the test harness for
# some SWE-Gym rows.  Like training, make the CUDA-13 NVSHMEM/NCCL libraries
# visible to this host-side Python process; otherwise all golden SWE-Gym rows
# fail before a fresh grading container is even created.
_nvidia_lib() {
  local _name="$1"
  local _d
  _d="$("${PYTHON_BIN}" -c "import nvidia.${_name},os;print(os.path.join(list(nvidia.${_name}.__path__)[0],'lib'))" 2>/dev/null)"
  if [[ -z "${_d}" || ! -d "${_d}" ]]; then
    _d="$(/usr/bin/python3 -c "import nvidia.${_name},os;print(os.path.join(list(nvidia.${_name}.__path__)[0],'lib'))" 2>/dev/null)"
  fi
  printf '%s' "${_d}"
}
_NVIDIA_LIB_PREPEND=""
for _nvidia_component in nvshmem nccl cudnn cu13; do
  _nvidia_dir="$(_nvidia_lib "${_nvidia_component}")"
  [[ -n "${_nvidia_dir}" && -d "${_nvidia_dir}" ]] && \
    _NVIDIA_LIB_PREPEND="${_nvidia_dir}:${_NVIDIA_LIB_PREPEND}"
done
[[ -n "${_NVIDIA_LIB_PREPEND}" ]] && \
  export LD_LIBRARY_PATH="${_NVIDIA_LIB_PREPEND}${LD_LIBRARY_PATH:-}"
# Transformer Engine's CUDA-wheel discovery assumes the ``nvidia`` namespace
# package has a concrete __file__, which is not true in this environment.  An
# explicit include path avoids that broken auto-discovery when SGLang imports
# its optional Megatron debug plugin.
NVTE_CUDA_INCLUDE_DIR="${NVTE_CUDA_INCLUDE_DIR:-${PROJECT_ROOT}/.venv/lib/python3.12/site-packages/nvidia/cu13/include}"
[[ -d "${NVTE_CUDA_INCLUDE_DIR}" ]] && export NVTE_CUDA_INCLUDE_DIR

[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python environment: ${PYTHON_BIN}" >&2; exit 1; }
[[ -r "${INPUT}" ]] || { echo "Missing input: ${INPUT}" >&2; exit 1; }
[[ "${GOLDEN}" == "0" || "${GOLDEN}" == "1" ]] || { echo "GOLDEN must be 0 or 1" >&2; exit 1; }
[[ "${HERO_GIT_PROGRESS_ENABLED}" == "0" || "${HERO_GIT_PROGRESS_ENABLED}" == "1" ]] || { echo "HERO_GIT_PROGRESS_ENABLED must be 0 or 1" >&2; exit 1; }
[[ "${CLAUDE_CODE_AUTO_COMPACT}" == "0" || "${CLAUDE_CODE_AUTO_COMPACT}" == "1" ]] || { echo "CLAUDE_CODE_AUTO_COMPACT must be 0 or 1" >&2; exit 1; }
case "${SGLANG_ROUTER_POLICY}" in
  random|round_robin|cache_aware|power_of_two|manual) ;;
  *) echo "Invalid SGLANG_ROUTER_POLICY=${SGLANG_ROUTER_POLICY}" >&2; exit 1 ;;
esac
if [[ "${GOLDEN}" != "1" ]]; then
  [[ -x "${HERO_BIN}" ]] || { echo "Missing HERO CLI: ${HERO_BIN}" >&2; exit 1; }
  [[ -d "${MODEL}" ]] || { echo "Missing HF checkpoint: ${MODEL}" >&2; exit 1; }
  [[ -x "${AGENT_CLI_DIR}/bin/claude" ]] || { echo "Missing Claude Code mount: ${AGENT_CLI_DIR}" >&2; exit 1; }
fi
for integer_name in NUM_SHARDS SGLANG_TP_SIZE CONCURRENCY GRADE_CONCURRENCY TASK_EXECUTION_TIMEOUT TASK_TIMEOUT MAX_CONTEXT_TOKENS MAX_RESPONSE_TOKENS \
  EVAL_TEST_TIMEOUT HERO_GATEWAY_POSTRUN_RESERVE_SECONDS MAX_RUNNING_REQUESTS MAX_TURNS CLAUDE_CODE_MAX_TURNS HERO_MAX_COMPLETIONS CLAUDE_CODE_AUTO_COMPACT_WINDOW \
  HERO_GIT_PROGRESS_CHECKPOINT_INTERVAL HERO_GATEWAY_WORKERS HERO_GATEWAY_INIT_WORKERS ROUTER_PORT WORKER_PORT_START \
  WORKER_NCCL_PORT_START HERO_ROLLOUT_PORT HERO_NODE_PORT; do
  integer_value="${!integer_name}"
  [[ "${integer_value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "${integer_name} must be a positive integer; got ${integer_value}" >&2
    exit 1
  }
done
[[ "${HERO_GATEWAY_PUBLISH_BEFORE_TEARDOWN}" == "0" || "${HERO_GATEWAY_PUBLISH_BEFORE_TEARDOWN}" == "1" ]] || {
  echo "HERO_GATEWAY_PUBLISH_BEFORE_TEARDOWN must be 0 or 1; got ${HERO_GATEWAY_PUBLISH_BEFORE_TEARDOWN}" >&2
  exit 1
}
if [[ "${GOLDEN}" != "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  AVAILABLE_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)"
  REQUIRED_GPUS=$((NUM_SHARDS * SGLANG_TP_SIZE))
  (( REQUIRED_GPUS <= AVAILABLE_GPUS )) || {
    echo "NUM_SHARDS=${NUM_SHARDS} * SGLANG_TP_SIZE=${SGLANG_TP_SIZE} needs ${REQUIRED_GPUS} GPUs, but only ${AVAILABLE_GPUS} are visible" >&2
    exit 1
  }
  if [[ -n "${SGLANG_GPU_IDS}" ]]; then
    IFS=',' read -r -a _SGLANG_GPU_ID_LIST <<<"${SGLANG_GPU_IDS}"
    (( ${#_SGLANG_GPU_ID_LIST[@]} == REQUIRED_GPUS )) || {
      echo "SGLANG_GPU_IDS must contain exactly ${REQUIRED_GPUS} GPU ids; got ${SGLANG_GPU_IDS}" >&2
      exit 1
    }
    for _gpu_id in "${_SGLANG_GPU_ID_LIST[@]}"; do
      [[ "${_gpu_id}" =~ ^[0-9]+$ ]] && (( _gpu_id < AVAILABLE_GPUS )) || {
        echo "Invalid GPU id in SGLANG_GPU_IDS: ${_gpu_id}" >&2
        exit 1
      }
    done
  fi
fi
if [[ "${CLAUDE_CODE_AUTO_COMPACT}" == "1" ]]; then
  (( CLAUDE_CODE_AUTO_COMPACT_WINDOW < MAX_CONTEXT_TOKENS )) || {
    echo "auto compact window must be below context limit when auto compact is enabled" >&2
    exit 1
  }
fi
(( MAX_RESPONSE_TOKENS < MAX_CONTEXT_TOKENS )) || {
  echo "max response must be below context limit" >&2
  exit 1
}

# Record the eval-only runtime, and reject resuming pre-fix results.
"${PYTHON_BIN}" "${SCRIPT_DIR}/eval_runtime.py" "${OUT}"
mkdir -p "${OUT}/settings" "${OUT}/sglang_logs"
export DOCKER_HOST="${HERO_LOCAL_DOCKER_HOST:-${DOCKER_HOST:-tcp://127.0.0.1:2377}}"
export HERO_LOCAL_DOCKER_HOST="${DOCKER_HOST}"
export HERO_LOCAL_DOCKER_DATA_ROOT="${HERO_LOCAL_DOCKER_DATA_ROOT:-${XDG_CACHE_HOME:-${HOME}/.cache}/hero/docker-data}"
export HERO_LOCAL_DOCKER_LOAD_CONCURRENCY="${HERO_LOCAL_DOCKER_LOAD_CONCURRENCY:-12}"
# On this 192-CPU/1.5-TiB host this gate can feed the 56-task eight-GPU
# profile while still limiting simultaneous Docker rm/create/start calls.
export HERO_LOCAL_DOCKER_RUN_CONCURRENCY="${HERO_LOCAL_DOCKER_RUN_CONCURRENCY:-24}"
# HERO has no separate boot gate, but eval re-grading uses Slime's official
# SWE-bench runner, which does.  Keep that runner's container-bootstrap gate
# aligned with the local Docker start gate.
export SWE_BOOT_CONCURRENCY="${SWE_BOOT_CONCURRENCY:-${HERO_LOCAL_DOCKER_RUN_CONCURRENCY}}"
# Bound the combined 56-rollout/16-grader pipeline with headroom for runtime
# handoff. Docker create/start remains separately protected at 24.
export HERO_LOCAL_DOCKER_ACTIVE_CONCURRENCY="${HERO_LOCAL_DOCKER_ACTIVE_CONCURRENCY:-80}"
# Official SWE-bench golden evaluation uses a distinct image per instance.
# Keep them for the complete dataset run: deleting/reloading them at C=32
# creates Docker API contention and can race the official grader.  RMI=0
# means the runtime never adds idle images to its eviction queue; the 500 GB
# watermark below remains a hard admission guard, so disk cannot be exhausted.
export HERO_LOCAL_DOCKER_IMAGE_CACHE_SIZE="${HERO_LOCAL_DOCKER_IMAGE_CACHE_SIZE:-512}"
export HERO_LOCAL_DOCKER_MIN_FREE_GB="${HERO_LOCAL_DOCKER_MIN_FREE_GB:-500}"
export HERO_LOCAL_DOCKER_RMI="${HERO_LOCAL_DOCKER_RMI:-0}"
export HERO_LOCAL_DOCKER_TRANSIENT_RETRIES="${HERO_LOCAL_DOCKER_TRANSIENT_RETRIES:-4}"
export HERO_RUN_ID="${HERO_RUN_ID:-eval-${DATASET}-${STAMP}}"
export FLASHINFER_USE_CUDA_NORM="${FLASHINFER_USE_CUDA_NORM:-1}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# The shared HERO evaluator uses official SWE-bench for both evaluation datasets.
SLIME_ROOT="${SLIME_ROOT:-${PROJECT_ROOT}/framework/slime}"
export HERO_SWEBENCH_PATH="${HERO_SWEBENCH_PATH:-${PROJECT_ROOT}/../harnesses/SWE-bench}"
export SWE_DOCKER_API_TIMEOUT_SEC="${SWE_DOCKER_API_TIMEOUT_SEC:-300}"
# This docker daemon only has usable DNS/outbound through host networking; the
# prepared SWE-bench harness reads this for official grading containers.
export SWE_BENCH_DOCKER_NETWORK="${SWE_BENCH_DOCKER_NETWORK:-host}"
export PYTHONPATH="${SLIME_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

hero_check_docker_storage
hero_check_available_memory
hero_cleanup_stale_containers

PIDS=()
cleanup() {
  local status=$?
  if ((${#PIDS[@]})); then
    kill "${PIDS[@]}" 2>/dev/null || true
    # SGLang multiprocessing children do not always leave promptly on TERM.
    # Bound teardown so serial baseline runs can advance to the next model.
    for _cleanup_try in {1..10}; do
      _cleanup_alive=0
      for _cleanup_pid in "${PIDS[@]}"; do
        kill -0 "${_cleanup_pid}" 2>/dev/null && _cleanup_alive=1
      done
      [[ "${_cleanup_alive}" == 0 ]] && break
      sleep 1
    done
    for _cleanup_pid in "${PIDS[@]}"; do
      kill -0 "${_cleanup_pid}" 2>/dev/null && kill -9 "${_cleanup_pid}" 2>/dev/null || true
    done
    wait "${PIDS[@]}" 2>/dev/null || true
  fi
  # EXIT/INT/TERM sweep: PIDS only covers processes we registered, so a crash
  # between launching an SGLang worker and recording its PID could orphan it.
  # Clean the same stale-proc set here so Ctrl-C mid-startup does not leave
  # GPU/ports occupied for the next re-run.
  cleanup_stale_eval_procs >/dev/null 2>&1 || true
  hero_cleanup_run_containers
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# --- Stale-process cleanup for re-runs -------------------------------------
# When a previous eval run is interrupted (Ctrl-C, crash, OOM) the SGLang
# workers / router / HERO rollout+gateway processes it spawned can be left
# behind holding the eval ports (ROUTER_PORT, WORKER_PORT_START+,
# WORKER_NCCL_PORT_START+, the gateway on 8080) and GPU memory.  The default
# ``port_must_be_free`` below refuses to start in that case, forcing a manual
# kill before the same script can be re-run.  That friction breaks the common
# workflow of "one eval finishes, start the next" or "I stopped it midway,
# just re-run".
#
# ``cleanup_stale_eval_procs`` removes only processes this eval owns:
#   - whatever is bound to the eval ports above, and
#   - SGLang workers serving *this* MODEL_PATH (matched narrowly so a
#     concurrent training run serving a different model is never touched), and
#   - HERO serve_rollout / serve_gateway pointing at *this* OUT's topology.
# Docker containers are already label-scoped by hero_cleanup_run_containers
# (ai.hero.run_id), so this only handles host-side SGLang/HERO processes.
#
# FORCE_CLEAN=1 (default) auto-cleans and continues; FORCE_CLEAN=0 reverts to
# the old fail-on-busy-port behaviour so a paranoid operator can opt out.
FORCE_CLEAN="${FORCE_CLEAN:-1}"

# Print PIDs listening on $1 (ipv4 or ipv6, any interface).  Empty if free.
_port_listener_pids() {
  local port="$1"
  # ss -ltnp is preferred; for each listening socket on $port emit the pid(s)
  # in the form "pid=1234" -> we keep just the numeric pid.
  ss -ltnp 2>/dev/null \
    | awk -v p=":${port}" 'index($0,p)""~/$/ || 0 {next} index($0,p){print}' >/dev/null 2>&1
  # The awk above is fragile across ss formats; use a simpler grab.
  ss -ltnp 2>/dev/null | grep -E "[:.]${port}[[:space:]]" \
    | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u
}

kill_port_procs() {
  # Kill processes bound to $1.  Best-effort; callers tolerate a misses.
  local port="$1" pid
  for pid in $(_port_listener_pids "${port}"); do
    [[ -n "${pid}" ]] || continue
    kill "${pid}" 2>/dev/null || true
  done
}

cleanup_stale_eval_procs() {
  local ports=("${ROUTER_PORT}" "${HERO_ROLLOUT_PORT}" "${HERO_NODE_PORT}")
  local i
  for ((i=0; i<NUM_SHARDS; i++)); do
    ports+=("$((WORKER_PORT_START + i))" "$((WORKER_NCCL_PORT_START + i))")
  done

  # Narrow model match: SGLang workers for THIS checkpoint only.  Matching on
  # the bare model basename avoids killing workers serving other models in a
  # concurrent run (e.g. a training server on a different checkpoint).
  local model_marker
  model_marker="$(basename "${MODEL}")"

  local stale_pids=()
  # 1. anything bound to an eval port
  for p in "${ports[@]}"; do
    local pid
    for pid in $(_port_listener_pids "${p}"); do
      [[ -n "${pid}" ]] && stale_pids+=("${pid}")
    done
  done
  # 2. SGLang workers for this model + the router child, and HERO procs for
  #    this OUT's topology (matched on the work-dir path, which is unique per
  #    run and present in every HERO invocation's argv).
  local out_marker="${OUT}"
  while IFS= read -r line; do
    [[ -n "${line}" ]] || continue
    stale_pids+=("${line}")
  # Match only the model-specific launch command.  Scheduler/detokenizer child
  # titles do not contain a model or run identifier, so matching those generic
  # titles would kill unrelated Qwen/baseline evaluations on this host.
  done < <(pgrep -f \
    "sglang.launch_server --model-path.*${model_marker}" 2>/dev/null)
  while IFS= read -r line; do
    [[ -n "${line}" ]] || continue
    stale_pids+=("${line}")
  done < <(pgrep -f "serve_rollout -c .*${out_marker}/settings/topology\.yaml|serve_gateway -c .*${out_marker}/settings/topology\.yaml" 2>/dev/null)

  # Dedup.
  local deduped=()
  if ((${#stale_pids[@]})); then
    printf '%s\n' "${stale_pids[@]}" | sort -u | while IFS= read -r pid; do
      [[ -n "${pid}" ]] && deduped+=("${pid}")
    done
  fi
  # Re-collect deduped into a plain array (the pipe above runs in a subshell).
  deduped=($(printf '%s\n' "${stale_pids[@]}" 2>/dev/null | sort -u))

  if ((!${#deduped[@]})); then
    return 0
  fi

  echo "[eval] found ${#deduped[@]} stale SGLang/HERO process(es) from a prior run; cleaning up" >&2
  if [[ "${FORCE_CLEAN}" != "1" ]]; then
    echo "[eval] FORCE_CLEAN=0: refusing to auto-clean. Stale PIDs: ${deduped[*]}" >&2
    echo "[eval] kill them manually or set FORCE_CLEAN=1, then re-run." >&2
    exit 1
  fi
  local pid
  for pid in "${deduped[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  # SIGKILL anything still alive after a grace period.
  sleep 2
  for pid in "${deduped[@]}"; do
    kill -0 "${pid}" 2>/dev/null && kill -9 "${pid}" 2>/dev/null || true
  done
  # Wait for the eval ports to actually free up (and GPU to drain) before
  # returning, so the subsequent port_must_be_free / worker launch sees a
  # clean slate.  Cap at ~30s so a wedged process does not hang the re-run.
  local waited=0
  while ((waited < 30)); do
    local busy=0 p
    for p in "${ports[@]}"; do
      ss -ltn 2>/dev/null | grep -qE "[:.]${p}[[:space:]]" && busy=1
    done
    [[ "${busy}" -eq 0 ]] && break
    sleep 1
    waited=$((waited + 1))
  done
}

# Re-run safety net: kill SGLang/router/HERO host processes a prior
# interrupted run left bound to the eval ports or serving this model/OUT,
# so the same script can be re-launched without a manual kill step.  Defined
# above and invoked here (after the function exists) so a Ctrl-C/Crash between
# runs auto-cleans before SGLang/HERO launch below.
cleanup_stale_eval_procs

wait_for_url() {
  local url="$1" attempts="${2:-300}"
  for ((i=1; i<=attempts; i++)); do
    curl -fsS --max-time 5 "${url}" >/dev/null 2>&1 && return 0
    sleep 2
  done
  echo "Timed out waiting for ${url}" >&2
  return 1
}

wait_for_worker_url() {
  local url="$1" pid="$2" log_file="$3" attempts="${4:-300}"
  for ((i=1; i<=attempts; i++)); do
    curl -fsS --max-time 5 "${url}" >/dev/null 2>&1 && return 0
    if ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}" 2>/dev/null || true
      echo "SGLang worker ${pid} exited before becoming healthy: ${url}" >&2
      tail -80 "${log_file}" >&2 || true
      return 1
    fi
    sleep 2
  done
  echo "Timed out waiting for ${url}" >&2
  tail -80 "${log_file}" >&2 || true
  return 1
}

port_must_be_free() {
  local port="$1"
  if ss -ltn 2>/dev/null | grep -qE "[:.]${port}[[:space:]]"; then
    echo "Port ${port} is already in use; refusing to interfere with an existing service." >&2
    exit 1
  fi
}

SGLANG_ROUTER_BASE_URL="${SGLANG_ROUTER_BASE_URL:-http://${ROUTER_HOST}:${ROUTER_PORT}}"
if [[ "${GOLDEN}" != "1" && "${SKIP_MODEL_SERVER:-0}" != "1" ]]; then
  SGLANG_CUDA_GRAPH_ARGS=()
  SGLANG_HTTP_ARGS=()
  SGLANG_ROUTER_ARGS=()
  SGLANG_WORKER_URLS=()
  SGLANG_REASONING_ARGS=(--reasoning-parser "${SGLANG_REASONING_PARSER}")
  # Baseline-only opt-out for non-reasoning checkpoints. The default Qwen3.5
  # command remains byte-for-byte equivalent and still passes qwen3 above.
  [[ "${SGLANG_REASONING_PARSER}" == "none" ]] && SGLANG_REASONING_ARGS=()
  [[ "${SGLANG_DISABLE_CUDA_GRAPH}" == "1" ]] && SGLANG_CUDA_GRAPH_ARGS+=(--disable-cuda-graph)
  [[ "${SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH}" == "1" ]] && SGLANG_CUDA_GRAPH_ARGS+=(--disable-piecewise-cuda-graph)
  [[ "${SGLANG_ENABLE_HTTP2}" == "1" ]] && SGLANG_HTTP_ARGS+=(--enable-http2)
  [[ "${SGLANG_ROUTER_DISABLE_HEALTH_CHECK}" == "1" ]] && SGLANG_ROUTER_ARGS+=(--disable-health-check)

  port_must_be_free "${ROUTER_PORT}"
  for ((index=0; index<NUM_SHARDS; index++)); do
    port_must_be_free "$((WORKER_PORT_START + index))"
    port_must_be_free "$((WORKER_NCCL_PORT_START + index))"
  done

  for ((index=0; index<NUM_SHARDS; index++)); do
    worker_port="$((WORKER_PORT_START + index))"
    worker_nccl_port="$((WORKER_NCCL_PORT_START + index))"
    worker_log="${OUT}/sglang_logs/worker_${worker_port}.log"
    worker_devices=""
    for ((tp_index=0; tp_index<SGLANG_TP_SIZE; tp_index++)); do
      gpu_index=$((index * SGLANG_TP_SIZE + tp_index))
      if [[ -n "${SGLANG_GPU_IDS}" ]]; then
        gpu_index="${_SGLANG_GPU_ID_LIST[${gpu_index}]}"
      fi
      worker_devices="${worker_devices:+${worker_devices},}${gpu_index}"
    done
    CUDA_VISIBLE_DEVICES="${worker_devices}" "${PYTHON_BIN}" -m sglang.launch_server \
      --model-path "${MODEL}" --served-model-name "${SERVED_MODEL_NAME}" \
      --host 127.0.0.1 --port "${worker_port}" --tp-size "${SGLANG_TP_SIZE}" \
      --nccl-port "${worker_nccl_port}" \
      --context-length "${MAX_CONTEXT_TOKENS}" \
      --mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}" \
      --max-running-requests "${MAX_RUNNING_REQUESTS}" \
      --tool-call-parser "${SGLANG_TOOL_CALL_PARSER}" \
      "${SGLANG_REASONING_ARGS[@]}" --trust-remote-code \
      "${SGLANG_HTTP_ARGS[@]}" \
      "${SGLANG_CUDA_GRAPH_ARGS[@]}" \
      >"${worker_log}" 2>&1 &
    worker_pid="$!"
    PIDS+=("${worker_pid}")
    wait_for_worker_url "http://127.0.0.1:${worker_port}/health_generate" \
      "${worker_pid}" "${worker_log}" "${SGLANG_START_ATTEMPTS:-450}"
    SGLANG_WORKER_URLS+=("http://127.0.0.1:${worker_port}")
  done

  # The router does not expose the worker chat-tokenization endpoint.
  # All replicas in this launch use the same model and chat template.
  export INFERENCE_TOKENIZATION_BASE_URL="${SGLANG_WORKER_URLS[0]}"

  if [[ "${SGLANG_DIRECT_WORKER}" == "1" ]]; then
    [[ "${NUM_SHARDS}" == "1" ]] || { echo "SGLANG_DIRECT_WORKER requires NUM_SHARDS=1" >&2; exit 1; }
    # A single healthy SGLang worker is itself a valid OpenAI-compatible
    # endpoint, so a router adds no routing value for the eval smoke path.
    export SGLANG_ROUTER_BASE_URL="${SGLANG_WORKER_URLS[0]}"
  else
    # Router's documented static-worker mode avoids the dynamic /workers
    # registration endpoint, whose transport differs between router releases.
    "${PYTHON_BIN}" -m sglang_router.launch_router \
      --host "${ROUTER_HOST}" --port "${ROUTER_PORT}" --log-level warn \
      --policy "${SGLANG_ROUTER_POLICY}" \
      --model-path "${MODEL}" --tokenizer-path "${MODEL}" \
      --worker-urls "${SGLANG_WORKER_URLS[@]}" "${SGLANG_ROUTER_ARGS[@]}" \
      >"${OUT}/sglang_logs/router.log" 2>&1 &
    PIDS+=("$!")
    wait_for_url "${SGLANG_ROUTER_BASE_URL}/health_generate" 60
  fi
fi

if [[ "${GOLDEN}" != "1" && "${SKIP_HERO:-0}" != "1" ]]; then
  export SGLANG_ROUTER_BASE_URL
  "${PYTHON_BIN}" - "${PROJECT_ROOT}/run_rl/_shared/config/topology.yaml" \
  "${OUT}/settings/topology.yaml" "${SGLANG_ROUTER_BASE_URL}" \
  "${MAX_RESPONSE_TOKENS}" "${TEMPERATURE}" "${TOP_P}" "${HERO_MAX_COMPLETIONS}" \
  "${SERVED_MODEL_NAME}" "${HERO_GATEWAY_WORKERS}" "${HERO_GATEWAY_INIT_WORKERS}" "${HERO_ROLLOUT_PORT}" \
  "${HERO_NODE_PORT}" <<'PY'
from pathlib import Path
import sys

source, destination = map(Path, sys.argv[1:3])
base_url, max_response_tokens, temperature, top_p, max_completions, served_model, gateway_workers, gateway_init_workers, rollout_port, node_port = sys.argv[3:13]
text = source.read_text()
replacements = {
    "${SGLANG_ROUTER_BASE_URL}": base_url,
    "${MAX_RESPONSE_TOKENS}": max_response_tokens,
    "${TEMPERATURE}": temperature,
    "${TOP_P}": top_p,
    "${HERO_MAX_COMPLETIONS}": max_completions,
}
for placeholder, value in replacements.items():
    if placeholder not in text:
        raise SystemExit(f"missing placeholder {placeholder!r} in {source}")
    text = text.replace(placeholder, value)
text = text.replace("max_init_workers: 16", f"max_init_workers: {gateway_init_workers}")
for worker_key in ("max_run_workers", "max_postrun_workers"):
    text = text.replace(f"{worker_key}: 16", f"{worker_key}: {gateway_workers}")
text = text.replace("model_served: Qwen/Qwen3.5-4B", f"model_served: {served_model}")
text = text.replace("port: 8080", f"port: {rollout_port}")
text = text.replace("http://127.0.0.1:8080", f"http://127.0.0.1:{rollout_port}")
text = text.replace("port: 8100", f"port: {node_port}")
text = text.replace("http://127.0.0.1:8100", f"http://127.0.0.1:{node_port}")
destination.write_text(text)
PY
fi
cp "${BASH_SOURCE[0]}" "${PROJECT_ROOT}/run_rl/_shared/runtime/docker_storage_guard.sh" \
  "${PROJECT_ROOT}/run_rl/_shared/runtime/local_docker_runtime.py" "${OUT}/settings/"
env | sort | awk -F= '$1 ~ /(KEY|TOKEN|SECRET|PASSWORD)$/ {print $1"=<redacted>"; next} {print}' \
  >"${OUT}/settings/environment.txt"

if [[ "${GOLDEN}" != "1" && "${SKIP_HERO:-0}" != "1" ]]; then
  cd "${OUT}"
  "${HERO_BIN}" serve_rollout -c "${OUT}/settings/topology.yaml" >"${OUT}/hero_rollout.log" 2>&1 &
  PIDS+=("$!")
  "${HERO_BIN}" serve_gateway -c "${OUT}/settings/topology.yaml" --node-id localhost-node-01 \
    >"${OUT}/hero_gateway.log" 2>&1 &
  PIDS+=("$!")
  wait_for_url "http://127.0.0.1:${HERO_ROLLOUT_PORT}/health" 90
fi

DRIVER_ARGS=(
  --input "${INPUT}"
  --image-dir "${IMAGE_DIR}"
  --work-dir "${OUT}"
  --model-name "${SERVED_MODEL_NAME}"
  --limit "${LIMIT}"
  --num-samples "${NUM_SAMPLES}"
  --concurrency "${CONCURRENCY}"
  --grade-concurrency "${GRADE_CONCURRENCY}"
  --timeout "${TASK_TIMEOUT}"
  --test-timeout "${EVAL_TEST_TIMEOUT}"
  --max-context-tokens "${MAX_CONTEXT_TOKENS}"
  --max-response-tokens "${MAX_RESPONSE_TOKENS}"
  --auto-compact-enabled "${CLAUDE_CODE_AUTO_COMPACT}"
  --auto-compact-window "${CLAUDE_CODE_AUTO_COMPACT_WINDOW}"
  --max-turns "${CLAUDE_CODE_MAX_TURNS}"
  --agent-cli-dir "${AGENT_CLI_DIR}"
  --agent-disallowed-tools "${CLAUDE_CODE_DISALLOWED_TOOLS}"
  --rollout-url "http://127.0.0.1:${HERO_ROLLOUT_PORT}"
  --git-progress-enabled "${HERO_GIT_PROGRESS_ENABLED}"
  --git-progress-checkpoint-interval "${HERO_GIT_PROGRESS_CHECKPOINT_INTERVAL}"
)
# 2026-08-13: old explicit allowlist forwarding intentionally disabled:
# [[ -n "${CLAUDE_CODE_TOOLS}" ]] && DRIVER_ARGS+=(--agent-tools "${CLAUDE_CODE_TOOLS}")
[[ -n "${INSTANCE_ID:-}" ]] && DRIVER_ARGS+=(--instance-id "${INSTANCE_ID}")
[[ -n "${RETRY_FAILED_FROM:-}" ]] && DRIVER_ARGS+=(--retry-failed-from "${RETRY_FAILED_FROM}")
[[ -n "${GOLD_PATCH_DIR:-}" ]] && DRIVER_ARGS+=(--gold-patch-dir "${GOLD_PATCH_DIR}")
[[ "${RESUME:-1}" == "1" ]] || DRIVER_ARGS+=(--no-resume)
[[ "${GOLDEN}" == "1" ]] && DRIVER_ARGS+=(--golden)
HEALTH_GATE="${HEALTH_GATE:-warn}"
DRIVER_ARGS+=(--health-gate "${HEALTH_GATE}")

echo "[eval] protocol=count-worker-postrun-single-workspace-20260922"
echo "[eval] dataset=${DATASET} golden=${GOLDEN} model=${MODEL} served_model=${SERVED_MODEL_NAME} out=${OUT}"
echo "[eval] gpu_replicas=${NUM_SHARDS} tp_size=${SGLANG_TP_SIZE} router_policy=${SGLANG_ROUTER_POLICY} gateway_workers=${HERO_GATEWAY_WORKERS}"
echo "[eval] Claude Code cli=${AGENT_CLI_DIR} tools=<default> disallowed_tools=${CLAUDE_CODE_DISALLOWED_TOOLS}"
echo "[eval] Git progress enabled=${HERO_GIT_PROGRESS_ENABLED} checkpoint_interval=${HERO_GIT_PROGRESS_CHECKPOINT_INTERVAL}"
echo "[eval] rollout_concurrency=${CONCURRENCY} grade_concurrency=${GRADE_CONCURRENCY} task_timeout=${TASK_TIMEOUT} execution_timeout=${TASK_EXECUTION_TIMEOUT} postrun_reserve=${HERO_GATEWAY_POSTRUN_RESERVE_SECONDS} gateway_init_workers=${HERO_GATEWAY_INIT_WORKERS} publish_before_teardown=${HERO_GATEWAY_PUBLISH_BEFORE_TEARDOWN} test_timeout=${EVAL_TEST_TIMEOUT} max_turns=${CLAUDE_CODE_MAX_TURNS} max_completions=${HERO_MAX_COMPLETIONS} auto_compact=${CLAUDE_CODE_AUTO_COMPACT} compact_window=${CLAUDE_CODE_AUTO_COMPACT_WINDOW} max_context_tokens=${MAX_CONTEXT_TOKENS} max_response_tokens=${MAX_RESPONSE_TOKENS} temperature=${TEMPERATURE} top_p=${TOP_P} max_running_requests=${MAX_RUNNING_REQUESTS}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/eval_swebench.py" "${DRIVER_ARGS[@]}" \
  2>&1 | tee "${OUT}/run.log"
driver_status="${PIPESTATUS[0]}"
# Do not fall off EOF while background SGLang/HERO jobs are still registered:
# bash may wait for those jobs indefinitely before running the EXIT trap.  An
# explicit exit runs cleanup immediately after the driver has written results.
exit "${driver_status}"
