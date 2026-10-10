#!/usr/bin/env bash
# Shared HERO/Slime training backend. Public entries live under run_rl/.
set -euo pipefail
TRAINING_MODEL_TYPE="${TRAINING_MODEL_TYPE:-dense}"
case "${TRAINING_MODEL_TYPE}" in
    dense|moe) ;;
    *) echo "TRAINING_MODEL_TYPE must be dense or moe" >&2; exit 2 ;;
esac

# Parse the complete launcher before starting long-lived services.
{
if [[ "${TRAINING_MODEL_TYPE}" == "moe" ]]; then
ray_job_parse_status() {
    local status_output="$1"
    local result_name="$2"
    local status_line
    local parsed_status=""
    local record_pattern="Status for job '[^']*': ([A-Z_]+)"
    local sentence_pattern="Job '[^']*' (pending|running|succeeded|failed|stopped)"

    # Ray has emitted both a record-style status ("Status for job ...:") and
    # a sentence-style status ("Job ... succeeded") across supported CLI
    # versions.  Keep the last recognized status when informational lines are
    # printed before it.
    while IFS= read -r status_line; do
        if [[ "${status_line}" =~ ${record_pattern} ]]; then
            parsed_status="${BASH_REMATCH[1]}"
        elif [[ "${status_line}" =~ ${sentence_pattern} ]]; then
            parsed_status="${BASH_REMATCH[1]^^}"
        fi
    done <<<"${status_output}"
    printf -v "${result_name}" '%s' "${parsed_status}"
}
else
ray_job_parse_status() {
    local status_output="$1"
    local result_name="$2"
    local status_line
    local parsed_status=""
    local status_pattern="Status for job '[^']*': ([A-Z_]+)"
    local terminal_pattern="Job '[^']*' (succeeded|failed|stopped)"

    # Keep the last status line, matching the old sed | tail behavior without
    # spawning helpers while the host is under process pressure.
    while IFS= read -r status_line; do
        if [[ "${status_line}" =~ ${status_pattern} ]]; then
            parsed_status="${BASH_REMATCH[1]}"
        elif [[ "${status_line}" =~ ${terminal_pattern} ]]; then
            parsed_status="${BASH_REMATCH[1]^^}"
        fi
    done <<<"${status_output}"
    printf -v "${result_name}" '%s' "${parsed_status}"
}
fi

ray_job_init_poll_clock() {
    # Bash has no sleep builtin. Start one tiny coprocess before Ray/model load
    # and keep it blocked in builtin read. Its silent stdout gives the parent a
    # stable fd for builtin read -t, so polling backoff never needs to fork.
    coproc RAY_JOB_POLL_CLOCK { IFS= read -r _; }
    RAY_JOB_POLL_FD="${RAY_JOB_POLL_CLOCK[0]}"
    RAY_JOB_POLL_CLOCK_PROCESS_PID="${RAY_JOB_POLL_CLOCK_PID}"
    PIDS+=("${RAY_JOB_POLL_CLOCK_PROCESS_PID}")
}

ray_job_poll_delay() {
    local ignored
    IFS= read -r -t "${RAY_JOB_POLL_INTERVAL_SEC}" \
        -u "${RAY_JOB_POLL_FD}" ignored || true
}

if [[ "${TRAINING_MODEL_TYPE}" == "moe" ]]; then
ray_job_poll() {
    local status_output=""
    local job_status=""
    local status_failures=0

    while true; do
        # Commands used as an if condition are exempt from errexit. This catches
        # both a regular Ray CLI error and a temporary fork(2) EAGAIN.
        if status_output="$("${RAY_CLI[@]}" job status \
            --address="${RAY_JOB_ADDRESS}" --log-style record \
            "${RAY_JOB_SUBMISSION_ID}" 2>&1)"; then
            ray_job_parse_status "${status_output}" job_status
        else
            status_failures=$((status_failures + 1))
            if (( status_failures <= RAY_JOB_STATUS_MAX_CONSECUTIVE_FAILURES \
                  || status_failures % RAY_JOB_STATUS_MAX_CONSECUTIVE_FAILURES == 0 )); then
                echo "[run] Ray job status query failed (${status_failures} consecutive attempt(s)): ${status_output}" >&2
            fi
            if (( status_failures == RAY_JOB_STATUS_MAX_CONSECUTIVE_FAILURES )); then
                echo "[run] Ray job status remains UNKNOWN; entering degraded monitoring without tearing down the possibly-running job." >&2
            fi
            ray_job_poll_delay
            continue
        fi

        case "${job_status}" in
            SUCCEEDED)
                echo "[run] Ray job ${RAY_JOB_SUBMISSION_ID} succeeded"
                return 0
                ;;
            FAILED|STOPPED)
                echo "${status_output}" >&2
                echo "[run] Ray job ${RAY_JOB_SUBMISSION_ID} ended with ${job_status}" >&2
                return 1
                ;;
            PENDING|RUNNING)
                status_failures=0
                ray_job_poll_delay
                ;;
            *)
                status_failures=$((status_failures + 1))
                if (( status_failures <= RAY_JOB_STATUS_MAX_CONSECUTIVE_FAILURES \
                      || status_failures % RAY_JOB_STATUS_MAX_CONSECUTIVE_FAILURES == 0 )); then
                    echo "[run] Could not parse Ray job status (${status_failures} consecutive attempt(s)): ${status_output}" >&2
                fi
                if (( status_failures == RAY_JOB_STATUS_MAX_CONSECUTIVE_FAILURES )); then
                    echo "[run] Ray job status remains UNKNOWN; entering degraded monitoring without tearing down the possibly-running job." >&2
                fi
                ray_job_poll_delay
                ;;
        esac
    done
}
else
ray_job_poll() {
    local status_output=""
    local job_status=""
    local status_failures=0

    while true; do
        # Commands used as an if condition are exempt from errexit. This catches
        # both a regular Ray CLI error and a temporary fork(2) EAGAIN.
        if status_output="$("${RAY_CLI[@]}" job status \
            --address="${RAY_JOB_ADDRESS}" --log-style record \
            "${RAY_JOB_SUBMISSION_ID}" 2>&1)"; then
            ray_job_parse_status "${status_output}" job_status
        else
            # Ray 2.53 returns non-zero for terminal FAILED/STOPPED jobs.
            # Parse that terminal output before treating it as a query error.
            ray_job_parse_status "${status_output}" job_status
            if [[ "${job_status}" != "FAILED" && "${job_status}" != "STOPPED" ]]; then
                status_failures=$((status_failures + 1))
                if (( status_failures <= RAY_JOB_STATUS_MAX_CONSECUTIVE_FAILURES \
                      || status_failures % RAY_JOB_STATUS_MAX_CONSECUTIVE_FAILURES == 0 )); then
                    echo "[run] Ray job status query failed (${status_failures} consecutive attempt(s)): ${status_output}" >&2
                fi
                if (( status_failures == RAY_JOB_STATUS_MAX_CONSECUTIVE_FAILURES )); then
                    echo "[run] Ray job status remains UNKNOWN; entering degraded monitoring without tearing down the possibly-running job." >&2
                fi
                ray_job_poll_delay
                continue
            fi
        fi

        case "${job_status}" in
            SUCCEEDED)
                echo "[run] Ray job ${RAY_JOB_SUBMISSION_ID} succeeded"
                return 0
                ;;
            FAILED|STOPPED)
                echo "${status_output}" >&2
                echo "[run] Ray job ${RAY_JOB_SUBMISSION_ID} ended with ${job_status}" >&2
                return 1
                ;;
            PENDING|RUNNING)
                status_failures=0
                ray_job_poll_delay
                ;;
            *)
                status_failures=$((status_failures + 1))
                if (( status_failures <= RAY_JOB_STATUS_MAX_CONSECUTIVE_FAILURES \
                      || status_failures % RAY_JOB_STATUS_MAX_CONSECUTIVE_FAILURES == 0 )); then
                    echo "[run] Could not parse Ray job status (${status_failures} consecutive attempt(s)): ${status_output}" >&2
                fi
                if (( status_failures == RAY_JOB_STATUS_MAX_CONSECUTIVE_FAILURES )); then
                    echo "[run] Ray job status remains UNKNOWN; entering degraded monitoring without tearing down the possibly-running job." >&2
                fi
                ray_job_poll_delay
                ;;
        esac
    done
}
fi

prorl_prune_runtime_images_enabled() {
    # Preserve the established local overlay2 cleanup. VFS workers use the
    # shared HERO/Slime image leases, so an unleased bulk image rm is unsafe.
    local prune_default=1
    if [[ "${HERO_LOCAL_DOCKER_STORAGE_DRIVER:-overlay2}" == "vfs" ]]; then
        prune_default=0
    fi
    [[ "${HERO_LOCAL_DOCKER_EXIT_PRUNE_UNUSED_SWE_IMAGES:-${prune_default}}" == "1" ]]
}
if [[ "${TRAINING_MODEL_TYPE}" == "moe" ]]; then

notify_ray_head_stopping() {
    # cluster multi-node submitters can publish a shared marker so worker pods
    # distinguish an intentional head teardown from an unexpected Ray failure.
    # Local launchers do not set this variable, leaving their overlay2 behavior
    # unchanged. Use a shell redirection instead of spawning `touch` because
    # cleanup may run while the host is under PID or memory pressure.
    local marker="${HERO_RAY_HEAD_STOPPING_MARKER:-}"
    [[ -n "${marker}" ]] || return 0
    if ! : > "${marker}"; then
        echo "[run] WARNING: could not publish Ray head stopping marker: ${marker}" >&2
        return 0
    fi
    echo "[run] Published Ray head stopping marker: ${marker}"
}
fi

# Tests source only these helpers, without starting HERO, Ray, or training.
if [[ "${SWEGYM_RUN_SH_FUNCTIONS_ONLY:-0}" == "1" ]]; then
    return 0 2>/dev/null || exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/../results}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)_${BASHPID}}"
RUN_ROOT="${RUN_ROOT:-${RESULTS_ROOT}/train/${TRAINING_METHOD:-HERO}/qwen35_${TRAINING_MODEL_SIZE:-4b}/${STAMP}_shared}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/runtime}"
export WANDB_DIR="${WANDB_DIR:-${RUN_ROOT}/logs/wandb}"
mkdir -p "${RUN_DIR}" "${WANDB_DIR}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python3}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi
PYTHON_BIN_DIR="$(cd -- "$(dirname -- "${PYTHON_BIN}")" &>/dev/null && pwd)"
export PATH="${PYTHON_BIN_DIR}:${PATH}"

is_path_like() {
    case "$1" in
        /*|./*|../*|~*) return 0 ;;
        *) return 1 ;;
    esac
}

detect_host_ip() {
    "${PYTHON_BIN}" - <<'PY'
import socket

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect(("8.8.8.8", 80))
    print(sock.getsockname()[0])
    sock.close()
except Exception:
    try:
        print(socket.gethostbyname(socket.gethostname()))
    except Exception:
        print("127.0.0.1")
PY
}

# ── External deps ──────────────────────────────────────────────────
SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/slime}"
if [[ "${TRAINING_MODEL_TYPE}" == "moe" ]]; then
if [ ! -f "${SLIME_DIR}/train.py" ] || [ ! -f "${SLIME_DIR}/train_async.py" ]; then
    echo "ERROR: Slime not found at ${SLIME_DIR}"
    echo "  git clone git@github.com:THUDM/slime.git ${SLIME_DIR}"
    exit 1
fi
else
if [ ! -f "${SLIME_DIR}/train_async.py" ]; then
    echo "ERROR: Slime not found at ${SLIME_DIR}"
    echo "  git clone git@github.com:THUDM/slime.git ${SLIME_DIR}"
    exit 1
fi
fi

MEGATRON_DIR="${MEGATRON_DIR:-${PROJECT_ROOT}/Megatron-LM}"
if [ ! -d "${MEGATRON_DIR}/megatron" ]; then
    echo "ERROR: Megatron-LM not found at ${MEGATRON_DIR}"
    echo "  git clone https://github.com/NVIDIA/Megatron-LM.git ${MEGATRON_DIR}"
    exit 1
fi

# ── Model ──────────────────────────────────────────────────────────
# Qwen3.5-4B: VLM checkpoint; we train text-only.  HF weights are loaded
# through slime_plugins.mbridge.qwen3_5 (text_config-aware) at convert-time.
HF_CHECKPOINT="${HF_CHECKPOINT:-Qwen/Qwen3.5-4B}"
REF_LOAD="${REF_LOAD:-${PROJECT_ROOT}/tmp/checkpoints/Qwen3.5-4B_torch_dist}"
RUN_ID="${RUN_ID:-swegym-slime-grpo-$(date -u +%Y%m%dT%H%M%SZ)}"
SAVE_ROOT="${SAVE_ROOT:-${RUN_ROOT}/ckpt}"
SAVE_DIR="${SAVE_DIR:-${SAVE_ROOT}}"
mkdir -p "$SAVE_DIR"
# The official example only saves Megatron checkpoints. Local launchers may
# opt in to an additional HuggingFace save by exporting SAVE_HF=<pattern>.
SAVE_HF_ARGS=()
if [[ -n "${SAVE_HF:-}" && "${SAVE_HF_DISABLE:-0}" != "1" ]]; then
    SAVE_HF_ARGS=(--save-hf "${SAVE_HF}")
    echo "Using save_hf pattern: ${SAVE_HF}"
fi
if [[ "${TRAINING_MODEL_TYPE}" == "moe" ]]; then
SAVE_ARGS=(--save "${SAVE_DIR}" --save-interval "${SAVE_INTERVAL:-10}")
case "${SAVE_DISABLE:-0}" in
    0)
        ;;
    1)
        SAVE_ARGS=()
        SAVE_HF_ARGS=()
        echo "Checkpoint saving disabled for this run"
        ;;
    *)
        echo "SAVE_DISABLE must be 0 or 1" >&2
        exit 2
        ;;
esac
fi
if is_path_like "$HF_CHECKPOINT" && [ ! -e "$HF_CHECKPOINT" ]; then
    echo "ERROR: HF checkpoint not found at $HF_CHECKPOINT"
    echo "  hf download Qwen/Qwen3.5-4B"
    exit 1
fi
if [ ! -d "$REF_LOAD" ] || [ ! -f "$REF_LOAD/latest_checkpointed_iteration.txt" ]; then
    echo "ERROR: Megatron torch_dist checkpoint not found at $REF_LOAD"
    echo "  Run bash examples/swegym_slime_grpo/convert_weights.sh first."
    exit 1
fi

# Allow model-specific launchers to select the matching Megatron architecture.
# Keep the 4B file as the default so existing launchers remain unchanged.
MODEL_ARGS_FILE="${MODEL_ARGS_FILE:-${SCRIPT_DIR}/model_args.sh}"
[[ -f "${MODEL_ARGS_FILE}" ]] || {
    echo "ERROR: model args file not found at ${MODEL_ARGS_FILE}" >&2
    exit 1
}
# shellcheck source=./model_args.sh
source "${MODEL_ARGS_FILE}"

# First run has an empty SAVE_DIR — slime's load_checkpoint asserts on empty.
# Pick REF_LOAD (torch_dist) until the first save lands.
if [ -f "$SAVE_DIR/latest_checkpointed_iteration.txt" ]; then
    LOAD_DIR="$SAVE_DIR"
else
    LOAD_DIR="$REF_LOAD"
fi

# ── Data ───────────────────────────────────────────────────────────
PROMPT_DATA="${PROMPT_DATA:-${PROJECT_ROOT}/../data/swe_train_640.jsonl}"
if [ ! -f "$PROMPT_DATA" ]; then
    echo "Missing training data: ${PROMPT_DATA}" >&2
    exit 1
fi

# ── Runtime configs ─────────────────────────────────────────────────
export AGENT_CLI_DIR="${AGENT_CLI_DIR:-${PROJECT_ROOT}/../cli/claude_code}"
export APPTAINER_IMAGE_DIR="${APPTAINER_IMAGE_DIR:-${PROJECT_ROOT}/tmp/swegym_apptainer_images}"
# Prefer apptainer in PATH (HPC modules etc.); fall back to /usr/bin for Ubuntu defaults.
export HERO_APPTAINER_BIN="${HERO_APPTAINER_BIN:-$(command -v apptainer || echo /usr/bin/apptainer)}"
SGLANG_ROUTER_PORT="${SGLANG_ROUTER_PORT:-9000}"
SGLANG_ROUTER_HOST="${SGLANG_ROUTER_HOST:-$(detect_host_ip)}"
export SGLANG_ROUTER_BASE_URL="${SGLANG_ROUTER_BASE_URL:-http://${SGLANG_ROUTER_HOST}:${SGLANG_ROUTER_PORT}}"
TOPOLOGY_TEMPLATE="${TOPOLOGY_TEMPLATE:-${SCRIPT_DIR}/topology.yaml}"
HERO_CONFIG_TEMPLATE="${HERO_CONFIG_TEMPLATE:-${SCRIPT_DIR}/hero_config.yaml}"
TOPOLOGY_PATH="${TOPOLOGY_PATH:-${RUN_DIR}/topology.yaml}"
CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${RUN_DIR}/hero_config.yaml}"
COMPILER_CACHE_ROOT="${COMPILER_CACHE_ROOT:-${RUN_DIR}/compiler_cache}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${COMPILER_CACHE_ROOT}/torchinductor}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${COMPILER_CACHE_ROOT}/triton}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# Render only explicit placeholders so literal $HOME / $... strings in the
# HERO task template remain untouched. Python is already a hard dependency.
mkdir -p "$(dirname "$TOPOLOGY_PATH")" "$(dirname "$CUSTOM_CONFIG_PATH")"
"${PYTHON_BIN}" - "$TOPOLOGY_TEMPLATE" "$TOPOLOGY_PATH" \
  "$HERO_CONFIG_TEMPLATE" "$CUSTOM_CONFIG_PATH" \
  "$SGLANG_ROUTER_BASE_URL" "$AGENT_CLI_DIR" "$APPTAINER_IMAGE_DIR" \
  "${CLAUDE_CODE_AUTO_COMPACT_SETTINGS:-}" \
  "${CLAUDE_CODE_MAX_TURNS:-}" \
  "${HERO_MAX_COMPLETIONS:-}" \
  "${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-}" \
  "${CLAUDE_CODE_TOOLS:-}" \
  "${CLAUDE_CODE_DISALLOWED_TOOLS:-}" \
  "${HERO_GIT_PROGRESS_ENABLED:-0}" \
  "${HERO_GIT_PROGRESS_CHECKPOINT_INTERVAL:-10}" \
  "${HERO_FAILURE_TOKEN_EFFICIENCY_ENABLED:-false}" \
  "${HERO_FAILURE_TOKEN_EFFICIENCY_CREDIT:-trajectory}" \
  "${HERO_FAILURE_NO_PATCH_RATIO:-0.5}" \
  "${HERO_FAILURE_EXTREME_RATIO:-2.0}" \
  "${HERO_FAILURE_MAX_ABS_PENALTY:-0.5}" \
  "${HERO_REQUEST_TIMEOUT:-1800}" \
  "${HERO_MIN_COMPLETE_ACCEPT_FRACTION:-0.6}" <<'PY'
from pathlib import Path
import os
import sys

topology_template, topology_path, hero_template, hero_path = map(Path, sys.argv[1:5])
values = dict(zip(
    ("SGLANG_ROUTER_BASE_URL", "AGENT_CLI_DIR", "APPTAINER_IMAGE_DIR",
     "CLAUDE_CODE_AUTO_COMPACT_SETTINGS", "CLAUDE_CODE_MAX_TURNS",
     "HERO_MAX_COMPLETIONS",
     "CLAUDE_CODE_MAX_OUTPUT_TOKENS", "CLAUDE_CODE_TOOLS",
     "CLAUDE_CODE_DISALLOWED_TOOLS", "HERO_GIT_PROGRESS_ENABLED",
     "HERO_GIT_PROGRESS_CHECKPOINT_INTERVAL",
     "HERO_FAILURE_TOKEN_EFFICIENCY_ENABLED",
     "HERO_FAILURE_TOKEN_EFFICIENCY_CREDIT",
     "HERO_FAILURE_NO_PATCH_RATIO", "HERO_FAILURE_EXTREME_RATIO",
     "HERO_FAILURE_MAX_ABS_PENALTY",
     "HERO_REQUEST_TIMEOUT",
     "HERO_MIN_COMPLETE_ACCEPT_FRACTION"),
    sys.argv[5:23],
))
values["HERO_MODEL_NAME"] = os.environ["HERO_MODEL_NAME"]
request_timeout = int(values["HERO_REQUEST_TIMEOUT"])
if request_timeout <= 0:
    raise SystemExit("HERO_REQUEST_TIMEOUT must be a positive integer")
complete_fraction = float(values["HERO_MIN_COMPLETE_ACCEPT_FRACTION"])
if not 0.0 <= complete_fraction <= 1.0:
    raise SystemExit("HERO_MIN_COMPLETE_ACCEPT_FRACTION must be between 0 and 1")
values["HERO_MAX_COMPLETIONS_SETTING"] = (
    f"  max_completions: {values['HERO_MAX_COMPLETIONS']}"
    if values["HERO_MAX_COMPLETIONS"] else ""
)
values["CLAUDE_CODE_MAX_OUTPUT_TOKENS_ENV"] = (
    "    env:\n"
    f"      CLAUDE_CODE_MAX_OUTPUT_TOKENS: \"{values['CLAUDE_CODE_MAX_OUTPUT_TOKENS']}\""
    if values["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] else ""
)
for source, destination in (
    (topology_template, topology_path),
    (hero_template, hero_path),
):
    text = source.read_text()
    for name, value in values.items():
        text = text.replace("${" + name + "}", value)
    destination.write_text(text)
PY

echo "Using topology: ${TOPOLOGY_PATH}"
echo "Using HERO config: ${CUSTOM_CONFIG_PATH}"
echo "Using Apptainer image dir: ${APPTAINER_IMAGE_DIR}"
echo "Using run id: ${RUN_ID}"
echo "Using save dir: ${SAVE_DIR}"
echo "Using SGLang router URL for HERO gateway: ${SGLANG_ROUTER_BASE_URL}"
echo "Using completion budget: max_turns=${CLAUDE_CODE_MAX_TURNS:-<unset>} max_completions=${HERO_MAX_COMPLETIONS:-<unset>}"

# Local-Docker launchers opt into node-safe Ray lifecycle handling. With no
# local backend configured, retain the official example's launch behavior.
LOCAL_DOCKER_MODE=0
if [[ -n "${HERO_LOCAL_DOCKER_HOST:-}" ]]; then
    LOCAL_DOCKER_MODE=1
    # The centralized sampler performs no Docker CLI forks, so it can check
    # substantially faster than the legacy per-container watchdog. Callers can
    # still override this resource-only interval explicitly.
    export HERO_LOCAL_DOCKER_RESOURCE_CHECK_INTERVAL_SEC="${HERO_LOCAL_DOCKER_RESOURCE_CHECK_INTERVAL_SEC:-5}"
    PRORL_NODE_LOCK_FILE="${PRORL_NODE_LOCK_FILE:-${TMPDIR:-/tmp}/hero-training.lock}"
    exec {PRORL_NODE_LOCK_FD}>"${PRORL_NODE_LOCK_FILE}"
    if ! flock -n "${PRORL_NODE_LOCK_FD}"; then
        echo "ERROR: another ProRL training launcher owns this node (${PRORL_NODE_LOCK_FILE})." >&2
        echo "Refusing to stop its Ray cluster. Wait for it to finish or inspect the lock owner with: fuser ${PRORL_NODE_LOCK_FILE}" >&2
        exit 1
    fi
fi

# ── Cleanup on exit ────────────────────────────────────────────────
PIDS=()
RAY_STARTED_BY_THIS_LAUNCHER=0

cleanup_local_docker_artifacts() {
    [[ "${LOCAL_DOCKER_MODE}" == "1" ]] || return 0

    local -a container_ids=()
    local -a image_ids=()
    local container_id image_id
    local cleanup_run_id="${HERO_RUN_ID:-${RUN_ID}}"

    mapfile -t container_ids < <(
        docker ps -aq \
            --filter "label=ai.hero.namespace=swegym-local-docker" \
            --filter "label=ai.hero.run_id=${cleanup_run_id}" \
            2>/dev/null || true
    )
    for container_id in "${container_ids[@]}"; do
        image_id="$(docker inspect --format '{{.Image}}' "${container_id}" 2>/dev/null || true)"
        [[ -n "${image_id}" ]] && image_ids+=("${image_id}")
    done

    if (( ${#container_ids[@]} > 0 )); then
        echo "Removing ${#container_ids[@]} stale HERO runtime container(s)..."
        docker rm -f "${container_ids[@]}" >/dev/null 2>&1 || true
    fi

    # Remove only images that were referenced by this launcher's HERO
    # runtimes. Docker will safely refuse an image still used elsewhere.
    if prorl_prune_runtime_images_enabled && (( ${#image_ids[@]} > 0 )); then
        mapfile -t image_ids < <(printf '%s\n' "${image_ids[@]}" | sort -u)
        echo "Removing ${#image_ids[@]} HERO runtime image(s)..."
        docker image rm "${image_ids[@]}" >/dev/null 2>&1 || true
    fi
}

if [[ "${TRAINING_MODEL_TYPE}" == "moe" ]]; then
cleanup() {
    echo "Shutting down..."
    notify_ray_head_stopping
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    if declare -p RAY_WORKER_IPS &>/dev/null; then
        for worker_ip in "${RAY_WORKER_IPS[@]}"; do
            ssh -o BatchMode=yes -o StrictHostKeyChecking=no "root@${worker_ip}" \
                "${PYTHON_BIN} -m ray.scripts.scripts stop --force" >/dev/null 2>&1 &
        done
        wait 2>/dev/null || true
    fi
    if [[ "${LOCAL_DOCKER_MODE}" == "1" && "${RAY_STARTED_BY_THIS_LAUNCHER}" == "1" ]] \
       && declare -p RAY_CLI &>/dev/null; then
        "${RAY_CLI[@]}" stop --force 2>/dev/null || true
    elif [[ "${LOCAL_DOCKER_MODE}" == "0" ]]; then
        ray stop --force 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    cleanup_local_docker_artifacts
}
else
cleanup() {
    echo "Shutting down..."
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    if [[ "${LOCAL_DOCKER_MODE}" == "1" && "${RAY_STARTED_BY_THIS_LAUNCHER}" == "1" ]] \
       && declare -p RAY_CLI &>/dev/null; then
        "${RAY_CLI[@]}" stop --force 2>/dev/null || true
    elif [[ "${LOCAL_DOCKER_MODE}" == "0" ]]; then
        ray stop --force 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    cleanup_local_docker_artifacts
}
fi
trap cleanup EXIT

# Recover from a launcher that was SIGKILLed or a previous gateway whose
# teardown failed before its EXIT trap ran. The node lock above guarantees
# that no other ProRL training launcher is active while this executes.
cleanup_local_docker_artifacts

# Initialize the no-fork polling clock before model workers and Docker tasks can
# put the host under memory/PID pressure.
if [[ "${LOCAL_DOCKER_MODE}" == "1" ]]; then
    ray_job_init_poll_clock
fi

# ── Step 1: HERO services (runs on host, CPU only) ───────────────
echo "=== Starting HERO rollout server (:8080) ==="
hero serve_rollout -c "${TOPOLOGY_PATH}" &
PIDS+=($!)
sleep 2

echo "=== Starting HERO gateway (:8100) ==="
hero serve_gateway -c "${TOPOLOGY_PATH}" --node-id localhost-node-01 &
PIDS+=($!)
sleep 2

curl -sf http://127.0.0.1:8080/health || { echo "HERO rollout server not healthy"; exit 1; }

if [[ "${TRAINING_MODEL_TYPE}" == "moe" ]]; then
# ── Step 2: Ray + Slime (manages SGLang engines + training) ───────
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-4}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
else
# ── Step 2: Ray + Slime (manages SGLang engines + training) ───────
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-4}"
fi
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-4}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-30000}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
if [[ "${TRAINING_MODEL_TYPE}" == "moe" ]]; then
PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-8}"
EXPERT_TENSOR_PARALLEL_SIZE="${EXPERT_TENSOR_PARALLEL_SIZE:-1}"
SLIME_COLOCATE="${SLIME_COLOCATE:-0}"
fi
ATTENTION_BACKEND="${ATTENTION_BACKEND:-auto}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-50000}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.8}"
if [[ "${TRAINING_MODEL_TYPE}" == "moe" ]]; then
# Keep SGLang on the regular PyTorch CUDA-graph path. The CUDA-13 TMS preload
# is scoped to Megatron actors; preloading it into SGLang's DP scheduler makes
# the scheduler's early cuMemCreate call fail.
export SGLANG_MEMORY_SAVER_CUDA_GRAPH="${SGLANG_MEMORY_SAVER_CUDA_GRAPH:-false}"
fi

for _parallel_value in TENSOR_MODEL_PARALLEL_SIZE CONTEXT_PARALLEL_SIZE MAX_TOKENS_PER_GPU SGLANG_CONTEXT_LENGTH; do
    [[ "${!_parallel_value}" =~ ^[1-9][0-9]*$ ]] || {
        echo "${_parallel_value} must be a positive integer" >&2
        exit 2
    }
done
"${PYTHON_BIN}" - "${SGLANG_MEM_FRACTION_STATIC}" <<'PY'
import sys
value = float(sys.argv[1])
if not 0.0 < value <= 1.0:
    raise SystemExit("SGLANG_MEM_FRACTION_STATIC must be in (0, 1]")
PY
if [[ "${TRAINING_MODEL_TYPE}" == "moe" ]]; then
ACTOR_WORLD_SIZE=$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))
MODEL_PARALLEL_SIZE=$((TENSOR_MODEL_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE))
if (( ACTOR_WORLD_SIZE % MODEL_PARALLEL_SIZE != 0 )); then
    echo "Actor world size (${ACTOR_WORLD_SIZE}) must be divisible by TP*CP*PP (${MODEL_PARALLEL_SIZE})" >&2
    exit 2
fi
else
MODEL_PARALLEL_SIZE=$((TENSOR_MODEL_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE))
if (( ACTOR_NUM_GPUS_PER_NODE % MODEL_PARALLEL_SIZE != 0 )); then
    echo "ACTOR_NUM_GPUS_PER_NODE (${ACTOR_NUM_GPUS_PER_NODE}) must be divisible by TP*CP (${TENSOR_MODEL_PARALLEL_SIZE}*${CONTEXT_PARALLEL_SIZE}=${MODEL_PARALLEL_SIZE})" >&2
    exit 2
fi
fi

if [[ "${TRAINING_MODEL_TYPE}" == "moe" ]]; then
# Ray sizing — derive total GPUs from the actor/rollout split.
RAY_NUM_NODES="${RAY_NUM_NODES:-8}"
RAY_NUM_GPUS_PER_NODE="${RAY_NUM_GPUS_PER_NODE:-8}"
else
# Ray sizing — derive total GPUs from the actor/rollout split.
RAY_NUM_GPUS="${RAY_NUM_GPUS:-$((ACTOR_NUM_GPUS_PER_NODE + ROLLOUT_NUM_GPUS))}"
fi
RAY_HEAD_IP="${RAY_HEAD_IP:-127.0.0.1}"
if [[ "${TRAINING_MODEL_TYPE}" == "moe" ]]; then
case "${SLIME_COLOCATE}" in
    0)
        REQUIRED_GPUS=$((ACTOR_WORLD_SIZE + ROLLOUT_NUM_GPUS))
        ;;
    1)
        REQUIRED_GPUS=$(( ACTOR_WORLD_SIZE > ROLLOUT_NUM_GPUS ? ACTOR_WORLD_SIZE : ROLLOUT_NUM_GPUS ))
        ;;
    *)
        echo "SLIME_COLOCATE must be 0 or 1" >&2
        exit 2
        ;;
esac
if (( REQUIRED_GPUS > RAY_NUM_NODES * RAY_NUM_GPUS_PER_NODE )); then
    echo "Requested actor/rollout topology needs ${REQUIRED_GPUS} GPUs but cluster capacity is $((RAY_NUM_NODES * RAY_NUM_GPUS_PER_NODE))" >&2
    exit 2
fi
fi
RAY_OBJECT_STORE_ARGS=()
if [[ -n "${RAY_OBJECT_STORE_MEMORY_BYTES:-}" ]]; then
    [[ "${RAY_OBJECT_STORE_MEMORY_BYTES}" =~ ^[1-9][0-9]*$ ]] || {
        echo "RAY_OBJECT_STORE_MEMORY_BYTES must be a positive integer (bytes)" >&2
        exit 2
    }
    RAY_OBJECT_STORE_ARGS=(--object-store-memory "${RAY_OBJECT_STORE_MEMORY_BYTES}")
fi

if [[ "${TRAINING_MODEL_TYPE}" == "moe" ]]; then
if [[ "${LOCAL_DOCKER_MODE}" == "1" ]]; then
    # Pin the local Ray head and workers to the same project interpreter.
    export PYTHONPATH="${MEGATRON_DIR}:${PROJECT_ROOT}/src:${PROJECT_ROOT}"
    RAY_CLI=("${PYTHON_BIN}" -m ray.scripts.scripts)
    echo "=== Starting Ray head on ${RAY_HEAD_IP} (${RAY_NUM_NODES}x${RAY_NUM_GPUS_PER_NODE} GPUs total) ==="
    if "${RAY_CLI[@]}" status &>/dev/null; then
        echo "ERROR: an active Ray cluster already exists on this node." >&2
        echo "Refusing to run 'ray stop --force' because it may own a live job." >&2
        exit 1
    fi
    "${RAY_CLI[@]}" stop --force 2>/dev/null || true
    sleep 1
    "${RAY_CLI[@]}" start --head --node-ip-address "$RAY_HEAD_IP" --num-gpus "$RAY_NUM_GPUS_PER_NODE" \
        --dashboard-host=0.0.0.0 "${RAY_OBJECT_STORE_ARGS[@]}" --disable-usage-stats
    RAY_STARTED_BY_THIS_LAUNCHER=1
else
    echo "=== Starting Ray head on ${RAY_HEAD_IP} (${RAY_NUM_NODES}x${RAY_NUM_GPUS_PER_NODE} GPUs total) ==="
    ray stop --force 2>/dev/null || true
    sleep 1
    ray start --head --node-ip-address "$RAY_HEAD_IP" --num-gpus "$RAY_NUM_GPUS_PER_NODE" \
        --dashboard-host=0.0.0.0 --disable-usage-stats
fi

case "${RAY_WORKERS_PRESTARTED:-0}" in
    0)
        HOSTFILE="${HOSTFILE:-${PROJECT_ROOT}/hostfile}"
        [[ -f "${HOSTFILE}" ]] || { echo "ERROR: missing HOSTFILE: ${HOSTFILE}" >&2; exit 1; }
        mapfile -t RAY_WORKER_IPS < <(awk 'NF && $1 !~ /^#/ {print $1}' "${HOSTFILE}" | awk -v head="${RAY_HEAD_IP}" '$1 != head' | head -n "$((RAY_NUM_NODES - 1))")
        (( ${#RAY_WORKER_IPS[@]} == RAY_NUM_NODES - 1 )) || {
            echo "ERROR: need $((RAY_NUM_NODES - 1)) worker hosts besides ${RAY_HEAD_IP} in ${HOSTFILE}" >&2
            exit 1
        }
        for worker_ip in "${RAY_WORKER_IPS[@]}"; do
            echo "Starting Ray worker on ${worker_ip}"
            ssh -o BatchMode=yes -o StrictHostKeyChecking=no "root@${worker_ip}" \
                "${PYTHON_BIN} -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true; ${PYTHON_BIN} -m ray.scripts.scripts start --address=${RAY_HEAD_IP}:6379 --node-ip-address=${worker_ip} --num-gpus=${RAY_NUM_GPUS_PER_NODE} --disable-usage-stats" &
        done
        wait
        ;;
    1)
        # Distributed schedulers commonly run the same command in every pod
        # and do not expose passwordless SSH. In that mode non-head pods start
        # their own Ray workers and the head only waits for the declared shape.
        RAY_WORKER_IPS=()
        "${PYTHON_BIN}" - "${RAY_HEAD_IP}" "${RAY_NUM_NODES}" <<'PY'
import sys
import time

import ray

address, expected_text = sys.argv[1:]
expected = int(expected_text)
ray.init(address=f"{address}:6379", logging_level="ERROR")
deadline = time.monotonic() + 600
while time.monotonic() < deadline:
    alive = [node for node in ray.nodes() if node.get("Alive")]
    if len(alive) >= expected:
        print(f"Ray cluster ready with {len(alive)} live nodes")
        break
    time.sleep(2)
else:
    raise SystemExit(f"Timed out waiting for {expected} Ray nodes")
ray.shutdown()
PY
        ;;
    *)
        echo "RAY_WORKERS_PRESTARTED must be 0 or 1" >&2
        exit 2
        ;;
esac
sleep 10
"${RAY_CLI[@]}" status
else
if [[ "${LOCAL_DOCKER_MODE}" == "1" ]]; then
    # Pin the local Ray head and workers to the same project interpreter.
    export PYTHONPATH="${MEGATRON_DIR}:${PROJECT_ROOT}/src:${PROJECT_ROOT}"
    RAY_CLI=("${PYTHON_BIN}" -m ray.scripts.scripts)
    echo "=== Starting Ray on ${RAY_HEAD_IP} (${RAY_NUM_GPUS} GPUs, object store ${RAY_OBJECT_STORE_MEMORY_BYTES:-Ray default}) ==="
    if "${RAY_CLI[@]}" status &>/dev/null; then
        echo "ERROR: an active Ray cluster already exists on this node." >&2
        echo "Refusing to run 'ray stop --force' because it may own a live job." >&2
        exit 1
    fi
    "${RAY_CLI[@]}" stop --force 2>/dev/null || true
    sleep 1
    "${RAY_CLI[@]}" start --head --node-ip-address "$RAY_HEAD_IP" --num-gpus "$RAY_NUM_GPUS" \
        "${RAY_OBJECT_STORE_ARGS[@]}" --disable-usage-stats
    RAY_STARTED_BY_THIS_LAUNCHER=1
else
    echo "=== Starting Ray on ${RAY_HEAD_IP} (${RAY_NUM_GPUS} GPUs) ==="
    ray stop --force 2>/dev/null || true
    sleep 1
    ray start --head --node-ip-address "$RAY_HEAD_IP" --num-gpus "$RAY_NUM_GPUS" --disable-usage-stats
fi
fi

# cuDNN lib path — probe the active venv instead of hardcoding python3.13.
if [ -z "${CUDNN_LIB:-}" ]; then
    CUDNN_LIB="$("${PYTHON_BIN}" -c 'import nvidia.cudnn, os; print(os.path.join(list(nvidia.cudnn.__path__)[0], "lib"))' 2>/dev/null || true)"
fi
RUNTIME_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
if [ -n "${CUDNN_LIB}" ] && [ -d "$CUDNN_LIB" ]; then
    RUNTIME_LD_LIBRARY_PATH="${CUDNN_LIB}:${RUNTIME_LD_LIBRARY_PATH}"
fi
if [[ "${TRAINING_MODEL_TYPE}" == "moe" ]]; then
if [[ "${LOCAL_DOCKER_MODE}" == "1" ]]; then
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_DIR}:${PROJECT_ROOT}/src:${PROJECT_ROOT}\",
    \"PATH\": \"${PYTHON_BIN_DIR}:${PATH}\",
    \"VIRTUAL_ENV\": \"${VIRTUAL_ENV:-${PROJECT_ROOT}/.venv}\",
    \"CUDA_HOME\": \"${CUDA_HOME:-/usr/local/cuda}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"WANDB_DIR\": \"${WANDB_DIR}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY:-}\",
    \"WANDB_MODE\": \"${WANDB_MODE:-online}\",
    \"WANDB_CONSOLE\": \"${WANDB_CONSOLE:-off}\",
    \"WANDB_INIT_TIMEOUT\": \"${WANDB_INIT_TIMEOUT:-300}\",
    \"TORCHINDUCTOR_CACHE_DIR\": \"${TORCHINDUCTOR_CACHE_DIR}\",
    \"TRITON_CACHE_DIR\": \"${TRITON_CACHE_DIR}\",
    \"FLASHINFER_USE_CUDA_NORM\": \"${FLASHINFER_USE_CUDA_NORM:-1}\",
    \"HERO_TRAIN_POLICY_EARLY_STOP_TRAJECTORIES\": \"${HERO_TRAIN_POLICY_EARLY_STOP_TRAJECTORIES:-0}\",
    \"HERO_GENERATION_ENTROPY_TOP_K\": \"${HERO_GENERATION_ENTROPY_TOP_K:-}\",
    \"GATE_THRESHOLD\": \"${GATE_THRESHOLD:-0.6}\",
    \"LD_LIBRARY_PATH\": \"${RUNTIME_LD_LIBRARY_PATH}\",
    \"SLIME_TMS_PRELOAD_LIBRARY\": \"${SLIME_TMS_PRELOAD_LIBRARY:-}\",
    \"SLIME_COLOCATED_WEIGHT_SYNC_CPU\": \"${SLIME_COLOCATED_WEIGHT_SYNC_CPU:-0}\",
    \"SLIME_VFS_MULTINODE\": \"${SLIME_VFS_MULTINODE:-0}\",
    \"SLIME_GATEWAY_TOPOLOGY\": \"${TOPOLOGY_PATH}\",
    \"SLIME_TRAIN_PHASE_TRACE_DIR\": \"${SLIME_TRAIN_PHASE_TRACE_DIR:-}\",
    \"SGLANG_MEMORY_SAVER_CUDA_GRAPH\": \"${SGLANG_MEMORY_SAVER_CUDA_GRAPH}\",
    \"PYTORCH_ALLOC_CONF\": \"max_split_size_mb:2048\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"max_split_size_mb:2048\",
    \"NVTE_CUDA_INCLUDE_DIR\": \"${NVTE_CUDA_INCLUDE_DIR:-/usr/local/cuda/include}\",
    \"NVTE_DEBUG\": \"1\",
    \"NVTE_DEBUG_LEVEL\": \"2\"
  }
}"
else
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_DIR}:${PROJECT_ROOT}/src\",
    \"PATH\": \"${PYTHON_BIN_DIR}:${PATH}\",
    \"VIRTUAL_ENV\": \"${VIRTUAL_ENV:-${PROJECT_ROOT}/.venv}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"WANDB_DIR\": \"${WANDB_DIR}\",
    \"WANDB_MODE\": \"${WANDB_MODE:-online}\",
    \"WANDB_CONSOLE\": \"${WANDB_CONSOLE:-off}\",
    \"WANDB_INIT_TIMEOUT\": \"${WANDB_INIT_TIMEOUT:-300}\",
    \"TORCHINDUCTOR_CACHE_DIR\": \"${TORCHINDUCTOR_CACHE_DIR}\",
    \"TRITON_CACHE_DIR\": \"${TRITON_CACHE_DIR}\",
    \"HERO_GENERATION_ENTROPY_TOP_K\": \"${HERO_GENERATION_ENTROPY_TOP_K:-}\",
    \"GATE_THRESHOLD\": \"${GATE_THRESHOLD:-0.6}\",
    \"LD_LIBRARY_PATH\": \"${RUNTIME_LD_LIBRARY_PATH}\",
    \"SLIME_TMS_PRELOAD_LIBRARY\": \"${SLIME_TMS_PRELOAD_LIBRARY:-}\",
    \"SLIME_COLOCATED_WEIGHT_SYNC_CPU\": \"${SLIME_COLOCATED_WEIGHT_SYNC_CPU:-0}\",
    \"SGLANG_MEMORY_SAVER_CUDA_GRAPH\": \"${SGLANG_MEMORY_SAVER_CUDA_GRAPH}\",
    \"PYTORCH_ALLOC_CONF\": \"max_split_size_mb:2048\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"max_split_size_mb:2048\",
    \"NVTE_DEBUG\": \"1\",
    \"NVTE_DEBUG_LEVEL\": \"2\"
  }
}"
fi
else
if [[ "${LOCAL_DOCKER_MODE}" == "1" ]]; then
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_DIR}:${PROJECT_ROOT}/src:${PROJECT_ROOT}\",
    \"PATH\": \"${PYTHON_BIN_DIR}:${PATH}\",
    \"VIRTUAL_ENV\": \"${VIRTUAL_ENV:-${PROJECT_ROOT}/.venv}\",
    \"CUDA_HOME\": \"${CUDA_HOME:-/usr/local/cuda}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"WANDB_DIR\": \"${WANDB_DIR}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY:-}\",
    \"WANDB_MODE\": \"${WANDB_MODE:-online}\",
    \"WANDB_INIT_TIMEOUT\": \"${WANDB_INIT_TIMEOUT:-300}\",
    \"TORCHINDUCTOR_CACHE_DIR\": \"${TORCHINDUCTOR_CACHE_DIR}\",
    \"TRITON_CACHE_DIR\": \"${TRITON_CACHE_DIR}\",
    \"FLASHINFER_USE_CUDA_NORM\": \"${FLASHINFER_USE_CUDA_NORM:-1}\",
    \"HERO_TRAIN_POLICY_EARLY_STOP_TRAJECTORIES\": \"${HERO_TRAIN_POLICY_EARLY_STOP_TRAJECTORIES:-0}\",
    \"HERO_GENERATION_ENTROPY_TOP_K\": \"${HERO_GENERATION_ENTROPY_TOP_K:-}\",
    \"SLIME_MIN_AVAILABLE_MEMORY_BEFORE_TRAIN_GB\": \"${SLIME_MIN_AVAILABLE_MEMORY_BEFORE_TRAIN_GB:-0}\",
    \"SLIME_HOST_MEMORY_CHECK_INTERVAL_SEC\": \"${SLIME_HOST_MEMORY_CHECK_INTERVAL_SEC:-15}\",
    \"GATE_THRESHOLD\": \"${GATE_THRESHOLD:-0.6}\",
    \"LD_LIBRARY_PATH\": \"${RUNTIME_LD_LIBRARY_PATH}\",
    \"PYTORCH_ALLOC_CONF\": \"max_split_size_mb:2048,expandable_segments:True\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"max_split_size_mb:2048,expandable_segments:True\",
    \"NVTE_CUDA_INCLUDE_DIR\": \"${NVTE_CUDA_INCLUDE_DIR:-/usr/local/cuda/include}\",
    \"NVTE_DEBUG\": \"1\",
    \"NVTE_DEBUG_LEVEL\": \"2\"
  }
}"
else
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_DIR}:${PROJECT_ROOT}/src\",
    \"PATH\": \"${PYTHON_BIN_DIR}:${PATH}\",
    \"VIRTUAL_ENV\": \"${VIRTUAL_ENV:-${PROJECT_ROOT}/.venv}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"WANDB_DIR\": \"${WANDB_DIR}\",
    \"WANDB_MODE\": \"${WANDB_MODE:-online}\",
    \"WANDB_INIT_TIMEOUT\": \"${WANDB_INIT_TIMEOUT:-300}\",
    \"TORCHINDUCTOR_CACHE_DIR\": \"${TORCHINDUCTOR_CACHE_DIR}\",
    \"TRITON_CACHE_DIR\": \"${TRITON_CACHE_DIR}\",
    \"HERO_GENERATION_ENTROPY_TOP_K\": \"${HERO_GENERATION_ENTROPY_TOP_K:-}\",
    \"SLIME_MIN_AVAILABLE_MEMORY_BEFORE_TRAIN_GB\": \"${SLIME_MIN_AVAILABLE_MEMORY_BEFORE_TRAIN_GB:-0}\",
    \"SLIME_HOST_MEMORY_CHECK_INTERVAL_SEC\": \"${SLIME_HOST_MEMORY_CHECK_INTERVAL_SEC:-15}\",
    \"GATE_THRESHOLD\": \"${GATE_THRESHOLD:-0.6}\",
    \"LD_LIBRARY_PATH\": \"${RUNTIME_LD_LIBRARY_PATH}\",
    \"PYTORCH_ALLOC_CONF\": \"max_split_size_mb:2048,expandable_segments:True\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"max_split_size_mb:2048,expandable_segments:True\",
    \"NVTE_DEBUG\": \"1\",
    \"NVTE_DEBUG_LEVEL\": \"2\"
  }
}"
fi
fi

# BEGIN HERO_ENV_FORWARDING
# Carry method parameters into the Ray job; nested actors inherit them too.
RUNTIME_ENV_JSON="$("${PYTHON_BIN}" - "${RUNTIME_ENV_JSON}" <<'PYENV'
import json
import os
import sys

runtime = json.loads(sys.argv[1])
for key in ("EFFICIENCY_RATIO", "EFFICIENCY_COEFFICIENT"):
    if os.environ.get(key):
        runtime.setdefault("env_vars", {})[key] = os.environ[key]
print(json.dumps(runtime))
PYENV
)"
# END HERO_ENV_FORWARDING

if [[ "${TRAINING_MODEL_TYPE}" == "moe" ]]; then
# BEGIN MULTINODE_RAY_NETWORK_ENV
# Match the reference 8-node recipe's explicit transport across Ray job
# boundaries. Other launchers retain their established environment behavior.
if [[ "${SLIME_VFS_MULTINODE:-0}" == "1" ]]; then
RUNTIME_ENV_JSON="$("${PYTHON_BIN}" - "${RUNTIME_ENV_JSON}" <<'PY'
import json
import os
import sys

runtime = json.loads(sys.argv[1])
env = runtime.setdefault("env_vars", {})
for key in ("NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME"):
    if os.environ.get(key):
        env[key] = os.environ[key]
if "TP_SOCKET_IFNAME" not in env and env.get("GLOO_SOCKET_IFNAME"):
    env["TP_SOCKET_IFNAME"] = env["GLOO_SOCKET_IFNAME"]
print(json.dumps(runtime))
PY
)"
fi
# END MULTINODE_RAY_NETWORK_ENV

fi
# Rollout sizing: 4 prompts × 16 trajectories = 64 trajectories/rollout.
# This matches the earlier high-util baseline and keeps request groups smaller
# so long tails do not collapse usable token throughput.
# With --dynamic-history each trajectory explodes into one sample per trace,
# so sample count per rollout is variable.
# The data source rounds epoch length up to whole rollout batches, covering
# every prompt and wrapping only a partial final batch when needed.
ROLLOUT_COUNT_ARGS=(--num-epoch "${NUM_EPOCH:-1}")
if [[ -n "${NUM_ROLLOUT:-}" ]]; then
    # An explicit step count takes precedence over epoch-derived length.
    ROLLOUT_COUNT_ARGS=(--num-rollout "${NUM_ROLLOUT}")
fi
CUSTOM_ADVANTAGE_ARGS=()
if [[ "${HERO_FAILURE_TOKEN_EFFICIENCY_ENABLED:-false}" == "true" \
      && "${HERO_FAILURE_TOKEN_EFFICIENCY_CREDIT:-trajectory}" == "turn" \
      && -z "${CUSTOM_ADVANTAGE_FUNCTION_PATH:-}" ]]; then
    CUSTOM_ADVANTAGE_FUNCTION_PATH="slime_bridge.turn_advantage.compute_turn_level_advantages"
fi
if [[ -n "${CUSTOM_ADVANTAGE_FUNCTION_PATH:-}" ]]; then
    CUSTOM_ADVANTAGE_ARGS=(
        --custom-advantage-function-path "${CUSTOM_ADVANTAGE_FUNCTION_PATH}"
    )
fi
# Keep standard Slime/GRPO whitening by default. Custom methods that already
# construct their final token-level advantage can opt out without changing the
# baseline launcher or framework code path.
NORMALIZE_ADVANTAGE_ARGS=()
case "${SLIME_NORMALIZE_ADVANTAGES:-1}" in
    1|true|True|TRUE|yes|Yes|YES)
        NORMALIZE_ADVANTAGE_ARGS=(--normalize-advantages)
        ;;
    0|false|False|FALSE|no|No|NO)
        ;;
    *)
        echo "SLIME_NORMALIZE_ADVANTAGES must be 0/1 (or false/true)" >&2
        exit 2
        ;;
esac
CUSTOM_ROLLOUT_LOG_ARGS=()
if [[ -n "${CUSTOM_ROLLOUT_LOG_FUNCTION_PATH:-}" ]]; then
    CUSTOM_ROLLOUT_LOG_ARGS=(
        --custom-rollout-log-function-path "${CUSTOM_ROLLOUT_LOG_FUNCTION_PATH}"
    )
fi
LOSS_NORMALIZATION_ARGS=()
case "${SLIME_LOSS_NORMALIZATION_BY_TRAJECTORY_ID:-0}" in
    0|false|False|FALSE|no|No|NO)
        ;;
    1|true|True|TRUE|yes|Yes|YES)
        LOSS_NORMALIZATION_ARGS=(--loss-normalization-by-trajectory-id)
        ;;
    *)
        echo "SLIME_LOSS_NORMALIZATION_BY_TRAJECTORY_ID must be 0/1 (or false/true)" >&2
        exit 2
        ;;
esac
if [[ "${TRAINING_MODEL_TYPE}" == "moe" ]]; then
OPTIMIZER_OFFLOAD_ARGS=()
case "${OPTIMIZER_CPU_OFFLOAD:-0}" in
    0)
        ;;
    1)
        OPTIMIZER_OFFLOAD_ARGS=(
            --optimizer-cpu-offload
            --overlap-cpu-optimizer-d2h-h2d
            --use-precision-aware-optimizer
        )
        ;;
    *)
        echo "OPTIMIZER_CPU_OFFLOAD must be 0 or 1" >&2
        exit 2
        ;;
esac
COLOCATE_ARGS=()
if [[ "${SLIME_COLOCATE}" == "1" ]]; then
    COLOCATE_ARGS=(--colocate)
fi
if [[ -z "${SLIME_TRAIN_ENTRY:-}" ]]; then
    if [[ "${SLIME_COLOCATE}" == "1" ]]; then
        SLIME_TRAIN_ENTRY="${SLIME_DIR}/train.py"
    else
        SLIME_TRAIN_ENTRY="${SLIME_DIR}/train_async.py"
    fi
fi
[[ -f "${SLIME_TRAIN_ENTRY}" ]] || {
    echo "Slime training entry not found: ${SLIME_TRAIN_ENTRY}" >&2
    exit 2
}
echo "=== Launching $(basename "${SLIME_TRAIN_ENTRY}") ==="
else
echo "=== Launching train_async.py ==="
fi
RAY_JOB_ADDRESS="http://${RAY_HEAD_IP}:8265"
if [[ "${LOCAL_DOCKER_MODE}" == "1" ]]; then
    RAY_JOB_SUBMISSION_ID="${RAY_JOB_SUBMISSION_ID:-${RUN_ID}}"
    RAY_JOB_SUBMIT=("${RAY_CLI[@]}" job submit --address="${RAY_JOB_ADDRESS}"
        --submission-id "${RAY_JOB_SUBMISSION_ID}" --no-wait)
else
    RAY_JOB_SUBMIT=(ray job submit --address="${RAY_JOB_ADDRESS}")
fi
# BEGIN TRAINING_SUBMISSION
TRAINING_ARGS=()
TRAINING_ARGS+=(
    --runtime-env-json="${RUNTIME_ENV_JSON}"
)
if [[ "${TRAINING_MODEL_TYPE}" == "moe" ]]; then
TRAINING_ARGS+=(
    -- "${PYTHON_BIN}" "${SLIME_TRAIN_ENTRY}"
    --actor-num-nodes "$ACTOR_NUM_NODES"
)
else
TRAINING_ARGS+=(
    -- "${PYTHON_BIN}" "${SLIME_DIR}/train_async.py"
    --actor-num-nodes 1
)
fi
TRAINING_ARGS+=(
    --actor-num-gpus-per-node "$ACTOR_NUM_GPUS_PER_NODE"
)
if [[ "${TRAINING_MODEL_TYPE}" == "moe" ]]; then
TRAINING_ARGS+=(
    --num-gpus-per-node "$RAY_NUM_GPUS_PER_NODE"
)
fi
TRAINING_ARGS+=(
    --rollout-num-gpus "$ROLLOUT_NUM_GPUS"
    --rollout-num-gpus-per-engine "$ROLLOUT_NUM_GPUS_PER_ENGINE"
)
if [[ "${TRAINING_MODEL_TYPE}" == "moe" ]]; then
TRAINING_ARGS+=(
    "${COLOCATE_ARGS[@]}"
)
fi
TRAINING_ARGS+=(
    "${MODEL_ARGS[@]}"
    --hf-checkpoint "$HF_CHECKPOINT"
    --ref-load "$REF_LOAD"
    --load "$LOAD_DIR"
)
if [[ "${TRAINING_MODEL_TYPE}" == "moe" ]]; then
TRAINING_ARGS+=(
    "${SAVE_ARGS[@]}"
)
else
TRAINING_ARGS+=(
    --save "$SAVE_DIR"
    --save-interval "${SAVE_INTERVAL:-10}"
)
fi
TRAINING_ARGS+=(
    "${SAVE_HF_ARGS[@]}"
    --update-weights-interval 1
)
if [[ "${TRAINING_MODEL_TYPE}" == "moe" ]]; then
TRAINING_ARGS+=(
    --update-weight-buffer-size "${UPDATE_WEIGHT_BUFFER_SIZE:-536870912}"
)
fi
TRAINING_ARGS+=(
    --rollout-function-path slime_bridge.rollout.generate_rollout_hero_async
    --custom-rm-path slime_bridge.reward.reward_func
    --custom-reward-post-process-path "${CUSTOM_REWARD_POST_PROCESS_PATH:-slime_bridge.reward_post_process.post_process_rewards}"
    "${CUSTOM_ADVANTAGE_ARGS[@]}"
    "${CUSTOM_ROLLOUT_LOG_ARGS[@]}"
    --custom-config-path "${CUSTOM_CONFIG_PATH}"
    --data-source-path slime_bridge.data_source.CeilEpochRolloutDataSourceWithBuffer
    --prompt-data "$PROMPT_DATA"
    --input-key prompt
    --label-key label
    --metadata-key metadata
    --rollout-shuffle
    --reward-key score
    "${ROLLOUT_COUNT_ARGS[@]}"
    --rollout-batch-size "$ROLLOUT_BATCH_SIZE"
    --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT"
    --rollout-max-response-len 16000
    --rollout-max-prompt-len 32000
    --dynamic-history
    --num-steps-per-rollout 1
    "${LOSS_NORMALIZATION_ARGS[@]}"
    --tensor-model-parallel-size "$TENSOR_MODEL_PARALLEL_SIZE"
    --sequence-parallel
)
if [[ "${TRAINING_MODEL_TYPE}" == "moe" ]]; then
TRAINING_ARGS+=(
    --pipeline-model-parallel-size "$PIPELINE_MODEL_PARALLEL_SIZE"
)
else
TRAINING_ARGS+=(
    --pipeline-model-parallel-size 1
)
fi
TRAINING_ARGS+=(
    --context-parallel-size "$CONTEXT_PARALLEL_SIZE"
)
if [[ "${TRAINING_MODEL_TYPE}" == "moe" ]]; then
TRAINING_ARGS+=(
    --expert-model-parallel-size "$EXPERT_MODEL_PARALLEL_SIZE"
    --expert-tensor-parallel-size "$EXPERT_TENSOR_PARALLEL_SIZE"
)
else
TRAINING_ARGS+=(
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
)
fi
TRAINING_ARGS+=(
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU"
    --log-probs-chunk-size 256
    --distributed-timeout-minutes 30
    --advantage-estimator grpo
    "${NORMALIZE_ADVANTAGE_ARGS[@]}"
    --use-tis
    --use-kl-loss
    --kl-loss-coef 0.001
    --kl-loss-type low_var_kl
    --entropy-coef 0.0
    --eps-clip 0.2
    --eps-clip-high 0.28
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)
if [[ "${TRAINING_MODEL_TYPE}" == "moe" ]]; then
TRAINING_ARGS+=(
    "${OPTIMIZER_OFFLOAD_ARGS[@]}"
)
fi
TRAINING_ARGS+=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend "$ATTENTION_BACKEND"
    --no-gradient-accumulation-fusion
    --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC"
    --sglang-context-length "$SGLANG_CONTEXT_LENGTH"
)
if [[ "${TRAINING_MODEL_TYPE}" == "moe" ]]; then
TRAINING_ARGS+=(
    --sglang-enable-dp-attention
    --sglang-dp-size "${SGLANG_DP_SIZE:-8}"
    --sglang-ep-size "${SGLANG_EP_SIZE:-8}"
    --sglang-enable-dp-lm-head
    --sglang-moe-dense-tp-size "${SGLANG_MOE_DENSE_TP_SIZE:-1}"
)
fi
TRAINING_ARGS+=(
    --sglang-tool-call-parser qwen3_coder
    --router-policy "${SGLANG_ROUTER_POLICY:-round_robin}"
    --use-wandb
    --wandb-project "${WANDB_PROJECT:-hero}"
    --wandb-group "${WANDB_GROUP:-swegym-qwen35-4b-async-grpo}"
    --sglang-router-port "$SGLANG_ROUTER_PORT"
)
"${RAY_JOB_SUBMIT[@]}" "${TRAINING_ARGS[@]}"
# END TRAINING_SUBMISSION

# The official blocking submission is already complete here. Local-Docker
# submissions are non-blocking, so monitor job state independently of the
# convenience WebSocket log stream.
if [[ "${LOCAL_DOCKER_MODE}" == "1" ]]; then
# Follow logs only as a convenience.  The Ray CLI uses a WebSocket for this
# stream, and that connection can close cleanly while the submitted job is
# still RUNNING.  Keep job lifetime tracking independent from the log stream;
# otherwise an innocent WebSocket disconnect reaches the EXIT trap above and
# tears down the live Ray cluster.
"${RAY_CLI[@]}" job logs --address="${RAY_JOB_ADDRESS}" \
    --follow "${RAY_JOB_SUBMISSION_ID}" &
RAY_JOB_LOG_PID=$!
PIDS+=("${RAY_JOB_LOG_PID}")

RAY_JOB_POLL_INTERVAL_SEC="${RAY_JOB_POLL_INTERVAL_SEC:-10}"
RAY_JOB_STATUS_MAX_CONSECUTIVE_FAILURES="${RAY_JOB_STATUS_MAX_CONSECUTIVE_FAILURES:-12}"
[[ "${RAY_JOB_POLL_INTERVAL_SEC}" =~ ^[1-9][0-9]*$ ]] || {
    echo "RAY_JOB_POLL_INTERVAL_SEC must be a positive integer" >&2
    exit 2
}
[[ "${RAY_JOB_STATUS_MAX_CONSECUTIVE_FAILURES}" =~ ^[1-9][0-9]*$ ]] || {
    echo "RAY_JOB_STATUS_MAX_CONSECUTIVE_FAILURES must be a positive integer" >&2
    exit 2
}

if ! ray_job_poll; then
    exit 1
fi
fi


exit $?
}
