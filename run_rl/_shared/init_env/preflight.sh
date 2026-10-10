#!/usr/bin/env bash
# Readiness check. Does not start training or load Docker images.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/framework/slime}"
MEGATRON_DIR="${MEGATRON_DIR:-${PROJECT_ROOT}/framework/Megatron-LM}"
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/../models}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python3}"
HERO_BIN="${HERO_BIN:-$(dirname "${PYTHON_BIN}")/hero}"
AGENT_CLI_DIR="${AGENT_CLI_DIR:-${PROJECT_ROOT}/../cli/claude_code}"
source "${PROJECT_ROOT}/run_rl/_shared/runtime/docker_storage_guard.sh"

check_file() {
  [[ -e "$1" ]] || { echo "[missing] $1" >&2; return 1; }
  echo "[ok] $1"
}

check_file "${PYTHON_BIN}"
check_file "${HERO_BIN}"
check_file "${SLIME_DIR}/train_async.py"
check_file "${MEGATRON_DIR}/megatron"
case "${TRAINING_MODEL_SIZE:-4b}" in
  4b) MODEL_NAME=Qwen3.5-4B ;;
  9b) MODEL_NAME=Qwen3.5-9B ;;
  35b) MODEL_NAME=Qwen3.5-35B-A3B ;;
  *) echo "Unknown TRAINING_MODEL_SIZE" >&2; exit 2 ;;
esac
check_file "${HF_CHECKPOINT:-${MODEL_ROOT}/${MODEL_NAME}}/config.json"
check_file "${REF_LOAD:-${MODEL_ROOT}/${MODEL_NAME}_torch_dist}/latest_checkpointed_iteration.txt"
check_file "${AGENT_CLI_DIR}/bin/claude"
check_file "${PROJECT_ROOT}/../data/swe_train_640.jsonl"

export DOCKER_HOST="${HERO_LOCAL_DOCKER_HOST:-${DOCKER_HOST:-tcp://127.0.0.1:2377}}"
export HERO_LOCAL_DOCKER_DATA_ROOT="${HERO_LOCAL_DOCKER_DATA_ROOT:-${XDG_CACHE_HOME:-${HOME}/.cache}/hero/docker-data}"
# A readiness check must not start/reconfigure Docker or alter host cgroups.
hero_validate_docker_data_root "${HERO_LOCAL_DOCKER_DATA_ROOT}"
_DOCKER_INFO="$(docker info --format '{{json .}}')"
"${PYTHON_BIN}" - "${_DOCKER_INFO}" <<'PYDOCKER'
import json, os, sys
info = json.loads(sys.argv[1])
expected_root = os.environ["HERO_LOCAL_DOCKER_DATA_ROOT"]
expected_driver = os.environ.get("HERO_LOCAL_DOCKER_STORAGE_DRIVER", "overlay2")
for field, expected in (("DockerRootDir", expected_root), ("Driver", expected_driver)):
    actual = info.get(field)
    if actual != expected:
        raise SystemExit(f"[mismatch] Docker {field}: expected {expected}, got {actual}")
print(f"[ok] Docker: {expected_driver}, {expected_root}")
PYDOCKER

# Mirror the training launcher's NVIDIA library probe: the venv
# ships nvidia.{nvshmem,nccl} as empty namespace stubs whose lib/ dir does not
# exist, so torch import-time dlopen of libnvshmem_host.so.3 fails unless we
# prepend the real (system) nvidia/*/lib dirs. Without this, `import sglang`
# (-> import torch) crashes on a fresh shell that has no LD_LIBRARY_PATH, even
# though the real training launcher sets it. Fall back to the system
# interpreter (/usr/local/lib/python3.12/dist-packages/nvidia/*) when the venv
# stub is empty, exactly like the launcher does.
_nvidia_lib() {
  local _name="$1" _d
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

export NVTE_CUDA_INCLUDE_DIR="${NVTE_CUDA_INCLUDE_DIR:-/usr/local/cuda/include}"
PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${SLIME_DIR}:${MEGATRON_DIR}" "${PYTHON_BIN}" - <<'PY'
import hero
import slime_bridge
import sglang
import slime
import megatron.core
from run_rl._shared.runtime.local_docker_runtime import ManagedDockerRuntime
print("[ok] Python imports: hero slime_bridge slime megatron.core sglang ManagedDockerRuntime")
PY

"${PYTHON_BIN}" "${SCRIPT_DIR}/../validate_training_data.py" --input "${PROJECT_ROOT}/../data/swe_train_640.jsonl" --check-images
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/prepare_harnesses.py" --check
echo "Preflight passed. No Docker image was downloaded or loaded."
