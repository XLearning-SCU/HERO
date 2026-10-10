#!/usr/bin/env bash
# Convert 4B, 9B or 35B-A3B weights; model-specific conversion behavior is retained.
set -euo pipefail
TRAINING_MODEL_SIZE="${TRAINING_MODEL_SIZE:-4b}"
case "${TRAINING_MODEL_SIZE}" in
35b)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/framework/slime}"
MEGATRON_DIR="${MEGATRON_DIR:-${PROJECT_ROOT}/framework/Megatron-LM}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python3}"
HF_CHECKPOINT="${HF_CHECKPOINT:-${PROJECT_ROOT}/../models/Qwen3.5-35B-A3B}"
TORCH_DIST_DIR="${TORCH_DIST_DIR:-${HF_CHECKPOINT}_torch_dist}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python: ${PYTHON_BIN}" >&2; exit 1; }
[[ -f "${HF_CHECKPOINT}/config.json" ]] || { echo "Missing HF checkpoint: ${HF_CHECKPOINT}" >&2; exit 1; }
UPSTREAM_CONVERTER="${SLIME_DIR}/tools/convert_hf_to_torch_dist.py"
MOE_CONVERTER="${SCRIPT_DIR}/moe_compat.py"
[[ -f "${UPSTREAM_CONVERTER}" ]] || { echo "Missing Slime converter under ${SLIME_DIR}" >&2; exit 1; }
[[ -f "${MOE_CONVERTER}" ]] || { echo "Missing MoE converter wrapper: ${MOE_CONVERTER}" >&2; exit 1; }
mkdir -p "${TORCH_DIST_DIR}"

# Match the training launcher's CUDA-13 runtime setup. The project venv may
# expose empty nvidia.* namespace packages, so fall back to the system Python
# packages that contain the actual NVSHMEM/NCCL/cuDNN/cu13 shared libraries.
_nvidia_lib() {
    local _name="$1"
    local _dir
    _dir="$("${PYTHON_BIN}" -c "import nvidia.${_name},os;print(os.path.join(list(nvidia.${_name}.__path__)[0],'lib'))" 2>/dev/null || true)"
    if [[ -z "${_dir}" || ! -d "${_dir}" ]]; then
        _dir="$(/usr/bin/python3 -c "import nvidia.${_name},os;print(os.path.join(list(nvidia.${_name}.__path__)[0],'lib'))" 2>/dev/null || true)"
    fi
    printf '%s' "${_dir}"
}
_cuda_lib_prepend=""
for _lib in nvshmem nccl cudnn cu13; do
    _lib_dir="$(_nvidia_lib "${_lib}")"
    if [[ -n "${_lib_dir}" && -d "${_lib_dir}" ]]; then
        _cuda_lib_prepend="${_lib_dir}:${_cuda_lib_prepend}"
    fi
done
[[ -n "${_cuda_lib_prepend}" ]] || {
    echo "Could not locate CUDA-13 nvidia.* runtime libraries" >&2
    exit 1
}
export LD_LIBRARY_PATH="${_cuda_lib_prepend}${LD_LIBRARY_PATH:-}"
export NVTE_CUDA_INCLUDE_DIR="${NVTE_CUDA_INCLUDE_DIR:-/usr/local/cuda/include}"
"${PYTHON_BIN}" -c 'import torch; print(f"PyTorch ready: {torch.__version__}")'

source "${SLIME_DIR}/scripts/models/qwen3.5-35B-A3B.sh"

# The current Megatron checkout implements Qwen3.5 gating through
# --attention-output-gate and --moe-shared-expert-gate.  Slime's model preset
# also carries this legacy compatibility switch, but the strict converter
# parser does not define it (the training parser ignores unknown arguments).
_converter_model_args=()
for _arg in "${MODEL_ARGS[@]}"; do
    [[ "${_arg}" == "--use-gated-attention" ]] || _converter_model_args+=("${_arg}")
done

echo "Converting ${HF_CHECKPOINT} -> ${TORCH_DIST_DIR}"
CUDA_DEVICE_MAX_CONNECTIONS=1 \
SLIME_UPSTREAM_CONVERTER="${UPSTREAM_CONVERTER}" \
PYTHONPATH="${MEGATRON_DIR}:${SLIME_DIR}:${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
"${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node 1 \
    "${MOE_CONVERTER}" --convert \
    "${_converter_model_args[@]}" \
    --hf-checkpoint "${HF_CHECKPOINT}" \
    --save "${TORCH_DIST_DIR}" \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --no-gradient-accumulation-fusion

echo "Done: ${TORCH_DIST_DIR}"

;;
4b|9b)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/framework/slime}"
MEGATRON_DIR="${MEGATRON_DIR:-${PROJECT_ROOT}/framework/Megatron-LM}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python3}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi
PYTHON_BIN_DIR="$(cd -- "$(dirname -- "${PYTHON_BIN}")" &>/dev/null && pwd)"
export PATH="${PYTHON_BIN_DIR}:${PATH}"

if [ ! -f "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" ]; then
    echo "ERROR: Slime not found at ${SLIME_DIR}. Clone it first:"
    echo "  git clone git@github.com:THUDM/slime.git ${SLIME_DIR}"
    exit 1
fi

MODEL_SIZE_LABEL="${TRAINING_MODEL_SIZE^^}"
HF_CHECKPOINT="${HF_CHECKPOINT:-Qwen/Qwen3.5-${MODEL_SIZE_LABEL}}"
OUTPUT_DIR="${TORCH_DIST_DIR:-${PROJECT_ROOT}/tmp/checkpoints/Qwen3.5-${MODEL_SIZE_LABEL}_torch_dist}"
mkdir -p "$OUTPUT_DIR"

# shellcheck source=./model_args.sh
source "${SCRIPT_DIR}/model_args.sh"

echo "Converting ${HF_CHECKPOINT} -> ${OUTPUT_DIR}"

CUDA_DEVICE_MAX_CONNECTIONS=1 \
PYTHONPATH="${MEGATRON_DIR}:${SLIME_DIR}:${PROJECT_ROOT}/src" \
torchrun --nproc_per_node 1 \
    "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$HF_CHECKPOINT" \
    --save "$OUTPUT_DIR" \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --no-gradient-accumulation-fusion

echo "Done: ${OUTPUT_DIR}"

;;
*) echo "TRAINING_MODEL_SIZE must be 4b, 9b or 35b" >&2; exit 2 ;;
esac
