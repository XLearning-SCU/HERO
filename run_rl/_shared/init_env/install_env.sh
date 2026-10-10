#!/usr/bin/env bash
# Install HERO's Python/training environment and the pinned Claude Code CLI.
# Local SWE Docker images are only verified; this script never downloads them.
set -euo pipefail

# pre_env.sh installs uv here for unprivileged users. Exports in that script do
# not persist when it is invoked as `bash pre_env.sh`, so make this standalone.
export PATH="${HOME}/.local/bin:${PATH}"
# This host exports a system site-packages directory through PYTHONPATH.  It
# would shadow packages installed in this project venv (notably sglang), so the
# installer must run with the venv's normal import precedence.
unset PYTHONPATH

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/framework/slime}"
MEGATRON_DIR="${MEGATRON_DIR:-${PROJECT_ROOT}/framework/Megatron-LM}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python3}"
SWEGYM_PACKAGE_SPEC="${SWEGYM_PACKAGE_SPEC:-${PROJECT_ROOT}/../harnesses/SWE-Bench-Package}"
PYTORCH_CUDA_INDEX="${PYTORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu130}"
# Override pyproject.toml's auto detection: it can select cu12 wheels even when
# the configured index and this training stack explicitly require CUDA 13.
export NVTE_CUDA_INCLUDE_DIR="${NVTE_CUDA_INCLUDE_DIR:-${CUDA_HOME:-/usr/local/cuda}/include}"
export UV_TORCH_BACKEND=cu130
export UV_CONSTRAINT="${SCRIPT_DIR}/constraints.txt"

command -v uv >/dev/null || { echo "uv is required" >&2; exit 1; }
[[ -f "${SLIME_DIR}/train_async.py" ]] || { echo "Slime missing: ${SLIME_DIR}" >&2; exit 1; }
[[ -d "${MEGATRON_DIR}/megatron" ]] || { echo "Megatron-LM missing: ${MEGATRON_DIR}" >&2; exit 1; }

if [[ ! -x "${PYTHON_BIN}" ]]; then
  uv venv --python 3.12 "${PROJECT_ROOT}/.venv"
fi

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/prepare_harnesses.py"
# Do not let the default package index select the CPU-only torch wheel.  SGLang
# and Transformer Engine require a matching CUDA stack on this CUDA 13 host.
uv pip install --python "${PYTHON_BIN}" --index-url "${PYTORCH_CUDA_INDEX}" \
  "torch==2.11.0+cu130" "torchvision==0.26.0+cu130" "torchaudio==2.11.0+cu130"
uv pip install --python "${PYTHON_BIN}" -e "${PROJECT_ROOT}[swebench]"
uv pip install --python "${PYTHON_BIN}" -e "${SLIME_DIR}"
uv pip install --python "${PYTHON_BIN}" -e "${MEGATRON_DIR}"
uv pip install --python "${PYTHON_BIN}" "${SWEGYM_PACKAGE_SPEC}"
uv pip install --python "${PYTHON_BIN}" --no-deps "mbridge==0.15.1"
# Megatron's default nvrx async checkpoint strategy imports this package during
# checkpoint load/save.  The Slime compatibility patch can fall back to mcore
# if it is absent, but installing the declared runtime dependency preserves
# the normal nvrx path after a fresh venv rebuild.
uv pip install --python "${PYTHON_BIN}" "nvidia-resiliency-ext==0.6.0"
uv pip install --python "${PYTHON_BIN}" "sglang[http2]==0.5.13"

if [[ "${INSTALL_TRAINING_EXTRAS:-1}" == "1" ]]; then
  uv pip install --python "${PYTHON_BIN}" "flash-linear-attention==0.5.0"
  # TE 2.5 ships a CUDA 12 core even with a CUDA 13 Torch build. Its two
  # runtime libraries are not declared by the upstream wheel; install them.
  uv pip install --python "${PYTHON_BIN}" ninja pybind11 setuptools wheel \
    "transformer-engine==2.5.0" "nvidia-cublas-cu12==12.9.1.4" "nvidia-cuda-runtime-cu12==12.9.79"
  # Build the extension against this environment's Torch, not a newer Torch
  # chosen by build isolation. Discard cached wheels built against another ABI.
  if ! "${PYTHON_BIN}" -c 'import transformer_engine.pytorch; import transformer_engine; assert transformer_engine.__version__ == "2.5.0"' >/dev/null 2>&1; then
    MAX_JOBS="${TE_MAX_JOBS:-4}" NVTE_BUILD_THREADS_PER_JOB="${TE_BUILD_THREADS:-2}" \
      uv --no-cache pip install --python "${PYTHON_BIN}" --no-build-isolation --no-deps \
        --reinstall-package transformer-engine-torch --no-binary transformer-engine-torch \
        "transformer-engine-torch==2.5.0"
  fi
fi

# Keep the framework checkout pristine by default. The historical ProRL patch
# targets Slime v0.3.0 exactly and does not apply to the newer original checkout
# under framework/slime. Enable it explicitly only after pinning a compatible
# Slime revision.
if [[ "${APPLY_SLIME_ROUTER_PATCH:-0}" == "1" ]]; then
  SLIME_DIR="${SLIME_DIR}" bash "${PROJECT_ROOT}/scripts/patch/patch_slime_router_tokens.sh"
fi
# Run the SGLang patch with the venv Python first on PATH. The patch script
# does `import sglang` to detect the installed version; if the system Python
# shadows the venv, it imports a system sglang (e.g. a dev build) with a
# mismatched __version__ and the patch is silently skipped, which later breaks
# text-only rollout on the Qwen3.5 VLM checkpoint (HTTP 400, zero trainable
# tokens). PYTHONPATH is already unset above; pin PATH too.
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
bash "${PROJECT_ROOT}/scripts/patch/patch_sglang_0513_token_metadata.sh"
# Demote Slime's numpy 1.x hard assert so actor init survives numpy 2.x (the
# cu130 stack requires 2.x). Idempotent; safe after every Slime reinstall.
SLIME_DIR="${SLIME_DIR}" PYTHON_BIN="${PYTHON_BIN}" \
  bash "${SCRIPT_DIR}/patch_slime_numpy_assert.sh"
# Preserve v0.3 Sample(group_id=index) semantics by aligning current Slime's
# rollout scheduling ID and trajectory loss ID.
PYTHON_BIN="${PYTHON_BIN}" bash "${SCRIPT_DIR}/patch_slime_bridge_group_id.sh"
compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]' || true)"
if [[ "${compute_cap}" == "10.0" && "${BUILD_FLASH_ATTN_SM100:-1}" == "1" ]]; then
  uv pip install --python "${PYTHON_BIN}" ninja pybind11 setuptools wheel
  TORCH_CUDA_ARCH_LIST=10.0 FLASH_ATTN_CUDA_ARCHS=100 \
    FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS="${FA_MAX_JOBS:-32}" NVCC_THREADS=4 \
    uv pip install --python "${PYTHON_BIN}" --no-build-isolation \
      "flash-attn==2.7.4.post1"
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_agent_cli.py"
"${PYTHON_BIN}" "${SCRIPT_DIR}/../validate_training_data.py" --input "${PROJECT_ROOT}/../data/swe_train_640.jsonl" --check-images

# Installation is not complete until the critical native extensions import.
export PYTHONDONTWRITEBYTECODE=1
"${PYTHON_BIN}" -c 'import torch, sglang, hero, slime, swegym; print("Core imports passed:", torch.__version__, sglang.__version__)'
if [[ "${INSTALL_TRAINING_EXTRAS:-1}" == "1" ]]; then
  NVTE_CUDA_INCLUDE_DIR="${NVTE_CUDA_INCLUDE_DIR:-${CUDA_HOME:-/usr/local/cuda}/include}" \
    "${PYTHON_BIN}" -c 'import transformer_engine.pytorch, fla, megatron.core; print("Training imports passed")'
fi

echo "Environment ready: ${PYTHON_BIN}"
echo "Training image archives were verified through ../docker/."
echo "No SWE image was downloaded or loaded during installation."
