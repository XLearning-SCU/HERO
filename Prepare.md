<h1 align="center">Preparing HERO</h1>

<p align="center">
  <a href="README.md">Overview</a> ·
  <a href="#environment">Environment</a> ·
  <a href="#models">Models</a> ·
  <a href="#training">Training</a> ·
  <a href="#evaluation">Evaluation</a>
</p>

Run all commands from the repository root. Follow steps 1–4, then choose
training or evaluation. Training starts from Qwen3.5; evaluation can use our
released HERO checkpoints.

## Prerequisites

- Linux x86-64; the host package helper targets Ubuntu 24.04.
- Python 3.12, CUDA 13.0 toolkit, and a compatible NVIDIA driver. Install CUDA
  using [NVIDIA's guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).
- Recommended: **300 GiB or more available host RAM per node**, with Docker
  `overlay2` on local SSD/NVMe.
- Our server: **1,500 GiB RAM, ~3.4 TiB local NVMe storage, and 192 logical CPUs**.
- Default training: 8 GPUs for 4B/9B; 8 nodes × 8 GPUs for 35B-A3B.
  GPU memory must accommodate the model and the configured context length.

```text
workspace/
├── HERO/                 # This repository
├── data/                 # Extracted task JSONL files
├── docker/               # Task image archives
├── models/               # HF and converted Megatron weights
├── cli/claude_code/       # Agent CLI and Node.js
├── harnesses/            # External grading sources
└── results/              # Training and evaluation outputs
```

<a id="environment"></a>

## 1. Prepare the Host

Install host utilities (use `sudo` for the two `apt-get` commands when needed):

```bash
apt-get update
apt-get install -y git git-lfs curl xz-utils build-essential python3-pip python3-venv
git lfs install
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
nvidia-smi
nvcc --version
PYTHON_BIN="$(uv python find 3.12)" bash run_rl/_shared/init_env/pre_env.sh
export DOCKER_HOST="${HERO_LOCAL_DOCKER_HOST:-tcp://127.0.0.1:2377}"
docker info --format 'driver={{.Driver}} root={{.DockerRootDir}}'
```

`pre_env.sh` prepares a dedicated Docker daemon and requires administrator
privileges when installing/configuring Docker. It does not install CUDA or the
NVIDIA driver. Docker data defaults to `${XDG_CACHE_HOME:-$HOME/.cache}/hero/docker-data`;
use a local filesystem, not shared/FUSE storage.

To reuse an existing daemon, export its `HERO_LOCAL_DOCKER_HOST` and actual
`HERO_LOCAL_DOCKER_DATA_ROOT`, plus `HERO_LOCAL_DOCKER_ALLOW_CUSTOM_DATA_ROOT=1`
and `HERO_LOCAL_DOCKER_AUTO_START=0`, before setup. Retain these exports for runs.

<a id="data-and-images"></a>

## 2. Prepare Data and Task Images

```bash
git lfs pull --include="data.tar.gz"
tar -xzf data.tar.gz -C ..
python3 scripts/prepare_images.py --dataset train --download
# Download the evaluation suites you need:
python3 scripts/prepare_images.py --dataset verified --download
python3 scripts/prepare_images.py --dataset multilingual --download
```

| Data file | Tasks | Image archives |
| --- | ---: | --- |
| `swe_train_640.jsonl` | 352 SWE-Gym + 238 Multi-SWE + 50 SWE-rebench | `../docker/swe_train_640/` |
| `swe_bench_verified.jsonl` | 500 | `../docker/swe-bench-verified/` |
| `swe_bench_multilingual.jsonl` | 300 | `../docker/swe-bench-multilingual/` |

The helper reads image names from the supplied JSONL, runs `docker pull` and
`docker save`, and reuses existing archives. Images come from `xingyaoww/`,
`mswebench/`, `swerebenchv2/`, and `swebench/`; no manual image build is needed.
The runtime loads archives when starting tasks. Remove `--download` to check
archive availability. Training images are also checked by the environment
installer, so prepare them even when using that installer for evaluation only.
See the extracted `../data/README.md` for data sources and licenses.

## 3. Install the Environment and Agent

```bash
bash run_rl/_shared/init_env/install_env.sh
source .venv/bin/activate
unset PYTHONPATH
hero --help
```

The script installs the bundled Slime and Megatron-LM, Torch 2.11.0/CUDA 13.0,
SGLang 0.5.13, training kernels, and conversion dependencies, then applies the
bundled compatibility patches. Shared dependencies are constrained, and Transformer
Engine is compiled against the installed Torch on first setup. It also downloads
Claude Code 2.1.111 and Node.js
22.11.0 into `../cli/claude_code/`. No separate ProRL installation is required.
Network access to PyPI, the PyTorch index, GitHub, nodejs.org, and npm is needed.

The tested training stack uses `flash-attn==2.7.4.post1`. The installer installs it
only on SM100 GPUs; on other GPUs, install it in the training venv if missing
or incompatible. Skip this if that version is already installed and imports successfully.
With the venv activated and the CUDA toolkit available:

```bash
python -c 'from importlib.metadata import version; print(version("flash-attn"))'
# Only if installation is needed; build against the installed Torch:
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
uv pip install --python .venv/bin/python ninja packaging setuptools wheel
FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=4 \
  uv --no-cache pip install --python .venv/bin/python --no-build-isolation --no-deps \
  --no-binary flash-attn "flash-attn==2.7.4.post1"
python -c 'import torch, flash_attn, flash_attn_2_cuda; print(flash_attn.__version__)'
```

This source-build recipe has not been validated from scratch on every GPU/CUDA combination.

Claude Code runs inside task containers and calls local SGLang through HERO;
this workflow does not need a personal Anthropic API key.

<a id="harnesses"></a>

### Grading harnesses

| Task family | Grading implementation | Installation |
| --- | --- | --- |
| SWE-Gym training | [SWE-Gym/SWE-Bench-Package](https://github.com/SWE-Gym/SWE-Bench-Package) | Automatically downloaded, patched, and installed |
| SWE-bench Verified / Multilingual evaluation | [SWE-bench/SWE-bench](https://github.com/SWE-bench/SWE-bench) | Automatically downloaded and patched under `../harnesses/` |
| Multi-SWE training | Bundled adapter + each row's `eval_script` | Included in the code, data, and task images |
| SWE-rebench training | Bundled adapter + each row's `test_patch` and `eval_script` | Included in the code, data, and task images |

External harness revisions are pinned in `scripts/prepare_harnesses.py`; no
manual version selection is needed. Multi-SWE training is distinct from
SWE-bench Multilingual evaluation. Check the prepared resources with:

```bash
python scripts/prepare_harnesses.py --check
../cli/claude_code/bin/node --version
PATH="$PWD/../cli/claude_code/bin:$PATH" ../cli/claude_code/bin/claude --version
```

<a id="models"></a>

## 4. Download Model Weights

```bash
uv tool install --python 3.12 huggingface_hub
export PATH="$HOME/.local/bin:$PATH"
```

### Qwen3.5 for training

| Size | Download | Local directory |
| --- | --- | --- |
| `4b` | [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | `../models/Qwen3.5-4B/` |
| `35b` | [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | `../models/Qwen3.5-35B-A3B/` |

Choose one size:

```bash
size=4b
case "$size" in
  4b) model=Qwen3.5-4B ;;
  35b) model=Qwen3.5-35B-A3B ;;
esac
hf download "Qwen/$model" --local-dir "../models/$model"
```

<a id="model-conversion"></a>

Training also requires Megatron weights. Convert once on an available GPU in
the same shell; skip this for evaluation:

```bash
# Match the launchers' CUDA library lookup for standalone conversion.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export NVTE_CUDA_INCLUDE_DIR="${NVTE_CUDA_INCLUDE_DIR:-$CUDA_HOME/include}"
for library in nvshmem nccl cudnn cu13; do
  library_dir="$(python -c "import nvidia.${library}; print(next(iter(nvidia.${library}.__path__)) + '/lib')" 2>/dev/null || true)"
  if [ ! -d "$library_dir" ]; then
    library_dir="$(/usr/bin/python3 -c "import nvidia.${library}; print(next(iter(nvidia.${library}.__path__)) + '/lib')" 2>/dev/null || true)"
  fi
  if [ -d "$library_dir" ]; then export LD_LIBRARY_PATH="$library_dir:${LD_LIBRARY_PATH:-}"; fi
done
TRAINING_MODEL_SIZE="$size" HF_CHECKPOINT="../models/$model" \
  TORCH_DIST_DIR="../models/${model}_torch_dist" \
  bash examples/swegym_slime_grpo/convert_weights.sh
```

### HERO for evaluation

| Model | Download |
| --- | --- |
| HERO-4B | [XLearning-SCU/HERO-4B](https://huggingface.co/XLearning-SCU/HERO-4B) |
| HERO-9B | [XLearning-SCU/HERO-9B](https://huggingface.co/XLearning-SCU/HERO-9B) |
| HERO-35B-A3B | [XLearning-SCU/HERO-35B-A3B](https://huggingface.co/XLearning-SCU/HERO-35B-A3B) |

```bash
model=HERO-4B  # Or HERO-9B / HERO-35B-A3B
hf download "XLearning-SCU/$model" --local-dir "../models/$model"
```

Keep the tokenizer and configuration files with the weights. The released
[HERO evaluation trajectories](https://huggingface.co/datasets/XLearning-SCU/HERO-Trajectories)
are optional reference artifacts, not runtime dependencies.

<a id="training"></a>
<a id="wandb"></a>

## 5. Train

```bash
source .venv/bin/activate
unset PYTHONPATH
size=4b  # Or 35b
TRAINING_MODEL_SIZE="$size" bash run_rl/_shared/init_env/preflight.sh
WANDB_MODE=disabled bash "run_rl/train_${size}.sh"
```

The launchers read the matching Qwen3.5 HF and `*_torch_dist` directories.
Use `HF_CHECKPOINT` and `REF_LOAD` to override them. Defaults are in the launch
scripts. For W&B, run `wandb login` and use `WANDB_MODE=online WANDB_PROJECT=hero`.

For 35B-A3B, prepare identical environment/resource paths on all eight nodes,
with passwordless SSH as `root` from the head to workers. Put one node IP per
line in a repository-root `hostfile`, including the head. Set `RAY_HEAD_IP` to
the head's reachable IP and run `run_rl/train_35b.sh` there.

<a id="evaluation"></a>

## 6. Evaluate

After downloading the HERO weights and evaluation images, start with one task:

```bash
source .venv/bin/activate
unset PYTHONPATH
MODEL=../models/HERO-4B \
  NUM_SHARDS=1 SGLANG_GPU_IDS=0 CONCURRENCY=1 GRADE_CONCURRENCY=1 LIMIT=1 \
  bash eval/run_eval_swebench_verified.sh
```

Check the output's `eval_result.json`, `results.jsonl`, and per-task logs, then
run a full benchmark:

```bash
MODEL=../models/HERO-4B bash eval/run_eval_swebench_verified.sh
MODEL=../models/HERO-4B bash eval/run_eval_swebench_multilingual.sh
```

Change `MODEL` for HERO-9B, HERO-35B-A3B, or your own exported HF checkpoint.
Evaluation uses a 131,072-token context and up to 200 agent turns. For your GPU
allocation, set `NUM_SHARDS` (replicas), `SGLANG_TP_SIZE` (GPUs per replica), and
`SGLANG_GPU_IDS` (a comma-separated list of exactly their product in GPU IDs).

<a id="outputs"></a>

## Outputs

- Training: `../results/train/<method>/<model>/<run>/`, with checkpoints in `ckpt/`.
- Evaluation: `../results/eval/<dataset>/<model>/<run>/`, with results and task logs.
- Override `RESULTS_ROOT`, or set `RUN_ROOT` / `OUT` for an individual training /
  evaluation run. Reuse `OUT` to resume the same evaluation protocol.

Preflight checks resources and imports; it does not execute training or an agent
task. Validate one task on your machine before starting a full evaluation.
