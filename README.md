<p align="center">
  <img src="assets/hero-logo.png" alt="HERO logo" width="760">
</p>

<p align="center">
  <strong>Doing More with Less Tokens: Hierarchical Reinforcement Learning for Efficient Coding Agents</strong>
</p>

<p align="center">
  <a href="Prepare.md"><img src="https://img.shields.io/badge/Setup-Preparation_Guide-2563eb.svg" alt="Preparation Guide"></a>
  <a href="https://arxiv.org/abs/2609.38885"><img src="https://img.shields.io/badge/arXiv-2609.38885-b31b1b.svg" alt="Paper"></a>
  <a href="https://huggingface.co/collections/XLearning-SCU/hero"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Models-f4c430.svg" alt="Hugging Face Models"></a>
  <a href="https://huggingface.co/datasets/XLearning-SCU/HERO-Trajectories"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Trajectories-f4c430.svg" alt="Hugging Face Trajectories"></a>
</p>

## 🔍 Overview

**HERO (HiErarchical ReinfOrcement learning)** trains coding agents to solve
software-engineering tasks efficiently while preserving resolution performance.
Following our [paper](https://arxiv.org/abs/2609.38885), it couples:

- **Hierarchical policy optimization:** prioritize task resolution and activate
  efficiency optimization once sufficient problem-solving capability is acquired.
- **Multi-granularity efficiency credit assignment:** encourage concise reasoning
  trajectories and penalize potentially unproductive turns using turn-level entropy.

This repository provides HERO training and evaluation on SWE-bench Verified
and SWE-bench Multilingual, with released HERO-4B, HERO-9B, and HERO-35B-A3B models.

## ⚡ Quick Start

Follow [Prepare.md](Prepare.md) to prepare the environment, task images, and model weights. Then launch HERO-4B training from the repository root:

```bash
source .venv/bin/activate
WANDB_MODE=disabled bash run_rl/train_4b.sh
```

For evaluation only, follow the [one-task evaluation setup](Prepare.md#evaluation).

## 🛠️ Setup

Use Linux x86-64, Python 3.12, CUDA 13.0, and local Docker storage.
Follow [Prepare.md](Prepare.md) in order; it covers every resource below.

| Resource | How to prepare it |
| --- | --- |
| Python/CUDA environment | [Host and environment setup](Prepare.md#environment); the installer prepares the Python stack, but NVIDIA drivers and CUDA must already be installed |
| Training and evaluation data | Extract the bundled Git LFS archive, `data.tar.gz`, into `../data/` |
| Task Docker images | [Image download commands](Prepare.md#data-and-images) read the registry names from the data and save archives under `../docker/` |
| Agent scaffold | Claude Code 2.1.111 + Node.js 22.11.0, downloaded automatically into `../cli/claude_code/` |
| Grading harnesses | [Harness table](Prepare.md#harnesses) explains SWE-Gym, SWE-bench, Multi-SWE, and SWE-rebench routing and installation |
| Model weights | [Model download table and conversion](Prepare.md#models); HF weights for evaluation, HF + Megatron weights for training |
| Training frameworks | Slime and Megatron-LM are bundled in `framework/` and installed by the environment script |

<a id="data-and-docker-images"></a>

## 🗂️ Data and Images

The dataset archive is stored with Git LFS; install Git LFS before cloning.
Extract the bundled dataset archive from the repository root into the sibling
`../data/` directory:

```bash
git lfs pull --include="data.tar.gz"
tar -xzf data.tar.gz -C ..
```

| File | Tasks | Use |
| --- | ---: | --- |
| `swe_train_640.jsonl` | 640 | Training: 352 SWE-Gym, 238 Multi-SWE, 50 SWE-ReBench |
| `swe_bench_verified.jsonl` | 500 | SWE-bench Verified evaluation |
| `swe_bench_multilingual.jsonl` | 300 | SWE-bench Multilingual evaluation |

The archive includes the three JSONL files and their source/license README.
[The image helper](Prepare.md#data-and-images) downloads their referenced
Docker images into `../docker/`.

<a id="training"></a>

## ▶️ Training

After completing [environment setup and weight conversion](Prepare.md), run from
the repository root:

```bash
source .venv/bin/activate
WANDB_MODE=disabled bash run_rl/train_4b.sh
```

Default GPU allocations by model size:

| Model | Default GPUs |
| --- | --- |
| Qwen3.5-4B | 1 node × 8 |
| Qwen3.5-9B | 1 node × 8 |
| Qwen3.5-35B-A3B | 8 nodes × 8 |

The launch scripts contain the default hyperparameters. For online logging,
run `wandb login` and use `WANDB_MODE=online WANDB_PROJECT=hero`.

The method implementation is in `run_rl/HERO/`; environment setup, model
recipes, and Docker execution live in `run_rl/_shared/`. Training and evaluation
do not depend on any optional baseline directory. See the
[method implementation](run_rl/HERO/README.md) for the advantage and loss hooks.

## 🤗 Released Models and Trajectories

| Model | Download |
| --- | --- |
| HERO-4B | [XLearning-SCU/HERO-4B](https://huggingface.co/XLearning-SCU/HERO-4B) |
| HERO-9B | [XLearning-SCU/HERO-9B](https://huggingface.co/XLearning-SCU/HERO-9B) |
| HERO-35B-A3B | [XLearning-SCU/HERO-35B-A3B](https://huggingface.co/XLearning-SCU/HERO-35B-A3B) |

Evaluation trajectories for SWE-bench Verified and SWE-bench Multilingual are
available at [HERO-Trajectories](https://huggingface.co/datasets/XLearning-SCU/HERO-Trajectories).
For training, start from the original [Qwen3.5 weights](Prepare.md#models).

<a id="evaluation"></a>

## 🧪 Evaluation

Download one of the HERO checkpoints above into `../models/HERO-4B/`, `../models/HERO-9B/`, or `../models/HERO-35B-A3B/`.
Then evaluate the selected model:

```bash
MODEL=../models/HERO-4B bash eval/run_eval_swebench_verified.sh
MODEL=../models/HERO-4B bash eval/run_eval_swebench_multilingual.sh
```

Set `MODEL` to the corresponding directory for HERO-9B or HERO-35B-A3B.
Evaluation uses a 131,072-token context and up to 200 agent turns. Start with the
[one-task command](Prepare.md#evaluation) to check your environment.

## 📂 Outputs

```text
../results/
├── train/HERO/<model>/<timestamp>_<pid>_<experiment>/
└── eval/<dataset>/<model>/<timestamp>_<pid>_<experiment>/
```

Training saves checkpoints, settings, and logs. Evaluation saves `results.jsonl`,
`eval_result.json`, and per-task artifacts. Set `RESULTS_ROOT` to change the output
root, or `RUN_ROOT` / `OUT` to select one training / evaluation run.

## 🙏 Acknowledgements

HERO builds on [ProRL-Agent-Server / Polar](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server), [Slime](https://github.com/THUDM/slime),
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM),
[SGLang](https://github.com/sgl-project/sglang), and the
[Qwen models](https://huggingface.co/Qwen). It uses tasks and grading resources
from SWE-Gym, Multi-SWE, SWE-ReBench, and SWE-bench.

## 📖 Citation

If you find HERO useful, please cite our [paper](https://arxiv.org/abs/2609.38885):

```bibtex
@misc{li2026hero,
  title         = {Doing More with Less Tokens: Hierarchical Reinforcement Learning for Efficient Coding Agents},
  author        = {Haobin Li and Liang Jiang and Zhenyu Huang and Mouxing Yang and Xi Peng},
  year          = {2026},
  eprint        = {2609.38885},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SE},
  url           = {https://arxiv.org/abs/2609.38885}
}
```

## 📄 License

HERO is distributed under the [Apache-2.0 License](LICENSE). Bundled frameworks
retain their license files; external datasets, models, and tools retain their
respective licenses and terms.
