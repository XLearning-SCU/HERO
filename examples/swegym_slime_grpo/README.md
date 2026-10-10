# Shared training support

Public training entries live under [run_rl](../../run_rl/README.md). Use those
entries to select the model, reward hooks, and local Docker configuration.

- `run.sh`: shared Dense/MoE backend; retains model-specific Ray polling,
  multi-node setup, offload, and training options.
- `model_args.sh`: 4B, 9B, and 35B-A3B presets selected by
  `TRAINING_MODEL_SIZE=4b|9b|35b`. Public launchers set this automatically.
- `convert_weights.sh`: checkpoint conversion with the same model selector.
- `moe_compat.py`: Megatron argument compatibility for 35B-A3B training and
  conversion (`--convert`); Dense training does not use this bridge.
- `topology.yaml`, `hero_config.yaml`: fallback Apptainer templates;
  public local Docker entries supply `run_rl/_shared/config/` instead.

Training uses the sibling `../data/swe_train_640.jsonl`. See the
[data and image layout](../../README.md#data-and-docker-images).

Convert a local Hugging Face checkpoint (choose `4b`, `9b`, or `35b`):

```bash
TRAINING_MODEL_SIZE=9b HF_CHECKPOINT=/path/to/Qwen3.5-9B \
  TORCH_DIST_DIR=/path/to/Qwen3.5-9B_torch_dist \
  PYTHON_BIN=/path/to/venv/bin/python3 \
  bash examples/swegym_slime_grpo/convert_weights.sh
```

Conversion requires the installed Slime/Megatron and CUDA dependencies.
Checkpoints, container images, and agent CLIs are external inputs.
The Apptainer fallback requires prebuilt images and an agent CLI directory.
