# Slime Bridge

`slime_bridge` connects [Slime](https://github.com/THUDM/slime)'s RL training
loop to a running HERO rollout server over HTTP. It lives **outside** the
`hero` package because Slime, Ray, Megatron, and torch are installed separately
— HERO depends on none of them.

## How it fits

Slime calls one entry point, `generate_rollout_hero_async`, wired in via
`--rollout-function-path`. From there the bridge:

- submits async task batches to `hero_rollout_url` (or a node derived from
  `hero_topology_path`) and collects each result through a local callback
  listener with a polling safety net;
- tracks rollout ids and policy versions, stamps HERO scheduler metadata
  (`group_id`, `policy_version`, `rollout_step`) onto every task, and keeps
  async admission bounded to the current Slime rollout request;
- converts each HERO `Trajectory` back into Slime `Sample`s (one per trace,
  mapping Slime 0.3.0 `group_id` to matching current `rollout_id` and
  `trajectory_id` partitions so siblings stay in one step and count once),
  dropping empty or oversized traces;
- computes dynamic-trace leave-one-trajectory-out advantages and zeroes out
  failed/aborted trajectories.

## Main files

- `config.py`: `HeroSlimeConfig` + `resolve_hero_slime_config`; also renders the
  task payload, the instruction, and the topology that points gateways at Slime's
  SGLang router.
- `rollout.py`: the async worker (submit → callback/poll → convert), the
  evaluation path, the acceptance filters, and the Slime entry point.
- `_messages.py`: prompt/message flattening shared by rollout + adapter.
- `adapter.py`: convert a HERO `SessionResult` into Slime `Sample`s.
- `data_source.py`: `CeilEpochRolloutDataSourceWithBuffer` — rounds the epoch
  length up so the dataset tail isn't skipped.
- `reward.py`: reward hook that reads the reward HERO already embedded.
- `reward_post_process.py`: trajectory-aware, group-normalized reward shaping.

## What the bridge owns

- Turn Slime samples + prompts into HERO task requests and submit async batches.
- Track rollout ids / policy versions and bound async admission to the current
  Slime rollout request.
- Filter unusable groups (zero trainable tokens, too few completed samples,
  logprob errors) with per-category metrics.
- Convert HERO trajectories back into Slime samples; compute dynamic-trace
  advantages.
- Run the evaluation path over `eval_datasets` and emit W&B metrics.

## Slime installation

HERO uses the bundled `framework/slime` and `framework/Megatron-LM` sources.
Install these checkouts into the same training environment as HERO; replacing
them with an arbitrary upstream version may lose required compatibility changes.
From the repository root, the full environment setup entry is:

```bash
bash run_rl/_shared/init_env/install_env.sh
```

Review the CUDA, Python, and external resource settings in that script before
running it on a new machine. The editable package installation steps are:

```bash
uv pip install -e '.[swebench]'
uv pip install -e framework/slime
uv pip install -e framework/Megatron-LM
```

These three commands alone do not install every training/CUDA dependency;
`install_env.sh` contains the version-specific setup. `SLIME_DIR` and
`MEGATRON_DIR` can select compatible external checkouts for the setup script.
The legacy router-token patch is opt-in there; it should not be blindly applied
to the bundled Slime checkout. HERO's basic service dependencies remain separate
from the heavy training dependencies.

See [HERO preparation](../../Prepare.md) for the complete supported setup and
image downloads. The service executable installed by this release is `hero`.
