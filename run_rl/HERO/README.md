# HERO: HiErarchical ReinfOrcement learning

HERO combines resolution and efficiency advantages with a fixed capability
threshold. This directory contains the method implementation; training entries
are in `run_rl/`.

## Entries and hyperparameters

| Model | Fixed-gate entry | Efficiency coefficient λ | Capability threshold τ |
|---|---|---|---|
| Qwen3.5-4B | [`run_rl/train_4b.sh`](../train_4b.sh) | 0.3 | 0.6 |
| Qwen3.5-9B | [`run_rl/train_9b.sh`](../train_9b.sh) | 0.3 | 0.6 |
| Qwen3.5-35B-A3B | [`run_rl/train_35b.sh`](../train_35b.sh) | 0.1 | 0.8 |

All entries set the context length to 65,536. Override the defaults through:

```bash
EFFICIENCY_COEFFICIENT=0.2 SCALING_GATE_THRESHOLD=0.7 bash run_rl/train_4b.sh
```

`launch.sh` selects HERO's reward and advantage hooks, then forwards to
`../_shared/run_*.sh` for model configuration and execution. Environment
setup and Docker runtime belong to `_shared/`; no other method directory is
required.

## Module-to-paper map

Equation references use the manuscript's LaTeX labels, so they remain stable if
the equation numbering changes.

| Paper component | Equation label / quantity | Implementation |
|---|---|---|
| Hierarchical Policy Optimization | `eq: hierarchical-rl`, A = A_res + w A_eff | [hierarchical_policy_optimization.py](hierarchical_policy_optimization.py): `compute_hierarchical_advantages`, `build_token_level_advantages` |
| Resolution advantage | A_res, standard GRPO group normalization | `resolution_advantages` in `compute_hierarchical_advantages`, using [group_statistics.py](group_statistics.py): `_standardize` |
| Efficiency gating | `eq: adaptive-weight`, w = λ gate(p, τ) | [efficiency_gating.py](efficiency_gating.py): `fixed_capability_gate`; λ is applied by the policy-optimization module |
| Resolution-first clipping | `eq: resolution-clipping`, Γ(A) | [resolution_first_clipping.py](resolution_first_clipping.py): `resolution_first_clipping` |
| Multi-granularity Efficiency Credit Assignment | `eq: efficiency-advantage`, A_eff = A_coarse + A_fine | [efficiency_credit_assignment.py](efficiency_credit_assignment.py), assembled in `compute_hierarchical_advantages` |
| Coarse-grained Trajectory Advantage / resolution-calibrated credit | `eq: coarse-advantage`, outcome-specific median budget and standardized utility | `coarse_grained_trajectory_advantages` in [efficiency_credit_assignment.py](efficiency_credit_assignment.py) |
| Fine-grained Turn Advantage / entropy-aware credit | `eq: fine-advantage`, low-entropy turn penalty | `fine_grained_turn_advantages` in [efficiency_credit_assignment.py](efficiency_credit_assignment.py) |
| Turn entropy estimation | h_ij, top-k probability renormalization | [rollout_metadata.py](rollout_metadata.py): `_turn_entropies` |
| Clipped policy surrogate | `eq: hierarchical-rl`, importance ratio ρ and PPO clipping ε | [Slime `compute_policy_loss`](../../framework/slime/slime/utils/ppo_utils.py), invoked by [Slime `policy_loss_function`](../../framework/slime/slime/backends/megatron_utils/loss.py) |
| Trace scheduling and loss denominator | implementation support | [trace_reward.py](trace_reward.py); original Slime DP scheduling and mask aggregation remain unchanged |

The loss links above are relative to this directory through the repository root:
see also `framework/slime/slime/utils/ppo_utils.py` and
`framework/slime/slime/backends/megatron_utils/loss.py`. No second or replacement
PPO loss is introduced by HERO. The method hooks construct token advantages;
Slime consumes them using its existing loss implementation and normalization.

## Data flow

1. `rollout_metadata.py` groups sibling traces into trajectories and checks
   reward, completion, token usage and entropy availability.
2. `compute_hierarchical_advantages` computes resolution advantages, the
   capability gate and coarse/fine efficiency credits.
3. Coarse credit is added to the scalar advantage; fine credit is carried as
   assistant-turn spans in `train_metadata`.
4. `trace_reward.py` assigns independent trace scheduling/loss identities after
   the fixed-gate reward computation.
5. `build_token_level_advantages` expands scalar credit, applies turn spans,
   performs resolution-first clipping, and slices for context parallelism.
6. Slime computes the clipped policy objective and the configured loss reduction.

`reward.py` exposes the full fixed-gate credit configuration to Slime;
`constants.py` holds numerical guards and serialized metadata keys.

## Preserved implementation details

These are implementation properties, not claims that every detail is specified
by the manuscript's simplified equations:

- Efficiency credit uses trajectories with both usable token and entropy data;
  at least four such trajectories are required in a group.
- Outcome-conditioned coarse credit requires at least three trajectories in an
  outcome subset, then standardizes eligible utilities using population variance
  plus the existing numerical epsilon.
- Resolution-first bounds are attached only to mixed-outcome efficiency-eligible
  groups. Homogeneous groups keep their existing unclipped advantage behavior.
- At τ=1, the fixed gate admits p=1.
- Entropy uses the existing top-10 renormalized estimate, completion-ID
  deduplication and linearly interpolated 20th percentile.
- Serialized metadata uses the `hero` namespace.
- Environment variables `EFFICIENCY_COEFFICIENT` (λ), `GATE_THRESHOLD` (τ), and
  the existing Slime hook names are retained as external interfaces. Python
  module/function names and experiment labels now use the paper terminology.

The Token-efficient Resolution Score (TRS) is an evaluation quantity in the
manuscript. This refactor does not add a TRS training reward or alter the binary
resolution reward to match that evaluation formula.
