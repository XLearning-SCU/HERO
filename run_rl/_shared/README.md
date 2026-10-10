# Shared training and evaluation infrastructure

This directory owns environment setup, model configuration, and execution.
It does not import any method package. Public entries select their reward and
advantage hooks before invoking the shared model launcher.

| Location | Responsibility |
| --- | --- |
| `run_4b.sh` | Common execution and 4B defaults |
| `run_9b.sh` | 9B overrides followed by common execution |
| `run_35b.sh` | 35B-A3B multi-node overrides followed by common execution |
| `init_env/` | Environment, agent CLI, preflight checks, and compatibility patches |
| `runtime/` | Docker coordination, resource checks, and lifecycle management |
| `config/` | Training/evaluation topology and task configuration |
| `eligibility.py` | Sample eligibility and reward access |
| `validate_training_data.py` | Training data validation |

HERO starts at `run_rl/train_{4b,9b,35b}.sh`, selects its hooks in
`run_rl/HERO/launch.sh`, and invokes the matching model launcher here.
The execution backend remains `examples/swegym_slime_grpo/run.sh`.
Evaluation uses the runtime and topology directly, without loading a training method.

Optional methods can be added under `run_rl/`. Each supplies its own entry
points and algorithm modules and depends only on this shared infrastructure
and the base repository. Removing an optional method does not affect HERO,
evaluation, or other methods.

The relocation preserves the original training parameters, reward formulas,
trace identities, and loss normalization. See [Prepare.md](../../Prepare.md)
for environment setup and resource layout.
