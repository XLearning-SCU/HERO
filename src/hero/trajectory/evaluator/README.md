# Trajectory Evaluators

Evaluators score a built `Trajectory` into an `EvalResult` (an outcome reward
and/or per-trace rewards, plus metadata). The gateway then merges that reward
onto the trajectory's traces.

## Main files

- `base.py`: the evaluator contract (`async evaluate(trajectory, **runtime) -> EvalResult`).
- `session_completed.py`: reward by terminal status.
- `test_on_output.py`: apply the agent's changes and grade test output.
- `swebench_harness.py`: grade a patch with the SWE-bench harness.
- `_patch_utils.py`: `BasePatchEvaluator` — the shared extract → filter → apply →
  test flow both grading evaluators build on.

## Built-in strategies

**`session_completed`** — reward `1.0` if the session reached `COMPLETED`, else
`0.0`. Needs no runtime; handy as a smoke-test signal.

**`test_on_output`** — for custom/toy tasks. It extracts the agent's git diff,
(optionally) applies it on a fresh runtime, runs a test command, and **grades by
matching parsed test output — not the exit code**: it reads
`PASSED`/`FAILED`/`ERROR`/`SKIPPED <node>` lines and rewards `1.0` only when the
parsed result **exactly equals** the expected map.

| config key | required | meaning |
|---|---|---|
| `test_command` | yes | the command to run |
| `expected_output_json` | yes | `{node: "PASSED", ...}` the output must match |
| `repo_dir` | no | where the diff/test run (default `/testbed`) |
| `patch_command` | no | how to extract the diff (default a `git diff`) |
| `test_timeout` / `apply_timeout` | no | timeouts |
| `exclude_patterns` | no | paths to drop from the diff |

**`swebench_harness`** — grades real SWE-bench-style patches with the SWE-bench
(or SWE-Gym) harness. Takes an `instance` dict plus the same patch config keys.

Both grading evaluators need a live runtime (and a `fresh_eval_runtime` when the
task sets `refresh_runtime`); an empty diff scores `0.0`.

## Adding an evaluator

Implement the base contract, return an `EvalResult`, and register the name in
`registry.py` (or pass a `"module:ClassName"` import path). Keep external
services, GPUs, and large datasets out of default unit tests.
