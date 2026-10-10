# Runtime Backends

`hero.runtime` gives each rollout session its own **sandbox** — one container
(Docker or Apptainer) that lives for the whole session. The gateway uses it to
run the prepare recipe, execute the agent and evaluator commands, move files in
and out, then tear it down.

## Mental model

- **One `RuntimeSpec` → one container**, shared across the init → run → eval
  stages of a session.
- The host session directory is **bind-mounted** to a fixed in-container path,
  `/hero/session` (`RUNTIME_SESSION_DIR`). Uploads/downloads under that path
  are plain host-side file copies (fast); paths outside it fall back to
  `docker cp` / `tar` streaming.
- Commands run in a login shell (`bash -lc`) with working directory
  `cwd or spec.workdir or /hero/session`.
- The factory verifies the chosen backend actually supports what the spec asks
  for (GPUs, CPU/memory limits, internet-off) before building it.

## Main files

- `models.py`: `RuntimeSpec`, `PrepareAction`, `ExecInput`, `ExecResult`.
- `base.py`: the `BaseRuntime` contract, the `/hero/session` path constants, and
  the bind-mount copy helpers.
- `docker.py`: `DockerRuntime` — the default backend.
- `apptainer.py`: `ApptainerRuntime` — daemonless, for clusters.
- `factory.py`: backend lookup + capability validation; also loads a custom
  backend via `RuntimeSpec.import_path`.

## The contract

A backend implements `start`, `stop`, `exec`, `upload_file`, `upload_dir`,
`download_file`, `download_dir` (plus `cancel`), hiding container details from
harnesses and evaluators. Well-known in-container paths (from `base.py`) are
`/hero/session` and, under it, `artifacts/`, `logs/`, `logs/agent/`,
`logs/eval/`, and `eval_artifacts/`.

## Prepare recipe

`RuntimeSpec.prepare` and `RuntimeSpec.eval_prepare` are ordered lists of
`PrepareAction` steps:

- `upload_file`: copy one host file in.
- `upload_dir`: copy one host directory in.
- `exec`: run a command inside the container.

`prepare` runs before the agent. `eval_prepare` runs before evaluation — and if
it's omitted, the eval runtime simply replays `prepare`.

## Docker vs Apptainer

Docker is the default for local examples and supports `--cpus` / `--memory`
limits. Apptainer is daemonless (good for clusters that forbid the Docker
socket), uses a host-backed overlay, and exposes GPUs with `--nv`. Both
bind-mount the session directory and run commands via `bash -lc`, so harnesses
and evaluators behave the same on either.
