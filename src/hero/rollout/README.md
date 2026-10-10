# Rollout Service

`hero.rollout` is the **central orchestrator**. A client (a trainer,
`hero submit`, or an example script) posts a task here; the rollout server fans
it out into one session per sample, schedules each onto a healthy gateway node,
collects the terminal result, and optionally persists it.

## Mental model

A task becomes N sessions, each placed on a gateway:

1. A `TaskRequest` is submitted — `POST /rollout/task/submit` returns
   immediately with `{task_id, status: "running"}`; you poll or get a callback.
2. The manager creates one session per `num_samples`.
3. The scheduler picks a healthy, non-draining gateway with spare capacity.
4. The pipeline dispatches the session and waits for the gateway's callback,
   **interleaving a poll** of the gateway as a live safety net (covers a dropped
   callback, or a status flip before the payload is serialized).
5. The terminal `SessionResult` is recorded; if `TaskRequest.callback_url` is
   set, the aggregate `TaskResult` is POSTed back to the trainer.
6. Results persist under `rollout.save_dir` when configured
   (`save_dir/task_<id>/ses_<sid>.json`).

The `timeout_seconds` budget starts when a session enters **INIT**, not at
dispatch — time spent queued on a busy gateway (`REGISTERED`) isn't charged
against the agent's wall-clock.

## Main files

- `server.py`: FastAPI app — task submit/get/status, node register/heartbeat/
  list/drain, the session-result callback, the `/tasks` + `/events`
  observability routes, and `/health`.
- `manager.py`: task lifecycle, session expansion, and the trainer-facing
  terminal callback.
- `pipeline.py`: dispatch to a gateway, wait (callback + interleaved poll),
  persist.
- `balancer.py`: `NodeScheduler` — registration, heartbeat health, pressure-based
  scheduling, draining.
- `models.py`: the public `TaskRequest`, `SessionResult`, `TaskResult`,
  `TaskStatus`, and node/heartbeat models.
- `timer.py`: per-stage timing helpers.

## HTTP endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/rollout/task/submit` | Submit a task (returns immediately) |
| GET | `/rollout/task/{task_id}` | Task status / progress |
| GET | `/rollout/status` | Service + fleet summary |
| POST | `/nodes/register`, `/nodes/{id}/heartbeat` | Gateway registration + heartbeat |
| GET / DELETE | `/nodes`, `/nodes/{id}` | List / inspect / drain nodes |
| POST | `/callbacks/session_result` | Gateway → rollout result callback |
| GET | `/tasks`, `/tasks/{id}/sessions`, `/events` | Observability (used by the dashboard) |
| GET | `/health` | Health check |

## Task request shape

```json
{
  "task_id": "example-task-001",
  "instruction": "Implement calculator.py and make the tests pass.",
  "num_samples": 8,
  "timeout_seconds": 900,
  "runtime": {"backend": "docker", "image": "...", "workdir": "/hero/session/workspace", "network": "host"},
  "agent": {"harness": "codex", "model_name": "openai/gpt-5.4"},
  "builder": {"strategy": "prefix_merging"},
  "evaluator": {"strategy": "test_on_output", "config": {"test_command": "...", "expected_output_json": {"test_calculator": "PASSED"}}},
  "callback_url": "http://trainer:9000/done",
  "metadata": {"group_id": "g1", "rollout_step": 42}
}
```

- `runtime`: the Docker/Apptainer sandbox (see [runtime](../runtime/README.md)).
  Optional — a node's `default_runtime` is used if omitted.
- `agent`: a preset harness, the generic `shell` harness, or a custom
  `import_path` (see [agent](../agent/README.md)).
- `builder` / `evaluator`: how completions become trajectories and how reward is
  attached (see [trajectory](../trajectory/README.md)).
- `callback_url` (optional): where the final `TaskResult` is POSTed.
- `metadata` (optional): free-form key/values carried through onto the
  trajectory — convenient for training fields like group id or policy version
  (it's an unvalidated pass-through dict, not a fixed schema).

## Scheduling

The scheduler prefers the least-loaded healthy node, comparing pressures in this
order — **run → post-run → init** — with a ready-buffer gap and total pressure as
tiebreakers. A node is eligible only while it is **healthy** (a heartbeat within
`heartbeat_interval × 2.5`), **not draining**, and **under its admission and
post-run-backlog limits**. A draining node stops receiving work and is removed
once its in-flight sessions finish.
