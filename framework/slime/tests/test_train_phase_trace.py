import json

import pytest

from slime.utils.train_phase_trace import TrainPhaseTrace


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_trace_scope_and_records(tmp_path, monkeypatch, enabled):
    root = tmp_path / "trace"
    monkeypatch.setenv("SLIME_VFS_MULTINODE", enabled)
    monkeypatch.setenv("SLIME_TRAIN_PHASE_TRACE_DIR", str(root))
    trace = TrainPhaseTrace(rank=57, rollout_id=1, step_id=0)
    trace.emit("train_start", num_microbatches=26)
    trace.emit("batch_ready", microbatch=1, token_shape=[1, 8192], total_lengths=[65536])
    if enabled == "0":
        assert not root.exists()
        return
    paths = list(root.glob("rank-57-*.jsonl"))
    assert len(paths) == 1
    records = [json.loads(line) for line in paths[0].read_text().splitlines()]
    assert [r["phase"] for r in records] == ["train_start", "batch_ready"]
    assert all(r["rank"] == 57 and r["rollout_id"] == 1 and r["step_id"] == 0 for r in records)
    assert records[1]["total_lengths"] == [65536]


def test_trace_io_failure_does_not_fail_training(tmp_path, monkeypatch):
    monkeypatch.setenv("SLIME_VFS_MULTINODE", "1")
    monkeypatch.setenv("SLIME_TRAIN_PHASE_TRACE_DIR", str(tmp_path))
    trace = TrainPhaseTrace(rank=0, rollout_id=1, step_id=0)
    trace.path = tmp_path  # Opening a directory as a file must fail safely.
    trace.emit("train_start")
    assert trace.path is None
    trace.emit("optimizer_return")


def test_missing_directory_is_noop(monkeypatch):
    monkeypatch.setenv("SLIME_VFS_MULTINODE", "1")
    monkeypatch.delenv("SLIME_TRAIN_PHASE_TRACE_DIR", raising=False)
    trace = TrainPhaseTrace(rank=0, rollout_id=1, step_id=0)
    assert trace.path is None
    trace.emit("train_start")
