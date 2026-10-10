from __future__ import annotations

import pytest
from types import SimpleNamespace

from slime.utils import reloadable_process_group as rpg


@pytest.fixture(autouse=True)
def default_comm_profile(monkeypatch):
    monkeypatch.delenv("SLIME_VFS_MULTINODE", raising=False)


@pytest.mark.unit
@pytest.mark.parametrize("method", ["_allgather_base", "_reduce_scatter_base", "allreduce"])
@pytest.mark.parametrize("enabled", [False, True])
def test_vfs_profile_preserves_enqueue_without_memory_sync(monkeypatch, method, enabled):
    events = []
    monkeypatch.setenv("SLIME_VFS_MULTINODE", str(int(enabled)))
    monkeypatch.setattr(rpg, "available_memory", lambda: {"free_GB": 2})
    monkeypatch.setattr(rpg, "clear_memory", lambda: events.append("device_sync"))
    work = object()

    def enqueue(*args, **kwargs):
        events.append("enqueue")
        return work

    wrapper = SimpleNamespace(group=SimpleNamespace(**{method: enqueue}))
    assert rpg.ReloadableProcessGroup._fwd(wrapper, method) is work
    assert events == (["enqueue"] if enabled else ["device_sync", "enqueue"])


@pytest.mark.unit
def test_vfs_profile_nested_wrappers_do_not_probe_cuda(monkeypatch):
    monkeypatch.setenv("SLIME_VFS_MULTINODE", "1")

    def forbidden(*args, **kwargs):
        pytest.fail("CUDA probe/cleanup inside VFS collective")

    for name in ("available_memory", "clear_memory", "print_memory"):
        monkeypatch.setattr(rpg, name, forbidden)
    error = RuntimeError("original collective failure")
    with pytest.raises(RuntimeError) as caught:
        with rpg._wrap_low_level_call(check_memory=False):
            with rpg._wrap_low_level_call():
                raise error
    assert caught.value is error


@pytest.mark.unit
def test_selected_comm_ops_skip_memory_check():
    skipped_ops = {
        "all_gather_into_tensor",
        "allgather_into_tensor_coalesced",
        "barrier",
        "broadcast_object_list",
        "reduce_scatter_tensor",
        "all_to_all_single",
        "isend",
        "irecv",
    }
    checked_ops = {
        "all_reduce",
        "all_gather",
        "broadcast",
        "reduce_scatter",
        "all_to_all",
        "send",
        "recv",
        "reduce_scatter_tensor_coalesced",
    }

    for op_name in skipped_ops:
        assert not rpg._should_check_memory_for_comm(op_name)

    for op_name in checked_ops:
        assert rpg._should_check_memory_for_comm(op_name)


@pytest.mark.unit
def test_wrap_low_level_call_can_skip_available_memory(monkeypatch):
    calls = []

    def fake_available_memory():
        calls.append("available_memory")
        return {"free_GB": 100}

    monkeypatch.setattr(rpg, "available_memory", fake_available_memory)

    with rpg._wrap_low_level_call(check_memory=False):
        pass

    assert calls == []


@pytest.mark.unit
def test_wrap_low_level_call_checks_available_memory_by_default(monkeypatch):
    calls = []

    def fake_available_memory():
        calls.append("available_memory")
        return {"free_GB": 100}

    monkeypatch.setattr(rpg, "available_memory", fake_available_memory)

    with rpg._wrap_low_level_call():
        pass

    assert calls == ["available_memory"]
