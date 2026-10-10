"""Regression tests for train-metadata transport into custom advantages."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from slime.ray import rollout as rollout_module
from slime.ray.rollout import RolloutManager
from slime.utils import data as rollout_data_module


NUM_GPUS = 0


def _args(transport: str = "object-store") -> SimpleNamespace:
    return SimpleNamespace(
        advantage_estimator="grpo",
        global_batch_size=4,
        micro_batch_size=1,
        use_dynamic_batch_size=False,
        max_tokens_per_gpu=None,
        balance_data=False,
        balance_by_flops=False,
        hidden_size=16,
        num_attention_heads=2,
        num_query_groups=2,
        vocab_size=32,
        ffn_hidden_size=64,
        num_experts=None,
        num_layers=2,
        kv_channels=8,
        rollout_data_transport=transport,
    )


def _split(monkeypatch: pytest.MonkeyPatch, metadata: list[dict], transport: str):
    manager_class = RolloutManager.__ray_metadata__.modified_class
    manager = object.__new__(manager_class)
    manager.args = _args(transport)
    manager.train_parallel_config = {
        "dp_size": 2,
        "cp_size": 1,
        "vpp_size": 1,
        "microbatch_group_size_per_vp_stage": 1,
    }
    data = {
        "tokens": [[0, 1], [2, 3], [4, 5], [6, 7]],
        "response_lengths": [2] * 4,
        "rewards": [0.25] * 4,
        "raw_reward": [0.25] * 4,
        "truncated": [0] * 4,
        "loss_masks": [[1, 1] for _ in range(4)],
        "sample_indices": list(range(4)),
        "rollout_ids": list(range(4)),
        "trajectory_ids": list(range(4)),
        "metadata": metadata,
    }
    put_calls: list[dict] = []

    def fake_put(value, **kwargs):
        put_calls.append(kwargs)
        return value

    monkeypatch.setattr(rollout_module.ray, "put", fake_put)
    refs = manager._split_train_data_by_dp(data)
    assert len(put_calls) == 2
    if transport == "nixl":
        assert all(call == {"_tensor_transport": "nixl"} for call in put_calls)
    else:
        assert all(not call for call in put_calls)
    return manager.args, refs


def _process_all(monkeypatch: pytest.MonkeyPatch, args, refs):
    monkeypatch.setattr(rollout_data_module.ray, "get", lambda value: value)
    return [rollout_data_module.process_rollout_data(args, refs, dp_rank, len(refs)) for dp_rank in range(len(refs))]


@pytest.mark.unit
@pytest.mark.parametrize("transport", ["object-store", "nixl"])
def test_dp_split_preserves_metadata_alignment_and_standard_advantage(monkeypatch: pytest.MonkeyPatch, transport: str) -> None:
    from run_rl.method.experiments.Gate_Coarse_Fine import reward

    metadata = [
        {
            "hero_efficiency_advantage_spans": [{"start": 1, "end": 2, "advantage": -0.5}],
            "hero_outcome_advantage_bound": "nonnegative",
        },
        {"hero_outcome_advantage_bound": "nonpositive"},
        {},
        {},
    ]
    args, refs = _split(monkeypatch, metadata, transport)
    outputs: dict[int, list[float]] = {}
    for shard in _process_all(monkeypatch, args, refs):
        assert shard["metadata"] == [metadata[i] for i in shard["sample_indices"]]
        shard["kl"] = [torch.zeros(length) for length in shard["response_lengths"]]
        reward.compute_advantages(args, shard)
        for index, advantage in zip(shard["sample_indices"], shard["advantages"], strict=True):
            outputs[index] = advantage.tolist()

    assert outputs == {
        0: [0.25, 0.0],
        1: [0.0, 0.0],
        2: [0.25, 0.25],
        3: [0.25, 0.25],
    }


@pytest.mark.unit
def test_prorl_prefers_direct_metadata_after_dp_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from run_rl.method.experiments.Prorl_task_Gate_Coarse_Fine import core

    monkeypatch.setenv("PRORL_TASK_ADV", "1")
    monkeypatch.setattr(core, "_global_masked_whiten", lambda chunks, masks: chunks)
    metadata = [
        {
            core.TASK_ADVANTAGE_KEY: 0.25,
            core.EFFICIENCY_ADVANTAGE_KEY: 0.0,
            "hero_efficiency_advantage_spans": [{"start": 1, "end": 2, "advantage": -0.5}],
            "hero_outcome_advantage_bound": "nonnegative",
        },
        {
            core.TASK_ADVANTAGE_KEY: 0.25,
            core.EFFICIENCY_ADVANTAGE_KEY: 0.0,
            "hero_outcome_advantage_bound": "nonpositive",
        },
        {core.TASK_ADVANTAGE_KEY: 0.25, core.EFFICIENCY_ADVANTAGE_KEY: 0.0},
        {core.TASK_ADVANTAGE_KEY: 0.25, core.EFFICIENCY_ADVANTAGE_KEY: 0.0},
    ]
    args, refs = _split(monkeypatch, metadata, "object-store")
    outputs: dict[int, list[float]] = {}
    for shard in _process_all(monkeypatch, args, refs):
        shard["kl"] = [torch.zeros(length) for length in shard["response_lengths"]]
        core.compute_advantages(args, shard)
        for index, advantage in zip(shard["sample_indices"], shard["advantages"], strict=True):
            outputs[index] = advantage.tolist()

    assert outputs[0] == [0.25, 0.0]
    assert outputs[1] == [0.0, 0.0]


@pytest.mark.unit
def test_rollout_logger_skip_list_includes_json_metadata() -> None:
    data_path = Path(rollout_module.__file__).parents[1] / "backends" / "megatron_utils" / "data.py"
    tree = ast.parse(data_path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "log_rollout_data")
    literal_lists = [{item.value for item in node.elts if isinstance(item, ast.Constant)} for node in ast.walk(function) if isinstance(node, ast.List)]
    assert any({"metadata", "source_names"} <= values for values in literal_lists)
