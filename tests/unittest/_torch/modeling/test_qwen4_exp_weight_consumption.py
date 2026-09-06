# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict
from tensorrt_llm._torch.models.checkpoints.hf.qwen4_exp_weight_mapper import Qwen4ExpHfWeightMapper
from tensorrt_llm._torch.models.checkpoints.hf.weight_loader import _LazySafetensorsWeights
from tensorrt_llm._torch.modules.qwen4_exp.ple_nvfp4 import Qwen4ExpNVFP4MmapEmbedding


def _mapper() -> Qwen4ExpHfWeightMapper:
    mapper = Qwen4ExpHfWeightMapper()
    mapper._config = SimpleNamespace(
        pretrained_config=SimpleNamespace(
            num_hidden_layers=2,
            linear_key_head_dim=4,
            linear_num_key_heads=2,
            linear_value_head_dim=4,
            linear_num_value_heads=4,
            hidden_size=64,
        ),
        mapping=SimpleNamespace(
            enable_attention_dp=False, tp_size=1, tp_rank=0, has_pp=lambda: False
        ),
        spec_config=None,
    )
    mapper._model = nn.Module()
    mapper._model.model = nn.Module()
    mapper._model.model.layers = nn.ModuleList([nn.Module(), nn.Module()])
    return mapper


def _gdn_weights() -> dict[str, torch.Tensor]:
    weights = {}
    for index, (projection, rows) in enumerate((("qkv", 32), ("z", 16), ("b", 4), ("a", 4))):
        prefix = f"model.language_model.layers.0.linear_attn.in_proj_{projection}"
        weights[f"{prefix}.weight"] = torch.full((rows, 64), index + 1).to(torch.float8_e4m3fn)
        weights[f"{prefix}.weight_scale"] = torch.full((rows, 2), 127, dtype=torch.uint8)
    return weights


@pytest.mark.parametrize("consumable", [False, True])
def test_preprocess_preserves_container_contract(consumable: bool) -> None:
    source = _gdn_weights()
    if consumable:
        source = ConsumableWeightsDict(source)
        source.checkpoint_dir = "/checkpoint/path"
    mapped = _mapper().preprocess_weights(source)
    assert isinstance(mapped, ConsumableWeightsDict) == consumable
    if consumable:
        assert len(source) == 0
        assert mapped.checkpoint_dir == "/checkpoint/path"
        source.clear()  # The model calls clear again after preprocessing.
    else:
        assert len(source) == 8
    assert len(mapped) == 4


def test_fused_gdn_tensors_release_after_last_prefix_reader() -> None:
    mapper = _mapper()
    source = ConsumableWeightsDict(_gdn_weights())
    source_ref = weakref.ref(source["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"])
    mapped = mapper.preprocess_weights(source)
    gc.collect()
    assert source_ref() is None
    prefix = "model.layers.0.linear_attn.in_proj_qkvz"
    weight_ref = weakref.ref(mapped[f"{prefix}.weight"])
    scale_ref = weakref.ref(mapped[f"{prefix}.weight_scale"])
    first_reader = mapper.filter_weights(prefix, mapped)
    second_reader = mapper.filter_weights(prefix, mapped)
    torch.testing.assert_close(first_reader["weight"].float()[:8], torch.ones(8, 64))
    assert mapped.mark_consumed(prefix) == 2
    assert mapper.filter_weights(prefix, mapped) == {}
    first_reader.clear()
    gc.collect()
    assert weight_ref() is second_reader["weight"]
    assert scale_ref() is second_reader["weight_scale"]
    second_reader.clear()
    gc.collect()
    assert weight_ref() is None
    assert scale_ref() is None
    assert len(mapped) == 2  # The B/A consumer has not run yet.


def test_indexed_grouped_prefix_consumption_preserves_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapper = _mapper()
    source = ConsumableWeightsDict(
        {
            "model.language_model.layers.1.group.weight": torch.ones(2),
            "model.language_model.layers.1.group.child.weight": torch.full((2,), 2.0),
            "model.language_model.layers.10.group.weight": torch.full((2,), 10.0),
        }
    )
    mapped = mapper.preprocess_weights(source)

    def forbidden_scan() -> None:
        raise AssertionError("Prefix lookup fell back to a full checkpoint scan")

    monkeypatch.setattr(mapped, "items", forbidden_scan)
    group = mapper.filter_weights("model.layers.1.group", mapped)
    assert set(group) == {"weight", "child.weight"}
    own_ref = weakref.ref(group["weight"])
    assert mapped.mark_consumed_keys(["model.layers.1.group.weight"]) == 1
    group.clear()
    gc.collect()
    assert own_ref() is None
    assert set(mapper.filter_weights("model.layers.1.group", mapped)) == {"child.weight"}
    assert mapped.mark_consumed("model.layers.1.group.child") == 1
    assert mapper.filter_weights("model.layers.1.group", mapped) == {}
    torch.testing.assert_close(
        mapper.filter_weights("model.layers.10.group", mapped)["weight"], torch.full((2,), 10.0)
    )


@pytest.mark.parametrize("first_layer", [0, 1])
def test_passthrough_storage_alias_survives_other_consumer(first_layer: int) -> None:
    backing = torch.arange(8, dtype=torch.bfloat16)
    backing_ref = weakref.ref(backing)
    source = ConsumableWeightsDict(
        {
            "model.language_model.layers.0.norm.weight": backing,
            "model.language_model.layers.1.norm.weight": backing.view(2, 4),
        }
    )
    del backing
    mapper = _mapper()
    mapped = mapper.preprocess_weights(source)
    assert mapped.mark_consumed(f"model.layers.{first_layer}.norm") == 1
    gc.collect()
    assert backing_ref() is not None
    remaining = f"model.layers.{1 - first_layer}.norm"
    reader = mapper.filter_weights(remaining, mapped)
    torch.testing.assert_close(reader["weight"].reshape(-1), torch.arange(8, dtype=torch.bfloat16))
    assert mapped.mark_consumed(remaining) == 1
    gc.collect()
    assert backing_ref() is not None
    reader.clear()
    gc.collect()
    assert backing_ref() is None


def test_hyper_connection_fusion_releases_after_consume() -> None:
    mapper = _mapper()
    hc = nn.Module()
    hc.input_mix_injection_offset = 4
    hc.hc_count = 2
    hc.input_mix_weight_down_block_inject = nn.Linear(16, 16, bias=False, dtype=torch.bfloat16)
    mapper._model.model.layers[0].attn_hyper_connection = hc
    source_prefix = "model.language_model.layers.0.attn_hyper_connection"
    source = ConsumableWeightsDict(
        {
            f"{source_prefix}.input_mix_weight_down.weight": torch.ones(
                4, 16, dtype=torch.bfloat16
            ),
            f"{source_prefix}.block_inject_weight.weight": torch.full(
                (2, 16), 2, dtype=torch.bfloat16
            ),
        }
    )
    mapped = mapper.preprocess_weights(source)
    prefix = "model.layers.0.attn_hyper_connection.input_mix_weight_down_block_inject"
    ref = weakref.ref(mapped[f"{prefix}.weight"])
    reader = mapper.filter_weights(prefix, mapped)
    expected = torch.cat(
        (torch.ones(4, 16), torch.full((2, 16), 2), torch.zeros(10, 16))
    ).bfloat16()
    torch.testing.assert_close(reader["weight"], expected)
    assert mapped.mark_consumed(prefix) == 1
    reader.clear()
    gc.collect()
    assert ref() is None


def test_preprocess_failure_keeps_source_ownership() -> None:
    source = ConsumableWeightsDict(_gdn_weights())
    del source["model.language_model.layers.0.linear_attn.in_proj_a.weight_scale"]
    source.checkpoint_dir = "/checkpoint/path"
    ref = weakref.ref(source["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"])
    with pytest.raises(ValueError, match="scales for every"):
        _mapper().preprocess_weights(source)
    gc.collect()
    assert len(source) == 7
    assert ref() is not None
    assert source.checkpoint_dir == "/checkpoint/path"


def test_lazy_ple_retains_mmap_after_source_and_consumers_clear(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapper = _mapper()
    ple = nn.Module()
    ple.ple_embedding = nn.Module()
    ple.ple_embedding.tp_size = 1
    embedding = Qwen4ExpNVFP4MmapEmbedding(4, 16, expected_shards=2)
    ple.ple_embedding.ngram_embedding = embedding
    mapper._model.model.layers[0].ple = ple
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    weights = {f"{prefix}.weight_scale_2": torch.tensor([2.0])}
    for index, (packed, scale) in enumerate(((0x21, 0.5), (0xE7, 2.0))):
        weights[f"{prefix}.shard_{index}.weight"] = torch.full((2, 8), packed, dtype=torch.uint8)
        weights[f"{prefix}.shard_{index}.weight_scale"] = torch.full((2, 1), scale).to(
            torch.float8_e4m3fn
        )
    weights["model.language_model.layers.0.marker.weight_scale_2"] = torch.tensor(0.25)
    save_file(weights, tmp_path / "model.safetensors")
    del weights
    calls = []

    def record_gpu_setup(checkpoint_dir: str, source_prefix: str) -> bool:
        assert len(embedding._shards) == 2
        calls.append((checkpoint_dir, source_prefix))
        return False  # This regression deliberately exercises only the CPU mmap path.

    monkeypatch.setattr(embedding, "enable_gpu_lookup", record_gpu_setup)
    with safe_open(tmp_path / "model.safetensors", framework="pt", device="cpu") as handle:
        source = _LazySafetensorsWeights(
            {name: handle.get_slice(name) for name in handle.keys()}, [handle]
        )
        source.checkpoint_dir = str(tmp_path)
        mapped = mapper.preprocess_weights(source)
        assert len(source) == 0
        assert source._handles == []
    del handle
    assert calls == [(str(tmp_path), prefix)]
    assert mapped.checkpoint_dir == str(tmp_path)
    assert len(mapped) == 1  # PLE was bound independently, not returned for generic loading.
    reader = mapper.filter_weights("model.layers.0.marker", mapped)
    assert reader["weight_scale_2"].shape == torch.Size([])
    assert reader["weight_scale_2"].item() == 0.25
    scalar_ref = weakref.ref(reader["weight_scale_2"])
    assert mapped.mark_consumed("model.layers.0.marker") == 1
    reader.clear()
    mapped.clear()
    source.clear()
    gc.collect()
    assert scalar_ref() is None
    result = embedding.gather(torch.tensor([0, 3], dtype=torch.int64))
    expected = torch.tensor([[0.5, 1.0] * 8, [24.0, -16.0] * 8], dtype=torch.bfloat16)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
