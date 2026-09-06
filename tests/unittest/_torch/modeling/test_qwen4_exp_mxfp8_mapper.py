# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tensorrt_llm._torch.models.checkpoints.hf.qwen4_exp_weight_mapper import Qwen4ExpHfWeightMapper


def _mapper(tp_size: int) -> Qwen4ExpHfWeightMapper:
    mapper = Qwen4ExpHfWeightMapper()
    mapper._config = SimpleNamespace(
        pretrained_config=SimpleNamespace(
            num_hidden_layers=1,
            linear_key_head_dim=4,
            linear_num_key_heads=2,
            linear_value_head_dim=4,
            linear_num_value_heads=4,
            hidden_size=64,
        ),
        mapping=SimpleNamespace(
            enable_attention_dp=False, tp_size=tp_size, tp_rank=0, has_pp=lambda: False
        ),
        spec_config=None,
    )
    return mapper


def _weights() -> dict[str, torch.Tensor]:
    weights = {}
    for index, (projection, rows) in enumerate((("qkv", 32), ("z", 16), ("b", 4), ("a", 4))):
        prefix = f"model.language_model.layers.0.linear_attn.in_proj_{projection}"
        weights[f"{prefix}.weight"] = torch.full((rows, 64), index + 1).to(torch.float8_e4m3fn)
        weights[f"{prefix}.weight_scale"] = (
            torch.arange(rows * 2).reshape(rows, 2).remainder(5).add(125 + index).to(torch.uint8)
        )
    return weights


def _dequantize(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return weight.float() * torch.exp2(scale.float() - 127).repeat_interleave(32, dim=-1)


@pytest.mark.parametrize("tp_size", [1, 2])
def test_gdn_mxfp8_fusion_preserves_values_and_rank_order(tp_size: int) -> None:
    source = _weights()
    mapped = _mapper(tp_size).preprocess_weights(source)
    decoded = {}
    for projection in ("qkv", "z", "b", "a"):
        prefix = f"model.language_model.layers.0.linear_attn.in_proj_{projection}"
        decoded[projection] = _dequantize(
            source[f"{prefix}.weight"], source[f"{prefix}.weight_scale"]
        )
    q, k, v = decoded["qkv"].split([8, 8, 16])
    for fused, components in (
        ("qkvz", (q, k, v, decoded["z"])),
        ("ba", (decoded["b"], decoded["a"])),
    ):
        prefix = f"model.layers.0.linear_attn.in_proj_{fused}"
        expected = torch.cat(
            [component.chunk(tp_size)[rank] for rank in range(tp_size) for component in components]
        )
        actual = _dequantize(mapped[f"{prefix}.weight"], mapped[f"{prefix}.weight_scale"])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(mapped) == 4


@pytest.mark.parametrize("fault", ["missing", "shape", "dtype", "orphan"])
def test_gdn_mxfp8_fusion_rejects_invalid_scales(fault: str) -> None:
    weights = _weights()
    key = "model.language_model.layers.0.linear_attn.in_proj_a.weight_scale"
    if fault == "missing":
        del weights[key]
    elif fault == "shape":
        weights[key] = weights[key][:, :1]
    elif fault == "dtype":
        weights[key] = weights[key].float()
    else:
        weights = {key: weights[key]}
    with pytest.raises(ValueError, match="scales|weight_scale"):
        _mapper(1).preprocess_weights(weights)


def test_lazy_projection_views_and_scalar_survive_source_close(tmp_path) -> None:
    weights = _weights()
    weights["model.language_model.layers.0.marker.weight_scale_2"] = torch.tensor(0.5)
    checkpoint = tmp_path / "model.safetensors"
    save_file(weights, checkpoint)
    expected = _mapper(1).preprocess_weights(weights)
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        lazy = {key: handle.get_slice(key) for key in handle.keys()}
        mapped = _mapper(1).preprocess_weights(lazy)
        lazy.clear()
    for key in expected:
        torch.testing.assert_close(mapped[key].float(), expected[key].float(), rtol=0, atol=0)
