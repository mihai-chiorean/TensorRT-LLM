# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU loader contract checks and genuine CPU-load/CUDA-placement regression."""

from types import ModuleType
from unittest.mock import Mock

import pytest
import test_mxfp8_sm12x_dispatch as dispatch_tests
import torch
from test_mxfp8_sm12x_dispatch import _mock_device, _mock_flashinfer
from torch import nn

linear_module = dispatch_tests.linear_module


@pytest.mark.parametrize("scale_name", ["weight_scale", "weight_scale_inv"])
@pytest.mark.parametrize("n,k", [(128, 128), (160, 160), (256, 288)])
def test_cpu_load_preserves_placement_contract(
    linear_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    scale_name: str,
    n: int,
    k: int,
) -> None:
    """Real load helpers keep CPU sources; only the native layout op is mocked."""
    native = _mock_device(monkeypatch, (12, 1))
    flashinfer = _mock_flashinfer(monkeypatch)
    flashinfer.block_scale_interleave.side_effect = AssertionError("FI received CPU scales")
    method = linear_module.MXFP8LinearMethod()
    layer = nn.Module()
    layer.tp_size, layer.tp_rank, layer.tp_mode = 1, 0, None
    layer.has_weight_only_quant = False
    method.create_weights(layer, k, n, bias=False, dtype=torch.bfloat16)
    weight = torch.randn(n, k).to(torch.float8_e4m3fn)
    scales = torch.randint(120, 132, (n, k // 32), dtype=torch.uint8)
    swizzled = torch.arange(layer.weight_scale.numel()).to(torch.uint8)
    native.block_scale_interleave.side_effect = lambda scale: swizzled

    def load_shard(
        weights: dict[str, torch.Tensor], name: str, device: torch.device, elm_packing: int
    ) -> torch.Tensor:
        return linear_module.load_weight_shard(weights[name], device=device)

    layer.load_shard = load_shard
    loaded_scales = method.load_weight_scales([{scale_name: scales}])
    assert loaded_scales[0].device.type == "cpu"
    assert loaded_scales[0].data_ptr() == scales.data_ptr()
    method.load_weights_vanilla(layer, [{"weight": weight, scale_name: scales}])
    native.block_scale_interleave.assert_called_once_with(scales)
    flashinfer.block_scale_interleave.assert_not_called()
    assert layer.weight.device.type == layer.weight_scale.device.type == "cpu"
    torch.testing.assert_close(layer.weight.view(torch.uint8), weight.view(torch.uint8))
    torch.testing.assert_close(layer.weight_scale, swizzled)
    for name, op in vars(native).items():
        if name != "block_scale_interleave":
            op.assert_not_called()


@pytest.mark.parametrize("n,k", [(128, 128), (160, 128), (160, 160), (256, 288)])
@pytest.mark.parametrize("load_device", ["cpu", "cuda"])
def test_cpu_load_then_cuda_placement_real_runtime(
    monkeypatch: pytest.MonkeyPatch, n: int, k: int, load_device: str
) -> None:
    """No AST or op substitutes: CPU scale packing, placement, and FI forward."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (12, 0),
        (12, 1),
    ):
        pytest.skip("requires SM120/121 and the installed TensorRT/FlashInfer runtime")
    import flashinfer

    from tensorrt_llm._torch.modules import linear as runtime_linear
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    monkeypatch.delenv("TRTLLM_MXFP8_GEMM_BACKEND", raising=False)
    # Exercise the Spark CPU-retention policy also when run on a discrete SM120.
    monkeypatch.setattr(runtime_linear, "is_device_integrated", lambda: True)
    torch.manual_seed(20260906)
    with torch.device("cpu"):
        layer = runtime_linear.Linear(
            k,
            n,
            bias=False,
            dtype=torch.bfloat16,
            reduce_output=False,
            quant_config=QuantConfig(quant_algo=QuantAlgo.MXFP8, group_size=32),
        )
        weight = torch.randn(n, k).to(torch.float8_e4m3fn)
        scales = torch.randint(123, 129, (n, k // 32), dtype=torch.uint8)
    assert layer.quant_method._use_flashinfer_sm12x
    device = torch.device("cuda", torch.cuda.current_device())
    if load_device == "cuda":
        layer.to(device)
    interleave = Mock(wraps=flashinfer.block_scale_interleave)
    monkeypatch.setattr(layer.quant_method, "_flashinfer_interleave", interleave)
    layer.load_weights([{"weight": weight, "weight_scale_inv": scales}])
    interleave.assert_not_called()
    assert layer.weight.device.type == layer.weight_scale.device.type == load_device
    assert weight.device.type == scales.device.type == "cpu"
    torch.testing.assert_close(layer.weight.view(torch.uint8).cpu(), weight.view(torch.uint8))
    cpu_swizzled = layer.weight_scale.detach().cpu().clone()
    layer.to(device)
    expected_scales = flashinfer.block_scale_interleave(scales.to(device))
    torch.testing.assert_close(layer.weight_scale, expected_scales, rtol=0, atol=0)
    torch.testing.assert_close(layer.weight_scale.cpu(), cpu_swizzled, rtol=0, atol=0)
    # The existing CUDA-source branch must still call FI, on the source device.
    device_scales = scales.to(device)
    layer.quant_method._store_scale(layer, device_scales)
    interleave.assert_called_once_with(device_scales)
    for m in (1, 4):
        x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
        actual = layer(x)
        quantized, activation_scales = flashinfer.mxfp8_quantize(x, is_sf_swizzled_layout=False)
        x_dequant = quantized.float().reshape(m, k // 32, 32) * torch.exp2(
            activation_scales.reshape(m, k // 32).float() - 127
        ).unsqueeze(-1)
        w_dequant = weight.float().reshape(n, k // 32, 32) * torch.exp2(
            scales.float() - 127
        ).unsqueeze(-1)
        expected = (
            x_dequant.reshape(m, k).double() @ w_dequant.reshape(n, k).to(device).double().t()
        ).bfloat16()
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
