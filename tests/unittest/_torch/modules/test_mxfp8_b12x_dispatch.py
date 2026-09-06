# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU-only opt-in dispatch checks using the production class and fake CUDA tensors."""

import sys
from contextlib import nullcontext
from types import ModuleType

import pytest
import test_mxfp8_sm12x_dispatch as dispatch_tests
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

linear_module = dispatch_tests.linear_module


@pytest.mark.parametrize("outer", [None, "auto", "flashinfer"])
@pytest.mark.parametrize("n,k", [(16384, 2560), (2560, 6144)])
@pytest.mark.parametrize("m", [1, 4, 16])
def test_b12x_validated_calls_preserve_operands_and_outer_intent(
    linear_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    outer: str | None,
    n: int,
    k: int,
    m: int,
) -> None:
    native = dispatch_tests._mock_device(monkeypatch, (12, 1))
    fi = dispatch_tests._mock_flashinfer(monkeypatch)
    monkeypatch.setenv("TRTLLM_MXFP8_FLASHINFER_BACKEND", "b12x")
    if outer is not None:
        monkeypatch.setenv("TRTLLM_MXFP8_GEMM_BACKEND", outer)
    method = linear_module.MXFP8LinearMethod()
    fi.autotune.side_effect = lambda: nullcontext()
    with FakeTensorMode(), torch.device("cuda:0"):
        layer = nn.Module()
        layer.dtype = torch.bfloat16
        method.create_weights(layer, k, n, bias=False, dtype=layer.dtype)
        weight, scales = layer.weight, layer.weight_scale
        x = torch.empty((1, m, k), dtype=torch.bfloat16)
        quantized = torch.empty((m, k), dtype=torch.float8_e4m3fn)
        act_scales = torch.empty(method._swizzled_scale_size(m, k), dtype=torch.uint8)
        fi.mxfp8_quantize.return_value = (quantized, act_scales)
        fi.mm_mxfp8.return_value = torch.empty((m, n), dtype=layer.dtype)
        bias = torch.empty(n, dtype=layer.dtype)
        # Selection is frozen, including through the missing-warmup fallback.
        monkeypatch.setenv("TRTLLM_MXFP8_FLASHINFER_BACKEND", "cutlass")
        method.disable_flashinfer_auto()
        for context in (
            nullcontext(),
            linear_module.flashinfer_mxfp8_autotune(),
            linear_module.flashinfer_mxfp8_decode_graph_capture(),
        ):
            with context:
                result = method.apply(layer, x, bias)
            assert result.shape == (1, m, n)
            args = fi.mm_mxfp8.call_args.args
            assert args[0] is quantized and args[2] is act_scales
            assert args[1]._base is weight and args[1].stride() == (1, k)
            assert args[3] is scales and layer.weight is weight
            assert fi.mm_mxfp8.call_args.kwargs == {
                "backend": "b12x",
                "out_dtype": torch.bfloat16,
                "use_8x4_sf_layout": False,
            }
        assert scales.ndim == 1
        assert scales.numel() == method._swizzled_scale_size(n, k)
    assert method.backend == (outer or "auto")
    assert method.needs_flashinfer_autotune
    assert not method.needs_native_autotune
    for op in vars(native).values():
        op.assert_not_called()


@pytest.mark.parametrize(
    "selector,capability,n,k,m,dtype,input_dtype,device",
    [
        (None, (12, 1), 2560, 6144, 4, torch.bfloat16, torch.bfloat16, "cuda:0"),
        ("cutlass", (12, 1), 2560, 6144, 4, torch.bfloat16, torch.bfloat16, "cuda:0"),
        ("b12x", (12, 0), 2560, 6144, 4, torch.bfloat16, torch.bfloat16, "cuda:0"),
        ("b12x", (12, 1), 2560, 6144, 2, torch.bfloat16, torch.bfloat16, "cuda:0"),
        ("b12x", (12, 1), 2560, 6144, 8, torch.bfloat16, torch.bfloat16, "cuda:0"),
        ("b12x", (12, 1), 2560, 6144, 17, torch.bfloat16, torch.bfloat16, "cuda:0"),
        ("b12x", (12, 1), 2560, 6144, 128, torch.bfloat16, torch.bfloat16, "cuda:0"),
        ("b12x", (12, 1), 128, 2560, 4, torch.bfloat16, torch.bfloat16, "cuda:0"),
        ("b12x", (12, 1), 2560, 6176, 4, torch.bfloat16, torch.bfloat16, "cuda:0"),
        ("b12x", (12, 1), 2560, 6144, 4, torch.float16, torch.float16, "cuda:0"),
        ("b12x", (12, 1), 2560, 6144, 4, torch.bfloat16, torch.float16, "cuda:0"),
        ("b12x", (12, 1), 2560, 6144, 4, torch.bfloat16, torch.bfloat16, "cuda:1"),
    ],
)
def test_other_flashinfer_calls_keep_cutlass(
    linear_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    selector: str | None,
    capability: tuple[int, int],
    n: int,
    k: int,
    m: int,
    dtype: torch.dtype,
    input_dtype: torch.dtype,
    device: str,
) -> None:
    dispatch_tests._mock_device(monkeypatch, capability)
    fi = dispatch_tests._mock_flashinfer(monkeypatch)
    if selector is not None:
        monkeypatch.setenv("TRTLLM_MXFP8_FLASHINFER_BACKEND", selector)
    method = linear_module.MXFP8LinearMethod()
    with FakeTensorMode(), torch.device(device):
        layer = nn.Module()
        layer.dtype = dtype
        method.create_weights(layer, k, n, bias=False, dtype=dtype)
        x = torch.empty((m, k), dtype=input_dtype)
        fi.mxfp8_quantize.return_value = (
            torch.empty_like(x, dtype=torch.float8_e4m3fn),
            torch.empty(method._swizzled_scale_size(m, k), dtype=torch.uint8),
        )
        fi.mm_mxfp8.return_value = torch.empty((m, n), dtype=dtype)
        method.apply(layer, x, None)
    assert fi.mm_mxfp8.call_args.kwargs["backend"] == "cutlass"


@pytest.mark.parametrize("capability", [(10, 0), (10, 3), (12, 0), (12, 1)])
def test_b12x_never_overrides_explicit_native(
    linear_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capability: tuple[int, int],
) -> None:
    dispatch_tests._mock_device(monkeypatch, capability)
    monkeypatch.setenv("TRTLLM_MXFP8_FLASHINFER_BACKEND", "b12x")
    monkeypatch.setenv("TRTLLM_MXFP8_GEMM_BACKEND", "trtllm")
    monkeypatch.setitem(sys.modules, "flashinfer", None)
    method = linear_module.MXFP8LinearMethod()
    assert method.backend == "trtllm"
    assert method._b12x_device is None
    assert method.use_cutlass is (capability[0] == 10)


def test_cpu_opt_in_is_inert(linear_module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("TRTLLM_MXFP8_FLASHINFER_BACKEND", "b12x")
    monkeypatch.setitem(sys.modules, "flashinfer", None)
    method = linear_module.MXFP8LinearMethod()
    assert method.backend == "trtllm" and method._b12x_device is None


@pytest.mark.parametrize("selector", ["", "auto", "B12X", "invalid"])
def test_invalid_selector_fails_early(
    linear_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    selector: str,
) -> None:
    dispatch_tests._mock_device(monkeypatch, (12, 1))
    monkeypatch.setenv("TRTLLM_MXFP8_FLASHINFER_BACKEND", selector)
    with pytest.raises(ValueError, match="TRTLLM_MXFP8_FLASHINFER_BACKEND"):
        linear_module.MXFP8LinearMethod()


def test_active_opt_in_requires_flashinfer_import(
    linear_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatch_tests._mock_device(monkeypatch, (12, 1))
    monkeypatch.setenv("TRTLLM_MXFP8_FLASHINFER_BACKEND", "b12x")
    monkeypatch.setitem(sys.modules, "flashinfer", None)
    with pytest.raises(RuntimeError, match="flashinfer-python"):
        linear_module.MXFP8LinearMethod()


@pytest.mark.parametrize("error", [ValueError("CUDA 13 required"), RuntimeError("launch failed")])
def test_public_flashinfer_errors_propagate_without_retry(
    linear_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    dispatch_tests._mock_device(monkeypatch, (12, 1))
    fi = dispatch_tests._mock_flashinfer(monkeypatch)
    monkeypatch.setenv("TRTLLM_MXFP8_FLASHINFER_BACKEND", "b12x")
    method = linear_module.MXFP8LinearMethod()
    with FakeTensorMode(), torch.device("cuda:0"):
        layer = nn.Module()
        layer.dtype = torch.bfloat16
        method.create_weights(layer, 6144, 2560, bias=False, dtype=layer.dtype)
        x = torch.empty((1, 6144), dtype=layer.dtype)
        fi.mxfp8_quantize.return_value = (
            torch.empty_like(x, dtype=torch.float8_e4m3fn),
            torch.empty(method._swizzled_scale_size(1, 6144), dtype=torch.uint8),
        )
        fi.mm_mxfp8.side_effect = error
        with pytest.raises(type(error), match=str(error)):
            method.apply(layer, x, None)
    fi.mm_mxfp8.assert_called_once()
    assert fi.mm_mxfp8.call_args.kwargs["backend"] == "b12x"
    assert method.backend == "auto"
