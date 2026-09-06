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
"""CPU-isolated dispatch checks without importing TensorRT native bindings."""

import ast
import importlib.util
import os
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

_ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def linear_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Execute the actual MXFP8 class with unrelated runtime imports excluded."""
    path = _ROOT / "tensorrt_llm/_torch/modules/linear.py"
    tree = ast.parse(path.read_text())
    names = {
        "_mxfp8_cutlass_op_available",
        "flashinfer_mxfp8_autotune",
        "flashinfer_mxfp8_decode_graph_capture",
        "MXFP8LinearMethod",
        "load_weight_shard",
        "load_weights_vanilla_helper",
        "copy_weight",
    }
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            body.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id.startswith("_FLASHINFER_MXFP8_")
            for target in node.targets
        ):
            body.append(node)

    module = ModuleType("linear_under_test")
    module.__dict__.update(
        torch=torch,
        F=torch.nn.functional,
        Parameter=nn.Parameter,
        os=os,
        ContextVar=ContextVar,
        contextmanager=contextmanager,
        LinearMethodBase=object,
        is_device_integrated=lambda: True,
        fp4_utils=SimpleNamespace(
            pad_up=lambda value, alignment: (value + alignment - 1) // alignment * alignment
        ),
        logger=SimpleNamespace(warning_once=Mock()),
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec"
        ),
        module.__dict__,
    )
    utils_name = "tensorrt_llm._torch.modules.mxfp8_utils"
    spec = importlib.util.spec_from_file_location(
        utils_name, _ROOT / "tensorrt_llm/_torch/modules/mxfp8_utils.py"
    )
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    monkeypatch.setitem(sys.modules, utils_name, utils)
    monkeypatch.delenv("TRTLLM_MXFP8_GEMM_BACKEND", raising=False)
    monkeypatch.delenv("TRTLLM_MXFP8_FLASHINFER_BACKEND", raising=False)
    return module


def _mock_device(monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int]) -> SimpleNamespace:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    native = SimpleNamespace(
        mxfp8_mxfp8_gemm=Mock(side_effect=AssertionError("unsupported native GEMM called")),
        mxfp8_mxfp8_gemm_autotuned=Mock(
            side_effect=AssertionError("unsupported native autotuner called")
        ),
        mxfp8_quantize=Mock(side_effect=AssertionError("native quantizer called")),
        block_scale_interleave=Mock(side_effect=AssertionError("native interleave called")),
    )
    monkeypatch.setattr(torch.ops, "trtllm", native)
    return native


def _mock_flashinfer(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    flashinfer = SimpleNamespace(
        mm_mxfp8=Mock(),
        mxfp8_quantize=Mock(),
        block_scale_interleave=Mock(side_effect=lambda scale: scale.flatten()),
        autotune=Mock(),
    )
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    return flashinfer


@pytest.mark.parametrize(
    "capability,expected",
    [
        ((9, 0), False),
        ((10, 0), True),
        ((10, 3), True),
        ((10, 7), True),
        ((11, 0), False),
        ((12, 0), False),
        ((12, 1), False),
        ((13, 0), False),
    ],
)
def test_native_gate_matches_cpp_sm100_family(
    linear_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capability: tuple[int, int],
    expected: bool,
) -> None:
    _mock_device(monkeypatch, capability)
    assert linear_module._mxfp8_cutlass_op_available() is expected


def test_native_gate_needs_op_and_cuda(
    linear_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_device(monkeypatch, (10, 0))
    monkeypatch.setattr(torch.ops, "trtllm", SimpleNamespace())
    assert not linear_module._mxfp8_cutlass_op_available()
    _mock_device(monkeypatch, (10, 0))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert not linear_module._mxfp8_cutlass_op_available()


@pytest.mark.parametrize("backend", [None, "auto", "flashinfer"])
def test_sm121_flashinfer_preserves_weights_and_avoids_native_ops(
    linear_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    backend: str | None,
) -> None:
    native = _mock_device(monkeypatch, (12, 1))
    flashinfer = _mock_flashinfer(monkeypatch)
    if backend is not None:
        monkeypatch.setenv("TRTLLM_MXFP8_GEMM_BACKEND", backend)
    method = linear_module.MXFP8LinearMethod()
    module = nn.Module()
    module.dtype = torch.bfloat16
    method.create_weights(module, 128, 128, bias=False, dtype=module.dtype)
    assert not method.use_cutlass and method._use_flashinfer_sm12x
    assert method.backend == ("flashinfer" if backend == "flashinfer" else "auto")
    scales = torch.full((128, 4), 127, dtype=torch.uint8)
    native.block_scale_interleave.side_effect = lambda scale: scale.flatten()
    method._store_scale(module, scales)
    native.block_scale_interleave.assert_called_once_with(scales)
    flashinfer.block_scale_interleave.assert_not_called()
    x = torch.ones((1, 2, 128), dtype=torch.bfloat16)
    quantized = x.reshape(2, 128).to(torch.float8_e4m3fn)
    activation_scales = torch.full((512,), 127, dtype=torch.uint8)
    flashinfer.mxfp8_quantize.return_value = (quantized, activation_scales)
    flashinfer.mm_mxfp8.return_value = torch.ones((2, 128), dtype=torch.bfloat16)
    weight_ptr = module.weight.data_ptr()
    bias = torch.ones(128, dtype=torch.bfloat16)
    method.enable_native_autotune()
    assert not method.needs_native_autotune
    assert not method.enable_flashinfer_auto()
    method.disable_flashinfer_auto()
    result = method.apply(module, x, bias)
    torch.testing.assert_close(result, torch.full_like(x, 2))
    args = flashinfer.mm_mxfp8.call_args.args
    assert args[0] is quantized and args[2] is activation_scales
    assert args[1].data_ptr() == weight_ptr and args[3] is module.weight_scale
    assert module.weight.dtype == torch.float8_e4m3fn
    assert flashinfer.mm_mxfp8.call_args.kwargs == {
        "out_dtype": torch.bfloat16,
        "use_8x4_sf_layout": False,
        "backend": "cutlass",
    }
    for name, op in vars(native).items():
        if name != "block_scale_interleave":
            op.assert_not_called()


@pytest.mark.parametrize(
    "n,k,dtype,supported",
    [
        (128, 128, torch.bfloat16, True),
        (16384, 2560, torch.bfloat16, True),
        (128, 160, torch.float16, True),
        (96, 2560, torch.bfloat16, False),
        (129, 2560, torch.bfloat16, False),
        (128, 64, torch.bfloat16, False),
        (128, 128, torch.float32, False),
    ],
)
def test_sm12x_shape_selection_before_scale_allocation(
    linear_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    n: int,
    k: int,
    dtype: torch.dtype,
    supported: bool,
) -> None:
    _mock_device(monkeypatch, (12, 0))
    _mock_flashinfer(monkeypatch)
    method = linear_module.MXFP8LinearMethod()
    with torch.device("meta"):
        module = nn.Module()
        method.create_weights(module, k, n, bias=False, dtype=dtype)
    assert method._use_flashinfer_sm12x is supported
    assert module.weight_scale.ndim == (1 if supported else 2)


@pytest.mark.parametrize(
    "backend,n,k",
    [(None, 96, 2560), ("flashinfer", 96, 2560), ("auto", 129, 128), ("trtllm", 128, 128)],
)
@pytest.mark.parametrize("fi_backend", ["cutlass", "b12x"])
def test_unsupported_sm121_shape_or_explicit_native_uses_exact_reference(
    linear_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    backend: str | None,
    n: int,
    k: int,
    fi_backend: str,
) -> None:
    native = _mock_device(monkeypatch, (12, 1))
    flashinfer = _mock_flashinfer(monkeypatch)
    if backend is not None:
        monkeypatch.setenv("TRTLLM_MXFP8_GEMM_BACKEND", backend)
    monkeypatch.setenv("TRTLLM_MXFP8_FLASHINFER_BACKEND", fi_backend)
    method = linear_module.MXFP8LinearMethod()
    module = nn.Module()
    module.dtype = torch.bfloat16
    method.create_weights(module, k, n, bias=False, dtype=module.dtype)
    packed = (torch.arange(n * k).reshape(n, k).remainder(7) - 3).to(torch.float8_e4m3fn)
    scales = (torch.arange(n * k // 32).reshape(n, k // 32).remainder(3) + 125).to(torch.uint8)
    module.weight.data.copy_(packed)
    method._store_scale(module, scales)
    x = torch.ones((2, k), dtype=torch.bfloat16)
    bias = torch.ones(n, dtype=torch.bfloat16)
    decoded = (
        packed.float() * torch.exp2(scales.float() - 127).repeat_interleave(32, dim=-1)
    ).bfloat16()
    expected = torch.nn.functional.linear(x, decoded, bias)
    actual = method.apply(module, x, bias)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(module.weight.float(), packed.float(), rtol=0, atol=0)
    assert module.weight_scale.shape == scales.shape
    assert not method.needs_flashinfer_autotune
    for op in vars(native).values():
        op.assert_not_called()
    flashinfer.mm_mxfp8.assert_not_called()
    flashinfer.mxfp8_quantize.assert_not_called()
    flashinfer.block_scale_interleave.assert_not_called()


def test_missing_flashinfer_defaults_to_reference_but_explicit_request_fails(
    linear_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_device(monkeypatch, (12, 1))
    monkeypatch.setitem(sys.modules, "flashinfer", None)
    method = linear_module.MXFP8LinearMethod()
    assert method.backend == "trtllm" and not method.use_cutlass
    monkeypatch.setenv("TRTLLM_MXFP8_GEMM_BACKEND", "flashinfer")
    with pytest.raises(RuntimeError, match="flashinfer-python"):
        linear_module.MXFP8LinearMethod()


@pytest.mark.parametrize("backend", ["", "invalid"])
def test_invalid_backend_still_rejected(
    linear_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    _mock_device(monkeypatch, (12, 1))
    monkeypatch.setenv("TRTLLM_MXFP8_GEMM_BACKEND", backend)
    with pytest.raises(ValueError, match="BACKEND"):
        linear_module.MXFP8LinearMethod()


@pytest.mark.parametrize(
    "name",
    [
        "test_mxfp8_flashinfer_call_contract",
        "test_mxfp8_auto_keeps_eager_native_and_captures_flashinfer",
        "test_mxfp8_auto_fallback_does_not_rearm_native_autotuning",
        "test_mxfp8_native_autotuner_dispatch",
    ],
)
@pytest.mark.parametrize("fi_backend", ["cutlass", "b12x"])
def test_existing_sm100_dispatch_regressions(
    linear_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    fi_backend: str,
) -> None:
    monkeypatch.setenv("TRTLLM_MXFP8_FLASHINFER_BACKEND", fi_backend)
    path = _ROOT / "tests/unittest/_torch/modules/test_mxfp8_linear.py"
    body = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in (name, "_mock_mxfp8_ops")
    ]
    namespace = dict(
        torch=torch,
        Mock=Mock,
        SimpleNamespace=SimpleNamespace,
        sys=sys,
        linear_module=linear_module,
        MXFP8LinearMethod=linear_module.MXFP8LinearMethod,
        flashinfer_mxfp8_decode_graph_capture=linear_module.flashinfer_mxfp8_decode_graph_capture,
    )
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    namespace[name](monkeypatch)
