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
"""CPU-isolated production engine warmup checks; no native imports or GPU execution.

The complete warmup and batch-admission methods are AST-loaded unchanged except
for two local imports supplied by the harness. Linear dispatch uses the existing
source-isolated fixture; FlashInfer, cache management, and scheduling are mocked.
"""

import ast
import os
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import test_mxfp8_sm12x_dispatch as dispatch_tests
import torch
from test_mxfp8_sm12x_dispatch import _mock_device, _mock_flashinfer
from torch import nn

_ROOT = Path(__file__).resolve().parents[4]
linear_module = dispatch_tests.linear_module


@pytest.fixture
def engine_module(linear_module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    path = _ROOT / "tensorrt_llm/_torch/pyexecutor/model_engine.py"
    tree = ast.parse(path.read_text())
    engine = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PyTorchModelEngine"
    )
    methods = [
        node
        for node in engine.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_run_autotuner_warmup", "_should_run_warmup_batch")
    ]
    assert len(methods) == 2
    for method in methods:
        method.body = [
            node
            for node in method.body
            if not (
                isinstance(node, ast.ImportFrom)
                and node.level == 2
                and node.module in ("custom_ops.torch_custom_ops", "modules.linear")
            )
        ]
    module = ModuleType("sm12x_warmup_engine_under_test")
    module.__dict__.update(
        os=os,
        torch=torch,
        MXFP8LinearMethod=linear_module.MXFP8LinearMethod,
        flashinfer_mxfp8_autotune=linear_module.flashinfer_mxfp8_autotune,
        MXFP8GemmRunner=SimpleNamespace(sync_all_tactic_caches=Mock()),
        AutoTuner=SimpleNamespace(get=Mock(return_value=Mock(profiling_cache={}))),
        autotune=Mock(side_effect=lambda **kwargs: nullcontext()),
        clear_memory_buffers=Mock(),
        logger=Mock(),
    )
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=body + methods, type_ignores=[])),
            str(path),
            "exec",
        ),
        module.__dict__,
    )
    monkeypatch.delenv("TLLM_AUTOTUNER_CACHE_PATH", raising=False)
    monkeypatch.setattr(torch.cuda, "empty_cache", Mock())
    monkeypatch.setattr(
        torch.cuda, "synchronize", Mock(side_effect=AssertionError("unexpected GPU work"))
    )
    return module


@pytest.mark.parametrize("backend", [None, "auto", "flashinfer"])
@pytest.mark.parametrize("out_features", [96, 128, 2560])
@pytest.mark.parametrize("fi_backend", ["cutlass", "b12x"])
def test_sm121_missing_warmup_preserves_backend_intent_and_scale_layout(
    linear_module: ModuleType,
    engine_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    backend: str | None,
    out_features: int,
    fi_backend: str,
) -> None:
    native = _mock_device(monkeypatch, (12, 1))
    flashinfer = _mock_flashinfer(monkeypatch)
    flashinfer.autotune.side_effect = lambda: nullcontext()
    if backend is not None:
        monkeypatch.setenv("TRTLLM_MXFP8_GEMM_BACKEND", backend)
    monkeypatch.setenv("TRTLLM_MXFP8_FLASHINFER_BACKEND", fi_backend)
    method = linear_module.MXFP8LinearMethod()
    layer = nn.Module()
    layer.dtype = torch.bfloat16
    in_features = 6144 if out_features == 2560 else 128
    method.create_weights(layer, in_features, out_features, bias=False, dtype=layer.dtype)
    layer.weight.data.copy_(torch.ones_like(layer.weight, dtype=torch.bfloat16))
    method._store_scale(
        layer, torch.full((out_features, in_features // 32), 127, dtype=torch.uint8)
    )
    weight_bytes = layer.weight.view(torch.uint8).clone()
    scale_bytes = layer.weight_scale.clone()
    weight_pointer = layer.weight.data_ptr()
    scale_pointer = layer.weight_scale.data_ptr()
    eligible = out_features >= 128
    assert method.needs_flashinfer_autotune is eligible
    assert method.backend == (backend or "auto")

    engine = SimpleNamespace(
        llm_args=SimpleNamespace(enable_autotuner=True),
        cuda_graph_runner=SimpleNamespace(enabled=True),
        model=SimpleNamespace(modules=lambda: [SimpleNamespace(quant_method=method)]),
        mapping=SimpleNamespace(tp_size=1, has_pp=lambda: False),
        dist=object(),
        kv_cache_manager_key="kv_cache",
        max_num_tokens=16,
        batch_size=16,
        max_seq_len=2,
        original_max_draft_len=0,
        is_draft_model=False,
        guided_decoder=None,
        max_total_draft_tokens=0,
        no_cuda_graph=lambda: nullcontext(),
        _is_distributed_forward=lambda: False,
        _create_warmup_request=Mock(return_value=object()),
        _release_batch_context=Mock(side_effect=lambda *args: nullcontext(None)),
        forward=Mock(side_effect=AssertionError("missing batch must not execute")),
        _release_megamoe_profiling_scratch=Mock(),
    )
    engine._should_run_warmup_batch = lambda *args: engine_module._should_run_warmup_batch(
        engine, *args
    )
    resources = SimpleNamespace(
        get_resource_manager=Mock(
            return_value=SimpleNamespace(get_num_available_tokens=lambda **kwargs: 16)
        )
    )
    should_raise = eligible and backend == "flashinfer"
    expected_error = (
        pytest.raises(RuntimeError, match="explicitly requested.*warmup forward could not run")
        if should_raise
        else nullcontext()
    )
    with expected_error:
        engine_module._run_autotuner_warmup(engine, resources)

    assert method.backend == (backend or "auto")
    assert method._use_flashinfer_sm12x is eligible
    assert method._b12x_shape_eligible is (fi_backend == "b12x" and out_features == 2560)
    assert not method._flashinfer_autotuned and not method.needs_native_autotune
    assert layer.weight.data_ptr() == weight_pointer
    assert layer.weight_scale.data_ptr() == scale_pointer
    torch.testing.assert_close(layer.weight.view(torch.uint8), weight_bytes, rtol=0, atol=0)
    torch.testing.assert_close(layer.weight_scale, scale_bytes, rtol=0, atol=0)
    assert layer.weight_scale.ndim == (1 if eligible else 2)
    engine.forward.assert_not_called()
    assert engine._create_warmup_request.call_count == (4 if eligible else 2)
    assert flashinfer.autotune.call_count == int(eligible)
    engine_module.autotune.assert_called_once_with(cache_path=None)
    engine_module.MXFP8GemmRunner.sync_all_tactic_caches.assert_not_called()
    engine_module.AutoTuner.get().cache_pp_recv.assert_not_called()
    engine_module.AutoTuner.get().cache_pp_send.assert_not_called()
    assert engine_module.clear_memory_buffers.call_count == int(not should_raise)

    # After a skipped optional pass, execute the chosen path outside tuning or capture.
    if not should_raise:
        x = torch.ones((1, in_features), dtype=torch.bfloat16)
        expected = torch.full((1, out_features), in_features, dtype=torch.bfloat16)
        flashinfer.mxfp8_quantize.return_value = (
            x.to(torch.float8_e4m3fn),
            torch.full((512,), 127, dtype=torch.uint8),
        )
        flashinfer.mm_mxfp8.return_value = expected.clone()
        actual = method.apply(layer, x, None)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert flashinfer.mxfp8_quantize.call_count == int(eligible)
        assert flashinfer.mm_mxfp8.call_count == int(eligible)
        if eligible:
            args = flashinfer.mm_mxfp8.call_args.args
            assert args[1].data_ptr() == weight_pointer
            assert args[3] is layer.weight_scale
            assert flashinfer.mm_mxfp8.call_args.kwargs["backend"] == "cutlass"
    for op in vars(native).values():
        op.assert_not_called()
