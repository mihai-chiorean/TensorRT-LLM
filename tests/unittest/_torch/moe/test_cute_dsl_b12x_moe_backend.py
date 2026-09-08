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
"""CuteDslB12xFusedMoE gating and dispatch tests."""

import sys
import types
from typing import Optional
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe import fused_moe_cute_dsl_b12x
from tensorrt_llm._torch.moe.fused_moe.fused_moe_cute_dsl_b12x import CuteDslB12xFusedMoE
from tensorrt_llm._torch.moe.fused_moe.fused_moe_cutlass import CutlassFusedMoE
from tensorrt_llm._torch.moe.fused_moe.impl_blocks import MoEWeightOwnerMixin
from tensorrt_llm._torch.moe.fused_moe.impl_contract import (
    MoEDeployment,
    MoEEnvironment,
    MoEProblem,
    MoERejectReason,
    canonical_quant,
)
from tensorrt_llm._torch.moe.fused_moe.impl_environment import MoEDep
from tensorrt_llm._torch.moe.fused_moe.quantization import (
    NVFP4CuteDslB12xFusedMoEMethod,
    NVFP4CutlassFusedMoEMethod,
)
from tensorrt_llm._torch.utils import ActivationType
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

pytestmark = pytest.mark.cpu_only


# Spelled out rather than read from _SUPPORTED_SM_VERSIONS: deriving the input
# from the value under test hides a narrowing of that set, which is the
# direction that silently drops hardware support.
SUPPORTED_SM = [120, 121]


def _deployment(
    sm: int,
    *,
    flashinfer: bool = True,
    ep_size: int = 1,
    use_dp: bool = False,
    parallel_size: Optional[int] = None,
    eplb: bool = False,
) -> MoEDeployment:
    """Declare the machine rather than patching the probes that read it."""
    return MoEDeployment(
        ep_size=ep_size,
        tp_size=1,
        parallel_size=ep_size if parallel_size is None else parallel_size,
        use_dp=use_dp,
        num_slots=8,
        eplb_enabled=eplb,
        env=MoEEnvironment(
            sm=sm,
            available_deps=(MoEDep.FLASHINFER.value,) if flashinfer else (),
        ),
    )


def _problem(
    quant_algo=QuantAlgo.NVFP4, dtype=torch.bfloat16, swiglu_gptoss_style=None
) -> MoEProblem:
    return MoEProblem(
        quant=canonical_quant(quant_algo),
        dtype_act=dtype,
        hidden_size=2048,
        intermediate_size=2048,
        num_experts=8,
        top_k=2,
        swiglu_gptoss_style=swiglu_gptoss_style,
    )


@pytest.mark.parametrize("sm_version", [80, 89, 90, 100, 103])
def test_can_implement_rejects_unsupported_sm(sm_version):
    verdict = CuteDslB12xFusedMoE.can_implement(_problem(), _deployment(sm_version))
    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.SM_UNSUPPORTED
    assert f"SM{sm_version}" in verdict.detail


@pytest.mark.parametrize("sm_version", SUPPORTED_SM)
@pytest.mark.parametrize("quant_algo", [QuantAlgo.NVFP4, QuantAlgo.W4A16_NVFP4])
def test_can_implement_accepts_supported_sm(sm_version, quant_algo):
    verdict = CuteDslB12xFusedMoE.can_implement(_problem(quant_algo), _deployment(sm_version))
    assert verdict.eligible
    assert verdict.reject_reason is None


@pytest.mark.parametrize(
    "quant_algo",
    [
        None,
        QuantAlgo.FP8,
        QuantAlgo.FP8_BLOCK_SCALES,
        QuantAlgo.W4A16_MXFP4,
        QuantAlgo.W4A8_MXFP4_FP8,
    ],
)
def test_can_implement_rejects_non_nvfp4(quant_algo):
    """Only NVFP4 is supported; everything else must be turned away."""
    verdict = CuteDslB12xFusedMoE.can_implement(_problem(quant_algo), _deployment(120))
    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.QUANT_UNSUPPORTED


def test_can_implement_rejects_swiglu_gptoss_style():
    verdict = CuteDslB12xFusedMoE.can_implement(
        _problem(swiglu_gptoss_style=True), _deployment(120)
    )
    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.ACTIVATION_UNSUPPORTED


@pytest.mark.parametrize("dtype", [torch.float32, torch.float8_e4m3fn])
def test_can_implement_rejects_unsupported_activation_dtype(dtype):
    verdict = CuteDslB12xFusedMoE.can_implement(_problem(dtype=dtype), _deployment(120))
    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.DTYPE_UNSUPPORTED


def test_can_implement_rejects_missing_flashinfer():
    verdict = CuteDslB12xFusedMoE.can_implement(_problem(), _deployment(120, flashinfer=False))
    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.DEP_MISSING


def test_can_implement_rejects_eplb():
    """EPLB is its own reject class, not a topology one: the machine is fine."""
    verdict = CuteDslB12xFusedMoE.can_implement(_problem(), _deployment(120, eplb=True))
    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.EPLB_UNSUPPORTED


def test_can_implement_rejects_expert_parallelism():
    verdict = CuteDslB12xFusedMoE.can_implement(_problem(), _deployment(120, ep_size=2))
    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.TOPOLOGY_UNSUPPORTED


def test_can_implement_rejects_attention_dp_without_expert_parallelism():
    """moe_tp == tp leaves ep_size at 1, so the EP gate alone would let this in."""
    verdict = CuteDslB12xFusedMoE.can_implement(
        _problem(), _deployment(120, ep_size=1, use_dp=True, parallel_size=2)
    )
    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.TOPOLOGY_UNSUPPORTED
    assert "attention-DP" in verdict.detail


# --------------------------------------------------------------------------
# Hybrid CUTLASS-prefill / b12x-decode dispatch predicate tests
#
# ``_route_to_cutlass`` is a pure shape predicate on its input ``x``; we test
# it via a stub that holds the class constant, sidestepping the full
# CutlassFusedMoE constructor (which needs a routing method, real model
# config, etc.).
# --------------------------------------------------------------------------


class _RoutePredicateStub:
    """Minimal carrier for the unbound dispatch predicate."""

    _PREFILL_VIA_CUTLASS_THRESHOLD = CuteDslB12xFusedMoE._PREFILL_VIA_CUTLASS_THRESHOLD

    _route_to_cutlass = CuteDslB12xFusedMoE._route_to_cutlass


def test_dispatch_routes_prefill_shape_via_cutlass():
    stub = _RoutePredicateStub()
    x = torch.empty(_RoutePredicateStub._PREFILL_VIA_CUTLASS_THRESHOLD, 1024)
    assert stub._route_to_cutlass(x) is True


def test_dispatch_just_below_threshold_takes_b12x():
    stub = _RoutePredicateStub()
    x = torch.empty(_RoutePredicateStub._PREFILL_VIA_CUTLASS_THRESHOLD - 1, 1024)
    assert stub._route_to_cutlass(x) is False


def test_dispatch_decode_shape_takes_b12x():
    stub = _RoutePredicateStub()
    x = torch.empty(1, 1024)
    assert stub._route_to_cutlass(x) is False


def test_w4a16_nvfp4_prefill_quantize_input_stays_on_b12x():
    moe = object.__new__(CuteDslB12xFusedMoE)
    moe.quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4)
    x = torch.empty(CuteDslB12xFusedMoE._PREFILL_VIA_CUTLASS_THRESHOLD, 1024)

    with patch.object(
        CutlassFusedMoE,
        "quantize_input",
        side_effect=AssertionError("W4A16_NVFP4 prefill must not route through CUTLASS"),
    ):
        out, out_sf = CuteDslB12xFusedMoE.quantize_input(moe, x)

    assert out is x
    assert out_sf is None


def _w4a16_nvfp4_module() -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
    class _RoutingMethod:
        experts_per_token = 4

    num_experts = 2
    hidden_size = 128
    logical_intermediate_size = 1856
    padded_intermediate_size = 1920
    module = torch.nn.Module()
    module.num_experts = num_experts
    module.hidden_size = hidden_size
    module.intermediate_size_per_partition = logical_intermediate_size
    module.moe_max_num_tokens = 8
    module.routing_method = _RoutingMethod()
    module.activation_type = ActivationType.Swiglu
    module.quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4)
    module.w3_w1_weight = torch.empty(
        num_experts,
        2 * padded_intermediate_size,
        hidden_size // 16,
        dtype=torch.int64,
    )
    module.w2_weight = torch.empty(
        num_experts,
        hidden_size,
        padded_intermediate_size // 16,
        dtype=torch.int64,
    )
    w3_w1_weight_scale = torch.ones(
        num_experts,
        2 * padded_intermediate_size,
        hidden_size // 16,
        dtype=torch.float8_e4m3fn,
    )
    w2_weight_scale = torch.ones(
        num_experts,
        hidden_size,
        padded_intermediate_size // 16,
        dtype=torch.float8_e4m3fn,
    )
    module.w3_w1_weight_scale = w3_w1_weight_scale.clone()
    module.w2_weight_scale = w2_weight_scale.clone()
    module.fc31_alpha = torch.tensor([0.25, 0.5], dtype=torch.float32)
    module.fc2_alpha = torch.tensor([0.125, 0.25], dtype=torch.float32)
    module.fc31_input_scale = torch.tensor(2.0, dtype=torch.float32)
    module.fc2_input_scale = torch.tensor(4.0, dtype=torch.float32)
    return module, w3_w1_weight_scale, w2_weight_scale


def _install_b12x_wrapper(monkeypatch: pytest.MonkeyPatch, wrapper: type) -> None:
    def _convert_sf_to_mma_layout(
        scales: torch.Tensor, *, m: int, k: int, num_groups: int
    ) -> torch.Tensor:
        del m, k, num_groups
        return scales

    flashinfer = types.ModuleType("flashinfer")
    flashinfer.B12xMoEWrapper = wrapper
    cute_dsl = types.ModuleType("flashinfer.cute_dsl")
    utils = types.ModuleType("flashinfer.cute_dsl.utils")
    utils.convert_sf_to_mma_layout = _convert_sf_to_mma_layout
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.cute_dsl", cute_dsl)
    monkeypatch.setitem(sys.modules, "flashinfer.cute_dsl.utils", utils)


def _transform_b12x_weights(module: torch.nn.Module) -> None:
    with patch.object(NVFP4CutlassFusedMoEMethod, "transform_weights", return_value=None):
        NVFP4CuteDslB12xFusedMoEMethod().transform_weights(module)


def _assert_w4a16_scale_contract(
    module: torch.nn.Module,
    w3_w1_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
) -> None:
    assert module._b12x_weights["fc2_input_scale"] is None
    assert torch.equal(
        module._b12x_weights["w1_weight_sf"].float(),
        w3_w1_weight_scale.float(),
    )
    assert torch.equal(
        module._b12x_weights["w2_weight_sf"].float(),
        w2_weight_scale.float(),
    )
    assert torch.allclose(
        module._b12x_weights["w1_alpha"],
        torch.tensor([0.5, 1.0], dtype=torch.float32),
    )
    assert torch.allclose(
        module._b12x_weights["w2_alpha"],
        torch.tensor([0.5, 1.0], dtype=torch.float32),
    )


def test_w4a16_nvfp4_post_load_default_preserves_scale_contract_and_omits_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _OldWrapper:
        calls = []

        def __init__(self, **kwargs: object) -> None:
            self._moe_output = None
            self.calls.append(kwargs)

    _install_b12x_wrapper(monkeypatch, _OldWrapper)
    module, w3_w1_weight_scale, w2_weight_scale = _w4a16_nvfp4_module()
    _transform_b12x_weights(module)
    assert _OldWrapper.calls
    wrapper_kwargs = _OldWrapper.calls[0]
    assert "enable_w4a16_tc_decode" not in wrapper_kwargs
    assert wrapper_kwargs["quant_mode"] == "w4a16"
    assert wrapper_kwargs["intermediate_size"] == 1920
    _assert_w4a16_scale_contract(module, w3_w1_weight_scale, w2_weight_scale)


def test_w4a16_nvfp4_post_load_keeps_capable_wrapper_default_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _TrueWrapper:
        calls = []

        def __init__(self, *, enable_w4a16_tc_decode: bool = True, **kwargs: object) -> None:
            self._moe_output = None
            self.enable_w4a16_tc_decode = enable_w4a16_tc_decode
            self.calls.append((enable_w4a16_tc_decode, kwargs))

    _install_b12x_wrapper(monkeypatch, _TrueWrapper)
    module, w3_w1_weight_scale, w2_weight_scale = _w4a16_nvfp4_module()
    _transform_b12x_weights(module)
    assert _TrueWrapper.calls[0][0] is True
    assert "enable_w4a16_tc_decode" not in _TrueWrapper.calls[0][1]
    _assert_w4a16_scale_contract(module, w3_w1_weight_scale, w2_weight_scale)


def test_w4a16_nvfp4_post_load_nonatomic_forwards_false_and_preserves_scale_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FalseWrapper:
        calls = []

        def __init__(self, *, enable_w4a16_tc_decode: bool = True, **kwargs: object) -> None:
            self._moe_output = None
            self.enable_w4a16_tc_decode = enable_w4a16_tc_decode
            self.calls.append((enable_w4a16_tc_decode, kwargs))

    _install_b12x_wrapper(monkeypatch, _FalseWrapper)
    module, w3_w1_weight_scale, w2_weight_scale = _w4a16_nvfp4_module()
    module._b12x_enable_w4a16_tc_decode = False
    _transform_b12x_weights(module)
    assert _FalseWrapper.calls[0][0] is False
    assert _FalseWrapper.calls[0][1]["quant_mode"] == "w4a16"
    _assert_w4a16_scale_contract(module, w3_w1_weight_scale, w2_weight_scale)


def test_w4a16_nvfp4_post_load_rejects_wrapper_that_ignores_nonatomic_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _IgnoringWrapper:
        def __init__(self, *, enable_w4a16_tc_decode: bool = True, **kwargs: object) -> None:
            self._moe_output = None
            self.enable_w4a16_tc_decode = True

    _install_b12x_wrapper(monkeypatch, _IgnoringWrapper)
    module, _, _ = _w4a16_nvfp4_module()
    module._b12x_enable_w4a16_tc_decode = False
    with pytest.raises(RuntimeError, match="did not retain"):
        _transform_b12x_weights(module)


def test_b12x_nonatomic_rejects_non_w4a16_quantization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FalseWrapper:
        def __init__(self, *, enable_w4a16_tc_decode: bool = True, **kwargs: object) -> None:
            self._moe_output = None
            self.enable_w4a16_tc_decode = enable_w4a16_tc_decode

    _install_b12x_wrapper(monkeypatch, _FalseWrapper)
    module, _, _ = _w4a16_nvfp4_module()
    module._b12x_enable_w4a16_tc_decode = False
    module.quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    with pytest.raises(RuntimeError, match="requires W4A16_NVFP4"):
        _transform_b12x_weights(module)


def test_nonatomic_post_load_is_idempotent_and_reuses_shared_cpu_output_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _CpuOutputWrapper:
        calls: list["_CpuOutputWrapper"] = []

        def __init__(self, *, enable_w4a16_tc_decode: bool = True, **kwargs: object) -> None:
            del kwargs
            self._moe_output = torch.empty((8, 128), dtype=torch.float32)
            self.enable_w4a16_tc_decode = enable_w4a16_tc_decode
            self.calls.append(self)

    class _LifecycleOwner(MoEWeightOwnerMixin, torch.nn.Module):
        def __init__(self) -> None:
            torch.nn.Module.__init__(self)

        def _get_quant_method(self) -> object:
            raise AssertionError("post-load lifecycle test installs its concrete quant method")

    def owner() -> tuple[_LifecycleOwner, torch.Tensor, torch.Tensor]:
        source, w1_scale, w2_scale = _w4a16_nvfp4_module()
        result = _LifecycleOwner()
        result.__dict__.update(source.__dict__)
        result._weights_created = True
        result._b12x_enable_w4a16_tc_decode = False
        result.quant_method = NVFP4CuteDslB12xFusedMoEMethod()
        return result, w1_scale, w2_scale

    _install_b12x_wrapper(monkeypatch, _CpuOutputWrapper)
    monkeypatch.setattr(fused_moe_cute_dsl_b12x, "_SHARED_MOE_OUTPUT_BUF", {})
    first, first_w1_scale, first_w2_scale = owner()
    with (
        patch.object(NVFP4CutlassFusedMoEMethod, "transform_weights", return_value=None),
        patch.object(NVFP4CuteDslB12xFusedMoEMethod, "cache_derived_state", return_value=None),
    ):
        first.post_load_weights()
        wrapper = first.b12x_wrapper
        output = wrapper._moe_output
        prepared_w1_scale = first._b12x_weights["w1_weight_sf"].clone()
        prepared_w2_scale = first._b12x_weights["w2_weight_sf"].clone()
        monkeypatch.setenv("TRTLLM_QWEN4_MTP_B12X_NONATOMIC", "0")
        first.post_load_weights()

        second, second_w1_scale, second_w2_scale = owner()
        second.post_load_weights()

    assert len(_CpuOutputWrapper.calls) == 2
    assert first.b12x_wrapper is wrapper
    assert first.b12x_wrapper._moe_output is output
    assert first.b12x_wrapper.enable_w4a16_tc_decode is False
    assert second.b12x_wrapper.enable_w4a16_tc_decode is False
    assert second.b12x_wrapper._moe_output is output
    assert torch.equal(first._b12x_weights["w1_weight_sf"], prepared_w1_scale)
    assert torch.equal(first._b12x_weights["w2_weight_sf"], prepared_w2_scale)
    _assert_w4a16_scale_contract(first, first_w1_scale, first_w2_scale)
    _assert_w4a16_scale_contract(second, second_w1_scale, second_w2_scale)


def test_dispatch_rejects_non_tensor():
    """Non-tensor inputs (e.g. Fp4QuantizedTensor) stay on the b12x path
    so the existing ValueError surfaces in quantize_input."""
    stub = _RoutePredicateStub()
    assert stub._route_to_cutlass(object()) is False
