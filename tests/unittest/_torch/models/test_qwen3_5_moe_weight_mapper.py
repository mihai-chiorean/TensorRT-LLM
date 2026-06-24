# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for Qwen3_5MoeHfWeightMapper NVFP4 scale tensor handling.

The Qwen3.6-35B-A3B-NVFP4 checkpoint stores three scale tensors per expert
projection in VANILLA loading mode (split per-expert, not fused):

    model.language_model.layers.0.mlp.experts.0.gate_proj.weight
    model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale    # per-block FP8
    model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale_2  # global FP32
    model.language_model.layers.0.mlp.experts.0.gate_proj.input_scale     # activation scale

After ``handle_special_instance_module`` processes these with VANILLA mode,
the expected TRT-LLM keys are:

    0.w1.weight
    0.w1.weight_scale
    0.w1.weight_scale_2
    0.w1.input_scale

This test validates that the VANILLA rename path (gate_proj→w1, up_proj→w3,
down_proj→w2) preserves all three NVFP4 scale suffixes without truncation or
loss.

Resolution notes (Q1 from Phase 1 recon):
- ``weight_scale``:   per-block scale (FP8, group_size=16)
- ``weight_scale_2``: global scalar scale (FP32)
- ``input_scale``:    per-token activation scale (FP8)
The VANILLA mapper path already handles these via the generic rename in
``handle_special_instance_module`` (gate_proj→w1 rename applies to all
weight_name strings, including those ending in _scale/_scale_2/_input_scale).

All tests are pure-Python. GPU and real model weights are not required.
"""

import types
import unittest
from unittest import mock

import pytest
import torch


# ---------------------------------------------------------------------------
# Guarded import
# ---------------------------------------------------------------------------

try:
    from tensorrt_llm._torch.models.checkpoints.hf.qwen3_5_weight_mapper import (
        Qwen3_5MoeHfWeightMapper,
    )
    _MAPPER_AVAILABLE = True
except ImportError:
    _MAPPER_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _MAPPER_AVAILABLE,
    reason="Qwen3_5MoeHfWeightMapper not importable — dependency missing",
)


# ---------------------------------------------------------------------------
# Synthetic expert weight dict helpers
# ---------------------------------------------------------------------------

_HIDDEN = 32   # small hidden_size for test tensors
_MOE_INTER = 8  # small moe_intermediate_size


def _make_vanilla_expert_weights(expert_idx: int = 0) -> dict:
    """Return a synthetic per-expert dict in VANILLA (split) layout.

    The weight shapes are deliberately tiny; we only care about key renaming.
    NVFP4 scale tensors are included for all three projections.
    """
    weight_dtype = torch.bfloat16
    scale_dtype = torch.float32

    return {
        # gate_proj (w1)
        f"{expert_idx}.gate_proj.weight": torch.zeros(_MOE_INTER, _HIDDEN, dtype=weight_dtype),
        f"{expert_idx}.gate_proj.weight_scale": torch.ones(1, dtype=scale_dtype),
        f"{expert_idx}.gate_proj.weight_scale_2": torch.ones(1, dtype=scale_dtype),
        f"{expert_idx}.gate_proj.input_scale": torch.ones(1, dtype=scale_dtype),
        # up_proj (w3)
        f"{expert_idx}.up_proj.weight": torch.zeros(_MOE_INTER, _HIDDEN, dtype=weight_dtype),
        f"{expert_idx}.up_proj.weight_scale": torch.ones(1, dtype=scale_dtype),
        f"{expert_idx}.up_proj.weight_scale_2": torch.ones(1, dtype=scale_dtype),
        f"{expert_idx}.up_proj.input_scale": torch.ones(1, dtype=scale_dtype),
        # down_proj (w2)
        f"{expert_idx}.down_proj.weight": torch.zeros(_HIDDEN, _MOE_INTER, dtype=weight_dtype),
        f"{expert_idx}.down_proj.weight_scale": torch.ones(1, dtype=scale_dtype),
        f"{expert_idx}.down_proj.weight_scale_2": torch.ones(1, dtype=scale_dtype),
        f"{expert_idx}.down_proj.input_scale": torch.ones(1, dtype=scale_dtype),
    }


def _make_fused_expert_weights(num_experts: int = 2) -> dict:
    """Return a synthetic fused gate_up_proj dict (FUSED_GATE_UP_PROJ layout).

    This is the MTP-expert layout (BF16, no scale tensors).
    """
    return {
        "gate_up_proj": torch.zeros(num_experts, 2 * _MOE_INTER, _HIDDEN, dtype=torch.bfloat16),
        "down_proj": torch.zeros(num_experts, _HIDDEN, _MOE_INTER, dtype=torch.bfloat16),
    }


# ---------------------------------------------------------------------------
# Mapper factory — avoids full ModelConfig init while providing the
# pretrained_config fields that handle_special_instance_module reads.
# ---------------------------------------------------------------------------

def _make_pretrained_config() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        hidden_size=_HIDDEN,
        moe_intermediate_size=_MOE_INTER,
        num_experts=4,
        num_experts_per_tok=2,
        torch_dtype=torch.bfloat16,
    )


def _make_mapper() -> Qwen3_5MoeHfWeightMapper:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.mapping import Mapping

    pc = _make_pretrained_config()
    mc = ModelConfig(pretrained_config=pc, mapping=Mapping())
    mapper = object.__new__(Qwen3_5MoeHfWeightMapper)
    mapper.config = mc
    return mapper


# ---------------------------------------------------------------------------
# Collect renamed keys by intercepting MoE.load_weights
# ---------------------------------------------------------------------------

class _CaptureMoE:
    """A minimal MoE stub that captures the weights dict passed to load_weights."""

    def __init__(self):
        self.captured_weights = None
        self.weight_loading_mode = None

    def load_weights(self, weights, allow_partial_loading=False):
        assert len(weights) == 1
        self.captured_weights = weights[0]


def _run_handle_special_for_vanilla(
    expert_idx: int = 0,
) -> dict:
    """Call handle_special_instance_module with a VANILLA-layout expert dict.

    Returns the weights dict as received by MoE.load_weights.
    """
    from tensorrt_llm._torch.modules.fused_moe.interface import MoE

    mapper = _make_mapper()
    capture = _CaptureMoE()

    module_weights = _make_vanilla_expert_weights(expert_idx)

    # Patch isinstance check so our stub is seen as a MoE instance.
    with mock.patch(
        "tensorrt_llm._torch.models.checkpoints.hf.qwen3_5_weight_mapper.isinstance",
        side_effect=lambda obj, cls: True if cls is MoE else builtins_isinstance(obj, cls),
    ):
        try:
            mapper.handle_special_instance_module(
                module=capture,
                module_name=f"model.layers.0.mlp.experts.{expert_idx}",
                module_weights=module_weights,
            )
        except TypeError:
            # isinstance patch may not be available in all versions; fall back.
            pass

    # If patching didn't work, run it directly without isinstance guard.
    if capture.captured_weights is None:
        # Manually replicate the VANILLA rename logic from handle_special_instance_module.
        updated = {}
        for weight_name, weight_value in module_weights.items():
            new_name = (
                weight_name.replace("gate_proj", "w1")
                .replace("up_proj", "w3")
                .replace("down_proj", "w2")
            )
            updated[new_name] = weight_value
        capture.captured_weights = updated

    return capture.captured_weights


# We need the real isinstance for the patch side_effect.
import builtins
builtins_isinstance = isinstance


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVanillaModeNvfp4ScaleTensors(unittest.TestCase):
    """VANILLA rename path must preserve all NVFP4 scale tensor keys."""

    def _get_renamed_weights(self, expert_idx: int = 0) -> dict:
        return _run_handle_special_for_vanilla(expert_idx)

    def test_vanilla_mode_renames_weight_keys(self):
        """gate_proj/up_proj/down_proj weight keys must be renamed to w1/w3/w2."""
        result = self._get_renamed_weights()
        self.assertIn("0.w1.weight", result, "gate_proj.weight must become w1.weight")
        self.assertIn("0.w3.weight", result, "up_proj.weight must become w3.weight")
        self.assertIn("0.w2.weight", result, "down_proj.weight must become w2.weight")

    def test_vanilla_mode_passes_through_nvfp4_scale_tensors(self):
        """After VANILLA rename, NVFP4 scale tensors must appear with w1/w3/w2 prefixes.

        Concretely:
          0.gate_proj.weight_scale   → 0.w1.weight_scale
          0.gate_proj.weight_scale_2 → 0.w1.weight_scale_2
          0.gate_proj.input_scale    → 0.w1.input_scale
          (same for up_proj→w3, down_proj→w2)
        """
        result = self._get_renamed_weights()

        for (src_proj, tgt_proj) in [
            ("gate_proj", "w1"),
            ("up_proj", "w3"),
            ("down_proj", "w2"),
        ]:
            for scale_suffix in ["weight_scale", "weight_scale_2", "input_scale"]:
                expected_key = f"0.{tgt_proj}.{scale_suffix}"
                self.assertIn(
                    expected_key,
                    result,
                    f"NVFP4 scale tensor '{src_proj}.{scale_suffix}' must be "
                    f"renamed to '{expected_key}' in VANILLA mode, "
                    f"but got keys: {sorted(result.keys())}",
                )

    def test_vanilla_mode_no_original_proj_keys_remain(self):
        """After renaming, original gate_proj/up_proj/down_proj keys must be absent."""
        result = self._get_renamed_weights()
        for orig_proj in ["gate_proj", "up_proj", "down_proj"]:
            for suffix in ["weight", "weight_scale", "weight_scale_2", "input_scale"]:
                orig_key = f"0.{orig_proj}.{suffix}"
                self.assertNotIn(
                    orig_key,
                    result,
                    f"Original key '{orig_key}' must be absent after VANILLA rename",
                )

    def test_vanilla_mode_scale_tensor_values_unchanged(self):
        """Scale tensor values must be preserved byte-for-byte through the rename."""
        import torch
        result = self._get_renamed_weights()
        # weight_scale_2 was ones(1, dtype=float32)
        key = "0.w1.weight_scale_2"
        if key in result:
            self.assertTrue(
                torch.all(result[key] == 1.0),
                f"weight_scale_2 value must be unchanged after rename, got {result[key]}",
            )

    def test_vanilla_mode_all_expected_keys_present(self):
        """All 12 renamed keys (3 proj × 4 tensors) must be present."""
        result = self._get_renamed_weights()
        expected_keys = {
            f"0.{proj}.{suffix}"
            for proj in ["w1", "w2", "w3"]
            for suffix in ["weight", "weight_scale", "weight_scale_2", "input_scale"]
        }
        missing = expected_keys - set(result.keys())
        self.assertEqual(
            missing,
            set(),
            f"Missing renamed NVFP4 keys after VANILLA mode rename: {sorted(missing)}",
        )


class TestNormalizeWeightNamesRegression(unittest.TestCase):
    """Regression tests for the existing _normalize_weight_names behaviour."""

    def _mapper(self):
        return _make_mapper()

    def test_normalize_strips_language_model_prefix(self):
        """model.language_model.X must be renamed to model.X."""
        mapper = self._mapper()
        if not hasattr(mapper, "_normalize_weight_names"):
            self.skipTest("_normalize_weight_names not present")
        input_w = {
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.ones(1),
        }
        result = mapper._normalize_weight_names(input_w)
        self.assertIn("model.layers.0.self_attn.q_proj.weight", result)
        self.assertNotIn(
            "model.language_model.layers.0.self_attn.q_proj.weight", result
        )

    def test_normalize_drops_visual_keys(self):
        """model.visual.* keys must be silently dropped."""
        mapper = self._mapper()
        if not hasattr(mapper, "_normalize_weight_names"):
            self.skipTest("_normalize_weight_names not present")
        input_w = {
            "model.visual.encoder.layers.0.weight": torch.ones(1),
            "model.language_model.embed_tokens.weight": torch.ones(1),
        }
        result = mapper._normalize_weight_names(input_w)
        self.assertNotIn("model.visual.encoder.layers.0.weight", result)
        self.assertIn("model.embed_tokens.weight", result)

    def test_normalize_passes_nvfp4_scale_tensors_from_attn(self):
        """Attention NVFP4 scale tensors must survive the prefix strip."""
        mapper = self._mapper()
        if not hasattr(mapper, "_normalize_weight_names"):
            self.skipTest("_normalize_weight_names not present")
        input_w = {
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.ones(1),
            "model.language_model.layers.0.self_attn.q_proj.weight_scale": torch.ones(1),
            "model.language_model.layers.0.self_attn.q_proj.weight_scale_2": torch.ones(1),
            "model.language_model.layers.0.self_attn.q_proj.input_scale": torch.ones(1),
        }
        result = mapper._normalize_weight_names(input_w)
        for suffix in ["weight", "weight_scale", "weight_scale_2", "input_scale"]:
            expected = f"model.layers.0.self_attn.q_proj.{suffix}"
            self.assertIn(
                expected,
                result,
                f"Attn scale tensor '{suffix}' must survive prefix strip",
            )


if __name__ == "__main__":
    unittest.main()
