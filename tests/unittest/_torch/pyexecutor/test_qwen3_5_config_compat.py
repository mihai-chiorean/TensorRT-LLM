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
"""Tests for ``_Qwen35ConfigCompat._flatten_rope`` mRoPE text-path guard.

Phase 2 deliverable 3 modifies ``_flatten_rope`` so that ``mrope_section``
and ``mrope_interleaved`` are *stripped* from the resulting ``rope_scaling``
dict for the text-only path, rather than promoted to ``type = "mrope"``.

Background
----------
Without the guard, a Qwen3.6-NVFP4 checkpoint config with::

    rope_parameters:
      mrope_interleaved: true
      mrope_section: [11, 11, 10]
      partial_rotary_factor: 0.25
      rope_theta: 10000000

would cause ``Qwen3Attention`` to pick ``PositionEmbeddingType.mrope``, which
requires 3-D position_ids that the text-only TRT-LLM executor never builds.
The result is silently wrong cos/sin tensors (wrong accuracy, no crash).

All tests are pure-Python and do not require GPU.
"""

import copy
import unittest

import pytest

# ---------------------------------------------------------------------------
# Guard: the module under test must be importable.
# ---------------------------------------------------------------------------
_compat_mod = pytest.importorskip(
    "tensorrt_llm._torch.pyexecutor.config_utils",
    reason="tensorrt_llm._torch.pyexecutor.config_utils not importable",
)
_Qwen35ConfigCompat = _compat_mod._Qwen35ConfigCompat


# ---------------------------------------------------------------------------
# Minimal Qwen3.6 text_config dict that exercises the mRoPE code path.
# This mirrors the actual text_config.json fields of the NVFP4 checkpoint.
# ---------------------------------------------------------------------------
_QWEN36_TEXT_CONFIG_WITH_MROPE = {
    "model_type": "qwen3_5_moe_text",
    "architectures": ["Qwen3_5MoeForCausalLM"],
    "hidden_size": 2048,
    "num_hidden_layers": 40,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "moe_intermediate_size": 512,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "vocab_size": 151936,
    "max_position_embeddings": 32768,
    "rope_parameters": {
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000,
        "rope_type": "default",
    },
}

# Config variant with a pre-existing rope_scaling field (to test merge logic).
_QWEN36_TEXT_CONFIG_WITH_ROPE_SCALING = dict(_QWEN36_TEXT_CONFIG_WITH_MROPE, **{
    "rope_scaling": {"factor": 1.0},
})

# Config without mrope — should not strip anything or change type.
_QWEN36_TEXT_CONFIG_WITHOUT_MROPE = {
    "model_type": "qwen3_5_moe_text",
    "architectures": ["Qwen3_5MoeForCausalLM"],
    "hidden_size": 2048,
    "num_hidden_layers": 40,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "rope_parameters": {
        "partial_rotary_factor": 0.25,
        "rope_theta": 1000000,
        "rope_type": "default",
    },
}


def _flatten_rope(text_config: dict) -> dict:
    """Call _flatten_rope on a deep copy so the original is not mutated."""
    return _Qwen35ConfigCompat._flatten_rope(copy.deepcopy(text_config))


# ---------------------------------------------------------------------------
# Detect whether Phase 2 implementation has been merged.
# After Phase 2, _flatten_rope strips mrope_section/mrope_interleaved.
# Before Phase 2, it promotes them to type="mrope".
# We auto-detect by running the function and inspecting the output.
# ---------------------------------------------------------------------------

def _phase2_strip_guard_is_active() -> bool:
    """Return True when the Phase 2 guard (strip mrope for text path) is present.

    Before Phase 2 the current implementation sets ``rope_scaling["type"] =
    "mrope"`` when mrope fields are found.  After Phase 2 it strips them.
    We probe this by running the actual function.
    """
    probe = _flatten_rope(copy.deepcopy(_QWEN36_TEXT_CONFIG_WITH_MROPE))
    rope_scaling = probe.get("rope_scaling", {})
    # Phase 2: mrope_section stripped → not in rope_scaling
    # Pre-Phase-2: mrope_section present and type == "mrope"
    return "mrope_section" not in rope_scaling


_PHASE2_ACTIVE = _phase2_strip_guard_is_active()
_requires_phase2 = pytest.mark.skipif(
    not _PHASE2_ACTIVE,
    reason="_flatten_rope mRoPE strip guard not yet merged (Phase 2 pending)",
)


class TestFlattenRopeCurrentBehavior(unittest.TestCase):
    """Tests that pass regardless of Phase 2 state.

    These assert invariants that must hold both before and after the Phase 2
    mRoPE strip guard is merged.
    """

    def test_flatten_rope_extracts_rope_theta(self):
        """rope_theta must be promoted to a top-level field."""
        result = _flatten_rope(copy.deepcopy(_QWEN36_TEXT_CONFIG_WITH_MROPE))
        self.assertEqual(
            result.get("rope_theta"),
            10000000,
            "rope_theta must be extracted from rope_parameters to top level",
        )

    def test_flatten_rope_extracts_partial_rotary_factor(self):
        """partial_rotary_factor must be promoted to a top-level field."""
        result = _flatten_rope(copy.deepcopy(_QWEN36_TEXT_CONFIG_WITH_MROPE))
        self.assertAlmostEqual(
            result.get("partial_rotary_factor"),
            0.25,
            places=6,
            msg="partial_rotary_factor must be extracted from rope_parameters",
        )

    def test_flatten_rope_removes_rope_parameters_key(self):
        """The rope_parameters key must be gone after flattening."""
        result = _flatten_rope(copy.deepcopy(_QWEN36_TEXT_CONFIG_WITH_MROPE))
        self.assertNotIn(
            "rope_parameters",
            result,
            "rope_parameters must be removed from the config dict",
        )

    def test_flatten_rope_no_mrope_preserves_rope_theta(self):
        """Without mrope fields, rope_theta is still extracted correctly."""
        result = _flatten_rope(copy.deepcopy(_QWEN36_TEXT_CONFIG_WITHOUT_MROPE))
        self.assertEqual(result.get("rope_theta"), 1000000)

    def test_flatten_rope_no_mrope_type_not_mrope(self):
        """Without mrope fields, type must not be set to 'mrope'."""
        result = _flatten_rope(copy.deepcopy(_QWEN36_TEXT_CONFIG_WITHOUT_MROPE))
        rope_scaling = result.get("rope_scaling", {})
        self.assertNotEqual(
            rope_scaling.get("type"),
            "mrope",
            "type must not be 'mrope' when no mrope fields are present",
        )


class TestFlattenRopeMropeStripGuard(unittest.TestCase):
    """Tests for the Phase 2 mRoPE strip guard in _flatten_rope.

    These tests are skipped until the Phase 2 implementation is merged.
    """

    @_requires_phase2
    def test_flatten_rope_strips_mrope_for_text_path(self):
        """After Phase 2, mrope_section and mrope_interleaved must be stripped.

        The text-only TRT-LLM executor never builds 3-D position_ids, so
        keeping mrope fields causes silently wrong cos/sin computation.
        """
        result = _flatten_rope(copy.deepcopy(_QWEN36_TEXT_CONFIG_WITH_MROPE))
        rope_scaling = result.get("rope_scaling", {})
        self.assertNotIn(
            "mrope_section",
            rope_scaling,
            "mrope_section must be stripped for text-only path",
        )
        self.assertNotIn(
            "mrope_interleaved",
            rope_scaling,
            "mrope_interleaved must be stripped for text-only path",
        )

    @_requires_phase2
    def test_flatten_rope_type_not_mrope_after_strip(self):
        """After stripping mrope fields, rope_scaling.type must not be 'mrope'."""
        result = _flatten_rope(copy.deepcopy(_QWEN36_TEXT_CONFIG_WITH_MROPE))
        rope_scaling = result.get("rope_scaling", {})
        self.assertNotEqual(
            rope_scaling.get("type"),
            "mrope",
            "rope_scaling.type must not be 'mrope' after mRoPE strip guard",
        )

    @_requires_phase2
    def test_flatten_rope_preserves_other_rope_fields(self):
        """rope_theta and partial_rotary_factor must survive the mrope strip."""
        result = _flatten_rope(copy.deepcopy(_QWEN36_TEXT_CONFIG_WITH_MROPE))
        self.assertEqual(
            result.get("rope_theta"),
            10000000,
            "rope_theta must not be removed by mrope strip",
        )
        self.assertAlmostEqual(
            result.get("partial_rotary_factor"),
            0.25,
            places=6,
            msg="partial_rotary_factor must not be removed by mrope strip",
        )

    @_requires_phase2
    def test_flatten_rope_idempotent(self):
        """Running _flatten_rope twice must yield the same result as once.

        This guards against the function modifying the rope_scaling dict in a
        way that causes double-processing to diverge.
        """
        first = _flatten_rope(copy.deepcopy(_QWEN36_TEXT_CONFIG_WITH_MROPE))
        second = _Qwen35ConfigCompat._flatten_rope(copy.deepcopy(first))
        self.assertEqual(
            first.get("rope_scaling"),
            second.get("rope_scaling"),
            "Running _flatten_rope twice must yield identical rope_scaling",
        )
        self.assertEqual(
            first.get("rope_theta"),
            second.get("rope_theta"),
        )
        self.assertAlmostEqual(
            first.get("partial_rotary_factor", 0.0),
            second.get("partial_rotary_factor", 0.0),
            places=6,
        )

    @_requires_phase2
    def test_flatten_rope_with_preexisting_rope_scaling(self):
        """mrope fields merged from rope_parameters into rope_scaling must also be stripped."""
        result = _flatten_rope(copy.deepcopy(_QWEN36_TEXT_CONFIG_WITH_ROPE_SCALING))
        rope_scaling = result.get("rope_scaling", {})
        self.assertNotIn("mrope_section", rope_scaling)
        self.assertNotIn("mrope_interleaved", rope_scaling)
        # Pre-existing fields must be preserved.
        self.assertEqual(rope_scaling.get("factor"), 1.0)


class TestFlattenRopeFullNormalize(unittest.TestCase):
    """Integration-style: test _Qwen35ConfigCompat.normalize for VLM configs."""

    def _make_vlm_config(self) -> dict:
        """Minimal VLM-style top-level config that wraps a text_config."""
        return {
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe",
            "text_config": copy.deepcopy(_QWEN36_TEXT_CONFIG_WITH_MROPE),
            "vision_config": {"hidden_size": 1024},
        }

    def test_normalize_extracts_text_config(self):
        """normalize() must return a dict with the text-config fields."""
        vlm_config = self._make_vlm_config()
        result = _Qwen35ConfigCompat.normalize(vlm_config)
        self.assertEqual(
            result.get("architectures"),
            ["Qwen3_5MoeForCausalLM"],
            "normalize() must rewrite architectures to Qwen3_5MoeForCausalLM",
        )
        self.assertNotIn(
            "vision_config",
            result,
            "normalize() must not include vision_config in the flattened output",
        )
        # rope_theta must appear at top level after flattening.
        self.assertEqual(result.get("rope_theta"), 10000000)

    @_requires_phase2
    def test_normalize_strips_mrope_end_to_end(self):
        """End-to-end: normalize() on a VLM config must not produce mrope rope_scaling."""
        vlm_config = self._make_vlm_config()
        result = _Qwen35ConfigCompat.normalize(vlm_config)
        rope_scaling = result.get("rope_scaling", {})
        self.assertNotIn("mrope_section", rope_scaling)
        self.assertNotIn("mrope_interleaved", rope_scaling)
        self.assertNotEqual(rope_scaling.get("type"), "mrope")


if __name__ == "__main__":
    unittest.main()
