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
"""Tests for Qwen3.5-MoE MTP layer (Phase 2 deliverables 4, 5, 6).

Deliverable 4: ``Qwen3_5MoeHfWeightMapper._normalize_weight_names`` MTP key
    normalization: ``mtp.*`` → ``model.layers.{num_hidden_layers}.*``

Deliverable 5: ``Qwen35MoeMTP`` class in
    ``tensorrt_llm/_torch/models/modeling_qwen3_5_mtp.py``
    - ``fc.in_features == 2 * hidden_size``, ``fc.out_features == hidden_size``
    - ``layers_0.layer_idx == num_hidden_layers (40)``
    - MTP sublayer is unquantized (mirrors NemotronHMTP pattern)

Deliverable 6: ``modeling_speculative.py`` MTPForCausalLM dispatch has a
    ``"qwen3_5_moe_text"`` case that returns ``Qwen35MoeMTP``.

All tests are pure-Python where possible. GPU is not required.
"""

import inspect
import types
import unittest

import pytest

# ---------------------------------------------------------------------------
# Guarded imports — skip gracefully when Phase 2 is not yet merged.
# ---------------------------------------------------------------------------

_mtp_mod = pytest.importorskip(
    "tensorrt_llm._torch.models.modeling_qwen3_5_mtp",
    reason="modeling_qwen3_5_mtp.py not yet present — Phase 2 not merged",
)
Qwen35MoeMTP = _mtp_mod.Qwen35MoeMTP

# ---------------------------------------------------------------------------
# Minimal Qwen3.6-35B-A3B config values (text-config after compat shim).
# ---------------------------------------------------------------------------
_HIDDEN_SIZE = 2048
_NUM_HIDDEN_LAYERS = 40  # MTP layer_idx == num_hidden_layers
_MTP_LAYER_IDX = _NUM_HIDDEN_LAYERS  # 40

_QWEN36_TEXT_CONFIG_DICT = {
    "architectures": ["Qwen3_5MoeForCausalLM"],
    "model_type": "qwen3_5_moe_text",
    "hidden_size": _HIDDEN_SIZE,
    "num_hidden_layers": _NUM_HIDDEN_LAYERS,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "intermediate_size": 4096,
    "moe_intermediate_size": 512,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "vocab_size": 151936,
    "max_position_embeddings": 32768,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000000.0,
    "partial_rotary_factor": 0.25,
    "torch_dtype": "bfloat16",
    # Hybrid layer config (full-attn at layers 3,7,11…39, linear-attn elsewhere)
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 64,
    "linear_value_head_dim": 64,
    "layer_types": (
        ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 10
    ),
}


def _make_pretrained_config() -> types.SimpleNamespace:
    """Return a minimal SimpleNamespace that behaves like a PretrainedConfig.

    Qwen35MoeMTP reads: hidden_size, rms_norm_eps, torch_dtype, num_hidden_layers.
    """
    import torch
    ns = types.SimpleNamespace(**_QWEN36_TEXT_CONFIG_DICT)
    ns.torch_dtype = torch.bfloat16
    return ns


def _make_model_config(quant_config=None):
    """Build a ModelConfig with the synthetic Qwen3.6 pretrained_config.

    Avoids any real checkpoint I/O or GPU initialisation.
    """
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig

    pc = _make_pretrained_config()
    qc = quant_config if quant_config is not None else QuantConfig()
    return ModelConfig(
        pretrained_config=pc,
        mapping=Mapping(),
        quant_config=qc,
    )


# ---------------------------------------------------------------------------
# Deliverable 5 — Qwen35MoeMTP instantiation tests
# ---------------------------------------------------------------------------

class TestQwen35MoeMTPInstantiation(unittest.TestCase):
    """Structural tests for Qwen35MoeMTP that do not require GPU or real weights."""

    def setUp(self):
        # Require at least a CPU-only torch; skip if CUDA is needed for init.
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")

    def _build_mtp(self, quant_config=None):
        """Instantiate Qwen35MoeMTP with a synthetic config.

        Pass an empty aux_stream_dict because we are not running actual forward
        passes, and the constructor should not require real CUDA streams.
        """
        model_config = _make_model_config(quant_config)
        return Qwen35MoeMTP(
            model_config=model_config,
            layer_idx=_MTP_LAYER_IDX,
            aux_stream_dict={},
        )

    def test_qwen35_moe_mtp_instantiation(self):
        """Qwen35MoeMTP must instantiate without error and expose fc / layers_0."""
        mtp = self._build_mtp()

        # fc: projects concat([embed_norm, hidden_norm]) → hidden
        self.assertTrue(
            hasattr(mtp, "fc"),
            "Qwen35MoeMTP must have a 'fc' attribute (projection layer)",
        )
        fc = mtp.fc
        self.assertEqual(
            fc.in_features,
            2 * _HIDDEN_SIZE,
            f"fc.in_features must be 2 * hidden_size ({2 * _HIDDEN_SIZE}), "
            f"got {fc.in_features}",
        )
        self.assertEqual(
            fc.out_features,
            _HIDDEN_SIZE,
            f"fc.out_features must be hidden_size ({_HIDDEN_SIZE}), "
            f"got {fc.out_features}",
        )

    def test_qwen35_moe_mtp_sublayer_layer_idx(self):
        """The inner decoder sublayer must use layer_idx == num_hidden_layers (40)."""
        mtp = self._build_mtp()

        # The decoder sublayer is stored as layers_0 or layers["0"].
        sublayer = getattr(mtp, "layers_0", None) or (
            mtp.layers["0"] if hasattr(mtp, "layers") else None
        )
        self.assertIsNotNone(
            sublayer,
            "Qwen35MoeMTP must have an inner decoder sublayer accessible as "
            "layers_0 or layers['0']",
        )
        self.assertEqual(
            sublayer.layer_idx,
            _MTP_LAYER_IDX,
            f"Inner sublayer layer_idx must be {_MTP_LAYER_IDX} "
            f"(== num_hidden_layers), got {sublayer.layer_idx}",
        )

    def test_qwen35_moe_mtp_sublayer_unquantized(self):
        """The MTP sublayer's quant_config.quant_algo must be None (unquantized).

        Background: the Qwen3.6-NVFP4 checkpoint stores MTP weights as BF16
        with no quantization. Because the TRT-LLM MoE backend requires the
        entire model to use a consistent quantization scheme, the MTP layer
        must explicitly override quant_algo=None, mirroring the
        NemotronHMTP._get_mtp_sublayer_quant_config pattern.
        """
        from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

        # Give the outer model a non-None quant_config so the sublayer override
        # has something to clear.
        outer_quant = QuantConfig(quant_algo=QuantAlgo.NVFP4)
        mtp = self._build_mtp(quant_config=outer_quant)

        sublayer = getattr(mtp, "layers_0", None) or (
            mtp.layers["0"] if hasattr(mtp, "layers") else None
        )
        if sublayer is None:
            self.skipTest("No inner sublayer found — cannot check quant_config")

        # The sublayer's model_config.quant_config.quant_algo should be None.
        sublayer_qc = getattr(sublayer, "model_config", None)
        if sublayer_qc is not None:
            sublayer_qc = sublayer_qc.quant_config
        elif hasattr(sublayer, "quant_config"):
            sublayer_qc = sublayer.quant_config
        else:
            self.skipTest("Cannot inspect sublayer quant_config")

        self.assertIsNone(
            sublayer_qc.quant_algo,
            f"MTP sublayer quant_algo must be None (unquantized), "
            f"got {sublayer_qc.quant_algo!r}",
        )

    def test_qwen35_moe_mtp_has_norm_layers(self):
        """Qwen35MoeMTP must have pre-fc normalization layers for embed and hidden."""
        mtp = self._build_mtp()
        has_enorm = hasattr(mtp, "pre_fc_norm_embedding") or hasattr(mtp, "enorm")
        has_hnorm = hasattr(mtp, "pre_fc_norm_hidden") or hasattr(mtp, "hnorm")
        self.assertTrue(
            has_enorm,
            "Qwen35MoeMTP must have a pre-fc embedding norm (pre_fc_norm_embedding or enorm)",
        )
        self.assertTrue(
            has_hnorm,
            "Qwen35MoeMTP must have a pre-fc hidden norm (pre_fc_norm_hidden or hnorm)",
        )


# ---------------------------------------------------------------------------
# Deliverable 6 — modeling_speculative.py MTP dispatch
# ---------------------------------------------------------------------------

class TestMTPDispatchQwen35MoeText(unittest.TestCase):
    """Verify the 'qwen3_5_moe_text' case in MTPForCausalLM.__init__."""

    def test_mtp_dispatch_recognizes_qwen3_5_moe_text_via_source(self):
        """The MTPForCausalLM dispatch must include a 'qwen3_5_moe_text' case.

        We use inspect.getsource to avoid instantiating the class (which
        requires real model + executor setup).  This is the lightest possible
        check that the dispatch branch exists.
        """
        spec_mod = pytest.importorskip(
            "tensorrt_llm._torch.models.modeling_speculative",
            reason="modeling_speculative not importable",
        )
        MTPForCausalLM = getattr(spec_mod, "MTPForCausalLM", None)
        self.assertIsNotNone(
            MTPForCausalLM,
            "MTPForCausalLM class must exist in modeling_speculative",
        )
        src = inspect.getsource(MTPForCausalLM.__init__)
        self.assertIn(
            "qwen3_5_moe_text",
            src,
            "MTPForCausalLM.__init__ must contain a 'qwen3_5_moe_text' dispatch branch",
        )
        # Also assert it references Qwen35MoeMTP.
        self.assertIn(
            "Qwen35MoeMTP",
            src,
            "MTPForCausalLM.__init__ dispatch for qwen3_5_moe_text must reference Qwen35MoeMTP",
        )

    def test_mtp_draft_model_dispatch_recognizes_qwen3_5_moe_text(self):
        """MTPDraftModel.__init__ must also handle 'qwen3_5_moe_text'."""
        spec_mod = pytest.importorskip(
            "tensorrt_llm._torch.models.modeling_speculative",
            reason="modeling_speculative not importable",
        )
        MTPDraftModel = getattr(spec_mod, "MTPDraftModel", None)
        if MTPDraftModel is None:
            self.skipTest("MTPDraftModel not found in modeling_speculative")
        src = inspect.getsource(MTPDraftModel.__init__)
        self.assertIn(
            "qwen3_5_moe_text",
            src,
            "MTPDraftModel.__init__ must contain a 'qwen3_5_moe_text' branch",
        )


# ---------------------------------------------------------------------------
# Deliverable 4 — weight mapper MTP key normalization
# ---------------------------------------------------------------------------

class TestQwen35MoeWeightMapperMtpKeyNormalization(unittest.TestCase):
    """Verify that mtp.* keys are normalized to model.layers.{num_hidden_layers}.*

    The weight mapper ``_normalize_weight_names`` currently strips
    ``model.language_model.*`` and drops ``model.visual.*``.  After Phase 2,
    it also maps ``mtp.*`` → ``model.layers.40.*`` so that the weight loader
    can place MTP weights in the correct slot.
    """

    def setUp(self):
        try:
            from tensorrt_llm._torch.models.checkpoints.hf.qwen3_5_weight_mapper import (
                Qwen3_5MoeHfWeightMapper,
            )
            self._mapper_class = Qwen3_5MoeHfWeightMapper
        except ImportError as exc:
            self.skipTest(f"Qwen3_5MoeHfWeightMapper not importable: {exc}")

    def _make_mapper(self):
        """Build a mapper instance with a minimal synthetic ModelConfig."""
        from tensorrt_llm._torch.model_config import ModelConfig
        from tensorrt_llm.mapping import Mapping

        pc = _make_pretrained_config()
        model_config = ModelConfig(
            pretrained_config=pc,
            mapping=Mapping(),
        )
        mapper = object.__new__(self._mapper_class)
        mapper.config = model_config
        return mapper

    def test_weight_mapper_normalizes_mtp_keys(self):
        """mtp.* checkpoint keys must be renamed to model.layers.40.* after normalization.

        Input key examples (from the actual checkpoint):
            mtp.layers.0.input_layernorm.weight
            mtp.fc.weight
            mtp.norm.weight
            mtp.pre_fc_norm_embedding.weight
            mtp.pre_fc_norm_hidden.weight

        Expected output keys:
            model.layers.40.input_layernorm.weight
            model.layers.40.fc.weight
            model.layers.40.norm.weight
            model.layers.40.pre_fc_norm_embedding.weight
            model.layers.40.pre_fc_norm_hidden.weight
        """
        import torch

        mapper = self._make_mapper()

        # Check if the Phase 2 MTP normalization is present.
        if not hasattr(mapper, "_normalize_weight_names"):
            self.skipTest("_normalize_weight_names not present — cannot test MTP key remap")

        input_weights = {
            "mtp.layers.0.input_layernorm.weight": torch.ones(1),
            "mtp.fc.weight": torch.ones(1),
            "mtp.norm.weight": torch.ones(1),
            "mtp.pre_fc_norm_embedding.weight": torch.ones(1),
            "mtp.pre_fc_norm_hidden.weight": torch.ones(1),
            # Non-MTP keys must pass through unchanged.
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.ones(1),
        }

        result = mapper._normalize_weight_names(input_weights)

        # MTP key remap check.
        expected_mtp_target = f"model.layers.{_NUM_HIDDEN_LAYERS}"
        for src_key in [
            "mtp.layers.0.input_layernorm.weight",
            "mtp.fc.weight",
            "mtp.norm.weight",
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
        ]:
            if src_key not in input_weights:
                continue
            suffix = src_key[len("mtp."):]  # strip leading "mtp."
            # mtp.layers.0.X → model.layers.40.X (layers.0 → top layer)
            # mtp.fc.weight   → model.layers.40.fc.weight
            expected_key = f"{expected_mtp_target}.{suffix}"
            self.assertIn(
                expected_key,
                result,
                f"MTP key '{src_key}' must be remapped to '{expected_key}', "
                f"got keys: {sorted(result.keys())}",
            )
            self.assertNotIn(
                src_key,
                result,
                f"Original MTP key '{src_key}' must not remain in normalized weights",
            )

        # Non-MTP key: model.language_model.layers.0.* → model.layers.0.*
        expected_non_mtp = "model.layers.0.self_attn.q_proj.weight"
        self.assertIn(
            expected_non_mtp,
            result,
            f"Non-MTP key must be normalized to '{expected_non_mtp}', "
            f"got: {sorted(result.keys())}",
        )

    def test_weight_mapper_current_behavior_drops_visual_keys(self):
        """Regression: model.visual.* keys must always be dropped (pre and post Phase 2)."""
        import torch

        mapper = self._make_mapper()
        if not hasattr(mapper, "_normalize_weight_names"):
            self.skipTest("_normalize_weight_names not present")

        input_weights = {
            "model.visual.encoder.weight": torch.ones(1),
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.ones(1),
        }
        result = mapper._normalize_weight_names(input_weights)
        self.assertNotIn(
            "model.visual.encoder.weight",
            result,
            "model.visual.* keys must be dropped by _normalize_weight_names",
        )
        self.assertIn(
            "model.layers.0.self_attn.q_proj.weight",
            result,
        )


if __name__ == "__main__":
    unittest.main()
