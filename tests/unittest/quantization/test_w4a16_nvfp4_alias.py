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
"""Tests for the W4A16_NVFP4 → NVFP4 alias (Phase 2 deliverables 1 + 2).

All tests are pure-Python and do not require GPU or model weights.

Deliverable 1: ``QuantAlgo.W4A16_NVFP4`` is added to the enum and maps to
    the same ``QuantMode`` bits as ``QuantAlgo.NVFP4``.

Deliverable 2: ``ModelConfig.load_modelopt_quant_config`` remaps the string
    ``"W4A16_NVFP4"`` to ``QuantAlgo.NVFP4`` for per-layer entries that come
    from a MIXED_PRECISION checkpoint, and appends ``"lm_head"`` to
    ``quant_config.exclude_modules`` at the point of remapping.
"""

import json
import tempfile
import unittest
from pathlib import Path

import pytest

from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantMode

# ---------------------------------------------------------------------------
# Guard: skip the whole module if Phase 2 implementation is not yet merged.
# Tests that depend on the new W4A16_NVFP4 enum member are individually
# guarded; the import-level guard below prevents hard import errors.
# ---------------------------------------------------------------------------
_W4A16_NVFP4_PRESENT = hasattr(QuantAlgo, "W4A16_NVFP4")


def _skip_if_not_implemented(fn):
    """Decorator: skip a test when Phase 2 enum value is not yet present."""
    return pytest.mark.skipif(
        not _W4A16_NVFP4_PRESENT,
        reason="QuantAlgo.W4A16_NVFP4 not present — Phase 2 not merged",
    )(fn)


# ---------------------------------------------------------------------------
# Helper: build a minimal hf_quant_config.json-style dict that mirrors the
# Qwen3.6-35B-A3B-NVFP4 checkpoint structure.
# ---------------------------------------------------------------------------

def _make_mixed_precision_quant_config(
    layer_quant_algo: str = "W4A16_NVFP4",
    group_size: int = 16,
    kv_cache_quant_algo: str = "FP8",
) -> dict:
    """Return a dict that matches the ``hf_quant_config.json`` format used by
    modelopt-exported MIXED_PRECISION checkpoints."""
    return {
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": kv_cache_quant_algo,
            "quantized_layers": {
                "model.language_model.layers.0.mlp.experts": {
                    "quant_algo": layer_quant_algo,
                    "group_size": group_size,
                },
                "lm_head": {
                    "quant_algo": layer_quant_algo,
                    "group_size": group_size,
                },
            },
        }
    }


def _write_quant_config_file(tmp_dir: Path, config_dict: dict) -> Path:
    path = tmp_dir / "hf_quant_config.json"
    path.write_text(json.dumps(config_dict))
    return path


# ---------------------------------------------------------------------------
# Deliverable 1 tests
# ---------------------------------------------------------------------------

class TestW4A16NvfP4EnumMember(unittest.TestCase):
    """QuantAlgo enum and QuantMode mapping for W4A16_NVFP4 (Deliverable 1)."""

    @_skip_if_not_implemented
    def test_w4a16_nvfp4_in_quant_algo_enum(self):
        """W4A16_NVFP4 must exist in QuantAlgo and be distinct from NVFP4."""
        algo = QuantAlgo.W4A16_NVFP4
        self.assertIsInstance(algo, QuantAlgo)
        self.assertNotEqual(algo, QuantAlgo.NVFP4,
                            "W4A16_NVFP4 must be a distinct enum member from NVFP4")

    @_skip_if_not_implemented
    def test_w4a16_nvfp4_quant_mode_matches_nvfp4(self):
        """QuantMode.from_quant_algo(W4A16_NVFP4) must produce the same bits as NVFP4."""
        mode_alias = QuantMode.from_quant_algo(QuantAlgo.W4A16_NVFP4)
        mode_nvfp4 = QuantMode.from_quant_algo(QuantAlgo.NVFP4)
        self.assertEqual(
            mode_alias,
            mode_nvfp4,
            f"Expected W4A16_NVFP4 QuantMode {mode_alias!r} == NVFP4 QuantMode {mode_nvfp4!r}",
        )

    @_skip_if_not_implemented
    def test_w4a16_nvfp4_quant_mode_has_nvfp4_flag(self):
        """The QuantMode bits for W4A16_NVFP4 must include the NVFP4 flag."""
        mode = QuantMode.from_quant_algo(QuantAlgo.W4A16_NVFP4)
        self.assertTrue(
            mode.has_nvfp4(),
            "QuantMode for W4A16_NVFP4 must have the NVFP4 flag set",
        )


# ---------------------------------------------------------------------------
# Deliverable 2 tests — load_modelopt_quant_config remapping
# ---------------------------------------------------------------------------

class TestModeloptRemapW4A16Nvfp4(unittest.TestCase):
    """load_modelopt_quant_config remap: W4A16_NVFP4 → NVFP4 (Deliverable 2)."""

    def setUp(self):
        # Guard at the test-class level so we get a clean skip message.
        if not _W4A16_NVFP4_PRESENT:
            self.skipTest("QuantAlgo.W4A16_NVFP4 not present — Phase 2 not merged")

        # Lazy import to avoid hard failure when model_config isn't updated yet.
        try:
            from tensorrt_llm._torch.model_config import ModelConfig
            self._ModelConfig = ModelConfig
        except ImportError as exc:
            self.skipTest(f"tensorrt_llm._torch.model_config unavailable: {exc}")

    def _load_quant_config(self, tmp_dir: Path, config_dict: dict):
        """Write a config file and call load_modelopt_quant_config.

        The signature is ``(quant_config_file, checkpoint_dir, moe_backend)``.
        We pass ``moe_backend='CUTLASS'`` to avoid the
        FP8_BLOCK_SCALES-specific ``exclude_modules`` injection.
        """
        quant_file = _write_quant_config_file(tmp_dir, config_dict)
        quant_config, layer_quant_config = self._ModelConfig.load_modelopt_quant_config(
            str(quant_file),
            str(tmp_dir),
            moe_backend="CUTLASS",
        )
        return quant_config, layer_quant_config

    def test_modelopt_remap_w4a16_nvfp4_to_nvfp4(self):
        """Per-layer W4A16_NVFP4 entries must be remapped to QuantAlgo.NVFP4."""
        config_dict = _make_mixed_precision_quant_config(layer_quant_algo="W4A16_NVFP4")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            quant_config, layer_quant_config = self._load_quant_config(
                tmp_path, config_dict
            )

        self.assertIsNotNone(
            layer_quant_config,
            "layer_quant_config must not be None for MIXED_PRECISION checkpoints",
        )
        # Check that all per-layer entries that were W4A16_NVFP4 are now NVFP4.
        for layer_name, cfg in layer_quant_config.items():
            if layer_name == "lm_head":
                # lm_head may be excluded; skip quant_algo check for it.
                continue
            self.assertEqual(
                cfg.quant_algo,
                QuantAlgo.NVFP4,
                f"Layer '{layer_name}' quant_algo expected NVFP4, got {cfg.quant_algo!r}",
            )

    def test_modelopt_remap_excludes_lm_head(self):
        """After W4A16_NVFP4 remap, 'lm_head' must appear in exclude_modules.

        Background: LMHead.__init__ bypasses create_weights (uses a raw
        Parameter) so quant scales are never allocated.  Remapping W4A16_NVFP4
        without excluding lm_head would cause a scale-load failure at runtime.
        """
        config_dict = _make_mixed_precision_quant_config(layer_quant_algo="W4A16_NVFP4")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            quant_config, layer_quant_config = self._load_quant_config(
                tmp_path, config_dict
            )

        exclude_modules = quant_config.exclude_modules or []
        # Accept both exact match and glob-style patterns containing "lm_head".
        lm_head_excluded = any(
            "lm_head" in m for m in exclude_modules
        )
        self.assertTrue(
            lm_head_excluded,
            f"'lm_head' must be in exclude_modules after W4A16_NVFP4 remap, "
            f"got exclude_modules={exclude_modules!r}",
        )

    def test_modelopt_remap_plain_nvfp4_unchanged(self):
        """Plain NVFP4 per-layer entries must pass through unchanged (regression guard)."""
        config_dict = _make_mixed_precision_quant_config(layer_quant_algo="NVFP4")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            quant_config, layer_quant_config = self._load_quant_config(
                tmp_path, config_dict
            )

        self.assertIsNotNone(layer_quant_config)
        for layer_name, cfg in layer_quant_config.items():
            if layer_name == "lm_head":
                continue
            self.assertEqual(
                cfg.quant_algo,
                QuantAlgo.NVFP4,
                f"Plain NVFP4 layer '{layer_name}' must stay NVFP4, got {cfg.quant_algo!r}",
            )


if __name__ == "__main__":
    unittest.main()
