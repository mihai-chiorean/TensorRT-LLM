# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest


def _flash_next_config() -> dict:
    # Config-only subset of Mia-AiLab/Qwen3.8-Flash-Next-NVFP4 at
    # 925d7be6c14c6c9442ef83e8f05b5a3c39304f69. No checkpoint tensors are loaded.
    return {
        "model_type": "qwen3_8_flash_next",
        "architectures": ["Qwen3_8FlashNextForConditionalGeneration"],
        "dtype": "bfloat16",
        "image_token_id": 248056,
        "video_token_id": 248057,
        "vision_start_token_id": 248053,
        "vision_end_token_id": 248054,
        "text_config": {
            "model_type": "qwen3_8_flash_next_text",
            "dtype": "bfloat16",
            "hidden_size": 2560,
            "intermediate_size": 12288,
            "num_hidden_layers": 48,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
            "tie_word_embeddings": False,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "mamba_ssm_dtype": "float32",
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 640,
            "shared_expert_intermediate_size": 640,
            "hc_count": 4,
            "hc_lowrank": 320,
            "ple_layer_ids": [2],
            "ple_embed_dim": 2560,
            "ple_conv_kernel_size": 4,
            "ngram_size": 3,
            "heads_per_ngram": 8,
            "ngram_vocab_size_base": 20000000,
            "make_ngram_vocab_size_divisible_by": 128,
            "split_ngram_parts": 128,
            "ple_embedding_dtype": "nvfp4",
            "output_gate_type": "sigmoid",
            "indexer_n_heads": 4,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 128,
            "indexer_budget": 2048,
            "indexer_compress_ratio": 4,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000,
                "rope_type": "default",
            },
            "mtp": {"hybrid": True, "num_hidden_layers": 1, "layer_types": ["full_attention"]},
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
        },
        "vision_config": {
            "model_type": "qwen3_5_vision",
            "dtype": "bfloat16",
            "depth": 27,
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_heads": 16,
            "out_hidden_size": 2560,
            "deepstack_visual_indexes": [],
        },
        "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                "model.language_model.layers.0.linear_attn.in_proj_qkv": {
                    "quant_algo": "MXFP8",
                    "group_size": 32,
                },
                "model.language_model.layers.0.mlp.experts": {
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                },
                "model.visual.blocks.0.mlp.linear_fc2": {
                    "quant_algo": "W4A16_NVFP4",
                    "group_size": 16,
                },
            },
        },
    }


@pytest.fixture
def flash_next_config() -> dict:
    config = _flash_next_config()
    config["text_config"]["layer_types"] *= 12
    return config


@pytest.mark.parametrize("language_model_only", [False, True])
def test_flash_next_load_normalizes_names_and_preserves_fields(
    tmp_path: Path, flash_next_config: dict, language_model_only: bool
) -> None:
    from tensorrt_llm._torch.pyexecutor.config_utils import is_qwen4_exp, load_pretrained_config

    flash_next_config["language_model_only"] = language_model_only
    (tmp_path / "config.json").write_text(json.dumps(flash_next_config))
    config = load_pretrained_config(str(tmp_path), trust_remote_code=False)

    if language_model_only:
        text_config = config
        assert not hasattr(config, "vision_config")
    else:
        assert config.model_type == "qwen4_exp"
        assert config.architectures == ["Qwen4ExpForConditionalGeneration"]
        assert config.vision_config.model_type == "qwen4_exp_vision"
        for key, value in flash_next_config["vision_config"].items():
            if key not in ("model_type", "dtype"):
                assert getattr(config.vision_config, key) == value
        for key in (
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
        ):
            assert getattr(config, key) == flash_next_config[key]
        text_config = config.text_config

    assert is_qwen4_exp(config)
    assert text_config.model_type == "qwen4_exp_text"
    assert text_config.architectures == ["Qwen4ExpForCausalLM"]
    for key, value in flash_next_config["text_config"].items():
        if key not in ("model_type", "dtype", "rope_parameters"):
            assert getattr(text_config, key) == value
    assert str(text_config.dtype).removeprefix("torch.") == "bfloat16"
    assert text_config.rope_scaling["rope_theta"] == 10000000
    assert text_config.partial_rotary_factor == 0.25
    assert text_config.rope_scaling["type"] == "mrope"
    assert text_config.rope_scaling["mrope_section"] == [11, 11, 10]
    assert text_config.rope_scaling["mrope_interleaved"] is True
    assert config.quantization_config == flash_next_config["quantization_config"]
    assert text_config.quantization_config == flash_next_config["quantization_config"]


@pytest.mark.parametrize("model_type", ["qwen3_8_flash_next", "qwen3_8_flash_next_text"])
def test_flash_next_flat_text_config(
    tmp_path: Path, flash_next_config: dict, model_type: str
) -> None:
    from tensorrt_llm._torch.pyexecutor.config_utils import load_pretrained_config

    fields = flash_next_config["text_config"]
    fields["model_type"] = model_type
    fields["architectures"] = ["Qwen3_8FlashNextForCausalLM"]
    (tmp_path / "config.json").write_text(json.dumps(fields))

    config = load_pretrained_config(str(tmp_path))
    assert config.model_type == "qwen4_exp_text"
    assert config.architectures == ["Qwen4ExpForCausalLM"]
    assert config.ple_embedding_dtype == "nvfp4"


@pytest.mark.parametrize("text_only", [False, True])
def test_flash_next_auto_config_alias(
    tmp_path: Path, flash_next_config: dict, text_only: bool
) -> None:
    from transformers import AutoConfig

    import tensorrt_llm._torch.configs  # noqa: F401

    fields = flash_next_config["text_config"] if text_only else flash_next_config
    if text_only:
        fields["architectures"] = ["Qwen3_8FlashNextForCausalLM"]
    (tmp_path / "config.json").write_text(json.dumps(fields))
    config = AutoConfig.from_pretrained(str(tmp_path), trust_remote_code=False)

    expected_type = "qwen4_exp_text" if text_only else "qwen4_exp"
    expected_arch = "Qwen4ExpForCausalLM" if text_only else "Qwen4ExpForConditionalGeneration"
    assert config.model_type == expected_type
    assert config.architectures == [expected_arch]


def test_flash_next_normalization_does_not_mutate_input(flash_next_config: dict) -> None:
    from tensorrt_llm._torch.configs import Qwen4ExpConfig

    original = deepcopy(flash_next_config)
    config = Qwen4ExpConfig.from_dict(flash_next_config)
    assert flash_next_config == original
    assert config.text_config.model_type == "qwen4_exp_text"
    assert config.vision_config.model_type == "qwen4_exp_vision"


@pytest.mark.parametrize("suffix", ["ForCausalLM", "ForConditionalGeneration"])
def test_flash_next_registries_reuse_qwen4_exp(suffix: str) -> None:
    from tensorrt_llm._torch.models._arch_index import MODEL_ARCH_TO_MODULE
    from tensorrt_llm._torch.models.checkpoints.auto_mapper import AutoCheckpointMapper
    from tensorrt_llm._torch.models.checkpoints.hf.qwen4_exp_weight_mapper import (
        Qwen4ExpHfWeightMapper,
    )
    from tensorrt_llm._torch.models.modeling_utils import get_registered_model_class
    from tensorrt_llm._torch.pyexecutor.config_utils import is_qwen4_exp

    architecture = f"Qwen3_8FlashNext{suffix}"
    assert MODEL_ARCH_TO_MODULE[architecture] == "modeling_qwen4_exp"
    assert get_registered_model_class(architecture) is get_registered_model_class(
        f"Qwen4Exp{suffix}"
    )
    assert isinstance(AutoCheckpointMapper.get("HF", architecture), Qwen4ExpHfWeightMapper)
    assert isinstance(AutoCheckpointMapper.get("MX", architecture), Qwen4ExpHfWeightMapper)
    assert is_qwen4_exp(SimpleNamespace(architectures=[architecture]))


@pytest.mark.parametrize("model_type", ["qwen3_8_flash_next", "qwen3_8_flash_next_text"])
def test_flash_next_requires_lazy_safetensors(tmp_path: Path, model_type: str) -> None:
    from tensorrt_llm._torch.models.checkpoints.hf.weight_loader import HfWeightLoader

    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    assert HfWeightLoader._requires_lazy_safetensors(str(tmp_path))


def test_flash_next_aliases_do_not_match_other_models() -> None:
    from tensorrt_llm._torch.pyexecutor.config_utils import (
        is_qwen4_exp,
        is_qwen4_exp_multimodal_config,
    )

    assert not is_qwen4_exp(SimpleNamespace(architectures=["Qwen3_5MoeForConditionalGeneration"]))
    assert not is_qwen4_exp_multimodal_config(
        {
            "model_type": "qwen3_5_moe",
            "text_config": {"hidden_size": 2560},
            "vision_config": {"depth": 27},
        }
    )
