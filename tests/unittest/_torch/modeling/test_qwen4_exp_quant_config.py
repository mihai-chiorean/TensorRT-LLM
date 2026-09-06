# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.configs.qwen4_exp_quant import normalize_qwen4_exp_quant_config_dict


def _policy(algorithm: str = "MXFP8", group_size: int = 32) -> SimpleNamespace:
    return SimpleNamespace(quant_algo=algorithm, group_size=group_size)


def _model_config(policies: dict | None) -> SimpleNamespace:
    return SimpleNamespace(
        quant_config_dict=policies,
        pretrained_config=SimpleNamespace(num_hidden_layers=48),
    )


def test_fuses_gdn_policies_without_changing_algorithms() -> None:
    prefix = "model.language_model.layers.0.linear_attn."
    policies = {prefix + f"in_proj_{name}": _policy() for name in ("qkv", "z", "a", "b")}
    policies[prefix + "out_proj"] = _policy()
    config = _model_config(policies)
    qkv_policy = policies[prefix + "in_proj_qkv"]
    a_policy = policies[prefix + "in_proj_a"]

    normalize_qwen4_exp_quant_config_dict(config)

    assert config.quant_config_dict is policies
    assert policies == {
        "model.layers.0.linear_attn.in_proj_qkvz": qkv_policy,
        "model.layers.0.linear_attn.in_proj_ba": a_policy,
        "model.layers.0.linear_attn.out_proj": _policy(),
    }
    assert policies["model.layers.0.linear_attn.in_proj_qkvz"] is qkv_policy
    original = deepcopy(policies)
    normalize_qwen4_exp_quant_config_dict(config)
    assert policies == original


@pytest.mark.parametrize("algorithm", ["MXFP8", "NVFP4", "W4A16_NVFP4", "FP8"])
def test_preserves_other_policies_and_runtime_paths(algorithm: str) -> None:
    indexer = "layers.3.self_attn.indexer.index_qk_proj"
    source_to_runtime = {
        "model.language_model.layers.0.mlp.experts": "model.layers.0.mlp.experts",
        "model.language_model.layers.0.mlp.gate": "model.layers.0.mlp.gate",
        "model.language_model.layers.0.mlp.shared_expert.gate_proj": "model.layers.0.mlp.shared_expert.gate_proj",
        "model.language_model.layers.0.mlp.shared_expert.up_proj": "model.layers.0.mlp.shared_expert.up_proj",
        "model.language_model.layers.0.mlp.shared_expert.down_proj": "model.layers.0.mlp.shared_expert.down_proj",
        f"model.language_model.{indexer}": f"model.{indexer}",
        "model.visual.blocks.0.mlp.linear_fc2": "model.visual.blocks.0.mlp.linear_fc2",
        "lm_head": "lm_head",
    }
    policy = _policy(algorithm, 16)
    config = _model_config(dict.fromkeys(source_to_runtime, policy))

    normalize_qwen4_exp_quant_config_dict(config)

    assert config.quant_config_dict == dict.fromkeys(source_to_runtime.values(), policy)
    assert all(value is policy for value in config.quant_config_dict.values())


@pytest.mark.parametrize(
    ("source", "runtime"),
    [
        ("mtp.layers.0.mlp.experts", "model.layers.48.mlp.experts"),
        ("mtp.layers.48.mlp.experts", "model.layers.48.mlp.experts"),
        ("mtp.fc_embedding", "model.layers.48.fc_embedding"),
        ("mtp.fc_hidden", "model.layers.48.fc_hidden"),
        ("mtp.pre_fc_norm_embedding", "model.layers.48.pre_fc_norm_embedding"),
        ("mtp.pre_fc_norm_hidden", "model.layers.48.pre_fc_norm_hidden"),
        (
            "mtp.hyper_connection_mixer.down",
            "model.layers.48.shared_head.hyper_connection_mixer.down",
        ),
        ("mtp.fc_embedding_extra", "mtp.fc_embedding_extra"),
    ],
)
def test_mtp_names_resolve_to_weight_mapper_runtime_paths(source: str, runtime: str) -> None:
    policy = _policy("W4A16_NVFP4", 16)
    config = _model_config({source: policy})
    normalize_qwen4_exp_quant_config_dict(config)
    assert config.quant_config_dict == {runtime: policy}


def test_mtp_relative_and_absolute_metadata_aliases_share_one_policy() -> None:
    policy = _policy("W4A16_NVFP4", 16)
    config = _model_config(
        {
            "mtp.layers.0.mlp.experts": policy,
            "mtp.layers.48.mlp.experts": deepcopy(policy),
        }
    )
    normalize_qwen4_exp_quant_config_dict(config)
    assert config.quant_config_dict == {"model.layers.48.mlp.experts": policy}
    normalize_qwen4_exp_quant_config_dict(config)
    assert config.quant_config_dict == {"model.layers.48.mlp.experts": policy}


def test_conflicting_mtp_aliases_fail_atomically() -> None:
    config = _model_config(
        {
            "mtp.layers.0.mlp.experts": _policy("W4A16_NVFP4", 16),
            "mtp.layers.48.mlp.experts": _policy("NVFP4", 16),
        }
    )
    original = deepcopy(config.quant_config_dict)
    with pytest.raises(ValueError, match="Conflicting.*quantization policies"):
        normalize_qwen4_exp_quant_config_dict(config)
    assert config.quant_config_dict == original


@pytest.mark.parametrize("layer", ["1", "47", "49", "96", "-1", "bad", ""])
def test_unknown_mtp_layer_is_rejected(layer: str) -> None:
    config = _model_config({f"mtp.layers.{layer}.mlp.experts": _policy("W4A16_NVFP4", 16)})
    original = deepcopy(config.quant_config_dict)
    with pytest.raises(ValueError, match="(Invalid|Unsupported).*MTP quantization key"):
        normalize_qwen4_exp_quant_config_dict(config)
    assert config.quant_config_dict == original


@pytest.mark.parametrize("num_mtp_layers", [0, 2])
def test_mtp_alias_is_bounded_to_one_trained_layer(num_mtp_layers: int) -> None:
    config = _model_config({"mtp.layers.48.mlp.experts": _policy("W4A16_NVFP4", 16)})
    config.pretrained_config.mtp_num_hidden_layers = num_mtp_layers
    with pytest.raises(ValueError, match="one trained MTP layer"):
        normalize_qwen4_exp_quant_config_dict(config)


@pytest.mark.parametrize("members", [("qkv", "z"), ("a", "b")])
@pytest.mark.parametrize("mismatch", ["quant_algo", "group_size", "scale_settings"])
def test_rejects_incompatible_gdn_policies_atomically(
    members: tuple[str, str], mismatch: str
) -> None:
    prefix = "model.language_model.layers.0.linear_attn.in_proj_"
    first, second = _policy(), _policy()
    setattr(
        second,
        mismatch,
        {"quant_algo": "NVFP4", "group_size": 16, "scale_settings": "different"}[mismatch],
    )
    config = _model_config({prefix + members[0]: first, prefix + members[1]: second})
    original = deepcopy(config.quant_config_dict)

    with pytest.raises(ValueError, match="Incompatible.*GDN quantization"):
        normalize_qwen4_exp_quant_config_dict(config)
    assert config.quant_config_dict == original


@pytest.mark.parametrize("projection", ["qkv", "z", "a", "b", "q", "k", "v"])
def test_rejects_incomplete_or_unsupported_gdn_sets(projection: str) -> None:
    config = _model_config({f"model.layers.0.linear_attn.in_proj_{projection}": _policy()})
    original = deepcopy(config.quant_config_dict)
    with pytest.raises(ValueError, match="(Incomplete|Unsupported).*GDN quantization"):
        normalize_qwen4_exp_quant_config_dict(config)
    assert config.quant_config_dict == original


def test_rejects_conflicting_canonical_policy() -> None:
    config = _model_config(
        {
            "model.language_model.layers.0.mlp.experts": _policy("NVFP4", 16),
            "model.layers.0.mlp.experts": _policy("MXFP8", 32),
        }
    )
    original = deepcopy(config.quant_config_dict)
    with pytest.raises(ValueError, match="Conflicting.*quantization policies"):
        normalize_qwen4_exp_quant_config_dict(config)
    assert config.quant_config_dict == original


def test_rejects_conflicting_fused_policy() -> None:
    prefix = "model.layers.0.linear_attn.in_proj_"
    config = _model_config(
        {
            prefix + "qkv": _policy(),
            prefix + "z": _policy(),
            prefix + "qkvz": _policy("NVFP4", 16),
        }
    )
    original = deepcopy(config.quant_config_dict)
    with pytest.raises(ValueError, match="Conflicting.*quantization policies"):
        normalize_qwen4_exp_quant_config_dict(config)
    assert config.quant_config_dict == original


@pytest.mark.parametrize("policies", [None, {}])
def test_no_layerwise_policies_is_a_noop(policies: dict | None) -> None:
    config = _model_config(policies)
    normalize_qwen4_exp_quant_config_dict(config)
    assert config.quant_config_dict is policies


def test_model_normalizes_quantization_before_decoder_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensorrt_llm._torch.models import modeling_qwen4_exp

    config = _model_config(
        {
            "model.language_model.layers.0.linear_attn.in_proj_qkv": _policy(),
            "model.language_model.layers.0.linear_attn.in_proj_z": _policy(),
        }
    )
    config.quant_config = SimpleNamespace(exclude_modules=None)

    class ConstructionReached(Exception):
        pass

    def check_config(model_config: SimpleNamespace) -> None:
        assert model_config.quant_config_dict == {
            "model.layers.0.linear_attn.in_proj_qkvz": _policy()
        }
        raise ConstructionReached

    monkeypatch.setattr(modeling_qwen4_exp, "Qwen4ExpModel", check_config)
    with pytest.raises(ConstructionReached):
        modeling_qwen4_exp.Qwen4ExpForCausalLM(config)
