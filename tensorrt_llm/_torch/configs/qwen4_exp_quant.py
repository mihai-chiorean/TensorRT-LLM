# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate Qwen4Exp checkpoint quantization names without changing policies."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.models.modeling_utils import QuantConfig


def _runtime_quant_name(name: str, num_hidden_layers: int, num_mtp_layers: int) -> str:
    if name.startswith("model.language_model."):
        return "model." + name.removeprefix("model.language_model.")
    if name.startswith("mtp.layers."):
        layer, separator, suffix = name.removeprefix("mtp.layers.").partition(".")
        if not layer.isdecimal() or not separator or not suffix:
            raise ValueError(f"Invalid Qwen4Exp MTP quantization key: {name!r}")
        # The single trained MTP layer is stored as mtp.layers.0. FlashNext
        # quantization metadata also names it by its absolute runtime index.
        if num_mtp_layers != 1 or int(layer) not in (0, num_hidden_layers):
            raise ValueError(
                f"Unsupported Qwen4Exp MTP quantization key {name!r}: "
                f"expected layer 0 or {num_hidden_layers} for one trained MTP layer"
            )
        return f"model.layers.{num_hidden_layers}.{suffix}"
    # Match the dedicated MTP projections and mixer in Qwen4ExpHfWeightMapper.
    mtp_modules = {
        "fc_embedding": "fc_embedding",
        "fc_hidden": "fc_hidden",
        "pre_fc_norm_embedding": "pre_fc_norm_embedding",
        "pre_fc_norm_hidden": "pre_fc_norm_hidden",
        "hyper_connection_mixer": "shared_head.hyper_connection_mixer",
    }
    if name.startswith("mtp."):
        module, separator, suffix = name.removeprefix("mtp.").partition(".")
        if module in mtp_modules:
            return f"model.layers.{num_hidden_layers}.{mtp_modules[module]}{separator}{suffix}"
    return name


def normalize_qwen4_exp_quant_config_dict(model_config: ModelConfig) -> None:
    """Map per-layer policies to runtime modules before model construction.

    GDN qkv/z and a/b projections share fused runtime Linears. Every member
    must have the same complete policy, including algorithm and group size;
    partial or conflicting policies fail instead of falling back to BF16.
    Other policies (including vision, routed/shared experts, and MTP) retain
    their algorithms. Mutate the dictionary only after all checks pass since
    ModelConfig itself is frozen. Already-normalized dictionaries are a no-op.
    """
    policies = model_config.quant_config_dict
    if not policies:
        return
    normalized: dict[str, QuantConfig] = {}

    def add_policy(name: str, policy: QuantConfig) -> None:
        if name in normalized and normalized[name] != policy:
            raise ValueError(f"Conflicting Qwen4Exp quantization policies for {name!r}")
        normalized[name] = policy

    for name, policy in policies.items():
        add_policy(
            _runtime_quant_name(
                name,
                model_config.pretrained_config.num_hidden_layers,
                getattr(model_config.pretrained_config, "mtp_num_hidden_layers", 1),
            ),
            policy,
        )

    fusion_groups = {"qkvz": ("qkv", "z"), "ba": ("a", "b")}
    prefixes = set()
    for name in normalized:
        prefix, separator, projection = name.rpartition(".linear_attn.in_proj_")
        if not separator:
            continue
        if projection in ("q", "k", "v"):
            raise ValueError(f"Unsupported split Qwen4Exp GDN quantization key: {name!r}")
        if projection in ("qkv", "z", "a", "b"):
            prefixes.add(prefix + separator)

    for prefix in sorted(prefixes):
        for fused, members in fusion_groups.items():
            names = [prefix + member for member in members]
            present = [name for name in names if name in normalized]
            if not present:
                continue
            if len(present) != len(names):
                raise ValueError(
                    f"Incomplete Qwen4Exp GDN quantization policies for {prefix + fused!r}: "
                    f"expected {names}, got {present}"
                )
            policy = normalized[names[0]]
            if any(normalized[name] != policy for name in names[1:]):
                raise ValueError(
                    f"Incompatible Qwen4Exp GDN quantization policies for {names}: "
                    "fused projections require matching algorithms, group sizes, and scale settings"
                )
            add_policy(prefix + fused, policy)
            for name in names:
                del normalized[name]

    policies.clear()
    policies.update(normalized)
