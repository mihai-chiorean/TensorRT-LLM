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
"""Qwen3.5-MoE MTP (Multi-Token Prediction) draft layer.

Architecture description
------------------------
The Qwen3.6-35B-A3B-NVFP4 checkpoint stores one MTP speculative-decoding
draft layer under the ``mtp.*`` top-level key namespace:

    mtp.pre_fc_norm_embedding.*   — RMSNorm applied to the input embedding
    mtp.pre_fc_norm_hidden.*      — RMSNorm applied to the main-model hidden state
    mtp.fc.*                      — Linear(2*hidden_size → hidden_size) projection
    mtp.layers.0.*                — A full-attention + MoE decoder layer
    mtp.norm.*                    — Final RMSNorm before the LM head

The weight mapper (Qwen3_5MoeHfWeightMapper._normalize_weight_names) remaps
these to ``model.layers.{num_hidden_layers}.*`` so the standard weight loader
can place them correctly.

This module implements ``Qwen35MoeMTP``, mirroring the
DeepseekV3MTP / NemotronHMTP patterns while adapting to the Qwen3Next
decoder-layer architecture.
"""

from dataclasses import replace
from typing import Dict, List, Optional

import torch
from torch import nn

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.speculative import SpecMetadata
from tensorrt_llm._torch.utils import AuxStreamType
from tensorrt_llm.models.modeling_utils import QuantConfig

from ..attention_backend import AttentionMetadata


class Qwen35MoeMTPHead(nn.Module):
    """Lightweight MTP logit head for Qwen3.5 MoE.

    Mirrors DeepseekV3MTPHead but accepts a pre-built ``norm`` so the weight
    loader maps checkpoint key ``model.layers.{N}.norm.*`` to the single
    ``Qwen35MoeMTP.norm`` attribute (rather than a nested ``shared_head.norm``
    attribute), keeping the weight key namespace consistent with the checkpoint.
    """

    def __init__(self, model_config: ModelConfig, norm: RMSNorm):
        super().__init__()
        self.model_config = model_config
        # Reference to the outer MTP layer's norm — NOT a new sub-module.
        # The weight for this norm is loaded via model.layers.N.norm.* on the
        # parent Qwen35MoeMTP.  Using object.__setattr__ bypasses nn.Module's
        # __setattr__ hook so PyTorch does NOT register norm as a sub-module of
        # shared_head (which would create model.layers.N.shared_head._norm.*).
        object.__setattr__(self, '_norm', norm)
        self.mapping_lm_head_tp = None

    @property
    def norm(self):
        return self._norm

    @torch.compile(options={"max-autotune": True})
    def _get_last_token_states(self, hidden_states: torch.Tensor,
                               attn_metadata: AttentionMetadata) -> torch.Tensor:
        last_tokens = torch.cumsum(
            attn_metadata.seq_lens_cuda, dim=0, dtype=torch.long) - 1
        return hidden_states[last_tokens]

    def forward(
        self,
        hidden_states: torch.Tensor,
        lm_head: Linear,
        attn_metadata: AttentionMetadata,
        return_context_logits: bool = False,
    ) -> torch.Tensor:
        if not return_context_logits:
            if attn_metadata is not None:
                hidden_states = self._get_last_token_states(
                    hidden_states, attn_metadata)
            else:
                hidden_states = hidden_states[-1].unsqueeze(0)

        enable_attention_dp = self.model_config.mapping.enable_attention_dp
        # Disable gather_output for the draft head (mirroring DeepseekV3MTPHead)
        if not enable_attention_dp:
            lm_head.gather_output = False
        logits = lm_head(hidden_states,
                         mapping_lm_head_tp=self.mapping_lm_head_tp,
                         is_spec_decoding_head=True)
        if not enable_attention_dp:
            lm_head.gather_output = True
        return logits


class Qwen35MoeMTP(nn.Module):
    """Qwen3.5-MoE MTP draft layer.

    Weight key namespace (after Qwen3_5MoeHfWeightMapper remapping):

        model.layers.{layer_idx}.pre_fc_norm_embedding.*  — embedding RMSNorm
        model.layers.{layer_idx}.pre_fc_norm_hidden.*     — hidden-state RMSNorm
        model.layers.{layer_idx}.fc.*                     — concat projection
        model.layers.{layer_idx}.layers.0.*               — decoder sub-layer
        model.layers.{layer_idx}.norm.*                   — final RMSNorm

    The sublayer at ``layers.0`` is a ``Qwen3NextFullAttentionDecoderLayer``
    with a forced unquantized quant_config (quant_algo=None), mirroring the
    NemotronHMTP._get_mtp_sublayer_quant_config pattern.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.model_config = model_config
        self.config = config
        self.layer_idx = layer_idx

        # ------------------------------------------------------------------ #
        # 1. Pre-FC normalization layers                                       #
        # ------------------------------------------------------------------ #
        # Weight keys: model.layers.{N}.pre_fc_norm_embedding.weight
        #              model.layers.{N}.pre_fc_norm_hidden.weight
        self.pre_fc_norm_embedding = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )
        self.pre_fc_norm_hidden = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

        # ------------------------------------------------------------------ #
        # 2. Projection: concat([embed_norm, hidden_norm]) → hidden           #
        # ------------------------------------------------------------------ #
        # Weight key: model.layers.{N}.fc.weight
        self.fc = Linear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
            dtype=config.torch_dtype,
        )

        # ------------------------------------------------------------------ #
        # 3. Decoder sub-layer (full-attention + MoE)                         #
        # ------------------------------------------------------------------ #
        # The MTP layers in the NVFP4 checkpoint are stored in BF16 without
        # quantization. Force quant_algo=None to avoid mismatched kernel
        # selection in the TRTLLM MoE backend.
        sublayer_quant_config = self._get_mtp_sublayer_quant_config(
            model_config)
        sublayer_model_config = replace(
            model_config,
            quant_config=sublayer_quant_config,
            spec_config=None,
        )

        # Import here to avoid circular imports at module load time.
        from tensorrt_llm._torch.models.modeling_qwen3_next import (
            Qwen3NextFullAttentionDecoderLayer,
        )

        # Use the provided stream if available; otherwise create one lazily.
        # Passing an empty aux_stream_dict is valid for structural tests that
        # do not run actual CUDA forward passes.
        aux_stream = aux_stream_dict.get(AuxStreamType.MoeChunkingOverlap, None)
        if aux_stream is None and torch.cuda.is_available():
            aux_stream = torch.cuda.Stream()
        # Weight key: model.layers.{N}.layers.0.*
        # Stored as self.layers["0"] so PyTorch's state_dict uses "layers.0.*"
        self.layers = nn.ModuleDict({
            "0":
            Qwen3NextFullAttentionDecoderLayer(
                model_config=sublayer_model_config,
                layer_idx=layer_idx,
                aux_stream=aux_stream,
            )
        })

        # ------------------------------------------------------------------ #
        # 4. Final RMSNorm (applied after the decoder sub-layer)              #
        # ------------------------------------------------------------------ #
        # Weight key: model.layers.{N}.norm.weight
        self.norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

        # ------------------------------------------------------------------ #
        # 5. Shared LM-head logit projection (references self.norm)           #
        # ------------------------------------------------------------------ #
        # Qwen35MoeMTPHead.norm is a property that returns self._norm, so
        # weight loading for "model.layers.{N}.norm.*" goes to self.norm,
        # NOT to self.shared_head (where it would be "shared_head.norm.*").
        self.shared_head = Qwen35MoeMTPHead(model_config, norm=self.norm)

    @staticmethod
    def _get_mtp_sublayer_quant_config(
            model_config: ModelConfig) -> Optional[QuantConfig]:
        """Return an unquantized QuantConfig for the MTP sub-layer.

        The Qwen3.6-NVFP4 checkpoint stores MTP weights in BF16, so the
        sub-layer must use quant_algo=None to avoid type errors in the
        quantized MoE kernel path.
        """
        quant_config = model_config.quant_config
        if quant_config is None:
            return None
        return QuantConfig(
            quant_algo=None,
            kv_cache_quant_algo=quant_config.kv_cache_quant_algo,
        )

    def forward(
        self,
        input_ids: torch.IntTensor,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        embed_tokens: Embedding,
        attn_metadata: AttentionMetadata,
        all_rank_num_tokens: Optional[List[int]] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        # 1. Normalize embedding tokens and hidden states.
        inputs_embeds = self.pre_fc_norm_embedding(embed_tokens(input_ids))
        hidden_norm = self.pre_fc_norm_hidden(hidden_states)

        # 2. Concatenate and project to hidden_size.
        hidden_states = self.fc(
            torch.cat([inputs_embeds, hidden_norm], dim=-1))

        # 3. Run through the decoder sub-layer.
        sublayer = self.layers["0"]
        hidden_states, residual = sublayer(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            residual=None,
            spec_metadata=spec_metadata,
            **kwargs,
        )

        # 4. Apply final norm.
        hidden_states, _ = self.norm(hidden_states, residual)

        if spec_metadata is not None:
            spec_metadata.maybe_capture_hidden_states(0, hidden_states, None)

        return hidden_states
