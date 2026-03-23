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

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm.functional import PositionEmbeddingType

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ..models.modeling_utils import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.fused_moe import MiniMaxM2MoeRoutingMethod, create_moe
from ..modules.linear import Linear
from ..modules.rms_norm import RMSNorm
from ..utils import AuxStreamType
from .modeling_utils import DecoderModel, DecoderModelForCausalLM, register_auto_model


class _EScoreCorrectionBiasHolder(nn.Module):
    """Holds e_score_correction_bias so the generic weight loader visits it with a narrow
    prefix (block_sparse_moe.e_score_correction_bias). This avoids mark_consumed deleting
    the whole block_sparse_moe prefix before gate and experts.backend load (see #11119).
    """

    def __init__(self, num_experts: int):
        super().__init__()
        self.e_score_correction_bias = nn.Parameter(
            torch.empty((num_experts), dtype=torch.float32), requires_grad=False
        )

    def load_weights(self, weights: List[Dict]):
        assert len(weights) == 1
        w = weights[0]
        src = w[""]
        self.e_score_correction_bias.copy_(src[:].to(self.e_score_correction_bias.dtype))


class MiniMaxMoE(nn.Module):
    """MoE block with sigmoid + bias correction routing (same as M2)."""

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        aux_stream: torch.cuda.Stream,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.enable_attention_dp = model_config.mapping.enable_attention_dp

        self.gate = Linear(
            self.hidden_dim, self.num_experts, bias=False, dtype=torch.float32, quant_config=None
        )

        reduce_results = True
        self.experts = create_moe(
            routing_method=MiniMaxM2MoeRoutingMethod(
                top_k=self.top_k,
                num_experts=self.num_experts,
                callable_e_score_correction_bias=lambda: self.e_score_correction_bias.e_score_correction_bias,
            ),
            num_experts=self.num_experts,
            aux_stream_dict={AuxStreamType.MoeChunkingOverlap: aux_stream},
            reduce_results=reduce_results,
            model_config=model_config,
            layer_idx=layer_idx,
        )
        self.e_score_correction_bias = _EScoreCorrectionBiasHolder(self.num_experts)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        all_rank_num_tokens = attn_metadata.all_rank_num_tokens
        hidden_states_f32 = hidden_states.to(torch.float32)
        router_logits = self.gate(hidden_states_f32)
        final_hidden_states = self.experts(
            hidden_states,
            router_logits,
            all_rank_num_tokens=all_rank_num_tokens,
            use_dp_padding=False,
        )
        return final_hidden_states


class MiniMaxAttention(Attention):
    """Full attention with QK norm across the full head_num * head_size dimension.

    Same implementation as MiniMaxM2Attention: performs allgather before QK norm
    when using tensor parallelism, then applies RoPE.
    """

    def __init__(
        self,
        *,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: Optional[int] = None,
    ):
        config = model_config.pretrained_config
        self.pretrained_config = config

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=False,
            pos_embd_params=PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gpt_neox,
                rope=RopeParams.from_config(config),
            ),
            rope_fusion=True,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
        )

        self.q_norm = RMSNorm(
            hidden_size=self.q_size * self.tp_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )
        self.k_norm = RMSNorm(
            hidden_size=self.kv_size * self.tp_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

    def apply_qk_norm(self, q, k):
        if self.qkv_proj.mapping.tp_size > 1:
            from ..distributed import allgather

            temp_q = allgather(q, self.qkv_proj.mapping)
            temp_k = allgather(k, self.qkv_proj.mapping)
            temp_q = self.q_norm(temp_q)
            temp_k = self.k_norm(temp_k)
            q = temp_q.reshape(-1, self.tp_size, self.q_size)[:, self.tp_rank, :].reshape(
                -1, self.q_size
            )
            k = temp_k.reshape(-1, self.tp_size, self.kv_size)[:, self.tp_rank, :].reshape(
                -1, self.kv_size
            )
        else:
            q = self.q_norm(q)
            k = self.k_norm(k)

        return q, k

    def apply_rope(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        position_ids: torch.Tensor,
    ):
        """Override to apply QK norm before RoPE."""
        q, k, v = self.split_qkv(q, k, v)
        q, k = self.apply_qk_norm(q, k)
        return super().apply_rope(q, k, v, position_ids)


class MiniMaxLightningAttention(nn.Module):
    """Lightning Attention - linear attention with block-based decay.

    This module does NOT use TRT-LLM's standard attention backend.
    It implements the linear attention mechanism from the MiniMax M2.5 model
    with block-based intra/inter computation for prefill and streaming state
    cache for decode.
    """

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: int,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.layer_idx = layer_idx
        self.head_dim = (
            getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        )
        self.num_attention_heads = config.num_attention_heads
        self.num_hidden_layers = config.num_hidden_layers
        self.block_size = config.block_size
        self.hidden_size = config.hidden_size

        qkv_size = self.num_attention_heads * self.head_dim * 3
        self.qkv_proj = Linear(
            config.hidden_size,
            qkv_size,
            bias=False,
            dtype=config.torch_dtype,
            quant_config=model_config.quant_config,
        )
        self.out_proj = Linear(
            self.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            dtype=config.torch_dtype,
            quant_config=model_config.quant_config,
        )
        self.output_gate = Linear(
            config.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=False,
            dtype=config.torch_dtype,
            quant_config=model_config.quant_config,
        )
        self.norm = RMSNorm(
            hidden_size=self.num_attention_heads * self.head_dim,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

        # Pre-compute decay factors (constants, not learnable parameters)
        slope_rate = self._get_slope_rate()
        query_decay, key_decay, diagonal_decay = self._decay_factors(slope_rate)

        self.register_buffer("slope_rate", slope_rate, persistent=False)
        self.register_buffer("query_decay", query_decay, persistent=False)
        self.register_buffer("key_decay", key_decay, persistent=False)
        self.register_buffer("diagonal_decay", diagonal_decay, persistent=False)

        # Linear attention state cache: [batch, heads, head_dim, head_dim]
        # Stored as a module attribute, managed per forward call
        self._linear_state = None

    def _get_slope_rate(self) -> torch.Tensor:
        """Compute per-head slope rates for exponential decay."""
        base = 1.0 / (2.0 ** (8.0 / self.num_attention_heads))
        exponent = torch.arange(self.num_attention_heads, dtype=torch.float32) + 1
        factor = 1.0 - self.layer_idx / (self.num_hidden_layers - 1 + 1e-5) + 1e-5

        rate = base**exponent
        rate = rate * factor
        # Shape: [num_heads, 1, 1]
        rate = rate[:, None, None]
        return rate

    def _decay_factors(self, slope_rate: torch.Tensor):
        """Compute block-level decay matrices from slope_rate."""
        block_size_range = torch.arange(self.block_size, dtype=torch.float32) + 1

        # query_decay: [num_heads, block_size, 1]
        query_decay = torch.exp(-slope_rate * block_size_range[:, None])
        # key_decay: [num_heads, block_size, 1]
        key_decay = torch.exp(-slope_rate * (self.block_size - block_size_range[:, None]))

        # diagonal_decay: [1, num_heads, block_size, block_size]
        diagonal_decay = block_size_range[:, None] - block_size_range[None, :]
        diagonal_decay = diagonal_decay[None, None, :, :]
        diagonal_decay = slope_rate * diagonal_decay
        diagonal_decay = torch.where(diagonal_decay >= 0, -diagonal_decay, float("-inf"))
        diagonal_decay = torch.exp(diagonal_decay)

        return query_decay, key_decay, diagonal_decay

    def _prefill(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> torch.Tensor:
        """Block-based prefill with intra/inter-block computation.

        Args:
            query_states: [batch, heads, seq_len, head_dim]
            key_states: [batch, heads, seq_len, head_dim]
            value_states: [batch, heads, seq_len, head_dim]

        Returns:
            attn_output: [batch, heads, seq_len, head_dim]
        """
        batch_size = query_states.shape[0]
        seq_len = query_states.shape[2]
        num_blocks = (seq_len + self.block_size - 1) // self.block_size

        # Initialize inter-block state: [batch, heads, head_dim, head_dim]
        attn_weights_inter = torch.zeros(
            batch_size,
            self.num_attention_heads,
            self.head_dim,
            self.head_dim,
            device=query_states.device,
            dtype=query_states.dtype,
        )

        attn_output = []
        for i in range(num_blocks):
            start_idx = i * self.block_size
            end_idx = min(start_idx + self.block_size, seq_len)
            current_block_size = end_idx - start_idx

            current_q = query_states[:, :, start_idx:end_idx]
            current_k = key_states[:, :, start_idx:end_idx]
            current_v = value_states[:, :, start_idx:end_idx]

            current_query_decay = self.query_decay[:, :current_block_size]
            current_key_decay = self.key_decay[:, -current_block_size:]
            current_diagonal_decay = self.diagonal_decay[
                :, :, :current_block_size, :current_block_size
            ]
            block_decay = torch.exp(-self.slope_rate * current_block_size)

            # Intra-block: (Q @ K^T) * diagonal_decay @ V
            attn_weights_intra = torch.matmul(current_q, current_k.transpose(-1, -2))
            attn_output_intra = torch.matmul(attn_weights_intra * current_diagonal_decay, current_v)

            # Inter-block: Q * query_decay @ state
            attn_output_inter = torch.matmul(current_q * current_query_decay, attn_weights_inter)

            current_attn_output = attn_output_inter + attn_output_intra
            attn_output.append(current_attn_output)

            # Update inter-block state for next block
            next_attn_weights_inter = torch.matmul(
                (current_k * current_key_decay).transpose(-1, -2), current_v
            )
            attn_weights_inter = attn_weights_inter * block_decay + next_attn_weights_inter

        # Save state for subsequent decode steps
        self._linear_state = attn_weights_inter

        return torch.cat(attn_output, dim=2)

    def _decode(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> torch.Tensor:
        """Streaming decode with [batch, heads, d, d] state cache.

        Args:
            query_states: [batch, heads, seq_len, head_dim] (seq_len typically 1)
            key_states: [batch, heads, seq_len, head_dim]
            value_states: [batch, heads, seq_len, head_dim]

        Returns:
            attn_output: [batch, heads, seq_len, head_dim]
        """
        ratio = torch.exp(-self.slope_rate)

        if self._linear_state is None:
            self._linear_state = torch.zeros(
                query_states.shape[0],
                self.num_attention_heads,
                self.head_dim,
                self.head_dim,
                device=query_states.device,
                dtype=query_states.dtype,
            )

        seq_len = query_states.shape[2]
        attn_output = []
        for i in range(seq_len):
            current_q = query_states[:, :, i : i + 1]
            current_k = key_states[:, :, i : i + 1]
            current_v = value_states[:, :, i : i + 1]

            current_kv = torch.matmul(current_k.transpose(-1, -2), current_v)
            self._linear_state = ratio * self._linear_state + current_kv
            current_attn_output = torch.matmul(current_q, self._linear_state)
            attn_output.append(current_attn_output)

        return torch.cat(attn_output, dim=2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.IntTensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for lightning attention.

        In TRT-LLM PyTorch backend, hidden_states is [num_tokens, hidden_size]
        (flattened batch). We determine prefill vs decode from attn_metadata.
        """
        num_tokens = hidden_states.shape[0]

        # QKV projection with SiLU activation
        qkv_states = F.silu(self.qkv_proj(hidden_states))

        # Reshape to [num_tokens, num_heads, 3 * head_dim] then split
        qkv_states = qkv_states.view(num_tokens, self.num_attention_heads, 3 * self.head_dim)
        query_states, key_states, value_states = torch.split(qkv_states, self.head_dim, dim=2)

        # Determine prefill vs decode.
        # During prefill, num_tokens > 1 per sequence; during decode, num_tokens == 1 per sequence.
        # For lightning attention there is no positional embedding, so we treat the
        # entire flattened token batch as a single sequence.
        is_prefill = num_tokens > 1

        # Reshape to [1, num_heads, num_tokens, head_dim] for batched matmul
        query_states = query_states.transpose(0, 1).unsqueeze(0)  # [1, heads, tokens, dim]
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        if is_prefill:
            attn_output = self._prefill(query_states, key_states, value_states)
        else:
            attn_output = self._decode(query_states, key_states, value_states)

        # Reshape back: [1, heads, tokens, dim] -> [tokens, heads * dim]
        attn_output = attn_output.squeeze(0).transpose(0, 1).contiguous()
        attn_output = attn_output.view(num_tokens, self.num_attention_heads * self.head_dim)

        # Output: RMSNorm(attn_out) * sigmoid(gate(input)) -> out_proj
        attn_output = self.norm(attn_output)
        attn_output = F.sigmoid(self.output_gate(hidden_states)) * attn_output
        attn_output = self.out_proj(attn_output)

        return attn_output


class MiniMaxDecoderLayer(DecoderLayer):
    """Decoder layer for MiniMax M2.5 with alternating attention types
    and weighted residual connections.
    """

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: int,
        aux_stream: torch.cuda.Stream,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]

        # Attention: choose between full attention and lightning attention
        if self.layer_type == "linear_attention":
            self.self_attn = MiniMaxLightningAttention(
                model_config=model_config,
                layer_idx=layer_idx,
            )
            self.attn_alpha_factor = config.linear_attn_alpha_factor
            self.attn_beta_factor = config.linear_attn_beta_factor
        else:
            self.self_attn = MiniMaxAttention(
                model_config=model_config,
                layer_idx=layer_idx,
            )
            self.attn_alpha_factor = config.full_attn_alpha_factor
            self.attn_beta_factor = config.full_attn_beta_factor

        # MoE (same for both layer types)
        self.block_sparse_moe = MiniMaxMoE(
            model_config=model_config, aux_stream=aux_stream, layer_idx=layer_idx
        )

        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )

        # MLP residual factors (same for both layer types)
        self.mlp_alpha_factor = config.mlp_alpha_factor
        self.mlp_beta_factor = config.mlp_beta_factor

        self.mapping = model_config.mapping

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Self Attention
        if self.layer_type == "linear_attention":
            attn_output = self.self_attn(
                hidden_states=hidden_states,
                position_ids=position_ids,
                attn_metadata=attn_metadata,
                **kwargs,
            )
        else:
            attn_output = self.self_attn(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                **kwargs,
            )

        # Weighted residual connection for attention
        hidden_states = residual * self.attn_alpha_factor + attn_output * self.attn_beta_factor

        # Post-attention layernorm + MLP with weighted residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.block_sparse_moe(hidden_states, attn_metadata)
        hidden_states = residual * self.mlp_alpha_factor + mlp_output * self.mlp_beta_factor

        # Return hidden_states and None for residual since weighted residual
        # is already applied (not using fused add-norm pattern)
        return hidden_states, None


class MiniMaxModel(DecoderModel):
    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__(model_config)
        quant_config = model_config.quant_config
        if quant_config is None or (
            (not quant_config.quant_mode.has_fp8_kv_cache())
            and (not quant_config.quant_mode.has_fp4_kv_cache())
        ):
            model_config.pretrained_config.torch_dtype = torch.bfloat16
        config = model_config.pretrained_config
        self.vocab_size = config.vocab_size
        self.aux_stream = torch.cuda.Stream()

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            enable_torch_compile_for_embedding=model_config.enable_torch_compile_for_embedding,
        )

        self.layers = nn.ModuleList(
            [
                MiniMaxDecoderLayer(model_config, layer_idx, self.aux_stream)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, "
                "and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        residual = None
        for decoder_layer in self.layers:
            hidden_states, residual = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
            )

        # Since weighted residual layers return residual=None,
        # just apply final norm directly
        if residual is not None:
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            hidden_states = self.norm(hidden_states)
        return hidden_states


@register_auto_model("MiniMaxForCausalLM")
class MiniMaxForCausalLM(DecoderModelForCausalLM[MiniMaxModel, PretrainedConfig]):
    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__(
            MiniMaxModel(model_config),
            config=model_config,
            hidden_size=model_config.pretrained_config.hidden_size,
            vocab_size=model_config.pretrained_config.vocab_size,
        )
