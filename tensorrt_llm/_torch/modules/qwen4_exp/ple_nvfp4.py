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
"""Bounded eager PLE lookup from CPU mmap-backed NVFP4 checkpoint shards."""

import math
import re
from typing import Protocol

import torch
from torch import nn

__all__ = ["Qwen4ExpNVFP4MmapEmbedding"]


class _SliceSource(Protocol):
    def __getitem__(self, index: slice) -> torch.Tensor: ...


def _cpu_view(source: torch.Tensor | _SliceSource, name: str) -> torch.Tensor:
    # The HF lazy loader uses safe_open(framework="pt", device="cpu"). Its
    # PySafeSlice[:] is a view of torch's file storage, not a tensor-sized copy.
    # Keeping the resulting tensor owns that mapping after weights.clear().
    value = source if isinstance(source, torch.Tensor) else source[:]
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise ValueError(f"PLE NVFP4 {name} must use CPU checkpoint storage")
    if not value.is_contiguous():
        raise ValueError(f"PLE NVFP4 {name} must be contiguous")
    return value.detach()


class Qwen4ExpNVFP4MmapEmbedding(nn.Module):
    """Retain packed CPU views; decode only requested rows into BF16.

    Shards contain uint8 ``[rows, embedding_dim / 2]`` low-nibble-first E2M1
    weights and float8_e4m3fn ``[rows, embedding_dim / 16]`` block scales.
    The shared float32 ``[1]`` scale multiplies every block. Storage is
    deliberately neither a parameter nor a buffer: model ``to``/``_apply``
    transformations must never move, cast, pin, or expand checkpoint tables.
    """

    _requires_standard_hf_loading = True
    supports_cuda_graph = False

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        expected_shards: int | None = None,
        max_gather_rows: int = 4096,
    ) -> None:
        super().__init__()
        if num_embeddings <= 0 or embedding_dim <= 0 or embedding_dim % 16:
            raise ValueError("PLE NVFP4 needs positive rows and a row width divisible by 16")
        if max_gather_rows <= 0 or (expected_shards is not None and expected_shards <= 0):
            raise ValueError("PLE NVFP4 shard count and gather chunk size must be positive")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.expected_shards = expected_shards
        self.max_gather_rows = max_gather_rows
        self._shards: tuple[tuple[torch.Tensor, torch.Tensor], ...] = ()
        self._global_scale: float | None = None

    def bind_shards(self, leaves: dict[str, torch.Tensor | _SliceSource]) -> None:
        """Validate shard metadata and retain CPU views without copying tables.

        Block-scale values are validated only when selected, avoiding an eager
        scan or float conversion of billions of mmap-backed scales.
        """
        parts: dict[int, dict[str, torch.Tensor | _SliceSource]] = {}
        for name, source in leaves.items():
            if not name.startswith("ngram_embedding.shard_"):
                continue
            match = re.fullmatch(
                r"ngram_embedding\.shard_(0|[1-9][0-9]*)\.(weight|weight_scale)", name
            )
            if match is None:
                raise ValueError(f"Invalid PLE NVFP4 shard key: {name}")
            parts.setdefault(int(match[1]), {})[match[2]] = source
        indices = sorted(parts)
        if not indices or indices != list(range(len(indices))):
            raise ValueError("PLE NVFP4 shards must be consecutively numbered from zero")
        if self.expected_shards is not None and len(parts) != self.expected_shards:
            raise ValueError(f"PLE NVFP4 expected {self.expected_shards} shards, got {len(parts)}")
        if "ngram_embedding.weight_scale" in leaves:
            raise ValueError("PLE NVFP4 requires block scales and weight_scale_2, not FP8 scaling")
        scale_source = leaves.get("ngram_embedding.weight_scale_2")
        if scale_source is None:
            raise ValueError("PLE NVFP4 is missing global weight_scale_2")
        scale = _cpu_view(scale_source, "weight_scale_2")
        if scale.dtype != torch.float32 or tuple(scale.shape) != (1,):
            raise ValueError("PLE NVFP4 weight_scale_2 must be float32 [1]")
        global_scale = float(scale.item())
        if not math.isfinite(global_scale) or global_scale <= 0:
            raise ValueError("PLE NVFP4 weight_scale_2 must be finite and positive")

        shards = []
        total_rows = 0
        for index in indices:
            part = parts[index]
            if set(part) != {"weight", "weight_scale"}:
                raise ValueError(f"PLE NVFP4 shard {index} needs both weight and weight_scale")
            weight = _cpu_view(part["weight"], f"shard_{index}.weight")
            scales = _cpu_view(part["weight_scale"], f"shard_{index}.weight_scale")
            if weight.dtype != torch.uint8:
                raise TypeError("PLE NVFP4 packed weights must be uint8")
            if (
                weight.ndim != 2
                or weight.shape[0] <= 0
                or weight.shape[1] != self.embedding_dim // 2
            ):
                raise ValueError(f"Invalid PLE NVFP4 packed weight shape: {tuple(weight.shape)}")
            if scales.dtype != torch.float8_e4m3fn:
                raise TypeError("PLE NVFP4 block scales must be float8_e4m3fn")
            if tuple(scales.shape) != (weight.shape[0], self.embedding_dim // 16):
                raise ValueError(f"Invalid PLE NVFP4 block scale shape: {tuple(scales.shape)}")
            total_rows += weight.shape[0]
            shards.append((weight, scales))
        if total_rows != self.num_embeddings:
            raise ValueError(
                f"PLE NVFP4 shards tile {total_rows} rows, expected {self.num_embeddings}"
            )
        self._shards = tuple(shards)
        self._global_scale = global_scale

    @staticmethod
    def check_eager(device: torch.device, *, is_cuda_graph: bool = False) -> None:
        """Reject graph execution before starting the CPU round trip."""
        if is_cuda_graph or (device.type == "cuda" and torch.cuda.is_current_stream_capturing()):
            raise RuntimeError("PLE NVFP4 mmap lookup is eager-only; disable CUDA graphs")

    def allocate_output(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Allocate BF16 selected-row output compatible with PLE prefetch."""
        self.check_eager(device)
        return torch.empty(shape, dtype=torch.bfloat16, device=device)

    def gather(
        self,
        input_ids: torch.Tensor,
        out: torch.Tensor | None = None,
        *,
        weight_scale: float | None = None,
    ) -> torch.Tensor:
        """Gather int32/int64 global IDs to BF16 on the input device.

        CPU scratch is bounded by ``max_gather_rows * embedding_dim``; only
        IDs and the selected BF16 rows cross the CPU/GPU boundary.
        """
        self.check_eager(input_ids.device)
        if not self._shards or self._global_scale is None:
            raise RuntimeError("PLE NVFP4 checkpoint shards have not been bound")
        if weight_scale is not None:
            raise ValueError("PLE NVFP4 uses its checkpoint scales, not an external weight_scale")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("PLE NVFP4 row IDs must be int32 or int64")
        if input_ids.device.type not in ("cpu", "cuda"):
            raise ValueError("PLE NVFP4 row IDs must be on CPU or CUDA")
        expected_shape = (*input_ids.shape, self.embedding_dim)
        if out is not None:
            if tuple(out.shape) != expected_shape:
                raise ValueError(f"PLE NVFP4 output shape must be {expected_shape}")
            if (
                out.dtype != torch.bfloat16
                or out.device != input_ids.device
                or not out.is_contiguous()
            ):
                raise ValueError("PLE NVFP4 output must be contiguous BF16 on the input-ID device")
        ids = input_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        if ids.numel() and (ids.min().item() < 0 or ids.max().item() >= self.num_embeddings):
            raise IndexError(f"PLE NVFP4 row IDs must be in [0, {self.num_embeddings})")
        output = self.allocate_output(expected_shape, input_ids.device) if out is None else out
        flat_output = output.view(-1, self.embedding_dim)
        lut = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
            dtype=torch.float32,
            device="cpu",
        )
        for start in range(0, ids.numel(), self.max_gather_rows):
            chunk_ids = ids[start : start + self.max_gather_rows]
            rows = torch.empty(
                (chunk_ids.numel(), self.embedding_dim), dtype=torch.bfloat16, device="cpu"
            )
            offset = 0
            for weight, scales in self._shards:
                end = offset + weight.shape[0]
                positions = torch.where((chunk_ids >= offset) & (chunk_ids < end))[0]
                if positions.numel():
                    local_ids = chunk_ids[positions] - offset
                    packed = weight.index_select(0, local_ids)
                    # Index bytes because CPU float8 indexing is not supported
                    # by every torch version; reinterpret before conversion.
                    blocks = (
                        scales.view(torch.uint8)
                        .index_select(0, local_ids)
                        .view(torch.float8_e4m3fn)
                        .float()
                    )
                    if not torch.isfinite(blocks).all() or (blocks < 0).any():
                        raise ValueError(
                            "PLE NVFP4 selected block scales must be finite and non-negative"
                        )
                    nibbles = torch.stack((packed & 15, packed >> 4), dim=-1).long()
                    values = lut[nibbles].reshape(-1, self.embedding_dim // 16, 16)
                    values = (values * blocks.unsqueeze(-1) * self._global_scale).to(torch.bfloat16)
                    if not torch.isfinite(values).all():
                        raise ValueError("PLE NVFP4 selected rows overflow BF16")
                    rows[positions] = values.reshape(-1, self.embedding_dim)
                offset = end
            # Blocking copy keeps the bounded CPU staging buffer alive until
            # transfer completes, including when invoked on a prefetch stream.
            flat_output[start : start + chunk_ids.numel()].copy_(rows, non_blocking=False)
        return output
