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
"""Graph-friendly GPU PLE gather from independently owned read-only file mappings.

This optional delegate does not replace the eager CPU NVFP4 backend. Bind the
checkpoint, prepare a supported CUDA device, and warm up gather before capture.
Keep the delegate alive until all its GPU work and captured graphs are destroyed.
Never mutate or truncate its checkpoint files. Rebinding is deliberately forbidden.

No host registration, pinning, CPU weight tensors, or protection changes to loader
mappings occur. NumPy only exposes addresses of fresh MAP_PRIVATE/PROT_READ buffers.
"""

import ctypes
import json
import math
import mmap
import os
import re
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import triton
import triton.language as tl

__all__ = ["Qwen4ExpNVFP4MmapGPUBackend"]

_MAX_METADATA_BYTES = 16 * 1024 * 1024
# Expert-dense checkpoints have larger indexes than individual tensor headers.
_MAX_INDEX_BYTES = 64 * 1024 * 1024
_MAX_INT64 = (1 << 63) - 1


@triton.jit
def _gather_rows(
    ids,
    output,
    pointers,
    row_starts,
    NUM_ROWS: tl.constexpr,
    NUM_SHARDS: tl.constexpr,
    WIDTH: tl.constexpr,
    GLOBAL_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    position = tl.program_id(0)
    row = tl.load(ids + position).to(tl.int64)
    valid_id = (row >= 0) & (row < NUM_ROWS)
    safe_row = tl.where(valid_id, row, 0)
    low = tl.full((), 0, tl.int32)
    high = tl.full((), NUM_SHARDS, tl.int32)
    while low < high:
        middle = (low + high) // 2
        end = tl.load(row_starts + middle + 1)
        right = safe_row >= end
        low = tl.where(right, middle + 1, low)
        high = tl.where(right, high, middle)
    local_row = safe_row - tl.load(row_starts + low)
    weights = tl.load(pointers + low * 2).to(tl.pointer_type(tl.uint8))
    scales = tl.load(pointers + low * 2 + 1).to(tl.pointer_type(tl.uint8))
    columns = tl.arange(0, BLOCK)
    mask = valid_id & (columns < WIDTH)
    packed = tl.load(weights + local_row * (WIDTH // 2) + columns // 2, mask, other=0)
    code = (packed.to(tl.int32) >> (4 * (columns % 2))) & 15
    magnitude = code & 7
    absolute = tl.where(
        magnitude < 4,
        magnitude * 0.5,
        tl.where(magnitude < 6, magnitude - 2.0, magnitude * 2.0 - 8.0),
    )
    decoded = tl.where((code & 8) != 0, -absolute, absolute)
    scale = (
        tl.load(scales + local_row * (WIDTH // 16) + columns // 16, mask, other=0)
        .to(tl.float8e4nv, bitcast=True)
        .to(tl.float32)
    )
    values = decoded * scale * GLOBAL_SCALE
    # NaN comparisons are false. Check the rounded BF16 too, since finite FP32
    # near BF16's upper rounding boundary can become infinity on conversion.
    rounded = values.to(tl.bfloat16).to(tl.float32)
    valid_value = (scale >= 0) & (scale <= 448) & (tl.abs(rounded) <= 3.3895313892515355e38)
    valid_row = valid_id & (tl.sum(((columns < WIDTH) & ~valid_value).to(tl.int32), 0) == 0)
    tl.store(
        output + position * WIDTH + columns,
        tl.where(valid_row, rounded, float("nan")),
        columns < WIDTH,
    )


class _TensorMetadata(NamedTuple):
    dtype: str
    shape: tuple[int, ...]
    offset: int
    size: int


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate metadata key: {key}")
        result[key] = value
    return result


def _read_json(raw: bytes) -> dict:
    value = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError("Checkpoint metadata must be a JSON object")
    return value


class _MappedFile:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.tensors: dict[str, _TensorMetadata] = {}
        with path.open("rb") as handle:
            file_size = os.fstat(handle.fileno()).st_size
            size_bytes = handle.read(8)
            if len(size_bytes) != 8:
                raise ValueError(f"Truncated safetensors header: {path}")
            header_size = struct.unpack("<Q", size_bytes)[0]
            if not (2 <= header_size <= _MAX_METADATA_BYTES and header_size + 8 <= file_size):
                raise ValueError(f"Invalid or oversized safetensors header: {path}")
            raw = handle.read(header_size)
            if len(raw) != header_size or not raw.startswith(b"{"):
                raise ValueError(f"Invalid safetensors header: {path}")
            header = _read_json(raw)
            data_start = 8 + header_size
            intervals = []
            for name, entry in header.items():
                if name == "__metadata__":
                    continue
                if not isinstance(entry, dict):
                    raise ValueError(f"Invalid tensor metadata: {name}")
                shape = entry.get("shape")
                offsets = entry.get("data_offsets")
                dtype = entry.get("dtype")
                if (
                    not isinstance(dtype, str)
                    or not isinstance(shape, list)
                    or any(type(dimension) is not int or dimension < 0 for dimension in shape)
                    or not isinstance(offsets, list)
                    or len(offsets) != 2
                    or any(type(offset) is not int for offset in offsets)
                ):
                    raise ValueError(f"Invalid tensor shape/dtype/offsets: {name}")
                begin, end = offsets
                if not (0 <= begin <= end <= file_size - data_start and end <= _MAX_INT64):
                    raise ValueError(f"Out-of-file tensor offsets: {name}")
                if end > begin:
                    intervals.append((begin, end))
                self.tensors[name] = _TensorMetadata(
                    dtype, tuple(shape), data_start + begin, end - begin
                )
            # Do not allow a selected tensor to alias another header entry.
            cursor = 0
            for begin, end in sorted(intervals):
                if begin != cursor:
                    raise ValueError(f"Overlapping or noncontiguous safetensors data: {path}")
                cursor = end
            if cursor != file_size - data_start:
                raise ValueError(f"Unindexed safetensors payload: {path}")
            self.mapping = mmap.mmap(
                handle.fileno(), 0, flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ
            )
        self.bytes_view = np.frombuffer(self.mapping, dtype=np.uint8)
        if self.bytes_view.flags.writeable:
            raise AssertionError("Expected an independently owned read-only mapping")

    def address(self, key: str, dtype: str, shape: tuple[int, ...], item_size: int) -> int:
        metadata = self.tensors[key]
        if metadata.dtype != dtype or metadata.shape != shape:
            raise ValueError(
                f"{key} requires {dtype} {shape}, got {metadata.dtype} {metadata.shape}"
            )
        if metadata.size != math.prod(shape) * item_size:
            raise ValueError(f"Tensor byte length does not match its shape: {key}")
        return int(self.bytes_view.ctypes.data) + metadata.offset


class _DeviceTables(NamedTuple):
    pointers: torch.Tensor
    row_starts: torch.Tensor


class Qwen4ExpNVFP4MmapGPUBackend:
    """Optional GPU delegate for Qwen4ExpNVFP4MmapEmbedding's CPU fallback.

    Args:
        num_embeddings: Total row count across consecutively numbered shards.
        embedding_dim: Decoded row width, a positive multiple of 16.
        expected_shards: Optional checkpoint shard count check; unequal sizes work.

    ``prefix`` in bind methods is the exact checkpoint namespace immediately
    before ``.shard_N.weight`` and ``.weight_scale_2``. The global scale is F32
    ``[1]`` and may reside in a different file from every weight and block scale.

    This object is not an nn.Module or registered buffer. Keep it alive with its
    owning embedding through graph teardown, including its per-device pointer
    tables. Checkpoint files must remain immutable. There is no unsafe unbind or
    close method. Bind/prepare are eager-only; gather never moves source tables.
    """

    supports_cuda_graph = True

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        expected_shards: int | None = None,
    ) -> None:
        if not (0 < num_embeddings <= _MAX_INT64 and embedding_dim > 0 and embedding_dim % 16 == 0):
            raise ValueError("NVFP4 requires positive int64 row count and width divisible by 16")
        if expected_shards is not None and expected_shards <= 0:
            raise ValueError("expected_shards must be positive")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.expected_shards = expected_shards
        self._files: dict[Path, _MappedFile] = {}
        self._addresses: tuple[tuple[int, int], ...] = ()
        self._row_starts: tuple[int, ...] = ()
        self._global_scale: float | None = None
        self._device_tables: dict[torch.device, _DeviceTables] = {}

    @staticmethod
    def _require_eager() -> None:
        if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("NVFP4 bind/prepare/validation must run outside CUDA graph capture")

    @staticmethod
    def _device(device: torch.device) -> torch.device:
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("GPU delegate requires CUDA; retain the existing eager CPU fallback")
        return torch.device(
            "cuda", torch.cuda.current_device() if device.index is None else device.index
        )

    @staticmethod
    def device_capabilities(device: torch.device) -> dict[str, int]:
        """Query eagerly; unsupported devices should keep using the CPU backend."""
        Qwen4ExpNVFP4MmapGPUBackend._require_eager()
        device = Qwen4ExpNVFP4MmapGPUBackend._device(device)
        with torch.cuda.device(device):
            torch.cuda.init()
        driver = ctypes.CDLL("libcuda.so.1")
        driver.cuDeviceGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]
        driver.cuDeviceGetAttribute.restype = ctypes.c_int
        attributes = {}
        # Stable CUDA Driver API attribute numbers from cuda.h.
        for name, attribute in (
            ("pageable_memory_access", 88),
            ("concurrent_managed_access", 89),
            ("pageable_memory_access_uses_host_page_tables", 100),
        ):
            value = ctypes.c_int()
            status = driver.cuDeviceGetAttribute(ctypes.byref(value), attribute, device.index)
            if status:
                raise RuntimeError(f"cuDeviceGetAttribute({attribute}) failed: {status}")
            attributes[name] = value.value
        return attributes

    def bind_checkpoint(self, checkpoint_dir: str | Path, prefix: str = "ngram_embedding") -> None:
        """Bind selected tensors using the HF index, or scan headers if unindexed.

        Only small JSON metadata and the four-byte global scale are CPU-read.
        Standard HF cache symlinks are supported. Explicit path traversal in an
        index is rejected; use bind_files for an intentional external file map.
        """
        self._require_eager()
        directory = Path(checkpoint_dir)
        index_path = directory / "model.safetensors.index.json"
        if index_path.exists():
            with index_path.open("rb") as handle:
                raw = handle.read(_MAX_INDEX_BYTES + 1)
            if len(raw) > _MAX_INDEX_BYTES:
                raise ValueError("Oversized checkpoint index")
            index = _read_json(raw).get("weight_map")
            if not isinstance(index, dict):
                raise ValueError("Checkpoint index requires a weight_map object")
            tensor_files = {}
            for name, filename in index.items():
                if not name.startswith(prefix.rstrip(".") + "."):
                    continue
                if not isinstance(filename, str) or not filename:
                    raise ValueError(f"Invalid checkpoint filename for {name}")
                relative = Path(filename)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Checkpoint index path traversal: {filename}")
                tensor_files[name] = directory / relative
            self.bind_files(tensor_files, prefix)
        else:
            files = {
                path.resolve(): _MappedFile(path.resolve())
                for path in sorted(directory.glob("*.safetensors"))
            }
            tensor_files = {}
            for path, mapped in files.items():
                for name in mapped.tensors:
                    if name.startswith(prefix.rstrip(".") + "."):
                        if name in tensor_files:
                            raise ValueError(f"Ambiguous unindexed tensor: {name}")
                        tensor_files[name] = path
            self._bind(tensor_files, prefix, files)

    def bind_files(
        self, tensor_files: Mapping[str, str | Path], prefix: str = "ngram_embedding"
    ) -> None:
        """Bind an explicit full tensor-key to safetensors-file mapping eagerly."""
        self._require_eager()
        self._bind(tensor_files, prefix, {})

    def _bind(
        self,
        tensor_files: Mapping[str, str | Path],
        prefix: str,
        files: dict[Path, _MappedFile],
    ) -> None:
        if self._addresses:
            raise RuntimeError("Cannot rebind mmap addresses; existing graphs may retain them")
        prefix = prefix.rstrip(".")
        if not prefix:
            raise ValueError("Checkpoint tensor prefix cannot be empty")
        selected = {
            name: Path(path).resolve()
            for name, path in tensor_files.items()
            if name.startswith(prefix + ".")
        }
        global_key = f"{prefix}.weight_scale_2"
        shards: dict[int, dict[str, str]] = {}
        for name in selected:
            if name == global_key:
                continue
            match = re.fullmatch(
                re.escape(prefix) + r"\.shard_(0|[1-9][0-9]*)\.(weight|weight_scale)", name
            )
            if match is None:
                raise ValueError(f"Unexpected NVFP4 checkpoint key: {name}")
            shards.setdefault(int(match[1]), {})[match[2]] = name
        if not shards or sorted(shards) != list(range(len(shards))):
            raise ValueError("NVFP4 shards must be consecutively numbered from zero")
        if self.expected_shards is not None and len(shards) != self.expected_shards:
            raise ValueError(f"Expected {self.expected_shards} shards, got {len(shards)}")
        if global_key not in selected:
            raise ValueError("Missing NVFP4 weight_scale_2")
        # Deduplicate physical file mappings and keep only files used by this PLE.
        files = {
            path: files[path] if path in files else _MappedFile(path)
            for path in set(selected.values())
        }
        for name, path in selected.items():
            if name not in files[path].tensors:
                raise ValueError(f"Index tensor {name} is missing from {path}")
        global_file = files[selected[global_key]]
        global_file.address(global_key, "F32", (1,), 4)
        global_scale = struct.unpack_from(
            "<f", global_file.mapping, global_file.tensors[global_key].offset
        )[0]
        if not math.isfinite(global_scale) or global_scale <= 0:
            raise ValueError("NVFP4 weight_scale_2 must be finite and positive")
        addresses = []
        starts = [0]
        for shard in range(len(shards)):
            keys = shards[shard]
            if set(keys) != {"weight", "weight_scale"}:
                raise ValueError(f"Shard {shard} needs both weight and weight_scale")
            weight_file = files[selected[keys["weight"]]]
            scale_file = files[selected[keys["weight_scale"]]]
            shape = weight_file.tensors[keys["weight"]].shape
            if len(shape) != 2 or shape[0] <= 0:
                raise ValueError(f"Invalid packed NVFP4 shape: {shape}")
            rows = shape[0]
            weight_address = weight_file.address(
                keys["weight"], "U8", (rows, self.embedding_dim // 2), 1
            )
            scale_address = scale_file.address(
                keys["weight_scale"], "F8_E4M3", (rows, self.embedding_dim // 16), 1
            )
            addresses.append((weight_address, scale_address))
            starts.append(starts[-1] + rows)
        if starts[-1] != self.num_embeddings:
            raise ValueError(
                f"NVFP4 shards contain {starts[-1]} rows, expected {self.num_embeddings}"
            )
        self._files = files
        self._addresses = tuple(addresses)
        self._row_starts = tuple(starts)
        self._global_scale = global_scale

    def prepare(self, device: torch.device) -> None:
        """Upload tiny pointer/offset tables once, synchronizing only during setup.

        Call before warmup and graph capture. This also makes tables ready for
        gather on another stream without per-call events or synchronization.
        """
        self._require_eager()
        if not self._addresses:
            raise RuntimeError("Bind a checkpoint before preparing the GPU delegate")
        device = self._device(device)
        if device in self._device_tables:
            return
        capabilities = self.device_capabilities(device)
        if not all(capabilities.values()):
            raise RuntimeError(
                f"Unsupported pageable host memory; retain CPU fallback: {capabilities}"
            )
        tables = _DeviceTables(
            torch.tensor(self._addresses, dtype=torch.int64, device=device),
            torch.tensor(self._row_starts, dtype=torch.int64, device=device),
        )
        torch.cuda.synchronize(device)
        self._device_tables[device] = tables

    def allocate_output(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Allocate contiguous BF16 GPU output for ``[*ids.shape, embedding_dim]``."""
        device = self._device(device)
        if not shape or shape[-1] != self.embedding_dim:
            raise ValueError("NVFP4 output shape must end with embedding_dim")
        return torch.empty(shape, dtype=torch.bfloat16, device=device)

    def gather(
        self,
        input_ids: torch.Tensor,
        out: torch.Tensor | None = None,
        *,
        weight_scale: float | None = None,
    ) -> torch.Tensor:
        """Gather contiguous int32/int64 GPU IDs to BF16 with one kernel launch.

        ``input_ids`` has any shape; ``out`` must be contiguous GPU BF16 with
        shape ``[*input_ids.shape, embedding_dim]``, on the same prepared device.
        Invalid IDs are masked before source loads. Invalid IDs, selected negative
        or NaN E4M3 scales, or BF16 overflow yield an ALL-NaN row, never silent zero.
        Values are not host-checked: use validate_output during eager validation.
        This differs deliberately from the CPU fallback's synchronous exceptions.
        """
        if weight_scale is not None:
            raise ValueError("NVFP4 uses its checkpoint scales, not an external weight_scale")
        if input_ids.dtype not in (torch.int32, torch.int64) or not input_ids.is_contiguous():
            raise ValueError("NVFP4 GPU IDs must be contiguous int32 or int64")
        device = self._device(input_ids.device)
        if device not in self._device_tables:
            raise RuntimeError("Call prepare(device) and warm up gather before CUDA graph capture")
        expected_shape = (*input_ids.shape, self.embedding_dim)
        if out is None:
            out = self.allocate_output(expected_shape, device)
        if (
            out.shape != expected_shape
            or out.dtype != torch.bfloat16
            or out.device != device
            or not out.is_contiguous()
        ):
            raise ValueError("NVFP4 output must have the expected BF16 shape, device, and layout")
        if not input_ids.numel():
            return out
        ids_begin = input_ids.data_ptr()
        out_begin = out.data_ptr()
        if max(ids_begin, out_begin) < min(
            ids_begin + input_ids.numel() * input_ids.element_size(),
            out_begin + out.numel() * out.element_size(),
        ):
            raise ValueError("NVFP4 output must not alias input ID storage")
        tables = self._device_tables[device]
        with torch.cuda.device(device):
            _gather_rows[(input_ids.numel(),)](
                input_ids,
                out,
                tables.pointers,
                tables.row_starts,
                NUM_ROWS=self.num_embeddings,
                NUM_SHARDS=len(self._addresses),
                WIDTH=self.embedding_dim,
                GLOBAL_SCALE=self._global_scale,
                BLOCK=triton.next_power_of_2(self.embedding_dim),
                num_warps=4,
                enable_fp_fusion=False,
            )
        return out

    @staticmethod
    def validate_output(output: torch.Tensor) -> None:
        """Synchronously reject nonfinite output during explicit eager validation.

        Never call inside capture. A successful initial sample is not a guarantee
        that later graph inputs/scales are valid; graph failures remain NaN rows.
        """
        Qwen4ExpNVFP4MmapGPUBackend._require_eager()
        if not bool(torch.isfinite(output).all().item()):
            raise ValueError("NVFP4 gather found an invalid ID, block scale, or BF16 overflow")
