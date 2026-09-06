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
"""CPU-only NVFP4 storage tests, independent of TensorRT native bindings."""

import gc
import importlib.util
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

# The storage has no TensorRT dependency. Load it directly so these tests can
# also run with CPU torch while a native runtime build is unavailable.
_ROOT = Path(__file__).resolve().parents[4]
_SPEC = importlib.util.spec_from_file_location(
    "ple_nvfp4", _ROOT / "tensorrt_llm/_torch/modules/qwen4_exp/ple_nvfp4.py"
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
Qwen4ExpNVFP4MmapEmbedding = _MODULE.Qwen4ExpNVFP4MmapEmbedding


def _leaves(width: int = 32) -> dict[str, torch.Tensor]:
    # Low nibble first: every signed E2M1 value occurs in each 16-value block.
    packed = torch.tensor([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE], dtype=torch.uint8)
    weights = packed.repeat(5, width // 16)
    scales = torch.tensor([0.5, 1, 1.5, 2, 3]).unsqueeze(1)
    scales = (scales * torch.arange(1, width // 16 + 1)).to(torch.float8_e4m3fn)
    # Insert out of shard order; each shard contributes weight AND scales.
    return {
        "ngram_embedding.shard_1.weight": weights[2:],
        "ngram_embedding.shard_0.weight": weights[:2],
        "ngram_embedding.shard_1.weight_scale": scales[2:],
        "ngram_embedding.shard_0.weight_scale": scales[:2],
        "ngram_embedding.weight_scale_2": torch.tensor([0.25], dtype=torch.float32),
    }


def _expected(ids: torch.Tensor, width: int = 32) -> torch.Tensor:
    decoded = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
    row_scales = torch.tensor([0.5, 1, 1.5, 2, 3])[ids.long()]
    block_scales = row_scales.unsqueeze(-1) * torch.arange(1, width // 16 + 1)
    stored_scales = block_scales.to(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
    return (decoded.repeat(width // 16) * stored_scales * 0.25).bfloat16()


@pytest.mark.parametrize("width", [16, 32, 160])
@pytest.mark.parametrize("chunk_rows", [1, 2, 4096])
def test_packed_values_scales_shard_boundaries_and_output_identity(
    width: int, chunk_rows: int
) -> None:
    table = Qwen4ExpNVFP4MmapEmbedding(5, width, expected_shards=2, max_gather_rows=chunk_rows)
    table.bind_shards(_leaves(width))
    ids = torch.tensor([[4, 0, 2], [2, 1, 3]], dtype=torch.int32)
    out = table.allocate_output((*ids.shape, width), ids.device)
    assert table.gather(ids, out=out) is out
    torch.testing.assert_close(out, _expected(ids, width), rtol=0, atol=0)


def test_safetensors_views_survive_closed_handles_and_weights_clear(tmp_path: Path) -> None:
    path = tmp_path / "tiny.safetensors"
    save_file({name: value.contiguous() for name, value in _leaves().items()}, path)
    table = Qwen4ExpNVFP4MmapEmbedding(5, 32)
    with safe_open(path, framework="pt", device="cpu") as handle:
        weights = {name: handle.get_slice(name) for name in handle.keys()}
        table.bind_shards(weights)
        # Pointer equality proves binding did not copy the packed table or scales.
        for index, (weight, scales) in enumerate(table._shards):
            assert (
                weight.data_ptr()
                == handle.get_tensor(f"ngram_embedding.shard_{index}.weight").data_ptr()
            )
            assert (
                scales.data_ptr()
                == handle.get_tensor(f"ngram_embedding.shard_{index}.weight_scale").data_ptr()
            )
        weights.clear()
    del weights, handle, weight, scales
    gc.collect()
    ids = torch.tensor([4, 2, 0])
    torch.testing.assert_close(table.gather(ids), _expected(ids), rtol=0, atol=0)


def test_storage_is_not_transformed_or_materialized() -> None:
    table = Qwen4ExpNVFP4MmapEmbedding(320001536, 160)
    assert list(table.parameters()) == [] and list(table.buffers()) == []
    assert table._shards == ()
    table = Qwen4ExpNVFP4MmapEmbedding(5, 32)
    leaves = _leaves()
    table.bind_shards(leaves)
    pointers = [(w.data_ptr(), s.data_ptr()) for w, s in table._shards]
    parent = torch.nn.ModuleList([table])
    parent.to(device="meta", dtype=torch.float64)
    parent.to_empty(device="cpu")
    assert pointers == [(w.data_ptr(), s.data_ptr()) for w, s in table._shards]
    assert all(w.device.type == s.device.type == "cpu" for w, s in table._shards)
    assert parent.state_dict() == {}
    torch.testing.assert_close(table.gather(torch.tensor([0])), _expected(torch.tensor([0])))


@pytest.mark.parametrize("ids", [torch.tensor(-1), torch.tensor([5]), torch.tensor([2**40])])
def test_invalid_ids(ids: torch.Tensor) -> None:
    table = Qwen4ExpNVFP4MmapEmbedding(5, 32)
    table.bind_shards(_leaves())
    with pytest.raises(IndexError, match="row IDs"):
        table.gather(ids)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bool, torch.uint8, torch.int16])
def test_invalid_index_dtype(dtype: torch.dtype) -> None:
    table = Qwen4ExpNVFP4MmapEmbedding(5, 32)
    table.bind_shards(_leaves())
    with pytest.raises(TypeError, match="row IDs"):
        table.gather(torch.tensor([0], dtype=dtype))


def test_empty_scalar_and_noncontiguous_indices() -> None:
    table = Qwen4ExpNVFP4MmapEmbedding(5, 32)
    table.bind_shards(_leaves())
    assert table.gather(torch.empty((0, 16), dtype=torch.int64)).shape == (0, 16, 32)
    for ids in (torch.tensor(4), torch.tensor([[0, 2], [1, 4]]).T):
        torch.testing.assert_close(table.gather(ids), _expected(ids), rtol=0, atol=0)


@pytest.mark.parametrize(
    "out",
    [
        torch.empty((1, 16), dtype=torch.bfloat16),
        torch.empty((1, 32), dtype=torch.float32),
        torch.empty((32, 2), dtype=torch.bfloat16).T,
        torch.empty((1, 32), device="meta", dtype=torch.bfloat16),
    ],
)
def test_invalid_outputs(out: torch.Tensor) -> None:
    table = Qwen4ExpNVFP4MmapEmbedding(5, 32)
    table.bind_shards(_leaves())
    with pytest.raises(ValueError, match="output"):
        table.gather(torch.zeros(out.shape[0], dtype=torch.long), out=out)


@pytest.mark.parametrize(
    "key,value,error",
    [
        ("ngram_embedding.shard_0.weight", torch.zeros((2, 16)), TypeError),
        ("ngram_embedding.shard_0.weight", torch.zeros((2, 8), dtype=torch.uint8), ValueError),
        ("ngram_embedding.shard_0.weight", torch.zeros((0, 16), dtype=torch.uint8), ValueError),
        (
            "ngram_embedding.shard_0.weight",
            torch.zeros((2, 16), dtype=torch.uint8, device="meta"),
            ValueError,
        ),
        ("ngram_embedding.shard_0.weight", torch.zeros((16, 2), dtype=torch.uint8).T, ValueError),
        ("ngram_embedding.shard_0.weight_scale", torch.ones((2, 2)), TypeError),
        (
            "ngram_embedding.shard_0.weight_scale",
            torch.ones((1, 2), dtype=torch.float8_e4m3fn),
            ValueError,
        ),
        ("ngram_embedding.weight_scale_2", torch.tensor([0.0]), ValueError),
        ("ngram_embedding.weight_scale_2", torch.tensor([-1.0]), ValueError),
        ("ngram_embedding.weight_scale_2", torch.tensor([float("nan")]), ValueError),
        ("ngram_embedding.weight_scale_2", torch.tensor([float("inf")]), ValueError),
        ("ngram_embedding.weight_scale_2", torch.tensor(1.0), ValueError),
        ("ngram_embedding.weight_scale_2", torch.ones(2), ValueError),
        ("ngram_embedding.weight_scale_2", torch.ones(1, dtype=torch.bfloat16), ValueError),
        ("ngram_embedding.weight_scale", torch.ones(1), ValueError),
    ],
)
def test_invalid_checkpoint_tensors(key: str, value: torch.Tensor, error: type[Exception]) -> None:
    leaves = _leaves()
    leaves[key] = value
    with pytest.raises(error):
        Qwen4ExpNVFP4MmapEmbedding(5, 32).bind_shards(leaves)


@pytest.mark.parametrize("missing", list(_leaves()))
def test_missing_shards_or_scales(missing: str) -> None:
    leaves = _leaves()
    del leaves[missing]
    with pytest.raises(ValueError):
        Qwen4ExpNVFP4MmapEmbedding(5, 32).bind_shards(leaves)


@pytest.mark.parametrize("replacement", ["shard_2.weight", "shard_01.weight", "shard_0.other"])
def test_invalid_shard_numbering_or_names(replacement: str) -> None:
    leaves = _leaves()
    leaves[f"ngram_embedding.{replacement}"] = leaves.pop("ngram_embedding.shard_0.weight")
    with pytest.raises(ValueError):
        Qwen4ExpNVFP4MmapEmbedding(5, 32).bind_shards(leaves)


@pytest.mark.parametrize("rows,shards", [(6, 2), (5, 128)])
def test_row_and_shard_counts(rows: int, shards: int) -> None:
    with pytest.raises(ValueError):
        Qwen4ExpNVFP4MmapEmbedding(rows, 32, expected_shards=shards).bind_shards(_leaves())


@pytest.mark.parametrize("bad_scale", [-1, float("nan")])
def test_scales_validated_on_selected_rows_only(bad_scale: float) -> None:
    leaves = _leaves()
    scales = leaves["ngram_embedding.shard_1.weight_scale"].float()
    scales[2, 0] = bad_scale
    leaves["ngram_embedding.shard_1.weight_scale"] = scales.to(torch.float8_e4m3fn)
    table = Qwen4ExpNVFP4MmapEmbedding(5, 32)
    table.bind_shards(leaves)
    torch.testing.assert_close(table.gather(torch.tensor([0])), _expected(torch.tensor([0])))
    with pytest.raises(ValueError, match="selected block scales"):
        table.gather(torch.tensor([4]))


def test_zero_scale_blocks_and_bf16_overflow() -> None:
    leaves = _leaves()
    leaves["ngram_embedding.shard_0.weight_scale"] = torch.zeros((2, 2), dtype=torch.float8_e4m3fn)
    table = Qwen4ExpNVFP4MmapEmbedding(5, 32)
    table.bind_shards(leaves)
    assert torch.count_nonzero(table.gather(torch.tensor([0]))) == 0
    leaves["ngram_embedding.weight_scale_2"] = torch.tensor([torch.finfo(torch.float32).max])
    table.bind_shards(leaves)
    with pytest.raises(ValueError, match="overflow"):
        table.gather(torch.tensor([4]))


def test_unbound_and_external_scaling_fail() -> None:
    table = Qwen4ExpNVFP4MmapEmbedding(5, 32)
    with pytest.raises(RuntimeError, match="not been bound"):
        table.gather(torch.tensor([0]))
    table.bind_shards(_leaves())
    with pytest.raises(ValueError, match="external"):
        table.gather(torch.tensor([0]), weight_scale=1.0)


def test_graph_guard_precedes_cpu_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    table = Qwen4ExpNVFP4MmapEmbedding(5, 32)
    with pytest.raises(RuntimeError, match="eager-only"):
        table.check_execution(torch.device("cuda"))
    with pytest.raises(RuntimeError, match="eager-only"):
        table.check_execution(torch.device("cpu"), is_cuda_graph=True)


def test_cpu_only_lookup_does_not_open_gpu_mappings(monkeypatch: pytest.MonkeyPatch) -> None:
    table = Qwen4ExpNVFP4MmapEmbedding(5, 32)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="Bind PLE"):
        table.enable_gpu_lookup("/unused", "embedding")
    table.bind_shards(_leaves())
    assert not table.enable_gpu_lookup("/unused", "embedding")
    assert not table.supports_cuda_graph
    assert table.gather(torch.tensor([0])).shape == (1, 32)


@pytest.mark.parametrize("stride", [1, 2])
def test_output_cannot_overwrite_ids_for_later_chunks(stride: int) -> None:
    table = Qwen4ExpNVFP4MmapEmbedding(5, 32, max_gather_rows=1)
    table.bind_shards(_leaves())
    storage = torch.zeros(64, dtype=torch.bfloat16)
    ids = storage.view(torch.int64)[::stride][:2]
    ids.copy_(torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="overlap input ID"):
        table.gather(ids, out=storage.view(2, 32))


def test_disjoint_views_of_shared_storage_are_allowed() -> None:
    table = Qwen4ExpNVFP4MmapEmbedding(5, 32, max_gather_rows=1)
    table.bind_shards(_leaves())
    storage = torch.zeros(72, dtype=torch.bfloat16)
    ids = storage[64:].view(torch.int64)
    ids.copy_(torch.tensor([0, 3]))
    expected = table.gather(ids.clone())
    output = storage[:64].view(2, 32)
    assert table.gather(ids, out=output) is output
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_selected_row_scratch_is_chunk_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    table = Qwen4ExpNVFP4MmapEmbedding(5, 160, max_gather_rows=7)
    table.bind_shards(_leaves(160))
    original_index_select = torch.Tensor.index_select
    selected_counts = []

    def checked_index_select(tensor: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
        selected_counts.append(index.numel())
        assert index.numel() <= 7
        return original_index_select(tensor, dim, index)

    monkeypatch.setattr(torch.Tensor, "index_select", checked_index_select)
    ids = torch.arange(99).remainder(5)
    torch.testing.assert_close(table.gather(ids), _expected(ids, 160), rtol=0, atol=0)
    assert len(selected_counts) > 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_gather_and_capture_rejection() -> None:
    table = Qwen4ExpNVFP4MmapEmbedding(5, 32, max_gather_rows=2)
    table.bind_shards(_leaves())
    ids = torch.tensor([[4, 2, 0]], device="cuda")
    out = table.allocate_output((*ids.shape, 32), ids.device)
    assert table.gather(ids, out=out) is out
    torch.testing.assert_close(out.cpu(), _expected(ids.cpu()), rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="eager-only"):
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            table.gather(ids, out=out)
