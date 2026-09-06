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
"""Standalone CPU/GPU checks without importing TRT or repository pytest plugins.

Run this file directly. GPU checks require PLE_NVFP4_GPU_TESTS=1. Set
PLE_NVFP4_RESULTS_JSON to store raw benchmark samples, and PLE_NVFP4_TEMP_DIR
to an owned disk-backed test directory (not tmpfs). No external checkpoint data,
global cache drops, pinning, or registration are used. Expected budget is below
256 MiB torch CUDA reservation and 1 GiB process RSS, including framework overhead.
"""

import ctypes
import gc
import hashlib
import importlib.util
import json
import mmap
import os
import resource
import statistics
import struct
import sys
import tempfile
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType
from unittest import mock

import torch
import triton
from safetensors import safe_open
from safetensors.torch import save_file

_ROOT = Path(__file__).resolve().parents[4]
_MODULE_PATH = Path(
    os.environ.get(
        "PLE_NVFP4_GPU_MODULE", _ROOT / "tensorrt_llm/_torch/modules/qwen4_exp/ple_nvfp4_gpu.py"
    )
)
_SPEC = importlib.util.spec_from_file_location("ple_nvfp4_gpu_under_test", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
_Backend = _MODULE.Qwen4ExpNVFP4MmapGPUBackend
_PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
_MIB = 1024 * 1024


def _wrapper_module() -> ModuleType:
    # Never stub or import tensorrt_llm: this private package exists only to
    # resolve the wrapper's relative helper import in the source-only test.
    package_name = "_ple_nvfp4_wrapper_integration_test"
    package = ModuleType(package_name)
    package.__path__ = []
    sys.modules[package_name] = package
    sys.modules[f"{package_name}.ple_nvfp4_gpu"] = _MODULE
    wrapper_path = Path(
        os.environ.get("PLE_NVFP4_WRAPPER_MODULE", _MODULE_PATH.with_name("ple_nvfp4.py"))
    )
    spec = importlib.util.spec_from_file_location(f"{package_name}.ple_nvfp4", wrapper_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Checkpoint:
    def __init__(
        self, root: Path, shards: int = 128, width: int = 160, global_scale: float = 1.3
    ) -> None:
        self.root = root
        self.width = width
        self.shards = shards
        self.global_scale = global_scale
        self.row_counts = [32 + index % 5 for index in range(shards)]
        self.starts = [0]
        self.leaves: dict[str, torch.Tensor] = {}
        self.tensor_files: dict[str, Path] = {}
        self.files: list[Path] = []
        self._groups: list[dict[str, torch.Tensor]] = [{}, {}, {}]
        generator = torch.Generator().manual_seed(121)
        for shard, rows in enumerate(self.row_counts):
            self.starts.append(self.starts[-1] + rows)
            packed = torch.randint(256, (rows, width // 2), generator=generator, dtype=torch.uint8)
            scales = torch.randint(
                127, (rows, width // 16), generator=generator, dtype=torch.uint8
            ).view(torch.float8_e4m3fn)
            for suffix, tensor, group in (
                ("weight", packed, shard % 3),
                ("weight_scale", scales, (shard + 1) % 3),
            ):
                key = f"{_PREFIX}.shard_{shard}.{suffix}"
                self.leaves[key] = tensor
                self._groups[group][key] = tensor
        self._groups[0][f"{_PREFIX}.weight_scale_2"] = torch.tensor(
            [global_scale], dtype=torch.float32
        )
        for group in self._groups:
            group["non_ple.weight"] = torch.arange(8, dtype=torch.float32)
        self.save()

    def save(self) -> None:
        self.files = []
        self.tensor_files = {}
        weight_map = {}
        for index, group in enumerate(self._groups, start=1):
            path = self.root / f"model-{index:05d}-of-00003.safetensors"
            save_file(group, path)
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
            self.files.append(path)
            for key in group:
                if key.startswith(_PREFIX):
                    self.tensor_files[key] = path
                    weight_map[key] = path.name
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )

    def backend(self) -> _Backend:
        backend = _Backend(self.starts[-1], self.width, expected_shards=self.shards)
        backend.bind_checkpoint(self.root, _PREFIX)
        return backend

    def reference(self, ids: torch.Tensor) -> torch.Tensor:
        flat_ids = ids.to(dtype=torch.int64, device="cpu").reshape(-1)
        output = torch.full((flat_ids.numel(), self.width), float("nan"), dtype=torch.bfloat16)
        lookup = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
        for shard, (begin, end) in enumerate(zip(self.starts, self.starts[1:])):
            positions = torch.where((flat_ids >= begin) & (flat_ids < end))[0]
            if not positions.numel():
                continue
            local_ids = flat_ids[positions] - begin
            packed = self.leaves[f"{_PREFIX}.shard_{shard}.weight"][local_ids].long()
            scales = (
                self.leaves[f"{_PREFIX}.shard_{shard}.weight_scale"]
                .view(torch.uint8)[local_ids]
                .view(torch.float8_e4m3fn)
                .float()
            )
            codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(1)
            values = (
                lookup[codes] * scales.repeat_interleave(16, dim=1) * self.global_scale
            ).bfloat16()
            valid = ((scales >= 0) & torch.isfinite(scales)).all(dim=1) & torch.isfinite(
                values
            ).all(dim=1)
            values[~valid] = float("nan")
            output[positions] = values
        return output.reshape(*ids.shape, self.width)


def _smaps(backend: _Backend) -> list[dict[str, str]]:
    addresses = {int(mapped.bytes_view.ctypes.data) for mapped in backend._files.values()}
    result = []
    current = None
    for line in Path("/proc/self/smaps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if fields and "-" in fields[0] and ":" not in fields[0]:
            begin, end = (int(value, 16) for value in fields[0].split("-"))
            current = None
            if any(begin <= address < end for address in addresses):
                current = {"permissions": fields[1]}
                result.append(current)
        elif current is not None and ":" in line:
            key, value = line.split(":", 1)
            if key in {"Rss", "Anonymous", "Private_Dirty", "Shared_Dirty", "Locked"}:
                current[key] = value.strip()
    return result


def _residency(backend: _Backend) -> list[dict[str, int]]:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    libc.mincore.restype = ctypes.c_int
    result = []
    for mapped in backend._files.values():
        pages = (len(mapped.mapping) + mmap.PAGESIZE - 1) // mmap.PAGESIZE
        vector = (ctypes.c_ubyte * pages)()
        if libc.mincore(int(mapped.bytes_view.ctypes.data), len(mapped.mapping), vector):
            raise OSError(ctypes.get_errno(), "mincore failed on own test fixture")
        result.append({"pages": pages, "resident": sum(value & 1 for value in vector)})
    return result


def _reclaim(backend: _Backend, checkpoint: _Checkpoint) -> list[dict[str, int]]:
    # Only an owned temporary fixture is accepted, never a production checkpoint.
    if not checkpoint.root.name.startswith("own-ple-gpu-"):
        raise ValueError("Refuse reclaim outside an owned test directory")
    for mapped in backend._files.values():
        if mapped.path.parent != checkpoint.root.resolve():
            raise ValueError("Refuse reclaim of an external mapping")
        mapped.mapping.madvise(mmap.MADV_DONTNEED)
        with mapped.path.open("rb") as handle:
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return _residency(backend)


def _temporary() -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory(
        prefix="own-ple-gpu-", dir=os.environ.get("PLE_NVFP4_TEMP_DIR")
    )


def _discard_cpu_fixture_views(wrapper: torch.nn.Module, checkpoint: _Checkpoint) -> None:
    # CPU fallback views are independent aliases of these same tiny files. Drop
    # their PTEs too, or fadvise cannot evict pages still mapped by the CPU path.
    if not checkpoint.root.name.startswith("own-ple-gpu-"):
        raise ValueError("Refuse reclaim outside an owned test directory")
    addresses = {tensor.data_ptr() for shard in wrapper._shards for tensor in shard}
    paths = {str(path.resolve()) for path in checkpoint.files}
    libc = ctypes.CDLL(None, use_errno=True)
    libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.madvise.restype = ctypes.c_int
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or fields[5] not in paths:
            continue
        begin, end = (int(value, 16) for value in fields[0].split("-"))
        if any(begin <= address < end for address in addresses):
            if libc.madvise(begin, end - begin, mmap.MADV_DONTNEED):
                raise OSError(ctypes.get_errno(), "madvise failed on own CPU fixture views")


def _rewrite_header(path: Path, mutate) -> None:
    raw = path.read_bytes()
    size = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + size])
    mutate(header)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw[8 + size :])


class MetadataTests(unittest.TestCase):
    def test_index_over_16_mib_retains_selected_ple(self) -> None:
        self.assertEqual(_MODULE._MAX_METADATA_BYTES, 16 * _MIB)
        self.assertEqual(_MODULE._MAX_INDEX_BYTES, 64 * _MIB)
        with _temporary() as directory:
            checkpoint = _Checkpoint(Path(directory), shards=1)
            path = checkpoint.root / "model.safetensors.index.json"
            # Legal JSON whitespace tests the real byte limit without allocating
            # a production-size expert dictionary in this tiny-memory test.
            target_size = 16 * _MIB + 4096
            remaining = target_size - path.stat().st_size
            padding = b" " * _MIB
            with path.open("ab") as handle:
                while remaining:
                    count = min(remaining, len(padding))
                    handle.write(padding[:count])
                    remaining -= count
            self.assertEqual(path.stat().st_size, target_size)
            backend = checkpoint.backend()
            self.assertEqual(backend._row_starts, tuple(checkpoint.starts))
            self.assertEqual(len(backend._addresses), 1)

    def test_index_and_header_caps_are_independent_and_reads_bounded(self) -> None:
        with _temporary() as directory:
            checkpoint = _Checkpoint(Path(directory), shards=1)
            index_path = checkpoint.root / "model.safetensors.index.json"
            raw_index = index_path.read_bytes()
            sizes = []
            for path in checkpoint.files:
                with path.open("rb") as handle:
                    sizes.append(struct.unpack("<Q", handle.read(8))[0])
            with mock.patch.object(_MODULE, "_MAX_INDEX_BYTES", len(raw_index)):
                with mock.patch.object(_MODULE, "_MAX_METADATA_BYTES", max(sizes)):
                    self.assertEqual(checkpoint.backend()._row_starts, tuple(checkpoint.starts))
                with mock.patch.object(_MODULE, "_MAX_METADATA_BYTES", max(sizes) - 1):
                    with self.assertRaisesRegex(ValueError, "header"):
                        checkpoint.backend()
            with mock.patch.object(_MODULE, "_MAX_INDEX_BYTES", len(raw_index) - 1):
                with self.assertRaisesRegex(ValueError, "Oversized checkpoint index"):
                    checkpoint.backend()
            opened = mock.mock_open(read_data=raw_index)
            with mock.patch.object(_MODULE, "_MAX_INDEX_BYTES", 32):
                with mock.patch.object(Path, "open", opened):
                    with self.assertRaisesRegex(ValueError, "Oversized checkpoint index"):
                        checkpoint.backend()
            opened.return_value.read.assert_called_once_with(33)

    def test_owned_read_only_mapping_does_not_change_loader_mapping(self) -> None:
        with _temporary() as directory:
            checkpoint = _Checkpoint(Path(directory), shards=2)
            key = f"{_PREFIX}.shard_0.weight"
            with safe_open(checkpoint.tensor_files[key], framework="pt", device="cpu") as handle:
                source = handle.get_tensor(key)
                source_pointer = source.data_ptr()
                with mock.patch.object(
                    torch.cuda, "init", side_effect=AssertionError("bind initialized CUDA")
                ):
                    backend = checkpoint.backend()
                self.assertNotEqual(backend._addresses[0][0], source_pointer)
                self.assertTrue(all(region["permissions"] == "r--p" for region in _smaps(backend)))
                # Modifying the old private view must neither fault nor alter the new mapping.
                original = checkpoint.leaves[key][0, 0].item()
                source[0, 0] = original ^ 255
                mapped = backend._files[checkpoint.tensor_files[key].resolve()]
                self.assertEqual(mapped.mapping[mapped.tensors[key].offset], original)

    def test_explicit_files_unindexed_headers_and_symlink(self) -> None:
        with _temporary() as directory:
            checkpoint = _Checkpoint(Path(directory), shards=3)
            backend = _Backend(checkpoint.starts[-1], 160)
            backend.bind_files(checkpoint.tensor_files, _PREFIX)
            (checkpoint.root / "model.safetensors.index.json").unlink()
            self.assertEqual(checkpoint.backend()._row_starts, backend._row_starts)
            target = checkpoint.files[0]
            link = checkpoint.root / "linked.safetensors"
            link.symlink_to(target.name)
            tensor_files = {
                key: link if path == target else path
                for key, path in checkpoint.tensor_files.items()
            }
            other = _Backend(checkpoint.starts[-1], 160)
            other.bind_files(tensor_files, _PREFIX)
            self.assertEqual(len(other._files), 3)

    def test_rebind_and_topology_checks(self) -> None:
        with _temporary() as directory:
            checkpoint = _Checkpoint(Path(directory), shards=2)
            backend = checkpoint.backend()
            with self.assertRaisesRegex(RuntimeError, "rebind"):
                backend.bind_checkpoint(checkpoint.root, _PREFIX)
            for rows, shards in ((checkpoint.starts[-1] + 1, 2), (checkpoint.starts[-1], 3)):
                with self.subTest(rows=rows, shards=shards), self.assertRaises(ValueError):
                    _Backend(rows, 160, expected_shards=shards).bind_checkpoint(
                        checkpoint.root, _PREFIX
                    )
            files = dict(checkpoint.tensor_files)
            del files[f"{_PREFIX}.shard_0.weight"]
            with self.assertRaisesRegex(ValueError, "both weight"):
                _Backend(checkpoint.starts[-1], 160).bind_files(files, _PREFIX)

    def test_global_scale_dtype_shape_and_values(self) -> None:
        for value in (
            torch.tensor(1.0),
            torch.ones(2),
            torch.ones(1, dtype=torch.bfloat16),
            torch.tensor([0.0]),
            torch.tensor([-1.0]),
            torch.tensor([float("nan")]),
            torch.tensor([float("inf")]),
        ):
            with self.subTest(value=value), _temporary() as directory:
                checkpoint = _Checkpoint(Path(directory), shards=1)
                checkpoint._groups[0][f"{_PREFIX}.weight_scale_2"] = value
                checkpoint.save()
                with self.assertRaises(ValueError):
                    checkpoint.backend()

    def test_corrupt_shape_dtype_offsets_overlap_and_header_size(self) -> None:
        key = f"{_PREFIX}.shard_0.weight"
        mutations = [
            lambda header: header[key].update(shape=[32, 79]),
            lambda header: header[key].update(dtype="I8"),
            lambda header: header[key].update(data_offsets=[0, 1 << 60]),
            lambda header: header[key].update(
                data_offsets=header["non_ple.weight"]["data_offsets"]
            ),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate), _temporary() as directory:
                checkpoint = _Checkpoint(Path(directory), shards=1)
                _rewrite_header(checkpoint.tensor_files[key], mutate)
                with self.assertRaises(ValueError):
                    checkpoint.backend()
        with _temporary() as directory:
            checkpoint = _Checkpoint(Path(directory), shards=1)
            checkpoint.files[0].write_bytes(struct.pack("<Q", 1 << 60))
            with self.assertRaisesRegex(ValueError, "header"):
                checkpoint.backend()

    def test_index_traversal_duplicate_keys_and_missing_tensor(self) -> None:
        with _temporary() as directory:
            checkpoint = _Checkpoint(Path(directory), shards=1)
            path = checkpoint.root / "model.safetensors.index.json"
            path.write_text('{"weight_map":{},"weight_map":{}}')
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                checkpoint.backend()
            path.write_text(
                json.dumps({"weight_map": {f"{_PREFIX}.weight_scale_2": "../outside.safetensors"}})
            )
            with self.assertRaisesRegex(ValueError, "traversal"):
                checkpoint.backend()
            checkpoint.save()
            files = dict(checkpoint.tensor_files)
            files[f"{_PREFIX}.shard_0.weight"] = checkpoint.files[2]
            with self.assertRaisesRegex(ValueError, "missing from"):
                _Backend(checkpoint.starts[-1], 160).bind_files(files, _PREFIX)


@unittest.skipUnless(
    os.environ.get("PLE_NVFP4_GPU_TESTS") == "1", "explicit tiny GPU opt-in required"
)
class GpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        torch.cuda.init()
        if not all(_Backend.device_capabilities(torch.device("cuda")).values()):
            raise unittest.SkipTest("pageable memory with host page tables required")

    def test_nan_guards_and_graph_recovery(self) -> None:
        with _temporary() as directory:
            checkpoint = _Checkpoint(Path(directory), shards=2)
            scales = checkpoint.leaves[f"{_PREFIX}.shard_0.weight_scale"].view(torch.uint8)
            scales[1, 0] = 0x7F
            scales[2, 0] = 0xB8
            scales[3, :] = 0x80  # negative zero is not a negative scale
            checkpoint.save()
            backend = checkpoint.backend()
            backend.prepare(torch.device("cuda"))
            rows = checkpoint.starts[-1]
            ids = torch.tensor([-1, rows, -(1 << 63), (1 << 63) - 1, 1, 2, 3, 0], device="cuda")
            output = backend.gather(ids)
            torch.cuda.synchronize()
            torch.testing.assert_close(
                output.cpu(), checkpoint.reference(ids.cpu()), rtol=0, atol=0, equal_nan=True
            )
            self.assertTrue(bool(torch.isnan(output[:6]).all()))
            self.assertTrue(bool(torch.isfinite(output[6:]).all()))
            with self.assertRaisesRegex(ValueError, "invalid ID"):
                backend.validate_output(output)
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    backend.gather(ids, out=output)
                ids.fill_(0)
                graph.replay()
                backend.validate_output(output)
                ids.fill_(-1)
                graph.replay()
                self.assertTrue(bool(torch.isnan(output).all()))
                ids.fill_(0)
                graph.replay()
                backend.validate_output(output)
            finally:
                torch.cuda.synchronize()
                graph.reset()

    def test_bf16_overflow_is_nan(self) -> None:
        with _temporary() as directory:
            checkpoint = _Checkpoint(
                Path(directory), shards=1, global_scale=torch.finfo(torch.float32).max
            )
            checkpoint.leaves[f"{_PREFIX}.shard_0.weight"].fill_(0x77)
            checkpoint.leaves[f"{_PREFIX}.shard_0.weight_scale"].view(torch.uint8).fill_(0x38)
            checkpoint.save()
            backend = checkpoint.backend()
            backend.prepare(torch.device("cuda"))
            output = backend.gather(torch.tensor([0], device="cuda"))
            self.assertTrue(bool(torch.isnan(output).all()))

    def test_api_errors_empty_scalar_noncontiguous_alias_and_capability(self) -> None:
        with _temporary() as directory:
            checkpoint = _Checkpoint(Path(directory), shards=2)
            backend = checkpoint.backend()
            ids = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
            with self.assertRaisesRegex(RuntimeError, "prepare"):
                backend.gather(ids)
            with mock.patch.object(_Backend, "device_capabilities", return_value={"pageable": 0}):
                with self.assertRaisesRegex(RuntimeError, "Unsupported"):
                    backend.prepare(torch.device("cuda"))
            backend.prepare(torch.device("cuda"))
            for bad_ids in (ids.cpu(), ids.float(), torch.arange(4, device="cuda")[::2]):
                with self.subTest(ids=bad_ids), self.assertRaises(ValueError):
                    backend.gather(bad_ids)
            with self.assertRaises(ValueError):
                backend.gather(ids, weight_scale=1.0)
            for out in (
                torch.empty(2, 160, device="cuda"),
                torch.empty(2, 159, dtype=torch.bfloat16, device="cuda"),
                torch.empty(160, 2, dtype=torch.bfloat16, device="cuda").T,
            ):
                with self.assertRaises(ValueError):
                    backend.gather(ids, out=out)
            shared = torch.zeros(2 * 160, dtype=torch.bfloat16, device="cuda")
            with self.assertRaisesRegex(ValueError, "alias"):
                backend.gather(shared.view(torch.int32)[:2], out=shared.view(2, 160))
            empty = backend.gather(torch.empty((2, 0), device="cuda", dtype=torch.int64))
            self.assertEqual(empty.shape, (2, 0, 160))
            scalar = backend.gather(torch.tensor(0, device="cuda", dtype=torch.int32))
            torch.testing.assert_close(
                scalar.cpu(), checkpoint.reference(torch.tensor(0)), rtol=0, atol=0
            )
            with mock.patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "outside CUDA graph"):
                    backend.validate_output(scalar)

    def test_width160_128_shards_changed_ids_cold_graph_and_timings(self) -> None:
        report = {
            "device": torch.cuda.get_device_name(),
            "compute_capability": torch.cuda.get_device_capability(),
            "torch": torch.__version__,
            "triton": triton.__version__,
            "helper_sha256": hashlib.sha256(_MODULE_PATH.read_bytes()).hexdigest(),
            "capabilities": _Backend.device_capabilities(torch.device("cuda")),
            "cases": [],
            "limitations": [
                "Tiny synthetic data, not full-model performance or memory-pressure validation.",
                "Cold means verified OS page-cache cold, not SSD/controller cache cold.",
                "Wall/event timings include launch overhead but exclude ID/output transfers.",
                "Torch CUDA accounting excludes driver/context allocations.",
                "Graph invalid values yield all-NaN rows, not synchronous Python exceptions.",
            ],
        }
        with _temporary() as directory:
            checkpoint = _Checkpoint(Path(directory))
            backend = checkpoint.backend()
            backend.prepare(torch.device("cuda"))
            report["shards"] = len(backend._addresses)
            report["row_counts"] = checkpoint.row_counts
            report["source_bytes"] = sum(path.stat().st_size for path in checkpoint.files)
            report["gpu_metadata_bytes"] = sum(
                tensor.numel() * tensor.element_size()
                for table in backend._device_tables.values()
                for tensor in table
            )
            generator = torch.Generator().manual_seed(2026)
            for batch in (16, 32, 128, 8192):
                cases = []
                for iteration in range(3):
                    ids_host = torch.randint(checkpoint.starts[-1], (batch,), generator=generator)
                    boundary_ids = (
                        checkpoint.starts[:-1]
                        if iteration % 2 == 0
                        else [end - 1 for end in checkpoint.starts[1:]]
                    )
                    ids_host[: min(batch, len(boundary_ids))] = torch.tensor(boundary_ids[:batch])
                    ids_host[-1] = checkpoint.starts[-1] - 1
                    ids_host[-2] = ids_host[0]
                    cases.append(ids_host.reshape(1, batch))
                references = [checkpoint.reference(ids_host) for ids_host in cases]
                ids_device = cases[0].to(
                    device="cuda", dtype=torch.int32 if batch == 32 else torch.int64
                )
                output = backend.allocate_output((1, batch, 160), ids_device.device)
                start = time.perf_counter()
                for _ in range(3):
                    backend.gather(ids_device, out=output)
                torch.cuda.synchronize()
                warmup_ms = (time.perf_counter() - start) * 1000
                torch.testing.assert_close(output.cpu(), references[0], rtol=0, atol=0)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    backend.gather(ids_device, out=output)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(graph, stream=stream):
                        backend.gather(ids_device, out=output)
                    trials = []
                    for ids_host, expected in zip(cases, references):
                        ids_device.copy_(ids_host)
                        output.fill_(float("nan"))
                        torch.cuda.synchronize()
                        before = _reclaim(backend, checkpoint)
                        self.assertTrue(all(item["resident"] == 0 for item in before))
                        timing = _timed(graph.replay)
                        after = _residency(backend)
                        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
                        self.assertTrue(all(item["resident"] > 0 for item in after))
                        trials.append({"before": before, "after": after, **timing})
                    warm = [_timed(graph.replay) for _ in range(20)]
                    eager = [
                        _timed(lambda: backend.gather(ids_device, out=output)) for _ in range(10)
                    ]
                    smaps = _smaps(backend)
                    self.assertTrue(smaps)
                    self.assertTrue(all(region["permissions"] == "r--p" for region in smaps))
                    self.assertTrue(
                        all(
                            region[key] == "0 kB"
                            for region in smaps
                            for key in ("Anonymous", "Private_Dirty", "Shared_Dirty", "Locked")
                        )
                    )
                    reclaimed = _reclaim(backend, checkpoint)
                    self.assertTrue(all(item["resident"] == 0 for item in reclaimed))
                    report["cases"].append(
                        {
                            "batch_rows": batch,
                            "width": 160,
                            "warmup_ms": warmup_ms,
                            "warm_graph_samples": warm,
                            "warm_eager_samples": eager,
                            "warm_graph_median_wall_us": statistics.median(
                                item["wall_us"] for item in warm
                            ),
                            "cold_graph_trials": trials,
                            "post_gpu_smaps": smaps,
                            "final_reclaimed_residency": reclaimed,
                        }
                    )
                finally:
                    torch.cuda.synchronize()
                    graph.reset()
            gc.collect()
            report["max_torch_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
            report["max_torch_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["process_max_rss_bytes"] = (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            )
            self.assertLess(report["max_torch_cuda_reserved_bytes"], 256 * _MIB)
            self.assertLess(report["process_max_rss_bytes"], 1024 * _MIB)
            report["numerics_graph_reclaim_pass"] = True
        output_path = os.environ.get("PLE_NVFP4_RESULTS_JSON")
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps({key: value for key, value in report.items() if key != "cases"}), flush=True
        )

    def test_wrapper_graph_cpu_fallback_and_reclaim(self) -> None:
        module = _wrapper_module()
        with _temporary() as directory:
            checkpoint = _Checkpoint(Path(directory))
            wrapper = module.Qwen4ExpNVFP4MmapEmbedding(
                checkpoint.starts[-1], 160, expected_shards=128, max_gather_rows=8
            )
            self.assertFalse(wrapper.supports_cuda_graph)
            leaves = {}
            with ExitStack() as stack:
                for path in checkpoint.files:
                    handle = stack.enter_context(safe_open(path, framework="pt", device="cpu"))
                    for key in handle.keys():
                        if key.startswith(_PREFIX + "."):
                            leaf = "ngram_embedding." + key[len(_PREFIX) + 1 :]
                            leaves[leaf] = handle.get_slice(key)
                wrapper.bind_shards(leaves)
            leaves.clear()
            gc.collect()

            ids_host = torch.tensor(
                [checkpoint.starts[index] for index in range(15)] + [checkpoint.starts[-1] - 1]
            ).reshape(4, 4)
            expected = checkpoint.reference(ids_host)
            torch.testing.assert_close(wrapper.gather(ids_host), expected, rtol=0, atol=0)
            self.assertTrue(wrapper.enable_gpu_lookup(str(checkpoint.root), _PREFIX))
            self.assertTrue(wrapper.supports_cuda_graph)
            backend = wrapper._gpu_backend
            self.assertEqual(backend._row_starts, tuple(checkpoint.starts))
            self.assertEqual(len(backend._addresses), 128)
            self.assertEqual(list(wrapper.parameters()), [])
            self.assertEqual(list(wrapper.buffers()), [])
            cpu_pointers = [tensor.data_ptr() for shard in wrapper._shards for tensor in shard]
            gpu_addresses = backend._addresses
            wrapper.to(device="cuda", dtype=torch.bfloat16)
            self.assertEqual(
                cpu_pointers, [tensor.data_ptr() for shard in wrapper._shards for tensor in shard]
            )
            self.assertTrue(
                all(tensor.device.type == "cpu" for shard in wrapper._shards for tensor in shard)
            )
            self.assertEqual(gpu_addresses, backend._addresses)
            torch.testing.assert_close(wrapper.gather(ids_host), expected, rtol=0, atol=0)
            with self.assertRaises(IndexError):
                wrapper.gather(torch.tensor([-1]))
            with self.assertRaisesRegex(RuntimeError, "cannot be rebound"):
                wrapper.bind_shards({})
            with self.assertRaisesRegex(RuntimeError, "cannot be rebound"):
                wrapper.enable_gpu_lookup(str(checkpoint.root), _PREFIX)
            for device in (
                torch.device("cpu"),
                torch.device("cuda", torch.cuda.current_device() + 1),
            ):
                with self.assertRaisesRegex(RuntimeError, "eager-only"):
                    wrapper.check_execution(device, is_cuda_graph=True)

            ids_device = ids_host.to("cuda")
            wrapper.check_execution(ids_device.device, is_cuda_graph=True)
            # PLE prefetch allocates flattened heads, then supplies a per-head view.
            flat_output = wrapper.allocate_output((4, 4 * 160), ids_device.device)
            output = flat_output.view(4, 4, 160)
            self.assertIs(wrapper.gather(ids_device, out=output), output)
            noncontiguous = ids_device.T
            self.assertFalse(noncontiguous.is_contiguous())
            torch.testing.assert_close(
                wrapper.gather(noncontiguous).cpu(),
                checkpoint.reference(ids_host.T),
                rtol=0,
                atol=0,
            )
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                wrapper.gather(ids_device, out=output)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            trials = []
            try:
                with torch.cuda.graph(graph, stream=stream):
                    wrapper.gather(ids_device, out=output)
                for changed_ids in (ids_host, ids_host.flip(1), torch.full_like(ids_host, -1)):
                    reference = checkpoint.reference(changed_ids)
                    ids_device.copy_(changed_ids)
                    output.fill_(float("nan"))
                    torch.cuda.synchronize()
                    _discard_cpu_fixture_views(wrapper, checkpoint)
                    before = _reclaim(backend, checkpoint)
                    self.assertTrue(all(item["resident"] == 0 for item in before))
                    timing = _timed(graph.replay)
                    after = _residency(backend)
                    torch.testing.assert_close(
                        output.cpu(), reference, rtol=0, atol=0, equal_nan=True
                    )
                    # Invalid IDs must not fault ANY source-file page back in.
                    expected_resident = not bool((changed_ids < 0).all())
                    self.assertEqual(any(item["resident"] > 0 for item in after), expected_resident)
                    trials.append(
                        {
                            "before": before,
                            "after": after,
                            "invalid_ids": not expected_resident,
                            **timing,
                        }
                    )
                ids_device.copy_(ids_host)
                graph.replay()
                backend.validate_output(output)
                smaps = _smaps(backend)
                self.assertTrue(smaps)
                self.assertTrue(all(region["permissions"] == "r--p" for region in smaps))
                self.assertTrue(
                    all(
                        region[key] == "0 kB"
                        for region in smaps
                        for key in ("Anonymous", "Private_Dirty", "Shared_Dirty", "Locked")
                    )
                )
                torch.cuda.synchronize()
                _discard_cpu_fixture_views(wrapper, checkpoint)
                reclaimed = _reclaim(backend, checkpoint)
                self.assertTrue(all(item["resident"] == 0 for item in reclaimed))
                # Explicit CPU fallback remains usable after GPU capture/reclaim.
                torch.testing.assert_close(wrapper.gather(ids_host), expected, rtol=0, atol=0)
            finally:
                torch.cuda.synchronize()
                graph.reset()
            report = {
                "wrapper_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                "helper_sha256": hashlib.sha256(_MODULE_PATH.read_bytes()).hexdigest(),
                "device": torch.cuda.get_device_name(),
                "checkpoint_prefix": _PREFIX,
                "shards": 128,
                "width": 160,
                "cpu_fallback_graph_reclaim_pass": True,
                "cold_graph_trials": trials,
                "post_gpu_smaps": smaps,
                "final_reclaimed_residency_before_cpu_fallback": reclaimed,
                "max_torch_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            }
            self.assertLess(report["max_torch_cuda_reserved_bytes"], 256 * _MIB)
            self.assertLess(report["process_max_rss_bytes"], 1024 * _MIB)
            output_path = os.environ.get("PLE_NVFP4_WRAPPER_RESULTS_JSON")
            if output_path:
                path = Path(output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(report, indent=2) + "\n")


def _timed(action) -> dict[str, float]:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    end.record()
    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    begin.record()
    action()
    end.record()
    end.synchronize()
    return {
        "wall_us": (time.perf_counter_ns() - started) / 1000,
        "cuda_event_us": begin.elapsed_time(end) * 1000,
    }


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
