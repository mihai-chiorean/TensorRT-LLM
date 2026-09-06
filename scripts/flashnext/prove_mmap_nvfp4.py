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
"""Tiny Linux proof: direct GPU NVFP4 decode from reclaimable safetensors mmap.

Run outside the source import tree, with installed torch/triton/safetensors:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python prove_mmap_nvfp4.py
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python prove_mmap_nvfp4.py --read-only-mappings

No TensorRT imports, model weights, host registration, pinning, or global cache
drops. Only self-created temporary files receive madvise/fadvise. Integer CPU
data_ptr values are cast to pointers *inside* Triton, bypassing its host-tensor
argument rejection. This requires pageable-memory access with host page tables;
it is not a portable fallback for discrete GPUs without those capabilities.

The graph owner retains CPU tensors (and hence their mmap storage) until all
replays finish and the graph is reset. Never truncate/unmap/replace the backing
files while GPU work can access them. IDs/output addresses and shard topology
are fixed at capture; ID contents can change. This is a proof, not runtime PLE.

MADV_DONTNEED removes process mappings; POSIX_FADV_DONTNEED additionally requests
file-cache eviction. mincore verifies residency without faulting pages in. A
zero-resident prelaunch sample proves page-cache cold, not cold SSD/controller
cache. Neither this tiny synthetic test nor its timings establish model wins.

Use --read-only-mappings to mprotect only this proof's own mappings PROT_READ.
The default preserves safetensors' mapping permissions. Compare Anonymous and
Private_Dirty in post_gpu_smaps: successful explicit MADV_DONTNEED alone does
not prove clean file-cache reclaim under OS pressure if GPU faults create COW.

CUDA file-backed Unified Memory reference:
https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html
"""

import argparse
import ctypes
import json
import mmap
import os
import resource
import statistics
import tempfile
import time
from pathlib import Path

import torch
import triton
import triton.language as tl
from safetensors import safe_open
from safetensors.torch import save_file

__all__ = []

_GLOBAL_SCALE = 1.3
_MIB = 1024 * 1024


@triton.jit
def _gather_nvfp4(
    weight_addresses,
    scale_addresses,
    ids,
    output,
    STARTS: tl.constexpr,
    WIDTH: tl.constexpr,
    GLOBAL_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    position = tl.program_id(0)
    row = tl.load(ids + position).to(tl.int64)
    columns = tl.arange(0, BLOCK)
    values = tl.full((BLOCK,), 0, tl.float32)
    for shard in tl.static_range(len(weight_addresses)):
        if (row >= STARTS[shard]) & (row < STARTS[shard + 1]):
            local_row = row - STARTS[shard]
            # These are scalar integer kernel arguments, not CUDA allocations.
            weights = weight_addresses[shard].to(tl.pointer_type(tl.uint8))
            scales = scale_addresses[shard].to(tl.pointer_type(tl.uint8))
            packed = tl.load(
                weights + local_row * (WIDTH // 2) + columns // 2,
                columns < WIDTH,
                other=0,
            ).to(tl.int32)
            code = (packed >> ((columns % 2) * 4)) & 15
            magnitude = code & 7
            # E2M1: 0, .5, 1, 1.5, 2, 3, 4, 6 (low nibble first).
            absolute = tl.where(
                magnitude < 4,
                magnitude * 0.5,
                tl.where(magnitude < 6, magnitude - 2.0, 2.0 * magnitude - 8.0),
            )
            decoded = tl.where((code & 8) != 0, -absolute, absolute)
            block_scale = (
                tl.load(
                    scales + local_row * (WIDTH // 16) + columns // 16,
                    columns < WIDTH,
                    other=0,
                )
                .to(tl.float8e4nv, bitcast=True)
                .to(tl.float32)
            )
            values = decoded * block_scale * GLOBAL_SCALE
    tl.store(output + position * WIDTH + columns, values, columns < WIDTH)


def _mapping_regions(path: Path) -> list[tuple[int, int]]:
    regions = []
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5] == str(path):
            start, end = (int(value, 16) for value in fields[0].split("-"))
            regions.append((start, end - start))
    if not regions:
        raise RuntimeError(f"No file-backed mapping found for own fixture {path}")
    return regions


def _libc() -> ctypes.CDLL:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.madvise.restype = ctypes.c_int
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    libc.mincore.restype = ctypes.c_int
    libc.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.mprotect.restype = ctypes.c_int
    return libc


class _Fixture:
    """Own tiny files and CPU mmap tensors; never accept external file paths."""

    def __init__(self, directory: Path, row_counts: tuple[int, ...], width: int):
        self.paths = []
        self.shards = []
        self.starts = [0]
        self.width = width
        self.libc = _libc()
        generator = torch.Generator().manual_seed(121)
        for shard, rows in enumerate(row_counts):
            path = (directory / f"own-nvfp4-{shard}.safetensors").resolve()
            packed = torch.randint(256, (rows, width // 2), dtype=torch.uint8, generator=generator)
            # Sample finite nonnegative E4M3 encodings, including subnormals.
            scale_bytes = torch.randint(
                127, (rows, width // 16), dtype=torch.uint8, generator=generator
            )
            save_file(
                {"weight": packed, "weight_scale": scale_bytes.view(torch.float8_e4m3fn)},
                str(path),
            )
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
            with safe_open(path, framework="pt", device="cpu") as handle:
                weights_host = handle.get_slice("weight")[:]
                scales_host = handle.get_slice("weight_scale")[:]
            # Retained tensors must own the file mapping after safe_open closes.
            regions = _mapping_regions(path)
            for tensor in (weights_host, scales_host):
                if tensor.device.type != "cpu" or not tensor.is_contiguous():
                    raise AssertionError("Expected contiguous CPU views")
                size = tensor.numel() * tensor.element_size()
                if not any(
                    start <= tensor.data_ptr() and tensor.data_ptr() + size <= start + length
                    for start, length in regions
                ):
                    raise AssertionError("safetensors returned a copy, not a file mapping")
            self.paths.append(path)
            self.shards.append((weights_host, scales_host))
            self.starts.append(self.starts[-1] + rows)

    def protect_read_only(self) -> None:
        for path in self.paths:
            for start, length in _mapping_regions(path):
                if self.libc.mprotect(start, length, mmap.PROT_READ):
                    raise OSError(ctypes.get_errno(), "mprotect failed on own fixture")

    def residency(self) -> list[dict[str, int]]:
        result = []
        for path in self.paths:
            pages = resident = 0
            for start, length in _mapping_regions(path):
                count = (length + mmap.PAGESIZE - 1) // mmap.PAGESIZE
                vector = (ctypes.c_ubyte * count)()
                if self.libc.mincore(start, length, vector):
                    raise OSError(ctypes.get_errno(), "mincore failed on own fixture")
                pages += count
                resident += sum(value & 1 for value in vector)
            result.append({"pages": pages, "resident": resident})
        return result

    def reclaim(self, evict_file_cache: bool) -> list[dict[str, int]]:
        # Caller synchronizes GPU first. Keep virtual addresses alive for graph replay.
        for path in self.paths:
            for start, length in _mapping_regions(path):
                if self.libc.madvise(start, length, mmap.MADV_DONTNEED):
                    raise OSError(ctypes.get_errno(), "madvise failed on own fixture")
            if evict_file_cache:
                with path.open("rb") as handle:
                    os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        return self.residency()

    def smaps(self) -> list[dict[str, str]]:
        result = []
        current = None
        for line in Path("/proc/self/smaps").read_text().splitlines():
            fields = line.split(maxsplit=5)
            if fields and "-" in fields[0] and ":" not in fields[0]:
                current = None
                if len(fields) == 6 and fields[5] in {str(path) for path in self.paths}:
                    current = {"file": Path(fields[5]).name, "permissions": fields[1]}
                    result.append(current)
            elif current is not None and ":" in line:
                key, value = line.split(":", 1)
                if key in {
                    "Rss",
                    "Locked",
                    "Anonymous",
                    "Private_Dirty",
                    "Shared_Dirty",
                    "VmFlags",
                }:
                    current[key] = value.strip()
        return result

    def reference(self, ids_host: torch.Tensor) -> torch.Tensor:
        lookup = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
        result = torch.zeros((ids_host.numel(), self.width), dtype=torch.bfloat16)
        for shard, (weights_host, scales_host) in enumerate(self.shards):
            selected = (ids_host >= self.starts[shard]) & (ids_host < self.starts[shard + 1])
            local_ids = ids_host[selected] - self.starts[shard]
            packed = weights_host[local_ids].long()
            codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(1)
            scales = scales_host.view(torch.uint8)[local_ids].view(torch.float8_e4m3fn).float()
            result[selected] = (
                lookup[codes] * scales.repeat_interleave(16, dim=1) * _GLOBAL_SCALE
            ).bfloat16()
        return result


class _GraphOwner:
    def __init__(self, fixture: _Fixture, batch_size: int):
        self.fixture = fixture
        self.ids_device = torch.zeros(batch_size, device="cuda", dtype=torch.int64)
        self.output_device = torch.empty(
            (batch_size, fixture.width), device="cuda", dtype=torch.bfloat16
        )
        self.graph = torch.cuda.CUDAGraph()

    def launch(self) -> None:
        _gather_nvfp4[(self.ids_device.numel(),)](
            tuple(weight.data_ptr() for weight, _ in self.fixture.shards),
            tuple(scale.data_ptr() for _, scale in self.fixture.shards),
            self.ids_device,
            self.output_device,
            STARTS=tuple(self.fixture.starts),
            WIDTH=self.fixture.width,
            GLOBAL_SCALE=_GLOBAL_SCALE,
            BLOCK=triton.next_power_of_2(self.fixture.width),
            num_warps=4,
            enable_fp_fusion=False,
        )

    def capture(self) -> None:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.launch()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        with torch.cuda.graph(self.graph, stream=stream):
            self.launch()
        torch.cuda.synchronize()

    def close(self) -> None:
        torch.cuda.synchronize()
        self.graph.reset()


def _capabilities() -> dict[str, int]:
    # CUDA Driver API enum values, stable ABI (cuda.h); no TRT/runtime import.
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuDeviceGetAttribute.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
    ]
    driver.cuDeviceGetAttribute.restype = ctypes.c_int
    attributes = {}
    for name, attribute in (
        ("pageable_memory_access", 88),
        ("concurrent_managed_access", 89),
        ("can_use_host_pointer_for_registered_mem", 91),
        ("pageable_memory_access_uses_host_page_tables", 100),
    ):
        value = ctypes.c_int()
        status = driver.cuDeviceGetAttribute(
            ctypes.byref(value), attribute, torch.cuda.current_device()
        )
        if status:
            raise RuntimeError(f"cuDeviceGetAttribute({attribute}) failed: {status}")
        attributes[name] = value.value
    return attributes


def _read_bytes() -> int:
    values = dict(line.split(":", 1) for line in Path("/proc/self/io").read_text().splitlines())
    return int(values["read_bytes"])


def _timed(action) -> dict[str, float | int]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Initialize event resources outside the measured interval.
    start.record()
    end.record()
    torch.cuda.synchronize()
    before_bytes = _read_bytes()
    before = time.perf_counter_ns()
    start.record()
    action()
    end.record()
    end.synchronize()
    elapsed = (time.perf_counter_ns() - before) / 1000
    return {
        "wall_us": elapsed,
        "cuda_event_us": start.elapsed_time(end) * 1000,
        "process_read_bytes_delta": _read_bytes() - before_bytes,
    }


def _id_cases(fixture: _Fixture, batch_size: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(2026)
    cases = []
    boundaries = [
        value
        for start, end in zip(fixture.starts, fixture.starts[1:])
        for value in (start, end - 1)
    ]
    for iteration in range(3):
        ids = torch.randint(fixture.starts[-1], (batch_size,), generator=generator)
        for position in range(len(boundaries)):
            ids[position] = boundaries[(position + iteration) % len(boundaries)]
        ids[-1] = ids[0]
        cases.append(ids)
    return cases


def _run(
    directory: Path,
    rows: tuple[int, ...],
    width: int,
    batch_size: int,
    repeats: int,
    read_only_mappings: bool,
) -> dict:
    torch.set_num_threads(1)
    torch.cuda.init()
    attributes = _capabilities()
    if not (
        attributes["pageable_memory_access"]
        and attributes["pageable_memory_access_uses_host_page_tables"]
        and attributes["concurrent_managed_access"]
    ):
        raise RuntimeError(f"Unsupported device; refusing raw CPU pointer launch: {attributes}")
    fixture = _Fixture(directory, rows, width)
    if read_only_mappings:
        fixture.protect_read_only()
    owner = _GraphOwner(fixture, batch_size)
    report = {
        "device": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "attributes": attributes,
        "rows": rows,
        "width": width,
        "batch_size": batch_size,
        "file_bytes": sum(path.stat().st_size for path in fixture.paths),
        "page_size": mmap.PAGESIZE,
        "read_only_mappings": read_only_mappings,
    }
    try:
        cases = _id_cases(fixture, batch_size)
        # Compute references before reclaim: never CPU-read source during a cold trial.
        references = [fixture.reference(ids) for ids in cases]
        owner.ids_device.copy_(cases[0])
        started = time.perf_counter()
        owner.launch()
        torch.cuda.synchronize()
        report["compile_and_first_launch_ms"] = (time.perf_counter() - started) * 1000
        torch.testing.assert_close(owner.output_device.cpu(), references[0], rtol=0, atol=0)
        report["warm_eager"] = _timed(owner.launch)

        trials = []
        for label, evict_cache in (("pte_cold_eager", False), ("file_cold_eager", True)):
            torch.cuda.synchronize()
            before = fixture.reclaim(evict_cache)
            smaps_before = fixture.smaps()
            timing = _timed(owner.launch)
            after = fixture.residency()
            torch.testing.assert_close(owner.output_device.cpu(), references[0], rtol=0, atol=0)
            trials.append(
                {
                    "mode": label,
                    "before": before,
                    "after": after,
                    "smaps_before": smaps_before,
                    **timing,
                }
            )

        owner.capture()
        for iteration, (ids, expected) in enumerate(zip(cases, references)):
            owner.ids_device.copy_(ids)
            owner.output_device.fill_(float("nan"))
            torch.cuda.synchronize()
            before = fixture.reclaim(evict_file_cache=True)
            smaps_before = fixture.smaps()
            timing = _timed(owner.graph.replay)
            after = fixture.residency()
            torch.testing.assert_close(owner.output_device.cpu(), expected, rtol=0, atol=0)
            trials.append(
                {
                    "mode": f"file_cold_graph_changed_ids_{iteration}",
                    "before": before,
                    "after": after,
                    "smaps_before": smaps_before,
                    **timing,
                }
            )

        warm = [_timed(owner.graph.replay) for _ in range(repeats)]
        report["warm_graph_median_wall_us"] = statistics.median(item["wall_us"] for item in warm)
        report["warm_graph_median_cuda_event_us"] = statistics.median(
            item["cuda_event_us"] for item in warm
        )
        report["warm_graph_repeats"] = repeats
        report["trials"] = trials
        report["post_gpu_smaps"] = fixture.smaps()
        report["post_gpu_anonymous_bytes"] = sum(
            int(region["Anonymous"].split()[0]) * 1024 for region in report["post_gpu_smaps"]
        )
        report["post_gpu_locked_bytes"] = sum(
            int(region["Locked"].split()[0]) * 1024 for region in report["post_gpu_smaps"]
        )
        report["post_gpu_dirty_bytes"] = sum(
            int(region[key].split()[0]) * 1024
            for region in report["post_gpu_smaps"]
            for key in ("Private_Dirty", "Shared_Dirty")
        )
        report["clean_file_backing_preserved"] = bool(report["post_gpu_smaps"]) and all(
            report[key] == 0
            for key in ("post_gpu_anonymous_bytes", "post_gpu_locked_bytes", "post_gpu_dirty_bytes")
        )
        torch.cuda.synchronize()
        report["final_reclaimed_residency"] = fixture.reclaim(evict_file_cache=True)
        report["final_reclaimed_smaps"] = fixture.smaps()
        report["max_torch_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["max_torch_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["process_max_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        report["numerical_and_graph_pass"] = True
        cold_trials = [trial for trial in trials if trial["mode"].startswith("file_cold")]
        report["all_file_cold_trials_verified"] = all(
            all(shard["resident"] == 0 for shard in trial["before"])
            and all(shard["resident"] > 0 for shard in trial["after"])
            for trial in cold_trials
        )
        report["reclaimed_after_graph"] = all(
            shard["resident"] == 0 for shard in report["final_reclaimed_residency"]
        ) and all(region["Rss"] == "0 kB" for region in report["final_reclaimed_smaps"])
        report["limitations"] = [
            "Synthetic data only; no full-model latency/throughput claim.",
            "Cold means verified OS page-cache cold, not SSD/controller cache cold.",
            "Reclaim is explicit on idle private files, not sustained memory-pressure testing.",
            "Private writable mappings may COW on GPU faults; inspect Anonymous/Private_Dirty.",
            "Process read_bytes may omit GPU-fault I/O charged to kernel workers.",
            "Torch GPU accounting excludes driver/context memory; RSS includes process overhead.",
            "Fixed graph shapes/pointers/shard topology; immutable mapped storage must outlive graph.",
            "Timing includes event/launch overhead, not an isolated kernel throughput benchmark.",
            "ID upload and output download are excluded; graph contains only the gather kernel.",
            "Proof inputs are finite valid NVFP4; production ID/scale validation is not implemented.",
        ]
        if report["post_gpu_locked_bytes"]:
            raise AssertionError("GPU access unexpectedly left locked source pages")
        if report["max_torch_cuda_reserved_bytes"] >= 256 * _MIB:
            raise AssertionError("Exceeded tiny-test CUDA allocator budget")
        if report["process_max_rss_bytes"] >= 1024 * _MIB:
            raise AssertionError("Exceeded tiny-test process RSS budget")
        return report
    finally:
        owner.close()


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-per-shard", type=int, nargs="+", default=[513, 769])
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--read-only-mappings", action="store_true")
    parser.add_argument(
        "--temp-dir", type=Path, default=None, help="Use a disk-backed directory, not tmpfs"
    )
    args = parser.parse_args()
    if not (
        1 <= len(args.rows_per_shard) <= 4
        and min(args.rows_per_shard) > 0
        and sum(args.rows_per_shard) <= 4096
    ):
        parser.error("Require 1-4 positive shards totaling at most 4096 rows")
    if not (16 <= args.width <= 4096 and args.width % 16 == 0):
        parser.error("Width must be a multiple of 16, at most 4096")
    if not (
        2 * len(args.rows_per_shard) + 1 <= args.batch_size <= 128 and 1 <= args.repeats <= 100
    ):
        parser.error(
            "Batch must cover shard boundaries plus a duplicate, at most 128; repeats 1-100"
        )
    with tempfile.TemporaryDirectory(prefix="own-nvfp4-proof-", dir=args.temp_dir) as directory:
        report = _run(
            Path(directory),
            tuple(args.rows_per_shard),
            args.width,
            args.batch_size,
            args.repeats,
            args.read_only_mappings,
        )
    print(json.dumps(report, indent=2))
    if not (report["all_file_cold_trials_verified"] and report["reclaimed_after_graph"]):
        raise SystemExit(
            "Numerics passed but cold/reclaim proof was inconclusive; inspect residency"
        )
    if args.read_only_mappings and not report["clean_file_backing_preserved"]:
        raise SystemExit("Read-only mappings did not preserve clean, unlocked file-backed pages")


if __name__ == "__main__":
    _main()
