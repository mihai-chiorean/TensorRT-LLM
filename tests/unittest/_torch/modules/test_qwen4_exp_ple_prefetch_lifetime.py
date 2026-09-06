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
"""Tiny GPU regression for aborted PLE prefetch output allocator ownership.

Run directly with PLE_PREFETCH_GPU_TESTS=1 to avoid native TensorRT imports.
PLE_PREFETCH_SOURCE optionally selects a source snapshot of ple.py. The actual
prefetch lifecycle methods are compiled unchanged from that file's AST; only
hashing and gather math are replaced by tiny fixtures. A negative control removes
the output record_stream call in memory, never modifying the source file.

Native CUDA allocator required. Each trial delays one stream for 100M GPU cycles
while another stream reallocates a 40 KiB buffer; no model or checkpoint loads.
Set PLE_PREFETCH_RESULTS_JSON to save raw outcomes with the reviewed source hash.
"""

import ast
import copy
import hashlib
import json
import os
import resource
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

_ROOT = Path(__file__).resolve().parents[4]
_SOURCE = Path(
    os.environ.get("PLE_PREFETCH_SOURCE", _ROOT / "tensorrt_llm/_torch/modules/qwen4_exp/ple.py")
)
_MIB = 1024 * 1024
_WIDTH = 160
_SMALL_ROWS = 128
_LARGE_ROWS = 256
_SENTINEL = 7.0
_RESULTS: dict[str, dict] = {}


def _lifecycle_class(remove_output_record: bool) -> tuple[type, int]:
    source = ast.parse(_SOURCE.read_text(), filename=str(_SOURCE))
    original = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen4ExpPLE"
    )
    names = {
        "_allocate_prefetch_buffer",
        "_get_prefetch_buffer",
        "start_prefetch",
        "abort_prefetch",
        "_consume_prefetched_embeddings",
    }
    methods = [
        copy.deepcopy(node)
        for node in original.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in methods} != names:
        raise AssertionError("PLE prefetch lifecycle changed; update the isolated test harness")
    removed = 0
    if remove_output_record:
        start = next(node for node in methods if node.name == "start_prefetch")
        body = []
        for node in start.body:
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "record_stream"
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id in {"prefetched", "output"}
            ):
                removed += 1
            else:
                body.append(node)
        start.body = body
    isolated = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            ast.ClassDef(
                name="IsolatedPLELifecycle",
                bases=[],
                keywords=[],
                body=methods,
                decorator_list=[],
            ),
        ],
        type_ignores=[],
    )
    namespace = {"torch": torch, "__name__": "_ple_prefetch_lifetime_test"}
    exec(compile(ast.fix_missing_locations(isolated), str(_SOURCE), "exec"), namespace)
    return namespace["IsolatedPLELifecycle"], removed


class _DelayedGather:
    def __init__(self) -> None:
        self.calls = 0

    @staticmethod
    def allocate_output(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        return torch.empty(shape, device=device, dtype=torch.bfloat16)

    def gather(
        self,
        ids: torch.Tensor,
        out: torch.Tensor,
        *,
        weight_scale: float | None = None,
    ) -> torch.Tensor:
        if weight_scale is not None or out.shape != (*ids.shape, _WIDTH):
            raise AssertionError("Unexpected prefetch gather contract")
        self.calls += 1
        if self.calls == 1:
            torch.cuda._sleep(100_000_000)
        # Deliberately do not retain a Python reference or register the output
        # stream here: that ownership belongs to the real start_prefetch method.
        return out.fill_(11.0 * self.calls)


def _exercise(remove_output_record: bool) -> dict:
    lifecycle, removed = _lifecycle_class(remove_output_record)
    # Reserve/warm the small allocation pool and kernels before the timed race;
    # a cudaMalloc retry must not accidentally synchronize the pending stream.
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    warmup = torch.empty(128 * 1024, dtype=torch.bfloat16, device="cuda")
    warmup.fill_(0)
    torch.cuda._sleep(1000)
    torch.cuda.synchronize()
    del warmup

    small_ids = torch.zeros((_SMALL_ROWS, 1), dtype=torch.int64, device="cuda")
    large_ids = torch.zeros((_LARGE_ROWS, 1), dtype=torch.int64, device="cuda")
    context = torch.zeros((1, 1), dtype=torch.int64, device="cuda")
    model = lifecycle()
    model.ple_embed_dim = _WIDTH
    model._prefetch_stream = torch.cuda.Stream()
    model._graph_prefetch_buffers = {}
    model._eager_prefetch_buffer = None
    model._prefetch_state = None
    model._prepare_ngram_lookup = lambda metadata, state: (state, metadata.ids)
    model.ple_embedding = SimpleNamespace(
        nvfp4_storage=False,
        ngram_embedding=_DelayedGather(),
        ngram_heads=1,
        head_dim_per_ngram=_WIDTH,
        _host_offload_weight_scale=None,
        _prepare_embedding_lookup=lambda ids, physical, ranks: (ids, ids.shape[0]),
        _finish_embedding_lookup=lambda values, semantic, physical, ranks: values,
    )
    small = SimpleNamespace(
        ids=small_ids, physical_tokens=_SMALL_ROWS, all_rank_num_tokens=None, is_cuda_graph=False
    )
    large = SimpleNamespace(
        ids=large_ids, physical_tokens=_LARGE_ROWS, all_rank_num_tokens=None, is_cuda_graph=False
    )
    first_done = torch.cuda.Event()
    torch.cuda.synchronize()
    began = time.perf_counter_ns()
    try:
        model.start_prefetch(small, context)
        old_pointer = model._eager_prefetch_buffer.data_ptr()
        first_done.record(model._prefetch_stream)
        model.abort_prefetch()
        if model._prefetch_state is not None:
            raise AssertionError("abort_prefetch did not discard its state")
        model.start_prefetch(large, context)
        new_pointer = model._eager_prefetch_buffer.data_ptr()
        guard = torch.empty((_SMALL_ROWS, _WIDTH), device="cuda", dtype=torch.bfloat16)
        guard_pointer = guard.data_ptr()
        guard.fill_(_SENTINEL)
        torch.cuda.current_stream().synchronize()
        pending_at_reallocation = not first_done.query()
        host_setup_us = (time.perf_counter_ns() - began) / 1000
        model._prefetch_stream.synchronize()
        guard_unchanged = bool((guard == _SENTINEL).all().item())
        overwritten_by_aborted_prefetch = bool((guard == 11.0).all().item())
        output, combined = model._consume_prefetched_embeddings(large)
        retry_correct = bool((output == 22.0).all().item())
        return {
            "removed_output_record_calls": removed,
            "old_pointer": old_pointer,
            "retry_pointer": new_pointer,
            "guard_pointer": guard_pointer,
            "old_buffer_reused": guard_pointer == old_pointer,
            "pending_at_reallocation": pending_at_reallocation,
            "host_setup_us": host_setup_us,
            "guard_unchanged": guard_unchanged,
            "overwritten_by_aborted_prefetch": overwritten_by_aborted_prefetch,
            "retry_correct": retry_correct,
            "combined_context_preserved": combined is context,
            "prefetch_consumed": model._prefetch_state is None,
        }
    finally:
        torch.cuda.synchronize()


@unittest.skipUnless(os.environ.get("PLE_PREFETCH_GPU_TESTS") == "1", "tiny GPU opt-in required")
class PrefetchLifetimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        torch.set_num_threads(1)
        torch.cuda.init()
        if torch.cuda.memory.get_allocator_backend() != "native":
            raise unittest.SkipTest("regression targets the native caching allocator")

    def test_negative_control_reuses_pending_output(self) -> None:
        result = _exercise(remove_output_record=True)
        _RESULTS["negative_control"] = result
        self.assertTrue(
            result["pending_at_reallocation"], "Delay completed before the race; inconclusive"
        )
        self.assertTrue(result["old_buffer_reused"], "Allocator did not reproduce the unsafe reuse")
        self.assertTrue(result["overwritten_by_aborted_prefetch"])
        self.assertTrue(result["retry_correct"])

    def test_real_methods_preserve_pending_output(self) -> None:
        result = _exercise(remove_output_record=False)
        _RESULTS["real_source"] = result
        self.assertTrue(
            result["pending_at_reallocation"], "Delay completed before the race; inconclusive"
        )
        self.assertFalse(
            result["old_buffer_reused"], "Pending prefetch output returned to allocator early"
        )
        self.assertTrue(result["guard_unchanged"], "Aborted prefetch corrupted a new allocation")
        self.assertTrue(result["retry_correct"])
        self.assertTrue(result["combined_context_preserved"])
        self.assertTrue(result["prefetch_consumed"])
        self.assertLess(torch.cuda.max_memory_reserved(), 256 * _MIB)
        self.assertLess(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, 1024 * _MIB)


if __name__ == "__main__":
    program = unittest.main(verbosity=2, exit=False)
    report = {
        "source_sha256": hashlib.sha256(_SOURCE.read_bytes()).hexdigest(),
        "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tests_passed": program.result.wasSuccessful(),
        "outcomes": _RESULTS,
        "max_torch_cuda_reserved_bytes": torch.cuda.max_memory_reserved()
        if torch.cuda.is_initialized()
        else 0,
        "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    print(json.dumps(report, indent=2), flush=True)
    output_path = os.environ.get("PLE_PREFETCH_RESULTS_JSON")
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if program.result.wasSuccessful() else 1)
