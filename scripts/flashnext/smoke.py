# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded text-only Flash Next load and generation smoke, not a TPOT benchmark."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm import LLM, RequestOutput


def _collect_stats(llm: LLM, result: RequestOutput) -> dict:
    # SpecSamplerBase.update_requests counts acceptance before stop truncation,
    # excluding the target bonus token. PyExecutor accumulates paired verified
    # counts and publishes spec_dec_totals in _handle_responses. Missing != zero.
    totals = result.spec_dec_totals
    return {
        "request_id": result.id,
        "request_spec_dec_totals": (
            None if totals is None else {"accepted": totals[0], "drafted": totals[1]}
        ),
        # Queue batches may include earlier requests; retain IDs and raw counters.
        # Do not sum iteration snapshots into completed-request acceptance totals.
        "iteration_stats": llm.get_stats(timeout=2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=Path(__file__).with_name("prompts.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--mtp", type=int, default=0)
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument(
        "--collect-stats",
        action="store_true",
        help="Enable diagnostic iteration/request statistics; instrumented, not a perf run",
    )
    parser.add_argument("--moe-backend", choices=("CUTLASS", "CUTEDSL"), default="CUTLASS")
    args = parser.parse_args()
    if not 0 < args.max_tokens <= 1024 or not 0 <= args.mtp <= 3:
        parser.error("max-tokens must be 1..1024 and mtp 0..3 for this smoke")
    config = json.loads((args.model / "config.json").read_text())
    if config.get("language_model_only") is not True:
        parser.error("prepare a text-only checkpoint view first with prepare_model.py")
    os.environ.setdefault("LLM_MODELS_ROOT", str(args.model.parent))
    os.environ.setdefault("TLLM_LOAD_WEIGHTS_NUM_WORKERS", "1")
    os.environ.setdefault("TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL", "1")

    import torch

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig, MoeConfig, MTPDecodingConfig

    prompts = json.loads(args.prompts.read_text())
    spec_config = MTPDecodingConfig(max_draft_len=args.mtp) if args.mtp else None
    stats_kwargs = (
        {"enable_iter_perf_stats": True, "enable_iter_req_stats": True}
        if args.collect_stats
        else {}
    )
    started = time.perf_counter()
    llm = LLM(
        model=str(args.model),
        attn_backend="TRTLLM",
        moe_config=MoeConfig(backend=args.moe_backend),
        max_batch_size=2,
        max_seq_len=2048,
        max_num_tokens=512,
        kv_cache_config=KvCacheConfig(
            max_gpu_total_bytes=1 << 30,
            free_gpu_memory_fraction=0.5,
            avg_seq_len=2048,
            enable_block_reuse=False,
            mamba_ssm_cache_dtype="float32",
        ),
        cuda_graph_config=None,
        disable_overlap_scheduler=True,
        enable_autotuner=args.autotune,
        speculative_config=spec_config,
        **stats_kwargs,
    )
    load_s = time.perf_counter() - started
    records = []
    try:
        for prompt in prompts:
            started = time.perf_counter()
            result = llm.generate(prompt, SamplingParams(max_tokens=args.max_tokens, temperature=0))
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            output = result.outputs[0]
            record = {
                "prompt": prompt,
                "text": output.text,
                "token_ids": list(output.token_ids),
                "elapsed_s": elapsed,
                "output_tokens_per_s_including_prefill": len(output.token_ids) / elapsed,
            }
            if args.collect_stats:
                record["stats"] = _collect_stats(llm, result)
            records.append(record)
            print(json.dumps(record), flush=True)
    finally:
        llm.shutdown()
    report = {
        "load_s": load_s,
        "text_only": True,
        "cuda_graphs": False,
        "mtp": args.mtp,
        "moe_backend_requested": args.moe_backend,
        "autotuner": args.autotune,
        "loader_workers": os.environ["TLLM_LOAD_WEIGHTS_NUM_WORKERS"],
        "sequential_loader_requested": os.environ["TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL"],
        "kv_cache_max_gpu_bytes_requested": 1 << 30,
        "kv_cache_free_fraction_requested": 0.5,
        "kv_cache_avg_seq_len_requested": 2048,
        "max_seq_len": 2048,
        "requests": records,
    }
    if args.collect_stats:
        resolved = llm.args.speculative_config
        report["stats_collection"] = {
            "instrumented": True,
            "retrieval_in_elapsed_s": False,
            "iteration_stats_scope": "Raw queue batches, not necessarily limited to this request",
            "request_totals_source": "RequestOutput.spec_dec_totals (accepted, drafted)",
            "request_totals_semantics": (
                "Cumulative verified draft tokens; accepted excludes the target bonus token. "
                "MTP acceptance is before EOS/output-limit truncation, not emitted-token count. "
                "Null means unavailable, not zero. Iteration counters are not summed."
            ),
            "speculative_config_source": (
                "LLM.args.speculative_config after load; frontend, not worker introspection"
            ),
            "speculative_config": None if resolved is None else resolved.model_dump(mode="json"),
            "spec_dec_mode": None if resolved is None else resolved.spec_dec_mode.name,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
