# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded text-only Flash Next load and generation smoke, not a TPOT benchmark."""

import argparse
import json
import os
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=Path(__file__).with_name("prompts.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--mtp", type=int, default=0)
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--moe-backend", choices=("CUTLASS", "CUTEDSL"), default="CUTLASS")
    args = parser.parse_args()
    if not 0 < args.max_tokens <= 1024 or not 0 <= args.mtp <= 3:
        parser.error("max-tokens must be 1..1024 and mtp 0..3 for this smoke")
    config = json.loads((args.model / "config.json").read_text())
    if config.get("language_model_only") is not True:
        parser.error("prepare a text-only checkpoint view first with prepare_model.py")
    os.environ.setdefault("LLM_MODELS_ROOT", str(args.model.parent))

    import torch

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig, MoeConfig, MTPDecodingConfig

    prompts = json.loads(args.prompts.read_text())
    spec_config = MTPDecodingConfig(num_nextn_predict_layers=args.mtp) if args.mtp else None
    started = time.perf_counter()
    llm = LLM(
        model=str(args.model),
        attn_backend="TRTLLM",
        moe_config=MoeConfig(backend=args.moe_backend),
        max_batch_size=2,
        max_seq_len=2048,
        max_num_tokens=512,
        kv_cache_config=KvCacheConfig(
            max_tokens=4096,
            free_gpu_memory_fraction=0.05,
            enable_block_reuse=False,
            mamba_ssm_cache_dtype="float32",
        ),
        cuda_graph_config=None,
        disable_overlap_scheduler=True,
        enable_autotuner=args.autotune,
        speculative_config=spec_config,
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
        "max_seq_len": 2048,
        "requests": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
