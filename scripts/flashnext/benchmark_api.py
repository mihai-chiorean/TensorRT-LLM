# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure raw-completion API latency using server-reported token counts."""

import argparse
import concurrent.futures
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def request(base_url: str, model: str, prompt: str, max_tokens: int, ignore_eos: bool) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": ignore_eos,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first = last = None
    usage = None
    text = []
    finish_reason = None
    done = False
    with urllib.request.urlopen(req, timeout=300) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(f"Generation failed: {event['error']}")
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                chunk = choice.get("text", "")
                if chunk:
                    last = time.perf_counter()
                    if first is None:
                        first = last
                    text.append(chunk)
    elapsed = time.perf_counter() - start
    if not done or usage is None or first is None or last is None or finish_reason is None:
        raise RuntimeError(
            "Incomplete response, missing usage/finish reason, or no text; cannot report throughput"
        )
    tokens = usage["completion_tokens"]
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
        raise RuntimeError("Invalid completion token count; cannot report throughput")
    if ignore_eos and tokens != max_tokens:
        raise RuntimeError(f"Fixed-output request produced {tokens} tokens, expected {max_tokens}")
    return {
        "prompt": prompt,
        "text": "".join(text),
        "usage": usage,
        "finish_reason": finish_reason,
        "elapsed_s": elapsed,
        "ttft_ms": (first - start) * 1000,
        "output_tokens_per_s": tokens / elapsed,
        # Speculative decoding emits several tokens per event. This is an
        # amortized decode estimate, not a per-token arrival distribution.
        "amortized_decode_ms": (last - first) * 1000 / (tokens - 1)
        if tokens > 1 and len(text) > 1
        else None,
        "text_events": len(text),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--prompts", type=Path, required=True, help="JSON array of raw prompt strings"
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prompts = json.loads(args.prompts.read_text())
    if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) for p in prompts):
        parser.error("prompts must be a nonempty JSON array of strings")
    if args.max_tokens <= 0 or args.concurrency <= 0:
        parser.error("max-tokens and concurrency must be positive")
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                request, args.base_url, args.model, prompt, args.max_tokens, args.ignore_eos
            )
            for prompt in prompts
        ]
        results = []
        errors = []
        for index, future in enumerate(futures):
            try:
                results.append(future.result())
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                RuntimeError,
                ValueError,
                KeyError,
            ) as error:
                errors.append(
                    {"request_index": index, "prompt": prompts[index], "error": str(error)}
                )
    elapsed = time.perf_counter() - started
    report = {
        "model": args.model,
        "concurrency": args.concurrency,
        "effective_max_concurrency": min(args.concurrency, len(prompts)),
        "max_tokens": args.max_tokens,
        "ignore_eos": args.ignore_eos,
        "elapsed_s": elapsed,
        "aggregate_output_tokens_per_s": sum(r["usage"]["completion_tokens"] for r in results)
        / elapsed
        if not errors
        else None,
        "errors": errors,
        "requests": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
