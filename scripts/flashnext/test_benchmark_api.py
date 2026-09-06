# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from benchmark_api import main, request


def _stream(events: list[dict], *, done: bool = True) -> bytes:
    output = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
    return output + (b"data: [DONE]\n\n" if done else b"")


def test_speculative_chunks_use_usage_tokens() -> None:
    events = [
        {"choices": [{"text": "one two three"}]},
        {"choices": [{"text": " four", "finish_reason": "length"}]},
        {"choices": [], "usage": {"completion_tokens": 17, "prompt_tokens": 2}},
    ]
    with (
        patch(
            "benchmark_api.urllib.request.urlopen", return_value=io.BytesIO(_stream(events))
        ) as urlopen,
        patch("benchmark_api.time.perf_counter", side_effect=[0, 1, 3, 4]),
    ):
        result = request("http://localhost:8000", "model", "prompt", 17, True)
    assert result["output_tokens_per_s"] == 17 / 4
    assert result["amortized_decode_ms"] == 2000 / 16
    assert result["text_events"] == 2
    assert result["text"] == "one two three four"
    assert result["ttft_ms"] == 1000
    payload = json.loads(urlopen.call_args.args[0].data)
    assert payload["ignore_eos"] is True
    assert payload["stream_options"] == {"include_usage": True}


def test_single_chunk_does_not_claim_zero_decode_latency() -> None:
    events = [
        {"choices": [{"text": "multiple tokens", "finish_reason": "length"}]},
        {"usage": {"completion_tokens": 10}},
    ]
    with patch("benchmark_api.urllib.request.urlopen", return_value=io.BytesIO(_stream(events))):
        result = request("http://localhost:8000", "model", "prompt", 10, False)
    assert result["amortized_decode_ms"] is None


@pytest.mark.parametrize(
    "fault", ["missing_usage", "missing_done", "error", "bad_count", "no_text", "missing_finish"]
)
def test_invalid_stream_cannot_produce_benchmark(fault: str) -> None:
    events = [
        {"choices": [{"text": "text", "finish_reason": "length"}]},
        {"usage": {"completion_tokens": 2}},
    ]
    if fault == "missing_usage":
        events.pop()
    elif fault == "error":
        events.append({"error": "generation failed"})
    elif fault == "bad_count":
        events[-1] = {"usage": {"completion_tokens": 0}}
    elif fault == "no_text":
        events.pop(0)
    elif fault == "missing_finish":
        del events[0]["choices"][0]["finish_reason"]
    with (
        patch(
            "benchmark_api.urllib.request.urlopen",
            return_value=io.BytesIO(_stream(events, done=fault != "missing_done")),
        ),
        pytest.raises(RuntimeError),
    ):
        request("http://localhost:8000", "model", "prompt", 10, False)


def test_fixed_output_requires_requested_token_count() -> None:
    events = [
        {"choices": [{"text": "text", "finish_reason": "stop"}]},
        {"usage": {"completion_tokens": 2}},
    ]
    with (
        patch("benchmark_api.urllib.request.urlopen", return_value=io.BytesIO(_stream(events))),
        pytest.raises(RuntimeError, match="Fixed-output"),
    ):
        request("http://localhost:8000", "model", "prompt", 10, True)


def test_failed_batch_retains_results_without_success_throughput(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps(["first", "second"]))
    output = tmp_path / "result.json"
    args = [
        "benchmark_api.py",
        "--base-url",
        "http://localhost",
        "--model",
        "model",
        "--prompts",
        str(prompts),
        "--output",
        str(output),
    ]
    with (
        patch.object(sys, "argv", args),
        patch(
            "benchmark_api.request",
            side_effect=[RuntimeError("failed"), {"usage": {"completion_tokens": 2}}],
        ),
        pytest.raises(SystemExit) as exc,
    ):
        main()
    assert exc.value.code == 1
    result = json.loads(output.read_text())
    assert result["aggregate_output_tokens_per_s"] is None
    assert len(result["requests"]) == 1
    assert result["errors"] == [{"request_index": 0, "prompt": "first", "error": "failed"}]
