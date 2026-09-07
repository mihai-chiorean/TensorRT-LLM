# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path
from typing import Callable

import pytest
from summarize_benchmark import summarize_reports


def _report(*, elapsed_s: float = 10.0) -> dict:
    return {
        "model": "flashnext-trt",
        "concurrency": 2,
        "effective_max_concurrency": 2,
        "max_tokens": 128,
        "ignore_eos": True,
        "elapsed_s": elapsed_s,
        "errors": [],
        "requests": [
            {
                "prompt": "first",
                "text": "first answer",
                "usage": {"prompt_tokens": 3, "completion_tokens": 128, "total_tokens": 131},
                "finish_reason": "length",
            },
            {
                "prompt": "second",
                "text": "second answer",
                "usage": {"prompt_tokens": 4, "completion_tokens": 128, "total_tokens": 132},
                "finish_reason": "length",
            },
        ],
    }


def _write_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report), encoding="utf-8")


def test_pools_token_counts_over_unequal_durations_and_hashes_inputs(tmp_path: Path) -> None:
    first = _report(elapsed_s=10.0)
    second = _report(elapsed_s=20.0)
    second["requests"][1]["text"] = "different answer"
    first_path, second_path = tmp_path / "first.json", tmp_path / "second.json"
    _write_report(first_path, first)
    _write_report(second_path, second)

    result = summarize_reports([first_path, second_path])

    assert result["total_requests"] == 4
    assert result["total_completion_tokens"] == 512
    assert result["total_walltime_s"] == 30.0
    assert result["pooled_output_tokens_per_s"] == 512 / 30
    assert result["per_round_output_tokens_per_s"] == [25.6, 12.8]
    assert result["repeat_exact_text_matches_vs_first"] == [2, 1]
    assert result["input_files"][0]["sha256"] == hashlib.sha256(first_path.read_bytes()).hexdigest()
    assert "not an engine comparison" in result["caveat"]
    assert "active capacity" in result["caveat"]


def test_rejects_reordered_prompts_and_mismatched_settings(tmp_path: Path) -> None:
    first_path, second_path = tmp_path / "first.json", tmp_path / "second.json"
    _write_report(first_path, _report())
    reordered = _report()
    reordered["requests"].reverse()
    _write_report(second_path, reordered)
    with pytest.raises(ValueError, match="prompts differ"):
        summarize_reports([first_path, second_path])

    for field, value in (("ignore_eos", False), ("max_tokens", 64)):
        mismatch = _report()
        mismatch[field] = value
        if field == "max_tokens":
            for request in mismatch["requests"]:
                request["usage"]["completion_tokens"] = value
                request["usage"]["total_tokens"] = request["usage"]["prompt_tokens"] + value
        _write_report(second_path, mismatch)
        with pytest.raises(ValueError, match=f"{field} differs"):
            summarize_reports([first_path, second_path])


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda report: report.update(errors=[{"error": "failed"}]),
            "errors must be an empty array",
        ),
        (lambda report: report["requests"][0].pop("usage"), "missing usage"),
        (lambda report: report.update(elapsed_s=float("nan")), "Non-finite JSON value"),
        (lambda report: report.update(elapsed_s=0), "positive finite"),
        (lambda report: report.update(elapsed_s=float("inf")), "Non-finite JSON value"),
        (lambda report: report["requests"][0].update(text=""), "missing completion text"),
        (
            lambda report: report["requests"][0]["usage"].update(completion_tokens=0),
            "positive integer",
        ),
        (
            lambda report: report["requests"][0]["usage"].update(completion_tokens=True),
            "positive integer",
        ),
        (
            lambda report: report["requests"][0]["usage"].update(total_tokens=999),
            "inconsistent usage.total_tokens",
        ),
        (lambda report: report["requests"][0].update(finish_reason="other"), "finish_reason"),
        (
            lambda report: report["requests"][0]["usage"].update(completion_tokens=127),
            "invalid completion token count",
        ),
    ],
)
def test_rejects_invalid_raw_evidence(
    tmp_path: Path, mutate: Callable[[dict], None], match: str
) -> None:
    first_path, invalid_path = tmp_path / "first.json", tmp_path / "invalid.json"
    _write_report(first_path, _report())
    invalid = _report()
    mutate(invalid)
    _write_report(invalid_path, invalid)
    with pytest.raises(ValueError, match=match):
        summarize_reports([first_path, invalid_path])


def test_rejects_duplicate_resolved_paths(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    _write_report(report_path, _report())

    with pytest.raises(ValueError, match="paths must be distinct"):
        summarize_reports([report_path, report_path])
