# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pool repeated benchmark_api reports for one fixed benchmark configuration."""

import argparse
import hashlib
import json
import math
from pathlib import Path

_CAVEAT = (
    "This summary pools repetitions for one deployment configuration; it is not an "
    "engine comparison and does not infer active capacity from client concurrency."
)


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"Non-finite JSON value is not allowed: {value}")


def _read_report(path: Path) -> tuple[dict, str]:
    """Read one strictly JSON-encoded benchmark report.

    Args:
        path: Report emitted by ``benchmark_api.py``.

    Returns:
        The parsed report object and SHA-256 digest of its parsed bytes.

    Raises:
        ValueError: If the file is not a JSON object or contains non-finite values.
    """
    contents = path.read_bytes()
    report = json.loads(contents, parse_constant=_reject_nonfinite)
    if not isinstance(report, dict):
        raise ValueError(f"{path}: benchmark report must be a JSON object")
    return report, hashlib.sha256(contents).hexdigest()


def _positive_int(value: object, name: str, path: Path) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{path}: {name} must be a positive integer")
    return value


def _positive_finite(value: object, name: str, path: Path) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{path}: {name} must be a positive finite number")
    return float(value)


def _validate_report(report: dict, path: Path) -> tuple[list[str], int, int, float]:
    """Validate a complete raw report and return its prompt and token totals."""
    model = report.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError(f"{path}: model must be a nonempty string")
    concurrency = _positive_int(report.get("concurrency"), "concurrency", path)
    effective_max_concurrency = _positive_int(
        report.get("effective_max_concurrency"), "effective_max_concurrency", path
    )
    max_tokens = _positive_int(report.get("max_tokens"), "max_tokens", path)
    if type(report.get("ignore_eos")) is not bool:
        raise ValueError(f"{path}: ignore_eos must be a boolean")
    errors = report.get("errors")
    if not isinstance(errors, list) or errors:
        raise ValueError(f"{path}: errors must be an empty array")
    requests = report.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError(f"{path}: requests must be a nonempty array")
    if effective_max_concurrency != min(concurrency, len(requests)):
        raise ValueError(f"{path}: effective_max_concurrency does not match the request count")

    prompts: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    for index, request in enumerate(requests):
        if not isinstance(request, dict):
            raise ValueError(f"{path}: request {index} must be an object")
        prompt = request.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError(f"{path}: request {index} has an invalid prompt")
        if not isinstance(request.get("text"), str) or not request["text"]:
            raise ValueError(f"{path}: request {index} is missing completion text")
        finish_reason = request.get("finish_reason")
        if finish_reason not in ("stop", "length"):
            raise ValueError(f"{path}: request {index} has an invalid finish_reason")
        usage = request.get("usage")
        if not isinstance(usage, dict):
            raise ValueError(f"{path}: request {index} is missing usage")
        request_prompt_tokens = _positive_int(
            usage.get("prompt_tokens"), f"request {index} usage.prompt_tokens", path
        )
        request_completion_tokens = _positive_int(
            usage.get("completion_tokens"), f"request {index} usage.completion_tokens", path
        )
        if request_completion_tokens > max_tokens or (
            report["ignore_eos"] and request_completion_tokens != max_tokens
        ):
            raise ValueError(f"{path}: request {index} has an invalid completion token count")
        if report["ignore_eos"] and finish_reason != "length":
            raise ValueError(
                f"{path}: request {index} must finish with length when ignore_eos is true"
            )
        total_tokens = usage.get("total_tokens")
        if total_tokens is not None and (
            type(total_tokens) is not int
            or total_tokens != request_prompt_tokens + request_completion_tokens
        ):
            raise ValueError(f"{path}: request {index} has inconsistent usage.total_tokens")
        prompts.append(prompt)
        prompt_tokens += request_prompt_tokens
        completion_tokens += request_completion_tokens
    return (
        prompts,
        prompt_tokens,
        completion_tokens,
        _positive_finite(report.get("elapsed_s"), "elapsed_s", path),
    )


def summarize_reports(paths: list[Path]) -> dict:
    """Summarize two or more compatible ``benchmark_api.py`` repetitions.

    Args:
        paths: Paths to raw reports for a single model and client configuration.

    Returns:
        JSON-serializable pooled throughput evidence.

    Raises:
        ValueError: If reports are invalid or do not describe the same configuration.
    """
    if len(paths) < 2:
        raise ValueError("Provide at least two benchmark reports")
    resolved_paths = [path.resolve() for path in paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("Benchmark report paths must be distinct")

    first_report, first_digest = _read_report(paths[0])
    prompts, prompt_tokens, completion_tokens, elapsed_s = _validate_report(first_report, paths[0])
    settings = {
        key: first_report[key]
        for key in ("model", "concurrency", "effective_max_concurrency", "ignore_eos", "max_tokens")
    }
    input_files = []
    per_round = []
    text_matches = []
    total_prompt_tokens = prompt_tokens
    total_completion_tokens = completion_tokens
    total_elapsed_s = elapsed_s
    first_texts = [request["text"] for request in first_report["requests"]]

    for index, path in enumerate(paths):
        report, digest = (first_report, first_digest) if index == 0 else _read_report(path)
        if index:
            round_prompts, round_prompt_tokens, round_completion_tokens, round_elapsed_s = (
                _validate_report(report, path)
            )
            if round_prompts != prompts:
                raise ValueError(f"{path}: request prompts differ from the first report")
            for key, expected in settings.items():
                if report.get(key) != expected:
                    raise ValueError(f"{path}: {key} differs from the first report")
            total_prompt_tokens += round_prompt_tokens
            total_completion_tokens += round_completion_tokens
            total_elapsed_s += round_elapsed_s
        else:
            round_completion_tokens, round_elapsed_s = completion_tokens, elapsed_s
        texts = [request["text"] for request in report["requests"]]
        input_files.append(
            {
                "path": str(path.resolve()),
                "sha256": digest,
            }
        )
        per_round.append(round_completion_tokens / round_elapsed_s)
        text_matches.append(sum(text == first for text, first in zip(texts, first_texts)))

    return {
        "caveat": _CAVEAT,
        **settings,
        "repetitions": len(paths),
        "requests_per_round": len(prompts),
        "total_requests": len(prompts) * len(paths),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_walltime_s": total_elapsed_s,
        "pooled_output_tokens_per_s": total_completion_tokens / total_elapsed_s,
        "per_round_output_tokens_per_s": per_round,
        "repeat_exact_text_matches_vs_first": text_matches,
        "input_files": input_files,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark repetition summarizer CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path, help="Raw benchmark_api JSON reports")
    parser.add_argument("--output", type=Path, help="Write the summary JSON to this file")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.resolve() in {
        path.resolve() for path in args.reports
    }:
        parser.error("Output path must not overwrite an input report")
    try:
        encoded = json.dumps(summarize_reports(args.reports), indent=2) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
