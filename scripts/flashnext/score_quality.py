# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Score saved benchmark_api reports offline, or export quality prompts only.

Final-answer rule: trim whitespace; optionally remove exactly one leading,
properly closed <think>...</think> block. Nested, repeated, misplaced, malformed,
or implicit-opening reasoning blocks fail. Never search reasoning for answers.
Python cases remain unscored/manual pending; generated code is never executed.

Scoring exit codes: 0 = all cases passed, 1 = failures/report errors,
2 = invalid input/CLI, 3 = no failures but manual scoring remains pending.
--prompts-output alone exports a JSON string array and performs no scoring.
The saved report does not retain raw SSE framing: this scorer checks saved
evidence, relying on benchmark_api's validation of stream completion.
"""

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

_DEFAULT_CASES = Path(__file__).with_name("quality_cases.json")
_THINK_MARKER = re.compile(r"<\s*/?\s*think\b", re.IGNORECASE)


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _strict_json(text: str) -> object:
    # Decimal preserves exact numeric comparisons, including values close to 80.
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=Decimal,
        )
    except InvalidOperation as error:
        raise ValueError("Invalid JSON numeric exponent") from error


def load_suite(path: Path = _DEFAULT_CASES) -> dict:
    """Load and validate the case/answer mapping without rendering prompts."""
    suite = _strict_json(path.read_text(encoding="utf-8"))
    if not isinstance(suite, dict) or type(suite.get("schema_version")) is not int:
        raise ValueError("Quality suite requires an integer schema_version")
    if suite["schema_version"] != 1:
        raise ValueError("Unsupported quality suite schema_version")
    cases, answers = suite.get("cases"), suite.get("expected_answers")
    if not isinstance(cases, list) or not cases or not isinstance(answers, dict):
        raise ValueError("Quality suite requires cases and expected_answers")
    ids, prompts = set(), set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Quality cases must be objects")
        case_id, prompt = case.get("id"), case.get("prompt")
        if not isinstance(case_id, str) or not case_id or case_id in ids:
            raise ValueError("Quality case IDs must be nonempty and unique")
        if not isinstance(prompt, str) or not prompt.strip() or prompt in prompts:
            raise ValueError("Quality prompts must be nonempty and unique")
        if case.get("answer_format") not in ("json", "python"):
            raise ValueError(f"Unsupported answer format for {case_id}")
        if not isinstance(answers.get(case_id), dict):
            raise ValueError(f"Missing or invalid expected answer for {case_id}")
        ids.add(case_id)
        prompts.add(prompt)
    if set(answers) != ids:
        raise ValueError("Expected answers must match case IDs exactly")
    return suite


def _final_answer(text: str) -> str:
    text = text.strip()
    markers = list(_THINK_MARKER.finditer(text))
    if markers:
        close = text.find("</think>", len("<think>"))
        if (
            len(markers) != 2
            or not text.startswith("<think>")
            or close < 0
            or markers[1].start() != close
        ):
            raise ValueError("Malformed or ambiguous reasoning block")
        text = text[close + len("</think>") :].strip()
    if not text:
        raise ValueError("Missing final answer")
    return text


def _matches(actual: object, expected: object, *, numeric_speed: bool = False) -> bool:
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and all(
                _matches(actual[key], value, numeric_speed=numeric_speed and key == "speed_kmh")
                for key, value in expected.items()
            )
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_matches(a, e) for a, e in zip(actual, expected))
        )
    if numeric_speed:
        return (
            type(actual) in (int, Decimal)
            and (not isinstance(actual, Decimal) or actual.is_finite())
            and actual == expected
        )
    return type(actual) is type(expected) and actual == expected


def _validate_record(record: dict, max_tokens: int) -> None:
    if "error" in record:
        raise ValueError("Request contains a protocol error")
    if record.get("finish_reason") != "stop":
        raise ValueError("Completion must finish with stop, not length or another reason")
    if not isinstance(record.get("text"), str):
        raise ValueError("Missing or invalid completion text")
    usage = record.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("Missing usage")
    for name in ("prompt_tokens", "completion_tokens"):
        if type(usage.get(name)) is not int or usage[name] <= 0:
            raise ValueError(f"Invalid usage.{name}")
    if usage["completion_tokens"] > max_tokens:
        raise ValueError("Completion count exceeds the requested max_tokens")
    if "total_tokens" in usage and (
        type(usage["total_tokens"]) is not int
        or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]
    ):
        raise ValueError("Inconsistent usage.total_tokens")


def score_report(suite: dict, report: object) -> dict:
    """Score exact-prompt records against a validated suite from load_suite."""
    if not isinstance(report, dict):
        raise ValueError("Benchmark report must be an object")
    records, errors = report.get("requests"), report.get("errors")
    if not isinstance(records, list) or not isinstance(errors, list):
        raise ValueError("Benchmark report requires requests and errors arrays")
    if report.get("ignore_eos") is not False:
        raise ValueError("Quality scoring requires ignore_eos=false")
    max_tokens = report.get("max_tokens")
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("Benchmark report requires a positive integer max_tokens")
    by_prompt = {case["prompt"]: [] for case in suite["cases"]}
    report_errors = ["Benchmark report contains request errors"] if errors else []
    error_prompts = {
        error.get("prompt")
        for error in errors
        if isinstance(error, dict) and isinstance(error.get("prompt"), str)
    }
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get("prompt"), str):
            report_errors.append(f"Invalid request record at index {index}")
        elif record["prompt"] not in by_prompt:
            report_errors.append(f"Unknown exact prompt at request index {index}")
        else:
            by_prompt[record["prompt"]].append(record)

    results = []
    totals = {"passed": 0, "failed": 0, "unscored": 0}
    for case in suite["cases"]:
        matches = by_prompt[case["prompt"]]
        status, reason = "failed", "Missing exact-prompt request record"
        if len(matches) > 1:
            reason = "Duplicate exact-prompt request records"
        elif case["prompt"] in error_prompts:
            reason = "Benchmark report contains an error for this prompt"
        elif len(matches) == 1:
            try:
                _validate_record(matches[0], max_tokens)
                answer = _final_answer(matches[0]["text"])
                if case["answer_format"] == "python":
                    status, reason = "unscored", "Manual Python review pending; code not executed"
                else:
                    actual = _strict_json(answer)
                    expected = suite["expected_answers"][case["id"]]
                    if _matches(actual, expected, numeric_speed=case["id"] == "math_speed"):
                        status, reason = "passed", "Exact final JSON answer matches"
                    else:
                        reason = "Final JSON answer has wrong values, types, or keys"
            except (ValueError, RecursionError) as error:
                reason = str(error)
        totals[status] += 1
        results.append({"id": case["id"], "status": status, "reason": reason})
    status = (
        "failed"
        if totals["failed"] or report_errors
        else ("incomplete" if totals["unscored"] else "passed")
    )
    return {
        "status": status,
        "total_cases": len(results),
        "totals": totals,
        "report_errors": report_errors,
        "cases": results,
        "reasoning_policy": "optional-single-leading-think-block-v1",
        "python_policy": "manual-pending-no-execution",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=_DEFAULT_CASES)
    parser.add_argument("--report", type=Path, help="Saved benchmark_api JSON report to score")
    parser.add_argument("--output", type=Path, help="Also write the scoring result to this file")
    parser.add_argument(
        "--prompts-output",
        type=Path,
        help="Export only case prompts as a JSON string array; usable without --report",
    )
    args = parser.parse_args(argv)
    if args.report is None and (args.prompts_output is None or args.output is not None):
        parser.error("Provide --report for scoring or --prompts-output for export only")
    inputs = {path.resolve() for path in (args.cases, args.report) if path is not None}
    outputs = [path.resolve() for path in (args.output, args.prompts_output) if path is not None]
    if inputs.intersection(outputs) or len(set(outputs)) != len(outputs):
        parser.error("Output paths must be distinct and must not overwrite an input")
    try:
        suite = load_suite(args.cases)
        result = None
        if args.report is not None:
            result = score_report(suite, _strict_json(args.report.read_text(encoding="utf-8")))
        if args.prompts_output is not None:
            args.prompts_output.parent.mkdir(parents=True, exist_ok=True)
            args.prompts_output.write_text(
                json.dumps([case["prompt"] for case in suite["cases"]], indent=2) + "\n",
                encoding="utf-8",
            )
        if result is not None:
            encoded = json.dumps(result, indent=2) + "\n"
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(encoded, encoding="utf-8")
            print(encoded, end="")
            return {"passed": 0, "failed": 1, "incomplete": 3}[result["status"]]
    except (OSError, ValueError, RecursionError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
