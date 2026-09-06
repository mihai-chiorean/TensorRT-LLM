# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from score_quality import load_suite, main, score_report


@pytest.fixture
def suite() -> dict:
    return load_suite()


@pytest.fixture
def report(suite: dict) -> dict:
    return {
        "ignore_eos": False,
        "max_tokens": 1024,
        "errors": [],
        "requests": [
            {
                "prompt": case["prompt"],
                "text": json.dumps(suite["expected_answers"][case["id"]])
                if case["answer_format"] == "json"
                else "def stable_unique(values):\n    return list(dict.fromkeys(values))",
                "usage": {"prompt_tokens": 40, "completion_tokens": 20, "total_tokens": 60},
                "finish_reason": "stop",
            }
            for case in suite["cases"]
        ],
    }


def _case(result: dict, case_id: str = "math_speed") -> dict:
    return next(case for case in result["cases"] if case["id"] == case_id)


def test_seven_json_passes_are_not_eight_case_pass(suite: dict, report: dict) -> None:
    report["requests"].reverse()
    result = score_report(suite, report)
    assert result["status"] == "incomplete"
    assert result["total_cases"] == 8
    assert result["totals"] == {"passed": 7, "failed": 0, "unscored": 1}
    assert not result["report_errors"]
    assert _case(result, "code_stable_unique")["status"] == "unscored"


@pytest.mark.parametrize(
    "text",
    [
        '  {"speed_kmh":80} \n',
        '{"speed_kmh":80.0}',
        '{"speed_kmh":8e1}',
        '<think>I first considered 79, then checked the units.</think>\n{"speed_kmh":80}',
        '  <think></think> {"speed_kmh":80}  ',
    ],
)
def test_exact_numeric_answer_and_single_leading_reasoning(
    suite: dict,
    report: dict,
    text: str,
) -> None:
    report["requests"][0]["text"] = text
    assert _case(score_report(suite, report))["status"] == "passed"


@pytest.mark.parametrize(
    "text",
    [
        '<think>{"speed_kmh":80}</think>{"speed_kmh":79}',
        '<think>{"speed_kmh":80}</think>',
        '<think>{"speed_kmh":80}',
        'reasoning</think>{"speed_kmh":80}',
        '<think>one</think><think>two</think>{"speed_kmh":80}',
        '<think><think>nested</think></think>{"speed_kmh":80}',
        'prefix<think>reasoning</think>{"speed_kmh":80}',
        '<think>reasoning</think>{"speed_kmh":80}</think>',
        '<THINK>reasoning</THINK>{"speed_kmh":80}',
        '<think >reasoning</think>{"speed_kmh":80}',
        '<think>reasoning</think >{"speed_kmh":80}',
        '<think>malformed <think</think>{"speed_kmh":80}',
        '<think>reasoning</think>Final answer: {"speed_kmh":80}',
    ],
)
def test_malformed_or_ambiguous_reasoning_never_matches_inner_answer(
    suite: dict,
    report: dict,
    text: str,
) -> None:
    report["requests"][0]["text"] = text
    assert _case(score_report(suite, report))["status"] == "failed"


@pytest.mark.parametrize(
    "text",
    [
        '{"speed_kmh":79}',
        '{"speed_kmh":true}',
        '{"speed_kmh":"80"}',
        '{"speed_kmh":null}',
        '{"speed_kmh":[80]}',
        '{"speed_kmh":{}}',
        '{"speed_kmh":80,"extra":1}',
        "{}",
        '[{"speed_kmh":80}]',
        '{"speed_kmh":80,"speed_kmh":80}',
        '{"speed_kmh":79,"speed_kmh":80}',
        '{"speed_kmh":NaN}',
        '{"speed_kmh":Infinity}',
        '{"speed_kmh":-Infinity}',
        '{"speed_kmh":1e9999}',
        '{"speed_kmh":1e9999999999999999999999999}',
        '{"speed_kmh":80.00000000000000001}',
        '{"speed_kmh":80} {"speed_kmh":80}',
        '```json\n{"speed_kmh":80}\n```',
        'Answer: {"speed_kmh":80}',
        '{"speed_kmh":80} trailing',
        "",
        "   ",
    ],
)
def test_strict_json_rejects_wrong_values_types_keys_and_extra_content(
    suite: dict,
    report: dict,
    text: str,
) -> None:
    report["requests"][0]["text"] = text
    assert _case(score_report(suite, report))["status"] == "failed"


@pytest.mark.parametrize(
    "index,text",
    [
        (1, '{"full_packs":13.0,"loose_pencils":4}'),
        (1, '{"full_packs":13,"loose_pencils":true}'),
        (2, '{"ids":["a04","a02"],"total":23}'),
        (3, '{"values":[-2,0.0,4,7]}'),
        (4, '{"result":[[1,true],[3,2]]}'),
        (6, '{"first_depot":"west","middle_crates":13,"last_depot":"north"}'),
        (7, '{"cedar":"Dee","maple":"Ana","oak":"null"}'),
    ],
)
def test_other_json_cases_have_exact_types_array_order_and_wrong_answer_controls(
    suite: dict,
    report: dict,
    index: int,
    text: str,
) -> None:
    report["requests"][index]["text"] = text
    result = score_report(suite, report)
    assert result["cases"][index]["status"] == "failed"
    assert result["totals"] == {"passed": 6, "failed": 1, "unscored": 1}


@pytest.mark.parametrize("fault", ["duplicate", "missing", "changed_prompt", "unknown"])
@pytest.mark.parametrize("index", [0, 5])
def test_exact_prompt_matching_rejects_ambiguous_and_missing_records(
    suite: dict,
    report: dict,
    fault: str,
    index: int,
) -> None:
    if fault == "duplicate":
        report["requests"].append(copy.deepcopy(report["requests"][index]))
    elif fault == "missing":
        report["requests"].pop(index)
    elif fault == "changed_prompt":
        report["requests"][index]["prompt"] += " "
    else:
        report["requests"].append({"prompt": "unknown", "text": "anything"})
    result = score_report(suite, report)
    assert result["status"] == "failed"
    if fault != "unknown":
        assert result["cases"][index]["status"] == "failed"
    else:
        assert result["report_errors"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("finish_reason", "length"),
        ("finish_reason", None),
        ("finish_reason", "content_filter"),
        ("finish_reason", "error"),
        ("usage", None),
        ("usage", {}),
        ("text", None),
        ("error", "server failed"),
    ],
)
def test_protocol_errors_fail_even_with_correct_answer(
    suite: dict,
    report: dict,
    field: str,
    value: object,
) -> None:
    report["requests"][0][field] = value
    assert _case(score_report(suite, report))["status"] == "failed"


@pytest.mark.parametrize(
    "field,value",
    [
        ("prompt_tokens", True),
        ("prompt_tokens", 0),
        ("prompt_tokens", -1),
        ("completion_tokens", False),
        ("completion_tokens", 0),
        ("completion_tokens", 2.5),
        ("completion_tokens", "20"),
        ("completion_tokens", 1025),
        ("total_tokens", True),
        ("total_tokens", 61),
    ],
)
def test_invalid_usage_is_not_quality_evidence(
    suite: dict,
    report: dict,
    field: str,
    value: object,
) -> None:
    report["requests"][0]["usage"][field] = value
    assert _case(score_report(suite, report))["status"] == "failed"


def test_missing_optional_total_and_object_key_order(suite: dict, report: dict) -> None:
    del report["requests"][1]["usage"]["total_tokens"]
    report["requests"][1]["text"] = '{"loose_pencils":4,"full_packs":13}'
    assert score_report(suite, report)["totals"]["passed"] == 7


def test_report_errors_cannot_be_hidden_by_a_success(suite: dict, report: dict) -> None:
    report["errors"] = [{"prompt": report["requests"][0]["prompt"], "error": "timeout"}]
    result = score_report(suite, report)
    assert result["status"] == "failed" and result["report_errors"]
    assert _case(result)["status"] == "failed"


@pytest.mark.parametrize("record", [None, [], {"prompt": 123}])
def test_malformed_extra_request_fails_report(suite: dict, report: dict, record: object) -> None:
    report["requests"].append(record)
    result = score_report(suite, report)
    assert result["status"] == "failed" and result["report_errors"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("ignore_eos", True),
        ("ignore_eos", None),
        ("ignore_eos", 0),
        ("max_tokens", True),
        ("max_tokens", 0),
        ("requests", {}),
        ("errors", None),
    ],
)
def test_invalid_report_metadata_rejected(
    suite: dict,
    report: dict,
    field: str,
    value: object,
) -> None:
    report[field] = value
    with pytest.raises(ValueError):
        score_report(suite, report)


def test_python_is_never_executed_or_claimed_correct(suite: dict, report: dict) -> None:
    report["requests"][5]["text"] = 'raise RuntimeError("DO NOT EXECUTE THIS")'
    with (
        patch("builtins.exec", side_effect=AssertionError("generated code executed")),
        patch("builtins.eval", side_effect=AssertionError("generated code evaluated")),
    ):
        result = score_report(suite, report)
    assert _case(result, "code_stable_unique")["status"] == "unscored"
    assert result["status"] == "incomplete"
    report["requests"][5]["finish_reason"] = "length"
    assert _case(score_report(suite, report), "code_stable_unique")["status"] == "failed"


@pytest.mark.parametrize("fault", ["duplicate_id", "duplicate_prompt", "missing_answer", "format"])
def test_invalid_suite_rejected(suite: dict, tmp_path: Path, fault: str) -> None:
    if fault == "duplicate_id":
        suite["cases"][1]["id"] = suite["cases"][0]["id"]
    elif fault == "duplicate_prompt":
        suite["cases"][1]["prompt"] = suite["cases"][0]["prompt"]
    elif fault == "missing_answer":
        del suite["expected_answers"]["math_speed"]
    else:
        suite["cases"][0]["answer_format"] = "unknown"
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(suite))
    with pytest.raises(ValueError):
        load_suite(path)


def test_cli_exports_exact_prompts_without_answers(suite: dict, tmp_path: Path) -> None:
    path = tmp_path / "prompts.json"
    assert main(["--prompts-output", str(path)]) == 0
    assert json.loads(path.read_text()) == [case["prompt"] for case in suite["cases"]]


def test_cli_distinct_failed_and_manual_pending_exits(
    report: dict,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    path, output = tmp_path / "report.json", tmp_path / "score.json"
    path.write_text(json.dumps(report))
    args = ["--report", str(path), "--output", str(output)]
    assert main(args) == 3
    result = json.loads(capsys.readouterr().out)
    assert result == json.loads(output.read_text())
    assert result["status"] == "incomplete" and result["totals"]["passed"] == 7
    report["requests"][0]["text"] = '{"speed_kmh":79}'
    path.write_text(json.dumps(report))
    assert main(args) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


@pytest.mark.parametrize("raw", ['{"requests":[],"requests":[]}', '{"x":NaN}', "[]", "{"])
def test_cli_rejects_malformed_report_json(tmp_path: Path, raw: str) -> None:
    path = tmp_path / "report.json"
    path.write_text(raw)
    with pytest.raises(SystemExit) as error:
        main(["--report", str(path)])
    assert error.value.code == 2


def test_cli_requires_mode_and_preserves_input_files(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 2
    path = tmp_path / "cases.json"
    original = json.dumps(load_suite())
    path.write_text(original)
    with pytest.raises(SystemExit) as error:
        main(["--cases", str(path), "--prompts-output", str(path)])
    assert error.value.code == 2
    assert path.read_text() == original
