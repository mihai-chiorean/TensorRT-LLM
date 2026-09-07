# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only smoke CLI tests; import no TensorRT or Torch runtime."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def smoke() -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "scripts/flashnext/smoke.py"
    spec = importlib.util.spec_from_file_location("flashnext_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("totals", [None, (0, 3), (2, 6)])
def test_raw_stats_preserve_counter_semantics(smoke: ModuleType, totals: tuple | None) -> None:
    raw = [
        {"iter": 3, "specDecodingStats": {"numDraftTokens": 3, "numAcceptedTokens": 2}},
        {"iter": 4, "requestStats": [{"id": 19, "stage": "GENERATION_COMPLETE"}]},
    ]
    llm = Mock()
    llm.get_stats.return_value = raw
    result = SimpleNamespace(id=19, spec_dec_totals=totals)
    stats = smoke._collect_stats(llm, result)
    llm.get_stats.assert_called_once_with(timeout=2)
    assert stats["iteration_stats"] is raw
    assert stats["request_id"] == 19
    assert stats["request_spec_dec_totals"] == (
        None if totals is None else {"accepted": totals[0], "drafted": totals[1]}
    )


@pytest.mark.parametrize("collect", [False, True])
@pytest.mark.parametrize("mtp", [0, 3])
def test_stats_opt_in_and_timing(
    smoke: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, collect: bool, mtp: int
) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"language_model_only": True}))
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps(["first", "second"]))
    output = tmp_path / "output.json"
    argv = [
        "smoke.py",
        "--model",
        str(tmp_path),
        "--prompts",
        str(prompts),
        "--output",
        str(output),
        "--mtp",
        str(mtp),
    ]
    if collect:
        argv.append("--collect-stats")
    monkeypatch.setattr(sys, "argv", argv)
    for name in (
        "LLM_MODELS_ROOT",
        "TLLM_LOAD_WEIGHTS_NUM_WORKERS",
        "TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL",
    ):
        monkeypatch.delenv(name, raising=False)

    events = []
    clock = iter([0, 10, 20, 22, 30, 33])

    def tick() -> int:
        value = next(clock)
        events.append(("clock", value))
        return value

    monkeypatch.setattr(smoke.time, "perf_counter", tick)
    resolved = Mock()
    resolved.spec_dec_mode.name = "MTP_EAGLE_ONE_MODEL"
    resolved.model_dump.return_value = {"max_draft_len": 3, "num_nextn_predict_layers": 1}
    llm = Mock()
    llm.args.speculative_config = resolved if mtp else None
    llm.generate.side_effect = [
        SimpleNamespace(
            id=i,
            spec_dec_totals=(0, 3) if mtp else None,
            outputs=[SimpleNamespace(text="answer", token_ids=[42])],
        )
        for i in (11, 12)
    ]

    def get_stats(timeout: float) -> list[dict]:
        events.append(("stats", timeout))
        return []

    llm.get_stats.side_effect = get_stats
    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(synchronize=lambda: events.append(("sync",)))
    fake_trt = ModuleType("tensorrt_llm")
    fake_trt.LLM = Mock(return_value=llm)
    fake_trt.SamplingParams = Mock()
    fake_api = ModuleType("tensorrt_llm.llmapi")
    fake_api.KvCacheConfig = Mock()
    fake_api.MoeConfig = Mock()
    fake_api.MTPDecodingConfig = Mock()
    for name, module in (
        ("torch", fake_torch),
        ("tensorrt_llm", fake_trt),
        ("tensorrt_llm.llmapi", fake_api),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    smoke.main()

    kwargs = fake_trt.LLM.call_args.kwargs
    assert {key: value for key, value in kwargs.items() if key.startswith("enable_iter_")} == (
        {"enable_iter_perf_stats": True, "enable_iter_req_stats": True} if collect else {}
    )
    assert kwargs["cuda_graph_config"] is None
    assert kwargs["disable_overlap_scheduler"] is True
    assert kwargs["enable_autotuner"] is False
    assert (kwargs["max_batch_size"], kwargs["max_seq_len"], kwargs["max_num_tokens"]) == (
        2,
        2048,
        512,
    )
    fake_api.MoeConfig.assert_called_once_with(backend="CUTLASS")
    fake_api.KvCacheConfig.assert_called_once_with(
        max_gpu_total_bytes=1 << 30,
        free_gpu_memory_fraction=0.5,
        avg_seq_len=2048,
        enable_block_reuse=False,
        mamba_ssm_cache_dtype="float32",
    )
    report = json.loads(output.read_text())
    assert report["load_s"] == 10
    assert [r["elapsed_s"] for r in report["requests"]] == [2, 3]
    assert ("stats_collection" in report) is collect
    assert all(("stats" in r) is collect for r in report["requests"])
    llm.shutdown.assert_called_once()
    if collect:
        assert events == [
            ("clock", 0),
            ("clock", 10),
            ("clock", 20),
            ("sync",),
            ("clock", 22),
            ("stats", 2),
            ("clock", 30),
            ("sync",),
            ("clock", 33),
            ("stats", 2),
        ]
        evidence = report["stats_collection"]
        assert evidence["retrieval_in_elapsed_s"] is False
        assert evidence["instrumented"] is True
        assert evidence["spec_dec_mode"] == ("MTP_EAGLE_ONE_MODEL" if mtp else None)
        assert evidence["speculative_config"] == (resolved.model_dump.return_value if mtp else None)
    else:
        llm.get_stats.assert_not_called()
        resolved.model_dump.assert_not_called()
