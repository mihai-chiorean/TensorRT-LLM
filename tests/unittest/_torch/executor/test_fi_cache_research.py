# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-isolated persistence tests; no native TRT/FI/CUDA imports or model loads."""

from __future__ import annotations

import ast
import builtins
import importlib.util
import json
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[4]
_DIRECTORY = _ROOT / "tensorrt_llm/_torch/pyexecutor"


class FakeTuner:
    def __init__(self) -> None:
        self.profiling_cache: dict = {}
        self._file_configs: dict = {}
        self._ranked_tactics_cache: dict = {}
        self._namespaced_records: dict = {}
        self._dirty_namespaces: set = set()
        self._dirty = False
        self._dirty_seq = 0
        self._observed_cache_generations: dict = {}
        self.is_tuning_mode = False
        self._active_tuning_contexts = 0
        self.load_result = True
        self.events: list[str] = []

    def load_configs(self, path: str) -> bool:
        self.events.append("load")
        if self.load_result:
            self._file_configs = {
                key: value
                for key, value in json.loads(Path(path).read_text()).items()
                if not key.startswith("_")
            }
        return self.load_result

    def save_configs(self, path: str) -> None:
        self.events.append("save")
        assert not Path(path).exists(), "FI must receive an absent staging destination"
        Path(path).write_text(
            json.dumps(
                {
                    "_metadata": {"flashinfer_version": "0.6.18"},
                    **self._file_configs,
                    **self.profiling_cache,
                }
            )
        )


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    spec = importlib.util.spec_from_file_location(
        "fi_cache_under_test", _DIRECTORY / "_fi_cache_research.py"
    )
    assert spec is not None and spec.loader is not None
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    monkeypatch.delenv(helper._LOAD, raising=False)
    monkeypatch.setenv(helper._SAVE, str(tmp_path / "output.json"))
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Qwen4ExpForCausalLM"]}))
    tuner = FakeTuner()
    fi = ModuleType("flashinfer")
    fi.__version__ = "0.6.18"
    autotuner = ModuleType("flashinfer.autotuner")
    autotuner.AutoTuner = SimpleNamespace(get=lambda: tuner)
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(
        current_device=lambda: 0, get_device_capability=lambda device: (12, 1)
    )
    monkeypatch.setitem(sys.modules, "flashinfer", fi)
    monkeypatch.setitem(sys.modules, "flashinfer.autotuner", autotuner)
    monkeypatch.setitem(sys.modules, "torch", torch)
    args = SimpleNamespace(
        speculative_config=SimpleNamespace(decoding_type="MTP", max_draft_len=3),
        disable_overlap_scheduler=True,
        enable_autotuner=True,
        moe_config=SimpleNamespace(backend="CUTLASS", disable_finalize_fusion=True),
        mm_encoder_only=False,
        sleep_config=None,
        cache_transceiver_config=None,
        kv_connector_config=None,
        dwdp_config=None,
    )
    mapping = SimpleNamespace(rank=0, local_rank=0, world_size=1, tp_size=1, pp_size=1, cp_size=1)
    return SimpleNamespace(
        helper=helper,
        tuner=tuner,
        fi=fi,
        torch=torch,
        args=args,
        mapping=mapping,
        directory=tmp_path,
        output=tmp_path / "output.json",
        logs=[],
    )


def _begin(setup: SimpleNamespace) -> object:
    return setup.helper.begin_fi_cache_research(
        setup.args, setup.mapping, str(setup.directory), setup.logs.append
    )


def _seed(setup: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> Path:
    source = setup.directory / "seed.json"
    source.write_text(
        json.dumps(
            {
                "_metadata": {"flashinfer_version": "0.6.18"},
                "mxfp8_gemm": ["runner", {"tile": [1, 2]}],
            }
        )
    )
    monkeypatch.setenv(setup.helper._LOAD, str(source))
    return source


def test_acquire_and_single_export(setup: SimpleNamespace) -> None:
    session = _begin(setup)
    assert setup.tuner.events == []
    setup.tuner.profiling_cache["mxfp8_gemm"] = ["runner", {"tile": [3, 4]}]
    session.save()
    assert json.loads(setup.output.read_text())["mxfp8_gemm"] == ["runner", {"tile": [3, 4]}]
    assert setup.tuner.events == ["save"]
    assert "mode=acquire" in setup.logs[0]
    assert "sha256=" in setup.logs[1] and "equality unverified" in setup.logs[1]
    assert not list(setup.directory.glob(".fi-cache-*"))
    with pytest.raises(RuntimeError, match="only once"):
        session.save()
    with pytest.raises(RuntimeError, match="fresh process"):
        setup.output.unlink()
        _begin(setup)


def test_seed_is_read_only_and_new_winners_are_not_frozen(
    setup: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = _seed(setup, monkeypatch)
    original = seed.read_bytes()
    session = _begin(setup)
    setup.tuner.profiling_cache["mxfp8_gemm"] = ["runner", {"tile": [7, 8]}]
    setup.tuner.profiling_cache["new_key"] = ["runner", -1]
    session.save()
    assert seed.read_bytes() == original
    assert setup.tuner.events == ["load", "save"]
    data = json.loads(setup.output.read_text())
    assert data["mxfp8_gemm"] == ["runner", {"tile": [7, 8]}]
    assert data["new_key"] == ["runner", -1]


@pytest.mark.parametrize(
    "field",
    [
        "profiling_cache",
        "_file_configs",
        "_ranked_tactics_cache",
        "_namespaced_records",
        "_dirty_namespaces",
        "_dirty",
        "_dirty_seq",
        "_observed_cache_generations",
    ],
)
def test_nonempty_singleton_is_not_cleared(setup: SimpleNamespace, field: str) -> None:
    setattr(setup.tuner, field, {"existing": 1})
    with pytest.raises(RuntimeError, match="empty singleton"):
        _begin(setup)
    assert getattr(setup.tuner, field) == {"existing": 1}
    assert setup.tuner.events == []


@pytest.mark.parametrize("field", ["is_tuning_mode", "_active_tuning_contexts"])
def test_active_tuner_rejected_at_load_and_save(setup: SimpleNamespace, field: str) -> None:
    setattr(setup.tuner, field, 1)
    with pytest.raises(RuntimeError, match="inactive"):
        _begin(setup)
    setup.helper._STARTED = False
    setattr(setup.tuner, field, 0)
    session = _begin(setup)
    setattr(setup.tuner, field, 1)
    with pytest.raises(RuntimeError, match="inactive"):
        session.save()
    assert not setup.output.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("rank", 1),
        ("world_size", 2),
        ("tp_size", 2),
        ("pp_size", 2),
        ("cp_size", 2),
    ],
)
def test_mapping_scope(setup: SimpleNamespace, field: str, value: int) -> None:
    setattr(setup.mapping, field, value)
    with pytest.raises(ValueError, match="TP1"):
        _begin(setup)


@pytest.mark.parametrize(
    "field,value",
    [
        ("disable_overlap_scheduler", False),
        ("enable_autotuner", False),
        ("mm_encoder_only", True),
        ("sleep_config", object()),
        ("cache_transceiver_config", object()),
        ("kv_connector_config", object()),
        ("dwdp_config", object()),
        ("speculative_config", None),
    ],
)
def test_args_scope(setup: SimpleNamespace, field: str, value: object) -> None:
    setattr(setup.args, field, value)
    with pytest.raises(ValueError):
        _begin(setup)


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("speculative_config", "max_draft_len", 2),
        ("speculative_config", "decoding_type", "AUTO"),
        ("moe_config", "backend", "CUTEDSL"),
        ("moe_config", "disable_finalize_fusion", False),
    ],
)
def test_nested_scope(setup: SimpleNamespace, section: str, field: str, value: object) -> None:
    setattr(getattr(setup.args, section), field, value)
    with pytest.raises(ValueError):
        _begin(setup)


@pytest.mark.parametrize(
    "config",
    [
        [],
        {},
        {"architectures": ["Other"]},
        {"architectures": ["Qwen4ExpForCausalLM"], "text_config": {}},
    ],
)
def test_checkpoint_scope(setup: SimpleNamespace, config: object) -> None:
    (setup.directory / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="text checkpoint"):
        _begin(setup)


@pytest.mark.parametrize(
    "model_type,architecture",
    [
        ("qwen4_exp", "Qwen4ExpForConditionalGeneration"),
        ("qwen3_8_flash_next", "Qwen3_8FlashNextForConditionalGeneration"),
    ],
)
def test_composite_text_checkpoint(
    setup: SimpleNamespace, model_type: str, architecture: str
) -> None:
    config = {
        "model_type": model_type,
        "architectures": [architecture],
        "language_model_only": True,
        "text_config": {"num_hidden_layers": 48},
        "vision_config": {"depth": 27},
    }
    (setup.directory / "config.json").write_text(json.dumps(config))
    _begin(setup)


@pytest.mark.parametrize(
    "overrides",
    [
        {"language_model_only": False},
        {"language_model_only": None},
        {"language_model_only": 1},
        {"text_config": {}},
        {"text_config": None},
        {"text_config": [1]},
        {"model_type": "other"},
        {"architectures": ["Qwen4ExpForCausalLM"]},
    ],
)
def test_composite_text_checkpoint_rejects_unsupported_scope(
    setup: SimpleNamespace, overrides: dict
) -> None:
    config = {
        "model_type": "qwen3_8_flash_next",
        "architectures": ["Qwen3_8FlashNextForConditionalGeneration"],
        "language_model_only": True,
        "text_config": {"num_hidden_layers": 48},
        "vision_config": {"depth": 27},
        **overrides,
    }
    (setup.directory / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="text checkpoint"):
        _begin(setup)


@pytest.mark.parametrize("kind", ["version", "device", "sm"])
def test_runtime_scope(setup: SimpleNamespace, kind: str) -> None:
    if kind == "version":
        setup.fi.__version__ = "0.6.19"
    elif kind == "device":
        setup.torch.cuda.current_device = lambda: 1
    else:
        setup.torch.cuda.get_device_capability = lambda device: (10, 0)
    with pytest.raises(RuntimeError):
        _begin(setup)
    assert setup.tuner.events == []


@pytest.mark.parametrize("kind", ["load_only", "empty_save", "empty_load", "existing", "symlink"])
def test_output_preflight(
    setup: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    if kind == "load_only":
        _seed(setup, monkeypatch)
        monkeypatch.delenv(setup.helper._SAVE)
    elif kind == "empty_save":
        monkeypatch.setenv(setup.helper._SAVE, "")
    elif kind == "empty_load":
        monkeypatch.setenv(setup.helper._LOAD, "")
    elif kind == "existing":
        setup.output.write_text("untouched")
    else:
        setup.output.symlink_to(setup.directory / "absent")
    with pytest.raises((ValueError, FileExistsError)):
        _begin(setup)
    assert setup.tuner.events == []


@pytest.mark.parametrize("kind", ["metadata", "json", "rejected", "missing"])
def test_bad_seed(setup: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    seed = _seed(setup, monkeypatch)
    if kind == "metadata":
        seed.write_text("{}")
    elif kind == "json":
        seed.write_text("{")
    elif kind == "rejected":
        setup.tuner.load_result = False
    else:
        seed.unlink()
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        _begin(setup)
    assert not setup.output.exists()


def test_seed_change_or_device_change_prevents_save(
    setup: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = _seed(setup, monkeypatch)
    session = _begin(setup)
    setup.torch.cuda.current_device = lambda: 1
    with pytest.raises(RuntimeError, match="device changed"):
        session.save()
    setup.torch.cuda.current_device = lambda: 0
    seed.write_text('{"_metadata": {}}')
    with pytest.raises(RuntimeError, match="seed changed"):
        session.save()
    assert not setup.output.exists()


@pytest.mark.parametrize("kind", ["race", "missing", "malformed", "io"])
def test_export_failure_no_clobber_or_staging_leak(
    setup: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    session = _begin(setup)

    def save(path: str) -> None:
        if kind == "race":
            setup.output.write_text("other writer")
            Path(path).write_text('{"_metadata": {}}')
        elif kind == "malformed":
            Path(path).write_text("{")
        elif kind == "io":
            raise OSError("disk full")

    monkeypatch.setattr(setup.tuner, "save_configs", save)
    with pytest.raises((OSError, ValueError)):
        session.save()
    assert not list(setup.directory.glob(".fi-cache-*"))
    assert not session._saved
    if kind == "race":
        assert setup.output.read_text() == "other writer"
    else:
        assert not setup.output.exists()


def _creator_nodes() -> tuple[list[ast.stmt], int, int, int]:
    tree = ast.parse((_DIRECTORY / "py_executor_creator.py").read_text())
    creator = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "create_py_executor"
    )
    body = creator.body
    begin = next(
        i
        for i, node in enumerate(body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "fi_cache_session"
            for target in node.targets
        )
    )
    save = next(
        i
        for i, node in enumerate(body)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "fi_cache_session is not None"
    )
    start = next(
        i
        for i, node in enumerate(body)
        if isinstance(node, ast.Expr) and ast.unparse(node.value) == "py_executor.start_worker()"
    )
    return body, begin, save, start


@pytest.mark.parametrize(
    "enabled,error_type",
    [
        (False, None),
        (True, None),
        (True, OSError),
        (True, ValueError),
        (True, RuntimeError),
        (True, TypeError),
    ],
)
def test_actual_creator_hook_order_and_failure(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, error_type: type[Exception] | None
) -> None:
    # Execute the actual two hook blocks in isolation; native construction is
    # represented by events. Static checks below pin their lifecycle placement.
    body, begin, save, start = _creator_nodes()
    assert "MODEL_ENGINE_MAIN" in ast.unparse(body[begin + 2])
    assert any("if estimating_kv_cache" in ast.unparse(node) for node in body[begin + 2 : save])
    assert not any("create_py_executor_instance(" in ast.unparse(node) for node in body[save:])
    assert save == start + 1
    assert isinstance(body[save + 1], ast.Return)
    monkeypatch.delenv("TRTLLM_FLASHNEXT_FI_CACHE_LOAD", raising=False)
    monkeypatch.delenv("TRTLLM_FLASHNEXT_FI_CACHE_SAVE", raising=False)
    if enabled:
        monkeypatch.setenv("TRTLLM_FLASHNEXT_FI_CACHE_SAVE", "output.json")
    events: list[str] = []
    error = error_type("export failed") if error_type else None

    def export() -> None:
        events.append("save")
        if error is not None:
            raise error

    def shutdown() -> None:
        assert "start" in events
        events.append("shutdown")

    def load(*args: object) -> SimpleNamespace:
        events.append("load")
        return SimpleNamespace(save=export)

    original_import = builtins.__import__

    def tracked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "_fi_cache_research":
            events.append("import")
            return SimpleNamespace(begin_fi_cache_research=load)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", tracked_import)
    namespace = {
        "os": __import__("os"),
        "llm_args": None,
        "mapping": None,
        "checkpoint_dir": None,
        "logger": SimpleNamespace(info=lambda message: None),
        "py_executor": SimpleNamespace(
            start_worker=lambda: events.append("start"), shutdown=shutdown
        ),
    }
    exec(
        compile(ast.Module(body=body[begin : begin + 2], type_ignores=[]), "creator-entry", "exec"),
        namespace,
    )
    events.extend(["profiling_constructor", "final_constructor"])
    final = ast.parse("def finish():\n    pass\n")
    final.body[0].body = body[start : save + 2]
    exec(compile(final, "creator-exit", "exec"), namespace)
    if error_type is not None:
        with pytest.raises(error_type, match="export failed") as caught:
            namespace["finish"]()
        assert caught.value is error
    else:
        assert namespace["finish"]() is namespace["py_executor"]
        events.append("returned")
    assert events == (
        (["import", "load"] if enabled else [])
        + ["profiling_constructor", "final_constructor"]
        + ["start"]
        + (["save"] if enabled else [])
        + (["shutdown"] if error_type else ["returned"])
    )


def test_export_failure_joins_started_worker_with_actual_lifecycle_methods() -> None:
    # Run the real start/shutdown methods with a CPU event loop and managers.
    # This checks thread ownership, not native CUDA resource teardown.
    tree = ast.parse((_DIRECTORY / "py_executor.py").read_text())
    executor_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PyExecutor"
    )
    methods = [
        node
        for node in executor_class.body
        if isinstance(node, ast.FunctionDef) and node.name in ("start_worker", "shutdown")
    ]
    assert len(methods) == 2
    isolated = ast.parse("class ExecutorHarness:\n    pass\n")
    isolated.body[0].body = methods
    namespace = {
        "threading": threading,
        "AsyncWorkerMixin": type("AsyncWorkerMixin", (), {}),
        "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    }
    exec(compile(isolated, "executor-lifecycle", "exec"), namespace)
    executor = namespace["ExecutorHarness"]()
    stop = threading.Event()
    executor.shutdown_event = threading.Event()
    executor.worker_lock = threading.Lock()
    executor.worker_started = False
    executor.dist = SimpleNamespace(pp_size=1, world_size=1)
    executor.sampler = object()
    executor.hang_detector = SimpleNamespace(detected=lambda: False)
    executor.executor_request_queue = SimpleNamespace(enqueue_shutdown_request=stop.set)
    executor._shutdown_sleep_wakeup_listeners = lambda: None
    events: list[str] = []
    executor.model_engine = SimpleNamespace(_release_cuda_graphs=lambda: events.append("graphs"))
    executor.draft_model_engine = None
    executor.resource_manager = SimpleNamespace(
        resource_managers={
            "cache": SimpleNamespace(shutdown=lambda: events.append("manager")),
        }
    )
    executor.virtual_memory_pools = None
    executor.dwdp_manager = None

    def event_loop() -> None:
        stop.wait(timeout=2)
        executor.shutdown_event.set()

    executor._event_loop_wrapper = event_loop
    error = OSError("export failed")

    def export() -> None:
        assert executor.worker_started
        assert executor.worker_thread.is_alive()
        raise error

    body, _, save, start = _creator_nodes()
    final = ast.parse("def finish():\n    pass\n")
    final.body[0].body = body[start : save + 2]
    namespace.update(py_executor=executor, fi_cache_session=SimpleNamespace(save=export))
    exec(compile(final, "creator-exit", "exec"), namespace)
    try:
        with pytest.raises(OSError, match="export failed") as caught:
            namespace["finish"]()
        assert caught.value is error
        assert stop.is_set()
        assert not executor.worker_started
        assert not executor.worker_thread.is_alive()
        assert events == ["graphs", "manager"]
    finally:
        stop.set()
        if hasattr(executor, "worker_thread"):
            executor.worker_thread.join(timeout=2)
