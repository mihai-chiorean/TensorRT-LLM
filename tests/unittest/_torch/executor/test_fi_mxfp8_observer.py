# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-isolated observer tests; fake tensors forbid data access."""

from __future__ import annotations

import ast
import dataclasses
import functools
import gc
import importlib.util
import json
import os
import sys
import threading
import weakref
from collections import namedtuple
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[4]
_HELPER = _ROOT / "tensorrt_llm/_torch/pyexecutor/_fi_mxfp8_observer.py"


class Dtype:
    def __str__(self) -> str:
        return "torch.bfloat16"


class Tensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.dtype = Dtype()
        self.device = SimpleNamespace(type="cuda", index=0)

    def size(self) -> tuple[int, ...]:
        return self.shape

    def stride(self) -> tuple[int, ...]:
        return (self.shape[1], 1) if len(self.shape) == 2 else (1,)

    def __iter__(self) -> None:
        raise AssertionError("tensor data read")

    def item(self) -> None:
        raise AssertionError("tensor data read")

    def cpu(self) -> None:
        raise AssertionError("tensor transfer")


@dataclasses.dataclass(frozen=True)
class Key:
    custom_op: str
    runner_class_name: str
    nearest_profile: tuple
    runner_hash: int

    @property
    def file_key(self) -> str:
        return str((self.custom_op, self.runner_class_name, self.nearest_profile, ()))


class Runner:
    def get_cache_key_extras(self, inputs: list) -> tuple:
        return ()


Cutlass = type("CutlassMxfp8GemmRunner", (Runner,), {"__module__": "flashinfer.gemm.gemm_base"})
B12x = type(
    "B12xMxfp8GemmRunner",
    (Runner,),
    {
        "__module__": "flashinfer.gemm.gemm_mm_mxfp8_cute_dsl",
    },
)


class Tuner:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._effective_skip_ops: set[str] = set()
        self._override_tuning_buckets = None
        self._override_round_up = False
        self.is_tuning_mode = False
        self.profiling_cache: dict = {}
        self._file_configs: dict = {}
        self.calls = 0
        self.result: tuple | None = None
        self.original_config_id: int | None = None
        self.effective_config_id: int | None = None
        self.original_inputs_id: int | None = None
        self.profile_on_call: tuple | None = None
        self.error: Exception | None = None

    def choose_one(
        self, custom_op: str, runners: list, tuning_config: object, inputs: list, **kwargs: object
    ) -> tuple:
        self.calls += 1
        self.original_config_id = id(tuning_config)
        self.original_inputs_id = id(inputs)
        if self.error is not None:
            raise self.error
        if self.profile_on_call is not None:
            key, tactic = self.profile_on_call
            self.profiling_cache[key] = (tactic, None)
        assert self.result is not None
        return self.result

    def _apply_tuning_overrides(self, config: object) -> object:
        return self.override_config

    def _get_input_sizes(self, inputs: list) -> tuple:
        return tuple(value.size() if isinstance(value, Tensor) else (0,) for value in inputs)

    def _get_cache_key(
        self, custom_op: str, runner: Runner, shapes: tuple, config: object, extras: tuple
    ) -> Key:
        self.effective_config_id = id(config)
        return Key(custom_op, type(runner).__name__, shapes, hash(runner))


def _serialize(value: object) -> object:
    return json.loads(json.dumps(value))


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    spec = importlib.util.spec_from_file_location("observer_under_test", _HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", raising=False)
    torch = ModuleType("torch")
    torch.Tensor = Tensor
    torch.dtype = Dtype
    torch.cuda = SimpleNamespace(
        current_device=lambda: 0, is_current_stream_capturing=lambda: False
    )
    fi = ModuleType("flashinfer.autotuner.autotuner")
    fi._tactic_to_json = _serialize
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "flashinfer.autotuner.autotuner", fi)
    tuner = Tuner()
    runner = Cutlass()
    tuner.result = (runner, ((128, 64), False))
    inputs = [
        Tensor((32, 2560)),
        Tensor((2560, 16384)),
        Tensor((4096,)),
        Tensor((1310720,)),
        Dtype(),
        Tensor((32, 16384)),
        Tensor((1024,)),
    ]
    return SimpleNamespace(
        module=module,
        tuner=tuner,
        runner=runner,
        inputs=inputs,
        config=object(),
        path=tmp_path / "trace.jsonl",
        torch=torch,
    )


def _install(setup: SimpleNamespace, seed: dict | None = None) -> object:
    return setup.module.Mxfp8Observer(
        setup.tuner, setup.path, seed or {}, "seed-hash" if seed else None, 0
    )


def _call(setup: SimpleNamespace, **kwargs: object) -> tuple:
    return setup.tuner.choose_one(
        "mxfp8_gemm", [setup.runner], setup.config, setup.inputs, **kwargs
    )


def _records(setup: SimpleNamespace, kind: str) -> list[dict]:
    return [
        record
        for line in setup.path.read_text().splitlines()
        if (record := json.loads(line))["kind"] == kind
    ]


def test_delegate_once_unchanged_and_tensor_free(setup: SimpleNamespace) -> None:
    observer = _install(setup)
    original_result = setup.tuner.result
    assert _call(setup) is original_result
    assert setup.tuner.calls == 1
    assert setup.tuner.original_inputs_id == id(setup.inputs)
    choice = _records(setup, "choice")[0]
    assert choice["selected_value"] == ["CutlassMxfp8GemmRunner", [[128, 64], False]]
    assert choice["inputs"][0]["shape"] == [32, 2560]
    refs = [weakref.ref(value) for value in setup.inputs if isinstance(value, Tensor)]
    setup.inputs = None
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert all(isinstance(value, str) for value in observer._seen)


def test_non_mxfp8_passthrough(setup: SimpleNamespace) -> None:
    _install(setup)
    result = setup.tuner.choose_one("other_op", [], None, [], unfamiliar=True)
    assert result is setup.tuner.result
    assert setup.tuner.calls == 1
    assert not _records(setup, "choice")


def test_original_exception_is_unchanged_and_not_retried(setup: SimpleNamespace) -> None:
    error = ValueError("FI original failed")
    setup.tuner.error = error
    _install(setup)
    with pytest.raises(ValueError) as caught:
        _call(setup)
    assert caught.value is error
    assert setup.tuner.calls == 1
    assert not _records(setup, "choice")


def test_seed_memory_precedence_is_evidence_not_assumed_hit(setup: SimpleNamespace) -> None:
    key = setup.tuner._get_cache_key(
        "mxfp8_gemm", setup.runner, setup.tuner._get_input_sizes(setup.inputs), setup.config, ()
    )
    seed = {key.file_key: ["CutlassMxfp8GemmRunner", 2]}
    setup.tuner._file_configs[key.file_key] = ("CutlassMxfp8GemmRunner", 2)
    setup.tuner.profiling_cache[key] = (3, None)
    setup.tuner.result = (setup.runner, 3)
    _install(setup, seed)
    seed[key.file_key][1] = 3
    _call(setup)
    record = _records(setup, "choice")[0]
    assert record["seed_member"] and not record["seed_equal"]
    assert record["candidates"][0]["memory_before"] == 3
    assert record["candidates"][0]["loaded_before"] == ["CutlassMxfp8GemmRunner", 2]


def test_second_runner_selection_and_seed_match(setup: SimpleNamespace) -> None:
    runner = B12x()
    key = setup.tuner._get_cache_key(
        "mxfp8_gemm", runner, setup.tuner._get_input_sizes(setup.inputs), setup.config, ()
    )
    setup.tuner.result = (runner, (1, 2))
    _install(setup, {key.file_key: ["B12xMxfp8GemmRunner", [1, 2]]})
    result = setup.tuner.choose_one(
        "mxfp8_gemm", [setup.runner, runner], setup.config, setup.inputs
    )
    assert result is setup.tuner.result
    record = _records(setup, "choice")[0]
    assert record["selected_index"] == 1 and record["seed_equal"]


def test_effective_override_key_original_config_delegated(setup: SimpleNamespace) -> None:
    setup.tuner._override_tuning_buckets = (16, 32)
    setup.tuner.override_config = object()
    _install(setup)
    _call(setup)
    assert setup.tuner.effective_config_id == id(setup.tuner.override_config)
    assert setup.tuner.original_config_id == id(setup.config)


def test_capture_phase_dedup_and_later_eager(setup: SimpleNamespace) -> None:
    observer = _install(setup)
    setup.torch.cuda.is_current_stream_capturing = lambda: True
    observer.set_phase("profiling_initialization")
    _call(setup)
    _call(setup)
    observer.set_phase("serving_initialization")
    _call(setup)
    observer.set_phase("serving")
    setup.torch.cuda.is_current_stream_capturing = lambda: False
    _call(setup)
    records = _records(setup, "choice")
    assert [r["capture"] for r in records] == [True, True, False]
    assert [r["phase"] for r in records] == [
        "profiling_initialization",
        "serving_initialization",
        "serving",
    ]
    assert setup.tuner.calls == 4
    assert _records(setup, "header")[0]["graph_replay_observed"] is False


def test_new_profile_beyond_actual_call_is_recorded(setup: SimpleNamespace) -> None:
    setup.tuner.is_tuning_mode = True
    key = Key("mxfp8_gemm", "CutlassMxfp8GemmRunner", ((128, 2560),), 123)
    setup.tuner.profile_on_call = (key, 7)
    _install(setup)
    _call(setup)
    record = _records(setup, "profiled_key")[0]
    assert record["file_key"] == key.file_key
    assert not record["seed_member"]
    assert record["value"] == ["CutlassMxfp8GemmRunner", 7]


@pytest.mark.parametrize(
    "kind", ["kwargs", "inputs", "runner", "extras", "device", "skip", "dtype", "tensor"]
)
def test_unsupported_call_fails_closed_before_delegate(setup: SimpleNamespace, kind: str) -> None:
    observer = _install(setup)
    if kind == "inputs":
        setup.inputs.pop()
    elif kind == "runner":
        setup.runner = Runner()
    elif kind == "extras":
        setup.runner.get_cache_key_extras = lambda inputs: (1,)
    elif kind == "device":
        setup.torch.cuda.current_device = lambda: 1
    elif kind == "skip":
        setup.tuner._effective_skip_ops.add("mxfp8_gemm")
    elif kind == "dtype":
        setup.inputs[4] = None
    elif kind == "tensor":
        setup.inputs[0] = None
    with pytest.raises(RuntimeError):
        _call(setup, **({"unknown": True} if kind == "kwargs" else {}))
    assert setup.tuner.calls == 0
    assert observer._failure is not None
    with pytest.raises(RuntimeError, match="previously failed"):
        _call(setup)


def test_tensor_tactic_rejected_without_iteration(setup: SimpleNamespace) -> None:
    setup.tuner.result = (setup.runner, Tensor((1,)))
    _install(setup)
    with pytest.raises(RuntimeError, match="tactic value"):
        _call(setup)
    assert setup.tuner.calls == 1


@pytest.mark.parametrize(
    "kind", ["record_limit", "byte_limit", "line_limit", "removed", "replaced"]
)
def test_output_failure_latched(
    setup: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    observer = _install(setup)
    if kind == "record_limit":
        monkeypatch.setattr(setup.module, "_MAX_RECORDS", 1)
    elif kind == "byte_limit":
        monkeypatch.setattr(setup.module, "_MAX_BYTES", observer._bytes + 1)
    elif kind == "line_limit":
        monkeypatch.setattr(setup.module, "_MAX_LINE_BYTES", 1)
    elif kind == "removed":
        setup.path.unlink()
    else:
        setup.path.rename(setup.path.with_suffix(".old"))
        setup.path.write_text("other writer")
    with pytest.raises((RuntimeError, OSError)):
        _call(setup)
    assert setup.tuner.calls == 1
    with pytest.raises(RuntimeError, match="previously failed"):
        _call(setup)
    assert setup.tuner.calls == 1
    if kind == "replaced":
        assert setup.path.read_text() == "other writer"


def test_existing_output_not_overwritten(setup: SimpleNamespace) -> None:
    setup.path.write_text("untouched")
    with pytest.raises(FileExistsError):
        _install(setup)
    assert setup.path.read_text() == "untouched"
    assert "choose_one" not in vars(setup.tuner)


def test_existing_wrapper_and_legacy_cache_rejected(
    setup: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", "1")
    with pytest.raises(RuntimeError, match="legacy"):
        _install(setup)
    monkeypatch.delenv("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE")
    setup.tuner.choose_one = setup.tuner.choose_one
    with pytest.raises(RuntimeError, match="already overridden"):
        _install(setup)


@pytest.mark.parametrize("kind", ["signature", "seed_records", "seed_bytes"])
def test_unsupported_installation_is_bounded(
    setup: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    seed = {"key": ["runner", 1]}
    if kind == "signature":
        setup.tuner.choose_one = lambda *args: None
    elif kind == "seed_records":
        monkeypatch.setattr(setup.module, "_MAX_RECORDS", 0)
    else:
        monkeypatch.setattr(setup.module, "_MAX_BYTES", 1)
    with pytest.raises(RuntimeError):
        _install(setup, seed)
    assert not setup.path.exists()


@pytest.mark.parametrize("enabled", [False, True])
def test_optional_persistence_integration(
    setup: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    # Scope validation is covered by the persistence suite; this isolates only
    # optional installation, phase propagation and staying installed after save.
    package = ModuleType("research_test_package")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, package.__name__ + "._fi_mxfp8_observer", setup.module)
    spec = importlib.util.spec_from_file_location(
        package.__name__ + "._fi_cache_research", _HELPER.with_name("_fi_cache_research.py")
    )
    assert spec is not None and spec.loader is not None
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    monkeypatch.setattr(helper, "_validate_scope", lambda *args: None)
    fi = ModuleType("flashinfer")
    fi.__version__ = "0.6.18"
    autotuner = ModuleType("flashinfer.autotuner")
    autotuner.AutoTuner = SimpleNamespace(get=lambda: setup.tuner)
    monkeypatch.setitem(sys.modules, "flashinfer", fi)
    monkeypatch.setitem(sys.modules, "flashinfer.autotuner", autotuner)
    setup.torch.cuda.get_device_capability = lambda device: (12, 1)
    for field in ("_active_tuning_contexts", "_dirty", "_dirty_seq"):
        setattr(setup.tuner, field, 0)
    for field in (
        "_ranked_tactics_cache",
        "_namespaced_records",
        "_dirty_namespaces",
        "_observed_cache_generations",
    ):
        setattr(setup.tuner, field, {})
    monkeypatch.delenv(helper._LOAD, raising=False)
    monkeypatch.setenv(helper._SAVE, str(setup.path.with_suffix(".cache")))
    if enabled:
        monkeypatch.setenv(helper._OBSERVE, str(setup.path))
    else:
        monkeypatch.delenv(helper._OBSERVE, raising=False)
    logs: list[str] = []
    session = helper.begin_fi_cache_research(
        None, SimpleNamespace(local_rank=0), "unused", logs.append
    )
    assert (session.observer is not None) == enabled
    session.set_phase("serving_initialization")
    _call(setup)

    def save(path: str) -> None:
        Path(path).write_text('{"_metadata": {}}')

    setup.tuner.save_configs = save
    session.save()
    _call(setup)
    if enabled:
        assert [r["phase"] for r in _records(setup, "choice")] == [
            "serving_initialization",
            "serving",
        ]
        assert any("diagnostic, not timing" in message for message in logs)
    else:
        assert not setup.path.exists()
        assert "choose_one" not in vars(setup.tuner)


def test_creator_phase_markers_precede_both_constructors() -> None:
    tree = ast.parse(_HELPER.with_name("py_executor_creator.py").read_text())
    creator = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "create_py_executor"
    )
    phase_calls = sorted(
        (node.lineno, ast.unparse(node))
        for node in ast.walk(creator)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "set_phase"
    )
    constructors = sorted(
        node.lineno
        for node in ast.walk(creator)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "create_py_executor_instance"
    )
    assert len(phase_calls) == len(constructors) == 2
    assert phase_calls[0][0] < constructors[0] < phase_calls[1][0] < constructors[1]
    assert "profiling_initialization" in phase_calls[0][1]
    assert "if estimating_kv_cache else 'serving_initialization'" in phase_calls[0][1]
    assert "serving_initialization" in phase_calls[1][1]


def test_pinned_fi_key_profile_and_tactic_methods(setup: SimpleNamespace) -> None:
    # Optional local-source contract test. Execute the pinned FI methods, not
    # copied bucket logic, with fake tensors and no FI/native imports.
    source = Path(
        os.environ.get(
            "FLASHNEXT_FI_AUTOTUNER_SOURCE",
            str(_ROOT.parent / "flashinfer-mtp-nonatomic-0618/flashinfer/autotuner/autotuner.py"),
        )
    )
    if not source.is_file():
        pytest.skip("pinned FI source checkout is not available")
    tree = ast.parse(source.read_text())
    original_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AutoTuner"
    )
    methods = [
        node
        for node in original_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in (
            "_get_cache_key",
            "_find_nearest_profile",
            "_find_nearest_profile_cached",
            "_get_input_sizes",
        )
    ]
    assert len(methods) == 4
    extras = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in ("ProfilingCacheKey", "_tactic_to_json")
    ]
    isolated = ast.parse("from __future__ import annotations\nclass ActualMethods:\n    pass\n")
    isolated.body[1].body = methods
    isolated.body[1:1] = extras
    namespace = {
        "__name__": __name__,
        "dataclass": dataclasses.dataclass,
        "functools": functools,
        "torch": setup.torch,
    }
    exec(compile(isolated, str(source), "exec"), namespace)
    actual = namespace["ActualMethods"]
    setup.tuner._get_input_sizes = actual()._get_input_sizes
    setup.tuner._get_cache_key = actual._get_cache_key
    dynamic = namedtuple("DynamicSpec", ["input_idx", "dim_idx", "map_to_tuning_buckets"])
    constraint = namedtuple("Constraint", ["input_idx", "dim_idx"])
    setup.config = SimpleNamespace(
        dynamic_tensor_specs=(dynamic((0, 5), (0, 0), lambda size: 16),),
        constraint_specs=(constraint(2, 0),),
    )
    _install(setup)
    setup.tuner.choose_one.__self__._serialize_tactic = namespace["_tactic_to_json"]
    _call(setup)
    record = _records(setup, "choice")[0]
    expected = actual._get_cache_key(
        "mxfp8_gemm", setup.runner, actual()._get_input_sizes(setup.inputs), setup.config, ()
    )
    assert record["file_key"] == expected.file_key
    assert record["candidates"][0]["nearest_profile"][4] == [0]
    assert record["candidates"][0]["nearest_profile"][0] == [16, 2560]
    assert record["candidates"][0]["nearest_profile"][5] == [16, 16384]
    assert record["candidates"][0]["nearest_profile"][2] == [-1]
    assert record["selected_value"][1] == [[128, 64], False]
