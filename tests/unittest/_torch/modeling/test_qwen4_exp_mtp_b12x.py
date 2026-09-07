# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-isolated MTP selector/constructor tests, not native kernel qualification."""

import ast
import copy
import fnmatch
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

_ROOT = Path(__file__).resolve().parents[4]
_ENV = "TRTLLM_QWEN4_MTP_B12X"


class _QuantPolicy(SimpleNamespace):
    def is_module_excluded_from_quantization(self, name: str) -> bool:
        return any(fnmatch.fnmatchcase(name, pattern) for pattern in self.exclude_modules)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        _frozen=True,
        moe_backend="CUTLASS",
        moe_max_num_tokens=512,
        max_num_tokens=512,
        use_cuda_graph=False,
        skip_create_weights_in_init=True,
        quant_config=_QuantPolicy(exclude_modules=[]),
        quant_config_dict={
            "model.layers.48.mlp.experts": _QuantPolicy(quant_algo="W4A16_NVFP4", group_size=16),
            "model.layers.0.mlp.experts": _QuantPolicy(quant_algo="NVFP4", group_size=16),
        },
        pretrained_config=SimpleNamespace(
            num_hidden_layers=48,
            mtp_num_hidden_layers=1,
            num_experts=512,
            hidden_size=2560,
            moe_intermediate_size=640,
            num_experts_per_tok=10,
            torch_dtype=torch.bfloat16,
            hc_count=4,
            rms_norm_eps=1e-6,
        ),
        mapping=SimpleNamespace(
            tp_size=1, pp_size=1, cp_size=1, moe_ep_size=1, enable_attention_dp=False
        ),
    )


@pytest.fixture
def model_source(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    path = _ROOT / "tensorrt_llm/_torch/models/modeling_qwen4_exp.py"
    tree = ast.parse(path.read_text())
    functions = {"_qwen4_exp_mtp_b12x_config", "Qwen4ExpMTP"}
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in functions
    ]
    helper_tree = ast.parse(
        (_ROOT / "tensorrt_llm/_torch/models/modeling_qwen3_next.py").read_text()
    )
    nodes.append(
        next(
            n
            for n in helper_tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_experts_excluded_from_quant"
        )
    )

    class FakeB12x(nn.Module):
        pass

    class FakeLayer(nn.Module):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__()
            self.args = args
            self.kwargs = kwargs

    module = ModuleType("tensorrt_llm._torch.models.mtp_selector_under_test")
    module.__package__ = "tensorrt_llm._torch.models"
    module.resolved_backend = FakeB12x
    module.wrap_backend = True
    module.constructed_configs = []

    class FakeDecoder(nn.Module):
        def __init__(self, config: SimpleNamespace, *args: object) -> None:
            super().__init__()
            module.constructed_configs.append(config)
            self.model_config = config
            self.mlp = nn.Module()
            self.mlp.experts = nn.Module()
            if module.wrap_backend:
                self.mlp.experts.backend = module.resolved_backend()

    module.__dict__.update(
        copy=copy,
        os=os,
        torch=torch,
        QuantAlgo=SimpleNamespace(W4A16_NVFP4="W4A16_NVFP4"),
        Qwen4ExpDecoderLayer=FakeDecoder,
        RMSNorm=FakeLayer,
        Linear=FakeLayer,
        Qwen4ExpMTPHead=FakeLayer,
        TensorParallelMode=SimpleNamespace(ROW="row"),
        AuxStreamType=SimpleNamespace(Attention="attention", MoeShared="shared"),
        EventType=SimpleNamespace(Main="main", MoeShared="shared"),
    )
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    body.extend(nodes)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec"
        ),
        module.__dict__,
    )
    backend = ModuleType("tensorrt_llm._torch.moe.fused_moe.fused_moe_cute_dsl_b12x")
    backend.CuteDslB12xFusedMoE = FakeB12x
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 1))
    monkeypatch.setattr(torch.cuda, "Event", Mock(return_value=object()))
    monkeypatch.delenv(_ENV, raising=False)
    return module


@pytest.mark.parametrize("selector", [None, "0"])
def test_default_is_identity_without_device_query(
    model_source: ModuleType, monkeypatch: pytest.MonkeyPatch, selector: str | None
) -> None:
    if selector is not None:
        monkeypatch.setenv(_ENV, selector)
    monkeypatch.setattr(torch.cuda, "is_available", Mock(side_effect=AssertionError))
    config = _config()
    assert model_source._qwen4_exp_mtp_b12x_config(config, 48) is config
    model_source.resolved_backend = nn.Module
    layer = model_source.Qwen4ExpMTP(config, 48, {"attention": None, "shared": None})
    assert layer.model_config is config


@pytest.mark.parametrize("selector", ["", "true", "b12x", "2"])
def test_invalid_selector_rejected(
    model_source: ModuleType, monkeypatch: pytest.MonkeyPatch, selector: str
) -> None:
    monkeypatch.setenv(_ENV, selector)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        model_source._qwen4_exp_mtp_b12x_config(_config(), 48)


@pytest.mark.parametrize("graphs", [False, True])
@pytest.mark.parametrize("frozen", [False, True])
def test_only_mtp_copy_changes_backend(
    model_source: ModuleType, monkeypatch: pytest.MonkeyPatch, graphs: bool, frozen: bool
) -> None:
    monkeypatch.setenv(_ENV, "1")
    config = _config()
    config.use_cuda_graph = graphs
    config._frozen = frozen
    original = copy.deepcopy(config)
    layer = model_source.Qwen4ExpMTP(config, 48, {"attention": None, "shared": None})
    copied = layer.model_config
    assert copied is not config
    assert copied.moe_backend == "CUTEDSL"
    assert copied._frozen is frozen
    assert config == original
    for name, value in vars(config).items():
        if name != "moe_backend":
            assert getattr(copied, name) is value
    assert model_source.constructed_configs == [copied]
    assert layer.shared_head.args[0] is copied
    assert copied.quant_config_dict["model.layers.48.mlp.experts"].quant_algo == "W4A16_NVFP4"
    assert copied.quant_config_dict["model.layers.0.mlp.experts"].quant_algo == "NVFP4"


@pytest.mark.parametrize("capability", [(8, 9), (10, 0), (10, 3), (12, 0), (13, 0)])
def test_other_devices_rejected(
    model_source: ModuleType, monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int]
) -> None:
    monkeypatch.setenv(_ENV, "1")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    with pytest.raises(ValueError, match="SM121"):
        model_source._qwen4_exp_mtp_b12x_config(_config(), 48)


def test_no_cuda_rejected(model_source: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV, "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="SM121"):
        model_source._qwen4_exp_mtp_b12x_config(_config(), 48)


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_experts", 256),
        ("hidden_size", 4096),
        ("moe_intermediate_size", 512),
        ("num_experts_per_tok", 8),
        ("mtp_num_hidden_layers", 2),
    ],
)
def test_other_geometry_rejected(
    model_source: ModuleType, monkeypatch: pytest.MonkeyPatch, field: str, value: int
) -> None:
    monkeypatch.setenv(_ENV, "1")
    config = _config()
    setattr(config.pretrained_config, field, value)
    with pytest.raises(ValueError, match="one E512"):
        model_source._qwen4_exp_mtp_b12x_config(config, 48)


@pytest.mark.parametrize("field", ["tp_size", "pp_size", "cp_size", "moe_ep_size"])
def test_parallel_topologies_rejected(
    model_source: ModuleType, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    monkeypatch.setenv(_ENV, "1")
    config = _config()
    setattr(config.mapping, field, 2)
    with pytest.raises(ValueError, match="TP1/PP1/CP1/EP1"):
        model_source._qwen4_exp_mtp_b12x_config(config, 48)


@pytest.mark.parametrize("layer_idx", [0, 47, 49])
def test_target_or_additional_layer_rejected(
    model_source: ModuleType, monkeypatch: pytest.MonkeyPatch, layer_idx: int
) -> None:
    monkeypatch.setenv(_ENV, "1")
    with pytest.raises(ValueError, match="one E512"):
        model_source._qwen4_exp_mtp_b12x_config(_config(), layer_idx)


@pytest.mark.parametrize(
    "policy",
    [
        None,
        {},
        {"model.layers.48.mlp.experts": _QuantPolicy(quant_algo="NVFP4", group_size=16)},
        {"model.layers.48.mlp.experts": _QuantPolicy(quant_algo=None, group_size=16)},
        {"model.layers.48.mlp.experts": _QuantPolicy(quant_algo="W4A16_NVFP4", group_size=32)},
    ],
)
def test_missing_or_wrong_quantization_rejected(
    model_source: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    policy: dict[str, _QuantPolicy] | None,
) -> None:
    monkeypatch.setenv(_ENV, "1")
    config = _config()
    config.quant_config_dict = policy
    with pytest.raises(ValueError, match="non-excluded W4A16_NVFP4"):
        model_source._qwen4_exp_mtp_b12x_config(config, 48)


@pytest.mark.parametrize(
    "pattern", ["model.layers.48.mlp.experts", "mtp.layers.0.mlp.experts", "*"]
)
def test_excluded_experts_rejected(
    model_source: ModuleType, monkeypatch: pytest.MonkeyPatch, pattern: str
) -> None:
    monkeypatch.setenv(_ENV, "1")
    config = _config()
    config.quant_config.exclude_modules = [pattern]
    with pytest.raises(ValueError, match="non-excluded W4A16_NVFP4"):
        model_source._qwen4_exp_mtp_b12x_config(config, 48)


@pytest.mark.parametrize("backend", ["CUTEDSL", "TRTLLM", "VANILLA"])
def test_non_cutlass_target_rejected(
    model_source: ModuleType, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    monkeypatch.setenv(_ENV, "1")
    config = _config()
    config.moe_backend = backend
    with pytest.raises(ValueError, match="target MoE backend"):
        model_source._qwen4_exp_mtp_b12x_config(config, 48)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_non_bf16_rejected(
    model_source: ModuleType, monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype
) -> None:
    monkeypatch.setenv(_ENV, "1")
    config = _config()
    config.pretrained_config.torch_dtype = dtype
    with pytest.raises(ValueError, match="BF16"):
        model_source._qwen4_exp_mtp_b12x_config(config, 48)


@pytest.mark.parametrize("wrap_backend", [False, True])
def test_resolver_fallback_fails_without_target_mutation(
    model_source: ModuleType, monkeypatch: pytest.MonkeyPatch, wrap_backend: bool
) -> None:
    monkeypatch.setenv(_ENV, "1")
    model_source.resolved_backend = nn.Module
    model_source.wrap_backend = wrap_backend
    config = _config()
    original = copy.deepcopy(config)
    with pytest.raises(RuntimeError, match="resolution fell back"):
        model_source.Qwen4ExpMTP(config, 48, {"attention": None, "shared": None})
    assert config == original


def test_real_model_config_freeze_method_preserved(
    model_source: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_ENV, "1")
    path = _ROOT / "tensorrt_llm/_torch/model_config.py"
    cls = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "ModelConfig"
    )
    setter = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__setattr__")
    frozen_cls = ast.ClassDef(
        name="FrozenConfig",
        bases=[],
        keywords=[],
        decorator_list=[],
        body=[
            ast.Assign(
                targets=[ast.Name(id="_frozen", ctx=ast.Store())], value=ast.Constant(value=False)
            ),
            setter,
        ],
    )
    namespace = {}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[frozen_cls], type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    config = namespace["FrozenConfig"]()
    source = _config()
    for key, value in vars(source).items():
        if key != "_frozen":
            setattr(config, key, value)
    config._frozen = True
    copied = model_source._qwen4_exp_mtp_b12x_config(config, 48)
    assert config.moe_backend == "CUTLASS"
    assert copied.moe_backend == "CUTEDSL"
    for item in (config, copied):
        with pytest.raises(AttributeError, match="instance is frozen"):
            item.moe_max_num_tokens = 4
