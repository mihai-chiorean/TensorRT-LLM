# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Mapping

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.models import modeling_utils
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict
from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper


class _PassthroughWeightMapper(BaseWeightMapper):
    def map_weights(self) -> None:
        pass

    def apply_callbacks(
        self,
        module: nn.Module,
        module_name: str,
        module_names_breakdown: list[str],
        weights: Mapping[str, torch.Tensor],
    ) -> list[dict[str, torch.Tensor]]:
        raise AssertionError("No callback is expected for a plain module")


class _RecordingWeights(ConsumableWeightsDict):
    def __init__(self, weights: dict[str, torch.Tensor]) -> None:
        super().__init__(weights)
        self.consumed_keys: list[str] = []

    def mark_consumed_keys(self, keys: Iterable[str]) -> int:
        consumed_keys = list(keys)
        self.consumed_keys.extend(consumed_keys)
        return super().mark_consumed_keys(consumed_keys)


def test_timing_metric_accumulates_and_records_failures(monkeypatch) -> None:
    perf_counter_values = iter([1.0, 1.25, 2.0, 2.5])
    monkeypatch.setattr(modeling_utils.time, "perf_counter", lambda: next(perf_counter_values))
    metrics = {}

    with modeling_utils.timing_metric("load_seconds", metrics):
        pass

    with pytest.raises(RuntimeError, match="load failed"):
        with modeling_utils.timing_metric("load_seconds", metrics):
            raise RuntimeError("load failed")

    assert metrics["load_seconds"] == pytest.approx(0.75)


def test_partial_load_marks_only_copied_parameters_consumed(monkeypatch) -> None:
    monkeypatch.setenv("TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL", "1")
    monkeypatch.setattr(modeling_utils.torch.cuda, "set_device", lambda _: None)

    layer = nn.Linear(2, 2)
    initial_bias = layer.bias.detach().clone()
    source_weight = torch.full_like(layer.weight, 3.0)
    model = nn.Module()
    model.add_module("layer", layer)
    weights = _RecordingWeights({"layer.weight": source_weight})

    modeling_utils._load_weights_impl_v2(
        model,
        weights,
        _PassthroughWeightMapper(),
        allow_partial_loading=True,
    )

    assert weights.consumed_keys == ["layer.weight"]
    torch.testing.assert_close(layer.weight, source_weight)
    torch.testing.assert_close(layer.bias, initial_bias)
