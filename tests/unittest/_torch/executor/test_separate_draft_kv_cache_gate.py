# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor import _util
from tensorrt_llm._torch.pyexecutor._util import CacheCost, KvCacheCreator
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType


def _make_creator(attn_backend: str, *, attention_dp: bool = False) -> KvCacheCreator:
    creator = object.__new__(KvCacheCreator)
    creator._llm_args = SimpleNamespace(attn_backend=attn_backend)
    creator._mapping = Mock(enable_attention_dp=attention_dp)
    creator._mapping.is_last_pp_rank.return_value = True
    creator._speculative_config = Mock()
    creator._speculative_config.spec_dec_mode.is_external_drafter.return_value = False
    creator._draft_model_engine = None
    return creator


@pytest.mark.parametrize(
    ("attn_backend", "expected"),
    [
        ("TRTLLM", True),
        ("FLASHINFER", False),
        ("VANILLA", False),
    ],
)
def test_separate_draft_cache_requires_supported_attention_backend(
    monkeypatch: pytest.MonkeyPatch,
    attn_backend: str,
    expected: bool,
) -> None:
    monkeypatch.setattr(_util, "should_use_separate_draft_kv_cache", lambda _: True)
    creator = _make_creator(attn_backend)

    assert creator._should_create_separate_draft_kv_cache() is expected


def test_separate_draft_cache_remains_disabled_with_attention_dp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_util, "should_use_separate_draft_kv_cache", lambda _: True)
    creator = _make_creator("TRTLLM", attention_dp=True)

    assert not creator._should_create_separate_draft_kv_cache()


@pytest.mark.parametrize(
    ("attn_backend", "expected_separate"),
    [
        ("TRTLLM", True),
        ("FLASHINFER", False),
    ],
)
def test_estimation_and_build_use_the_same_backend_gate(
    monkeypatch: pytest.MonkeyPatch,
    attn_backend: str,
    expected_separate: bool,
) -> None:
    monkeypatch.setattr(_util, "should_use_separate_draft_kv_cache", lambda _: True)
    creator = _make_creator(attn_backend)
    creator._model_engine = Mock()
    creator._model_engine.model.model_config = Mock()
    creator._kv_cache_manager_cls = Mock()
    creator._kv_cache_config = Mock()
    creator._is_kv_cache_manager_v2 = True
    creator._is_encoder_decoder = Mock(return_value=False)
    creator._get_effective_draft_config = Mock(return_value=Mock())
    creator._get_num_draft_layers = Mock(return_value=1)
    creator._per_manager_cache_cost = Mock(return_value=CacheCost(slope=1))

    estimated_cost = creator._get_kv_size_per_token()

    assert estimated_cost == CacheCost(slope=2 if expected_separate else 1)
    assert creator._per_manager_cache_cost.call_count == (2 if expected_separate else 1)
    assert creator._needs_gpu_kv_cache_budget_split() is expected_separate

    target_manager = object()
    draft_manager = object()
    creator._skip_est = False
    creator._max_seq_len = 1024
    creator._kv_connector_manager = None
    creator._create_kv_cache_manager = Mock(return_value=target_manager)
    creator._create_one_model_draft_kv_cache_manager = Mock(return_value=draft_manager)
    creator._split_kv_cache_budget_for_draft = Mock(
        side_effect=lambda _, target, draft: (target, draft or Mock())
    )

    resources = {}
    creator.build_managers(resources, estimating_kv_cache=False)

    expected_draft_manager = draft_manager if expected_separate else None
    assert resources[ResourceManagerType.DRAFT_KV_CACHE_MANAGER] is expected_draft_manager
    assert creator._create_one_model_draft_kv_cache_manager.called is expected_separate
    assert creator._split_kv_cache_budget_for_draft.called is expected_separate
