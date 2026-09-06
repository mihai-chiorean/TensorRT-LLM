# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NVFP4 PLE construction and mapper integration, using only tiny weights."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tensorrt_llm._torch.models.checkpoints.hf.qwen4_exp_weight_mapper import Qwen4ExpHfWeightMapper
from tensorrt_llm._torch.modules.qwen4_exp import ple
from tensorrt_llm._torch.modules.qwen4_exp.ple_nvfp4 import Qwen4ExpNVFP4MmapEmbedding
from tensorrt_llm.mapping import Mapping


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        ple_embedding_dtype="nvfp4",
        ngram_size=2,
        heads_per_ngram=1,
        vocab_size=16,
        eos_token_id=2,
        ngram_vocab_size_base=3,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=2,
    )


def test_nvfp4_auto_selects_storage_before_dense_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("NVFP4 must never construct a dense or pinned table")

    monkeypatch.setenv("TRTLLM_QWEN4_EXP_PLE_HOST_OFFLOAD", "0")
    monkeypatch.setattr(ple, "Embedding", forbidden)
    monkeypatch.setattr(ple, "Qwen4ExpPinnedHostEmbedding", forbidden)
    config = _config()
    config.ngram_size = 3
    config.heads_per_ngram = 8
    config.ngram_vocab_size_base = 20_000_000
    config.make_ngram_vocab_size_divisible_by = 128
    module = ple.Qwen4ExpNGramEmbedding(config, embedding_dim=2560, dtype=torch.bfloat16)
    assert module.nvfp4_storage and module.host_offload
    assert isinstance(module.ngram_embedding, Qwen4ExpNVFP4MmapEmbedding)
    assert module.head_dim_per_ngram == 160 and module.ngram_heads == 16
    assert module.ngram_embedding.num_embeddings > 320_000_000
    assert list(module.ngram_embedding.parameters()) == []
    assert module.ngram_embedding._shards == ()


def test_nvfp4_tp_guard() -> None:
    with pytest.raises(NotImplementedError, match="TP=1"):
        ple.Qwen4ExpNGramEmbedding(
            _config(), 160, dtype=torch.bfloat16, mapping=Mapping(world_size=2, tp_size=2)
        )


def test_nvfp4_activation_dtype_guard() -> None:
    with pytest.raises(TypeError, match="BF16"):
        ple.Qwen4ExpNGramEmbedding(_config(), 160, dtype=torch.float16)


def test_mapper_binds_paired_lazy_shards_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = ple.Qwen4ExpNGramEmbedding(_config(), 160, dtype=torch.bfloat16)
    leaves = {
        "ngram_embedding.shard_1.weight": torch.full((2, 80), 0x72, dtype=torch.uint8),
        "ngram_embedding.shard_0.weight": torch.full((2, 80), 0x91, dtype=torch.uint8),
        "ngram_embedding.shard_1.weight_scale": torch.full((2, 10), 2, dtype=torch.float8_e4m3fn),
        "ngram_embedding.shard_0.weight_scale": torch.full((2, 10), 0.5, dtype=torch.float8_e4m3fn),
        "ngram_embedding.weight_scale_2": torch.tensor([0.25]),
        "layer_multipliers": torch.tensor([11, 13]),
        "ngram_heads_offsets": torch.tensor([0]),
        "ngram_heads_vocab_sizes": torch.tensor([3]),
    }
    path = tmp_path / "ple.safetensors"
    save_file(leaves, path)
    mapper = Qwen4ExpHfWeightMapper()
    monkeypatch.setattr(mapper, "_ngram_module_for_prefix", lambda _: module)
    with safe_open(path, framework="pt", device="cpu") as handle:
        weights = {name: handle.get_slice(name) for name in handle.keys()}
        mapper._load_ngram_tables({"model.layers.1.ple": weights})
        weights.clear()
    module.to(dtype=torch.bfloat16)
    ids = torch.tensor([[0], [2], [3], [1]])
    result = module.embed(ids)
    expected = torch.tensor([[0.0625, -0.0625], [0.5, 3], [0.5, 3], [0.0625, -0.0625]])
    expected = expected.repeat(1, 80).unsqueeze(1).bfloat16()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(module.layer_multipliers, torch.tensor([11, 13]))
    assert mapper.should_skip_module("model.layers.1.ple.ple_embedding.ngram_embedding")


@pytest.mark.parametrize("source_root", ["model.language_model", "model"])
def test_mapper_retains_original_ple_namespace(
    source_root: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class CheckpointWeights(dict):
        checkpoint_dir = str(tmp_path)

    source_prefix = f"{source_root}.layers.1.ple.ple_embedding.ngram_embedding"
    weights = CheckpointWeights(
        {f"{source_prefix}.shard_0.weight": torch.zeros(1, 80, dtype=torch.uint8)}
    )
    mapper = Qwen4ExpHfWeightMapper()
    mapper._config = SimpleNamespace(
        pretrained_config=SimpleNamespace(
            num_hidden_layers=48,
            linear_key_head_dim=128,
            linear_num_key_heads=16,
            linear_value_head_dim=128,
            linear_num_value_heads=48,
        ),
        mapping=Mapping(),
        spec_config=None,
    )
    captured = {}

    def capture(ngram: dict, **kwargs: object) -> None:
        captured.update(ngram=ngram, **kwargs)

    monkeypatch.setattr(mapper, "_load_ngram_tables", capture)
    assert mapper.preprocess_weights(weights) == {}
    assert captured["checkpoint_dir"] == str(tmp_path)
    assert captured["source_prefixes"] == {"model.layers.1.ple": source_prefix}
    assert set(captured["ngram"]) == {"model.layers.1.ple"}


@pytest.mark.parametrize("entry_point", ["forward", "start_prefetch"])
def test_ple_rejects_graph_metadata_before_work(entry_point: str) -> None:
    # Reject before reading metadata fields or touching a CUDA stream/state.
    module = ple.Qwen4ExpPLE.__new__(ple.Qwen4ExpPLE)
    torch.nn.Module.__init__(module)
    module.ple_embedding = ple.Qwen4ExpNGramEmbedding(_config(), 160, dtype=torch.bfloat16)
    metadata = SimpleNamespace(is_cuda_graph=True)
    empty = torch.empty(0)
    with pytest.raises(RuntimeError, match="eager-only"):
        if entry_point == "forward":
            module.forward(empty, metadata, empty, empty)
        else:
            module.start_prefetch(metadata, empty)
