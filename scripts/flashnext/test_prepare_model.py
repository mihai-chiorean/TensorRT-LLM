# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file


def _checkpoint(path: Path) -> Path:
    path.mkdir()
    (path / "config.json").write_text('{"model_type": "qwen3_8_flash_next"}\n')
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "model-00001.safetensors"}})
    )
    save_file({"weight": np.zeros((4, 4), dtype=np.float32)}, path / "model-00001.safetensors")
    save_file({"amax": np.zeros((1,), dtype=np.float32)}, path / "amax.safetensors")
    return path


def _run(source: Path, destination: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("prepare_model.py")),
            "--source",
            str(source),
            "--destination",
            str(destination),
            "--text-only",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_checkpoint_view_preserves_source_and_omits_calibration(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path / "source")
    original = (source / "config.json").read_bytes()
    destination = tmp_path / "view"
    result = _run(source, destination)
    assert result.returncode == 0, result.stderr
    assert (source / "config.json").read_bytes() == original
    assert json.loads((destination / "config.json").read_text())["language_model_only"] is True
    assert (destination / "model-00001.safetensors").is_symlink()
    assert not (destination / "amax.safetensors").exists()
    assert _run(source, destination).returncode != 0


@pytest.mark.parametrize("fault", ["truncated", "missing_key", "outside_path", "empty_index"])
def test_invalid_checkpoint_never_creates_view(tmp_path: Path, fault: str) -> None:
    source = _checkpoint(tmp_path / "source")
    index = {"weight_map": {"weight": "model-00001.safetensors"}}
    if fault == "truncated":
        shard = source / "model-00001.safetensors"
        shard.write_bytes(shard.read_bytes()[:-1])
    elif fault == "missing_key":
        index["weight_map"] = {"absent": "model-00001.safetensors"}
    elif fault == "outside_path":
        index["weight_map"]["weight"] = "../model-00001.safetensors"
    else:
        index["weight_map"] = {}
    (source / "model.safetensors.index.json").write_text(json.dumps(index))
    destination = tmp_path / "view"
    assert _run(source, destination).returncode != 0
    assert not destination.exists()
