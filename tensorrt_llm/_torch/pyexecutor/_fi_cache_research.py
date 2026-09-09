# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private, single-worker FlashNext cache acquisition, not a tactic freeze."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flashinfer.autotuner import AutoTuner

    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
    from tensorrt_llm.mapping import Mapping

_LOAD = "TRTLLM_FLASHNEXT_FI_CACHE_LOAD"
_SAVE = "TRTLLM_FLASHNEXT_FI_CACHE_SAVE"
_ALLOW_OVERLAP = "TRTLLM_FLASHNEXT_FI_CACHE_ALLOW_OVERLAP"
_ENTRY_LOCK = threading.Lock()
_STARTED = False


def _read_cache(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict) or not isinstance(data.get("_metadata"), dict):
        raise ValueError(f"FI research cache requires save-produced metadata: {path}")
    return data, hashlib.sha256(raw).hexdigest()


def _require_idle(tuner: AutoTuner) -> None:
    if tuner.is_tuning_mode or tuner._active_tuning_contexts:
        raise RuntimeError("FI research cache requires an inactive AutoTuner")


def _validate_scope(llm_args: TorchLlmArgs, mapping: Mapping, checkpoint_dir: str | None) -> None:
    if mapping.rank != 0 or any(
        getattr(mapping, field) != 1 for field in ("world_size", "tp_size", "pp_size", "cp_size")
    ):
        raise ValueError("FI research cache requires rank-zero TP1/PP1/CP1")
    spec = llm_args.speculative_config
    if (
        spec is None
        or spec.decoding_type != "MTP"
        or type(spec.max_draft_len) is not int
        or spec.max_draft_len not in (1, 3)
    ):
        raise ValueError("FI research cache requires explicit MTP draft length 1 or 3")
    allow_overlap = os.environ.get(_ALLOW_OVERLAP, "0")
    if allow_overlap not in ("0", "1"):
        raise ValueError(f"{_ALLOW_OVERLAP} must be 0 or 1")
    if allow_overlap == "1" and spec.max_draft_len != 3:
        raise ValueError("FI overlap research requires MTP draft length 3")
    if not llm_args.disable_overlap_scheduler and allow_overlap != "1":
        raise ValueError("FI research cache requires overlap OFF unless explicitly opted in")
    if not llm_args.enable_autotuner:
        raise ValueError("FI research cache requires global autotuner ON")
    if llm_args.moe_config.backend != "CUTLASS" or not llm_args.moe_config.disable_finalize_fusion:
        raise ValueError("FI research cache requires target CUTLASS with finalize fusion OFF")
    if llm_args.mm_encoder_only or any(
        getattr(llm_args, field) is not None
        for field in (
            "sleep_config",
            "cache_transceiver_config",
            "kv_connector_config",
            "dwdp_config",
        )
    ):
        raise ValueError(
            "FI research cache excludes sleep, disaggregation, DWDP and multimodal encoding"
        )
    if checkpoint_dir is None:
        raise ValueError("FI research cache requires a local text checkpoint config")
    config = json.loads((Path(checkpoint_dir) / "config.json").read_text())
    if not isinstance(config, dict):
        raise ValueError("FI research cache requires a Qwen4Exp text checkpoint")
    flat_text = (
        config.get("architectures") in (["Qwen4ExpForCausalLM"], ["Qwen3_8FlashNextForCausalLM"])
        and "vision_config" not in config
        and "text_config" not in config
    )
    # config_utils flattens these composite checkpoints only when text-only
    # execution is explicitly requested; raw JSON retains the vision config.
    composite_text = (
        (config.get("model_type"), config.get("architectures"))
        in (
            ("qwen4_exp", ["Qwen4ExpForConditionalGeneration"]),
            ("qwen3_8_flash_next", ["Qwen3_8FlashNextForConditionalGeneration"]),
        )
        and config.get("language_model_only") is True
        and isinstance(config.get("text_config"), dict)
        and bool(config["text_config"])
    )
    if not (flat_text or composite_text):
        raise ValueError("FI research cache requires a Qwen4Exp text checkpoint")


class FICacheResearchSession:
    """Save once after final initialization; loaded records may still be extended."""

    def __init__(
        self,
        tuner: AutoTuner,
        output: Path,
        source: Path | None,
        source_sha: str | None,
        device: int,
        log: Callable[[str], None],
    ) -> None:
        self._tuner = tuner
        self._output = output
        self._source = source
        self._source_sha = source_sha
        self._device = device
        self._log = log
        self._saved = False

    def save(self) -> None:
        import torch

        if self._saved:
            raise RuntimeError("FI research cache export may run only once")
        if torch.cuda.current_device() != self._device:
            raise RuntimeError("FI research cache CUDA device changed during initialization")
        _require_idle(self._tuner)
        if self._source is not None and _read_cache(self._source)[1] != self._source_sha:
            raise RuntimeError("FI research cache seed changed during initialization")
        # FI merges existing output files. Save to a fresh staging path, then
        # publish without replacing even a concurrently created destination.
        with tempfile.TemporaryDirectory(prefix=".fi-cache-", dir=self._output.parent) as directory:
            staged = Path(directory) / "cache.json"
            self._tuner.save_configs(str(staged))
            data, digest = _read_cache(staged)
            os.link(staged, self._output)
        self._saved = True
        records = sum(not key.startswith("_") for key in data)
        self._log(
            f"FI research cache saved pid={os.getpid()} device={self._device} "
            f"path={self._output} sha256={digest} records={records} "
            f"profiling_records={len(self._tuner.profiling_cache)}; "
            "persistence only, exercised-key equality unverified"
        )


def begin_fi_cache_research(
    llm_args: TorchLlmArgs,
    mapping: Mapping,
    checkpoint_dir: str | None,
    log: Callable[[str], None],
) -> FICacheResearchSession:
    """Called only by the opt-in creator branch, before either model engine."""
    global _STARTED

    output_name = os.environ.get(_SAVE)
    source_name = os.environ.get(_LOAD)
    if not output_name or (_LOAD in os.environ and not source_name):
        raise ValueError("FI research cache requires nonempty SAVE and, when supplied, LOAD paths")
    output = Path(output_name).absolute()
    source = Path(source_name).absolute() if source_name else None
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"FI research cache output must be unique: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(
            f"FI research cache output directory does not exist: {output.parent}"
        )
    _validate_scope(llm_args, mapping, checkpoint_dir)
    with _ENTRY_LOCK:
        if _STARTED:
            raise RuntimeError(
                "FI research cache requires a fresh process for each model initialization"
            )
        _STARTED = True

    import flashinfer
    import torch
    from flashinfer.autotuner import AutoTuner

    if flashinfer.__version__ != "0.6.18":
        raise RuntimeError("FI research cache supports only FlashInfer 0.6.18")
    device = torch.cuda.current_device()
    if device != mapping.local_rank or torch.cuda.get_device_capability(device) != (12, 1):
        raise RuntimeError("FI research cache requires the already-selected local SM121 device")
    tuner = AutoTuner.get()
    _require_idle(tuner)
    for field in (
        "profiling_cache",
        "_file_configs",
        "_ranked_tactics_cache",
        "_namespaced_records",
        "_dirty_namespaces",
        "_dirty",
        "_dirty_seq",
        "_observed_cache_generations",
    ):
        if getattr(tuner, field):
            raise RuntimeError(f"FI research cache requires empty singleton state: {field}")
    source_sha = None
    if source is not None:
        _, source_sha = _read_cache(source)
        if tuner.load_configs(str(source)) is not True:
            raise RuntimeError("FI research cache seed metadata rejected by FlashInfer")
        if _read_cache(source)[1] != source_sha:
            raise RuntimeError("FI research cache seed changed while loading")
    mode = "seeded" if source else "acquire"
    log(
        f"FI research cache begin pid={os.getpid()} device={device} version={flashinfer.__version__} "
        f"mode={mode} seed={source} sha256={source_sha} "
        f"loaded_records={len(tuner._file_configs)} output={output}"
    )
    return FICacheResearchSession(tuner, output, source, source_sha, device, log)
