# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded host-side dispatch evidence for pinned FI; never enable for timing."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flashinfer.autotuner import AutoTuner

_RUNNERS = {
    ("flashinfer.gemm.gemm_base", "CutlassMxfp8GemmRunner"),
    ("flashinfer.gemm.gemm_mm_mxfp8_cute_dsl", "B12xMxfp8GemmRunner"),
}
_MAX_RECORDS = 4096
_MAX_BYTES = 8 * 1024 * 1024
_MAX_LINE_BYTES = 64 * 1024


class Mxfp8Observer:
    """Observe this fresh singleton through initialization and later eager calls.

    Records describe selection during capture, not execution on graph replay.
    Deduplication keeps first occurrences, not per-call timing or frequency.
    """

    def __init__(
        self, tuner: AutoTuner, path: Path, seed: dict, seed_sha: str | None, device: int
    ) -> None:
        import torch
        from flashinfer.autotuner.autotuner import _tactic_to_json

        original = tuner.choose_one
        if list(inspect.signature(original).parameters) != [
            "custom_op",
            "runners",
            "tuning_config",
            "inputs",
            "kwargs",
        ]:
            raise RuntimeError("FI MXFP8 observer: unsupported choose_one signature")
        if "choose_one" in vars(tuner):
            raise RuntimeError("FI MXFP8 observer: choose_one is already overridden")
        if os.environ.get("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", "0") != "0":
            raise RuntimeError("FI MXFP8 observer excludes bundled legacy cache lookup")
        self._tuner = tuner
        self._original = original
        self._torch = torch
        self._serialize_tactic = _tactic_to_json
        self._device = device
        self._path = path
        if len(seed) > _MAX_RECORDS:
            raise RuntimeError("FI MXFP8 observer seed exceeds record budget")
        seed_json = json.dumps(seed)
        if len(seed_json.encode()) > _MAX_BYTES:
            raise RuntimeError("FI MXFP8 observer seed exceeds byte budget")
        self._seed = json.loads(seed_json)
        self._phase = "model_initialization"
        self._seen: set[str] = set()
        self._records = 0
        self._bytes = 0
        self._calls = 0
        self._failure: str | None = None
        with path.open("x") as stream:
            stat = os.fstat(stream.fileno())
            self._identity = (stat.st_dev, stat.st_ino)
        self._emit(
            {
                "kind": "header",
                "schema": 1,
                "pid": os.getpid(),
                "device": device,
                "seed_sha256": seed_sha,
                "seed_available": seed_sha is not None,
                "autotuner_source_sha256": hashlib.sha256(
                    Path(inspect.getfile(type(tuner))).read_bytes()
                ).hexdigest(),
                "max_records": _MAX_RECORDS,
                "max_bytes": _MAX_BYTES,
                "diagnostic_not_timing": True,
                "graph_replay_observed": False,
                "deduplication": "first occurrence per phase and complete record",
            }
        )
        tuner.choose_one = self.choose_one

    def _emit(self, record: dict) -> None:
        payload = json.dumps(record, sort_keys=True, allow_nan=False, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode()).hexdigest()
        if digest in self._seen:
            return
        line = (
            json.dumps(
                {"sequence": self._records, "first_call": self._calls, **record},
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode()
        if (
            self._records >= _MAX_RECORDS
            or self._bytes + len(line) > _MAX_BYTES
            or len(line) > _MAX_LINE_BYTES
        ):
            raise RuntimeError("FI MXFP8 observer exhausted its bounded output budget")
        fd = os.open(self._path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
        with os.fdopen(fd, "ab", buffering=0) as stream:
            stat = os.fstat(stream.fileno())
            if (stat.st_dev, stat.st_ino) != self._identity:
                raise RuntimeError("FI MXFP8 observer output file was replaced")
            if stat.st_size != self._bytes:
                raise RuntimeError("FI MXFP8 observer output file size changed externally")
            if stream.write(line) != len(line):
                raise OSError("FI MXFP8 observer short write")
        self._seen.add(digest)
        self._records += 1
        self._bytes += len(line)

    def set_phase(self, phase: str) -> None:
        with self._tuner._lock:
            self._check_health()
            self._phase = phase
            self._emit({"kind": "phase", "phase": phase, "calls_so_far": self._calls})

    def _check_health(self) -> None:
        if self._failure is not None:
            raise RuntimeError(f"FI MXFP8 observer previously failed: {self._failure}")

    def _tensor_metadata(self, value: object) -> dict:
        if not isinstance(value, self._torch.Tensor):
            raise RuntimeError("FI MXFP8 observer expects tensor inputs except out_dtype")
        if value.device.type != "cuda" or value.device.index != self._device:
            raise RuntimeError("FI MXFP8 observer input is not on the selected CUDA device")
        return {
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }

    def _tactic(self, tactic: object) -> object:
        # These runners return only integers or nested tuples/lists of them.
        # Reject tensor/foreign iterables before FI's generic serializer can
        # iterate them and accidentally read tensor data.
        def check(value: object) -> None:
            if type(value) in (int, bool, str, float, type(None)):
                return
            if isinstance(value, (tuple, list)):
                for item in value:
                    check(item)
                return
            raise RuntimeError("FI MXFP8 observer: unsupported tactic value")

        check(tactic)
        return self._serialize_tactic(tactic)

    def _profile_records(self) -> None:
        for key, (tactic, _) in self._tuner.profiling_cache.items():
            if key.custom_op == "mxfp8_gemm":
                value = [key.runner_class_name, self._tactic(tactic)]
                self._emit(
                    {
                        "kind": "profiled_key",
                        "phase": self._phase,
                        "file_key": key.file_key,
                        "value": value,
                        "seed_member": key.file_key in self._seed,
                        "seed_equal": self._seed.get(key.file_key) == value,
                    }
                )

    def choose_one(
        self, custom_op: str, runners: list, tuning_config: object, inputs: list, **kwargs: object
    ) -> tuple:
        if custom_op != "mxfp8_gemm":
            return self._original(custom_op, runners, tuning_config, inputs, **kwargs)
        # Same RLock as FI: snapshot and delegation see one coherent cache state.
        with self._tuner._lock:
            self._check_health()
            try:
                return self._observe(custom_op, runners, tuning_config, inputs, kwargs)
            except (OSError, ValueError, RuntimeError, TypeError) as error:
                self._failure = str(error)
                raise

    def _observe(
        self, custom_op: str, runners: list, tuning_config: object, inputs: list, kwargs: dict
    ) -> tuple:
        tuner = self._tuner
        if kwargs or len(inputs) != 7 or not runners or len(runners) > 2:
            raise RuntimeError("FI MXFP8 observer: unsupported call overload")
        if custom_op in tuner._effective_skip_ops:
            raise RuntimeError("FI MXFP8 observer excludes skip_ops bypass")
        if self._torch.cuda.current_device() != self._device:
            raise RuntimeError("FI MXFP8 observer CUDA device changed")
        metadata = [
            self._tensor_metadata(value) if i != 4 else {"dtype": str(value)}
            for i, value in enumerate(inputs)
        ]
        if not isinstance(inputs[4], self._torch.dtype):
            raise RuntimeError("FI MXFP8 observer expects an output dtype at input 4")
        a_shape, b_shape, out_shape = (metadata[i]["shape"] for i in (0, 1, 5))
        if (
            len(a_shape) != 2
            or len(b_shape) != 2
            or a_shape[1] != b_shape[0]
            or out_shape != [a_shape[0], b_shape[1]]
        ):
            raise RuntimeError("FI MXFP8 observer requires A[M,K], B[K,N], out[M,N]")
        effective = tuning_config
        if tuner._override_tuning_buckets is not None or tuner._override_round_up:
            effective = tuner._apply_tuning_overrides(tuning_config)
        shapes = tuple(tuner._get_input_sizes(inputs))
        candidates = []
        keys = []
        for runner in runners:
            cls = type(runner)
            if (cls.__module__, cls.__name__) not in _RUNNERS:
                raise RuntimeError("FI MXFP8 observer: unsupported runner class")
            extras = runner.get_cache_key_extras(inputs)
            if not isinstance(extras, tuple) or len(extras) != 0:
                raise RuntimeError("FI MXFP8 observer: unsupported runner key extras")
            key = tuner._get_cache_key(custom_op, runner, shapes, effective, extras)
            keys.append(key)
            memory = tuner.profiling_cache.get(key)
            loaded = tuner._file_configs.get(key.file_key)
            candidates.append(
                {
                    "module": cls.__module__,
                    "class": cls.__name__,
                    "qualname": cls.__qualname__,
                    "file_key": key.file_key,
                    "nearest_profile": key.nearest_profile,
                    "memory_before": self._tactic(memory[0]) if memory is not None else None,
                    "memory_present": memory is not None,
                    "loaded_before": [loaded[0], self._tactic(loaded[1])]
                    if loaded is not None
                    else None,
                    "seed_member": key.file_key in self._seed,
                }
            )
        self._calls += 1
        capture = bool(self._torch.cuda.is_current_stream_capturing())
        tuning = bool(tuner.is_tuning_mode)
        # Pass the original config: FI applies its own effective override once.
        result = self._original(custom_op, runners, tuning_config, inputs)
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("FI MXFP8 observer: unsupported choose_one return")
        runner, tactic = result
        selected = next((i for i, candidate in enumerate(runners) if candidate is runner), None)
        if selected is None:
            raise RuntimeError("FI MXFP8 observer: selected runner not in candidates")
        key = keys[selected]
        value = [type(runner).__name__, self._tactic(tactic)]
        self._emit(
            {
                "kind": "choice",
                "phase": self._phase,
                "capture": capture,
                "tuning": tuning,
                "m": a_shape[0],
                "n": b_shape[1],
                "k": a_shape[1],
                "device": self._device,
                "inputs": metadata,
                "candidates": candidates,
                "selected_index": selected,
                "file_key": key.file_key,
                "selected_value": value,
                "seed_member": key.file_key in self._seed,
                "seed_equal": self._seed.get(key.file_key) == value,
            }
        )
        if tuning:
            self._profile_records()
        return result
