# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Indexed chunk autotuning must preserve input state until the final launch."""

from typing import Any

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules.fla import chunk_delta_h


@pytest.mark.parametrize("exception", [None, RuntimeError("compile failed")])
@pytest.mark.parametrize("use_varlen", [False, True])
def test_indexed_state_hooks_restore_selected_rows(
    exception: Exception | None, use_varlen: bool
) -> None:
    backing = torch.arange(8 * 29, dtype=torch.float32).view(8, 29)
    before = backing.clone()
    pool = backing[:, :24].view(8, 2, 3, 4)
    indices = torch.tensor([5, 1, -1, 7], dtype=torch.int32)
    args = {
        "h0": pool,
        "h0_i": indices,
        "k": torch.empty(1 if use_varlen else 2, 0),
        "cu_seqlens": torch.arange(3) if use_varlen else None,
    }

    chunk_delta_h._save_indexed_autotune_state(args)
    saved_indices, saved = args["_indexed_state_snapshot"]
    assert saved.shape == (2, 2, 3, 4)
    assert saved.numel() == 2 * pool[0].numel()
    pool.index_fill_(0, saved_indices, -7)
    chunk_delta_h._restore_indexed_autotune_state(args, exception)
    assert "_indexed_state_snapshot" not in args
    torch.testing.assert_close(backing, before, rtol=0, atol=0)

    chunk_delta_h._save_indexed_autotune_state(args, reset_only=True)
    assert "_indexed_state_snapshot" not in args
    torch.testing.assert_close(backing, before, rtol=0, atol=0)


@pytest.mark.parametrize("pool,indices", [(None, None), (torch.ones(1, 4), None)])
def test_indexed_state_hooks_ignore_nonindexed_calls(
    pool: torch.Tensor | None, indices: torch.Tensor | None
) -> None:
    args = {"h0": pool, "h0_i": indices}
    chunk_delta_h._save_indexed_autotune_state(args)
    chunk_delta_h._restore_indexed_autotune_state(args, None)
    assert "_indexed_state_snapshot" not in args


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_indexed_state_snapshot_allocation_is_batch_bounded() -> None:
    # A padded outer stride must not make index_select materialize the pool.
    backing = torch.ones(32, 48 * 128 * 128 + 1024, device="cuda")
    pool = backing[:, : 48 * 128 * 128].view(32, 48, 128, 128)
    args = {
        "h0": pool,
        "h0_i": torch.tensor([17, 3, -1, 31], device="cuda", dtype=torch.int32),
        "k": torch.empty(1, 0, device="cuda"),
        "cu_seqlens": torch.tensor([0, 16, 32], device="cuda"),
    }
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated = torch.cuda.memory_allocated()
    chunk_delta_h._save_indexed_autotune_state(args)
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - allocated
    assert extra < 2 * pool[0].numel() * pool.element_size() + 1024**2
    pool[17].zero_()
    pool[3].zero_()
    chunk_delta_h._restore_indexed_autotune_state(args, None)
    assert "_indexed_state_snapshot" not in args
    assert torch.all(backing == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("nonzero_initial_state", [False, True])
@pytest.mark.parametrize("use_varlen", [False, True])
@torch.no_grad()
def test_cold_indexed_chunk_prefill_then_decode(
    monkeypatch: pytest.MonkeyPatch, nonzero_initial_state: bool, use_varlen: bool
) -> None:
    from test_gdn_replay_recurrent import _seq_ref_step

    from tensorrt_llm._torch.modules.fla.chunk import chunk_gated_delta_rule
    from tensorrt_llm._torch.modules.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops import fused_gdn_post_conv

    # A fresh in-process tuning key is required even when JIT binaries are cached.
    tuner = chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64.fn
    monkeypatch.setattr(tuner, "cache", {})
    monkeypatch.setattr(tuner, "cache_results", False)
    original_pre, original_post = tuner.pre_hook, tuner.post_hook
    calls = {"save": 0, "restore": 0}

    def pre(args: dict[str, Any], reset_only: bool = False) -> None:
        original_pre(args, reset_only=reset_only)
        if not reset_only and args["h0_i"] is not None:
            assert args["_indexed_state_snapshot"][1].shape == (1, 48, 128, 128)
            calls["save"] += 1

    def post(args: dict[str, Any], exception: Exception | None) -> None:
        original_post(args, exception=exception)
        assert "_indexed_state_snapshot" not in args
        if args["h0_i"] is not None:
            calls["restore"] += 1

    monkeypatch.setattr(tuner, "pre_hook", pre)
    monkeypatch.setattr(tuner, "post_hook", post)
    torch.manual_seed(20260906)
    hk, hv, dim, tokens = 16, 48, 128, 16
    raw_cpu = torch.randn(tokens + 1, (2 * hk + hv) * dim, dtype=torch.bfloat16) * 0.5
    ba_cpu = torch.randn(tokens + 1, 2 * hv, dtype=torch.bfloat16) * 0.5
    a_log_cpu = torch.full((hv,), -1.5)
    bias_cpu = torch.randn(hv) * 0.1
    initial = (
        torch.randn(hv, dim, dim) * 0.1 if nonzero_initial_state else torch.zeros(hv, dim, dim)
    )
    q_cpu = raw_cpu[:, : hk * dim].view(tokens + 1, hk, dim)
    k_cpu = raw_cpu[:, hk * dim : 2 * hk * dim].view(tokens + 1, hk, dim)
    v_cpu = raw_cpu[:, 2 * hk * dim :].view(tokens + 1, hv, dim)
    g_cpu = -a_log_cpu.exp() * F.softplus(ba_cpu[:, hv:].float() + bias_cpu)
    beta_cpu = ba_cpu[:, :hv].float().sigmoid()
    ref_out, ref_states = _seq_ref_step(initial, q_cpu, k_cpu, v_cpu, g_cpu, beta_cpu, dim**-0.5)
    backing = torch.full((3, hv * dim * dim + 1024), 0.125, device="cuda")
    pool = backing[:, : hv * dim * dim].view(3, hv, dim, dim)
    pool[1].copy_(initial)
    raw, ba, a_log, bias = raw_cpu.cuda(), ba_cpu.cuda(), a_log_cpu.cuda(), bias_cpu.cuda()
    b, a = ba[:, :hv], ba[:, hv:]
    indices = torch.tensor([1] if use_varlen else [1, -1, 2], device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, tokens], device="cuda", dtype=torch.int64) if use_varlen else None
    q, k, v, g, beta = fused_gdn_post_conv(
        raw[:tokens].t().contiguous(),
        None,
        a[:tokens],
        b[:tokens],
        a_log,
        bias,
        hk,
        dim,
        hv,
        dim,
    )

    def prefill() -> torch.Tensor:
        out, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=pool,
            initial_state_indices=indices,
            inplace_indexed_state_update=True,
            cu_seqlens=cu,
            head_first=False,
            use_qk_l2norm_in_kernel=False,
        )
        return out[0]

    def nonindexed_prefill() -> None:
        source = initial.unsqueeze(0).cuda()
        out, state = chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=source,
            output_final_state=True,
            cu_seqlens=cu,
            head_first=False,
            use_qk_l2norm_in_kernel=False,
        )
        torch.testing.assert_close(source[0].cpu(), initial, rtol=0, atol=0)
        torch.testing.assert_close(out[0].float().cpu(), ref_out[:tokens], atol=0.006, rtol=0.02)
        torch.testing.assert_close(state[0].cpu(), ref_states[tokens - 1], atol=0.006, rtol=0.02)

    if not nonzero_initial_state:
        nonindexed_prefill()
        assert calls == {"save": 0, "restore": 0}

    cold_out = prefill()
    cold_state = pool[1].clone()
    assert calls["save"] > 0 and calls["save"] == calls["restore"]
    torch.testing.assert_close(cold_out.float().cpu(), ref_out[:tokens], atol=0.006, rtol=0.02)
    torch.testing.assert_close(cold_state.cpu(), ref_states[tokens - 1], atol=0.006, rtol=0.02)

    pool[1].copy_(initial)
    hook_counts = calls.copy()
    warm_out = prefill()
    assert calls == hook_counts  # No snapshot/copy hooks on the steady path.
    torch.testing.assert_close(warm_out, cold_out, rtol=0, atol=0)
    torch.testing.assert_close(pool[1], cold_state, rtol=0, atol=0)

    row = raw[tokens:]
    decode_out = fused_sigmoid_gating_delta_rule_update(
        A_log=a_log,
        a=a[tokens:],
        dt_bias=bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=row[:, : hk * dim].view(1, 1, hk, dim),
        k=row[:, hk * dim : 2 * hk * dim].view(1, 1, hk, dim),
        v=row[:, 2 * hk * dim :].view(1, 1, hv, dim),
        b=b[tokens:],
        initial_state_source=pool,
        initial_state_indices=indices,
        cu_seqlens=torch.tensor([0, 1], device="cuda", dtype=torch.int64),
        use_qk_l2norm_in_kernel=True,
    )
    torch.testing.assert_close(decode_out[0].float().cpu(), ref_out[tokens:], atol=0.006, rtol=0.02)
    torch.testing.assert_close(pool[1].cpu(), ref_states[-1], atol=0.006, rtol=0.02)
    assert torch.all(pool[0] == 0.125) and torch.all(pool[2] == 0.125)
    assert torch.all(backing[:, hv * dim * dim :] == 0.125)
    if nonzero_initial_state:
        nonindexed_prefill()
        assert calls == hook_counts
