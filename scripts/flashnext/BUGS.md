<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Flash Next Compatibility and Bug Ledger

Baseline: TensorRT-LLM `70feda63959fedf0f5f12c5e8c771e5392ce2d25`.
Checkpoint: `925d7be6c14c6c9442ef83e8f05b5a3c39304f69`.
No new NVIDIA issues or PRs have been opened for this experiment.

## Upstream Compatibility Gaps

| Item | Evidence | Resolution and validation |
| --- | --- | --- |
| Old Flash Next identifiers | Checkpoint config uses Qwen3.8 Flash Next names; current main implements the architecture as Qwen4Exp. | `f74e6657` registers narrow aliases and normalizes config names. 56 isolated config tests, including existing regressions; full-runtime tests pending. |
| Mixed precision namespace/fusion | Quant policies reference source modules while runtime combines GDN and QKV projections. MTP metadata repeats relative 0 and absolute 48; weights only use relative 0. | `f74e6657` coalesces identical policies and rejects conflicts; no quantization-algorithm replacement. |
| GDN MXFP8 scale fusion | Mapper combines Q/K/V/Z and B/A weights without corresponding per-32-element UE8M0 scales. Lazy slices also lack tensor metadata assumed by mapper operations. | `2c844dd9` gives scales the identical TP row ordering and exposes mmap tensor views. Seven isolated tests pass, including real SafeTensors and scalar lifetime. |
| Packed NVFP4 PLE | Existing dense/scaled-FP8 PLE loader cannot retain paired FP4 code/scale shards. BF16 expansion is about 95.4 GiB for the table alone. | `23872afa` adds bounded CPU gather; direct read-only GPU delegate is being validated. No full-model memory/performance claim yet. |

## Upstream Defects Found

### MXFP8 Native Architecture Gate

- Status: `ed8858ea`, component-tested; not yet full-model tested.
- Site: `_torch/modules/linear.py`, `_mxfp8_cutlass_op_available`.
- Root cause: Python tests compute-capability major >=10, admitting SM121;
  native MXFP8 GEMM dispatch only supports the SM100 family.
- Consequence: after a current native build exposes the operator, SM121 can
  enter an unsupported GEMM instead of a supported backend or reference path.
- Fix: align native eligibility and use FlashInfer 0.6.18 CUTLASS MXFP8 on
  validated SM120/121 shapes. Keep an explicit BF16 dequantization fallback
  for unsupported shapes such as GDN B/A output width 96.
- Tests: 36 isolated dispatch/engine tests pass, including SM100 regression cases.
  Actual edited methods ran on SM121 at `(M,N,K)` = `(1,96,2560)`,
  `(1,128,2560)` and `(4,16384,2560)` with the expected paths and zero observed
  maximum difference from the corresponding numerical references.
- Numerical caveat: FlashInfer was compared with dequantized MXFP8 operands,
  not unquantized BF16 activations. The reference fallback intentionally
  retains the existing higher-precision activation behavior.
- Raw evidence: sibling `flashnext-results/mxfp8-sm121-captured-results.json`
  and its named logs/repro scripts. Smoke timings are not full-model benchmarks.

### PLE Asynchronous Prefetch Buffer Lifetime

- Status: independently identified, reproduced on SM121, fixed in `7f4d3f0e`.
- Site: `_torch/modules/qwen4_exp/ple.py`, `start_prefetch`, `abort_prefetch`,
  `_get_prefetch_buffer`.
- Root cause: only lookup IDs are recorded on the prefetch stream. Abort drops
  prefetch state without joining that stream; a larger retry can replace the
  cached output allocation while the old gather is still writing it.
- Fix: record the output buffer on the prefetch stream before launch, allowing
  the allocator to defer reuse. Do not introduce a per-request synchronization.
- Scope: relevant to asynchronous host-gather paths, including the new GPU
  NVFP4 delegate. The initial blocking CPU copy did not expose this window.
- Test: delayed prefetch, abort, buffer growth and forced allocator reuse.
  Removing the fix reproduces early reuse and guard corruption; the fixed
  version prevents reuse and preserves the guard and retry output. Two GPU
  regressions pass. Evidence: `ple-prefetch-lifetime-20260906.json` in the
  sibling results directory. Full-model validation remains pending.

## Defects Caught in This Fork

These are experimental implementation mistakes, not claims about upstream.

| Item | Reproduction | Fix and state |
| --- | --- | --- |
| CPU gather output aliases IDs | Two CPU int64 IDs share storage with BF16 output and `max_gather_rows=1`. Writing the first chunk corrupts IDs needed for the next. Independently reproduced. | Reject overlapping address spans before gathering, including strided ID views. Two overlap regressions plus a disjoint-shared-storage case pass. |
| GPU loader index limit too small | Actual pinned index is 33,074,286 bytes; initial helper capped both index and per-file header at 16 MiB. | Separate bounded 64 MiB index allowance from 16 MiB per-file header limit. Oversized-index and bounded-read regressions pass; actual checkpoint parsing pending transfer. |
| SM12x automatic backend loses user intent | Converting default/auto to explicit `flashinfer` makes a missing engine warmup raise instead of retaining a supported untuned path. Reference-only shapes also requested unnecessary tuning. | Preserve `auto`, exclude reference-only shapes from tuning, and keep the prepared FlashInfer layout/default tactic on SM12x if tuning cannot run. Six independent engine regressions pass in `ed8858ea`. |
| Writable GPU mmap destroys reclaimability | Isolated probe found GPU access through default writable SafeTensors mappings created private-dirty/anonymous pages. | GPU delegate opens independent read-only mappings. Small-file tests observed zero anonymous/private-dirty/locked pages and successful own-file reclaim. Full-model pressure validation pending. |
| Benchmark can misstate speculative performance | Text-event count is not token count; a one-event response cannot expose a decode interval. Failed streams could discard a whole batch's evidence. | Use server token usage, null unobservable decode estimates, validate termination/fixed length, preserve errors and partial results. 15 API/checkpoint helper tests pass. |

## Build and Admission Findings

- The first complete native compile stopped on a Git LFS pointer in
  `trtllmGen_bmm_export/KernelMetaInfo.h`. Its pinned 7,286,260-byte object was
  fetched and verified against SHA256
  `08050119e223e4685acf4d337854f964d8eba22ef4878741550ca007c6762349`.
  Resuming the existing build completed successfully. No C++ logic patch.
- Docker declared memory and memory-swap limits of 84 GiB, but the effective
  host cgroup initially allowed unlimited swap. Reapplying the Docker limits
  set `memory.swap.max=0`; actual readings confirm the correction and no OOM
  events. Root cause of the initial discrepancy is not established.
- Header-only audit found that Qwen4Exp's mapper returns a plain dictionary,
  losing lazy loader consumption and indexed prefix lookup. About 2.074 GiB
  of fused CPU tensors remains referenced until loading completes. Fixed in
  `2d95b4db`; 24 real-runtime consumption/GDN/PLE regressions pass. Peak memory
  reduction remains unmeasured; no measured OOM claim.
- QSA short-prompt eager geometry passed nine standalone SM121 kernel checks.
  Native row-range Top-K additionally passed exact-set/padding checks and
  changed-input graph replay at K=512, including the radix dispatch boundary.
  Complete dense/sparse attention-layer integration remains pending.
  Prefill graphs encounter data-dependent boolean indexing, and captured
  dense/sparse selection cannot change merely by changing replay lengths.
  Keep graphs off for initial validation; test threshold crossings and MTP
  state rollback separately. These probes do not prove full-model correctness.

## Unvalidated Performance Opportunities

1. FlashInfer 0.6.18 B12x MXFP8: six real GDN projection shapes (M=1/4/16)
   passed fractional-weight, independent dequantized-reference and changed-input
   graph tests. B12x and CUTLASS outputs were bit-identical. Three repetitions
   of 100 iterations found large component gains, but concurrent CPU build,
   differing cache behavior and per-call scope preclude a full-model claim.
   Raw results and the report are in the sibling results directory under
   `mxfp8-b12x-graph-*`. Test an opt-in selector after a full-model baseline.
2. Cache or pad the small N=96 GDN B/A projection if its reference path is
   material in the full trace. Do not spend memory or change quantization
   behavior without a measured benefit.
3. Compare existing CUTLASS and B12x MoE backends after coherent generation;
   backend names alone do not establish equal numerics or better throughput.
4. MTP length, graph capture and recurrent-state precision are tuning axes,
   but only after no-MTP correctness and stable memory admission.
