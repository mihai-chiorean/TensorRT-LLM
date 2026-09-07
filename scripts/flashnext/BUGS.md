<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Flash Next Compatibility and Bug Ledger

Baseline: TensorRT-LLM `70feda63959fedf0f5f12c5e8c771e5392ce2d25`.
Checkpoint: `925d7be6c14c6c9442ef83e8f05b5a3c39304f69`.
No new NVIDIA issues or PRs have been opened for this experiment.

## Latest Integration Evidence

The entries below retain historical discovery-stage test counts. As of
September 7, 02:35 UTC, the combined text-only path loads and generates on
Spark SM121, with the exact source and artifact prefixes in
`../../FLASHNEXT_STATUS.md`. CUTLASS eager, B12x dense eager, and B12x dense
decode graphs each completed 32 fixed-output API requests. The small strict
quality suite remains six passed, one arithmetic failure, one unscored.
This qualifies short text serving at a 2048-token limit, not multimodal or
long-context support. The cache workaround and CPU-source scale fix have
therefore progressed beyond their earlier pending full-model retries.

MTP3 also completed two instrumented 32-token requests with 46/57 draft tokens
accepted, no OOM, and clean shutdown. Eager MTP then completed 42 API requests;
MTP with decode graphs completed 74. Both MTP configurations scored 5/2/1 in
the small strict suite: inventory arithmetic and an unwanted Markdown fence
around a correct JSON-filter answer. Preserve those failures. All 64 fixed-output
requests in the extended MTP graph pilot had valid usage and exact lengths.

### Decode Graph Repeatability Remains Open

The no-MTP graph API pilot produced coherent output and higher throughput,
but C1 repetitions matched zero of eight response texts; C2 repetitions matched
all eight. Six of eight quality texts differed from eager while strict outcomes
were unchanged. Cause is unlocalized: do not infer benign rounding, corruption,
or numerical equivalence solely from these API results. QSA's short-sequence
graph family already exists; its Python threshold branch is not independently
evidence of a bug. Preserve this as an unresolved correctness qualification,
with raw reports under `20260906-trt-graphs-1914` in sibling results.
Eager MTP repeats matched 8/8 at C1 but 1/8 at C2. MTP graphs' first two pairs
show the same match counts. The vLLM reference matched 0/8 at both concurrencies,
with prefix caching enabled. These controls do not establish a common cause.
The CPU-validated request-order probe is prepared but has not been run live.

### Additional Source Findings, Deferred

- MTP graph warmup at limit 2048/draft3 can produce duplicate short-family
  keys instead of a near-limit long key. An absent key falls back to eager;
  this is not evidence of wrong-family replay. Boundary workload qualification
  and a focused key-generation regression remain future work.
- The W4A16 loader can replace unequal gate/up global scales with their maximum
  without preserving each half's effective scale. A source-extracted CPU
  counterexample reproduces this conditional defect. It does **not** affect
  this checkpoint's MTP experts: direct inspection found all 512 gate/up pairs
  bitwise equal and all 1536 gate/up/down globals finite and positive. No large
  weight payloads or GPU operations were used for that inspection. Do not add
  a workaround for a condition this checkpoint does not trigger.

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
| MXFP8 scale layout rejects CPU sources | First full-model smoke fails at layer 0's GDN projection: integrated-GPU `load_weight_shard` deliberately retains CPU scales, while the new FlashInfer interleave branch requires CUDA. | `c03087c7` reuses native CPU scale packing and retains FlashInfer for CUDA sources. Eight actual runtime cases pass, including byte-exact padded layouts, CPU-source loading into CUDA parameters and 16 numerical forwards; 42 CPU checks pass. Introduced by this fork's dispatch path, not an upstream loading failure. Full-model retry pending. |
| Writable GPU mmap destroys reclaimability | Isolated probe found GPU access through default writable SafeTensors mappings created private-dirty/anonymous pages. | GPU delegate opens independent read-only mappings. Small-file tests observed zero anonymous/private-dirty/locked pages and successful own-file reclaim. Full-model pressure validation pending. |
| Benchmark can misstate speculative performance | Text-event count is not token count; a one-event response cannot expose a decode interval. Failed streams could discard a whole batch's evidence. | Use server token usage, null unobservable decode estimates, validate termination/fixed length, preserve errors and partial results. 15 API/checkpoint helper tests pass. |
| Smoke underbudgets recurrent states | The second run loaded all modules, then V2 Mamba rejected a 212,893,030-byte GPU quota below its 347,713,584-byte live-state/page minimum. | `4c2ec307` requests 1 GiB and fraction 0.5; `a2fbebd2` removes the token-derived restriction and sets explicit average sequence length 2048. Native constraint floors can exceed requested quotas. The old 0.05 fraction was too small. This is a harness configuration error, not a reason to weaken the runtime guard. Full-model retry pending. |

## Build and Admission Findings

- The third full-model run completed loading and warmup, then stalled on its
  first request. Although `max_tokens=4096` was requested, V2 cache sizing
  advertised only 64 usable tokens and lowered its sequence limit to 64;
  the LLM arguments still advertised 2048. One worker thread consumed about
  one CPU core while GPU activity was consistent with unrelated Isaac Sim.
  The owned guard stopped cleanly; no OOM occurred. A cache-only repro admits
  prefill, then exhausts growth at capacity 65, after 28 simulated output
  tokens. It does not independently reproduce the scheduler retry loop.
  Removing `max_tokens` and using `avg_seq_len=2048` passes complete allocation
  lifecycles for two 37+384-token requests. Six configurations were compared;
  host-cache disabling alone does not fix the capacity collapse. C++
  `StorageManager::computeSlotCountForLevel` also raises requested
  quota to its minimum-slot floor, so requested byte caps do not alone prove
  an upper bound on actual native allocation.

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

## Additional Correctness Findings

### GDN Autotuning Corrupts Indexed State

- Status: reproduced on SM121, fixed in `cdb81425`; not an SM121-only claim.
- Site: `_torch/modules/fla/chunk_delta_h.py`, chunk-H autotuner and indexed
  final-state alias to `h0`; production caller is `mamba/gdn_mixer.py`.
- Root cause: benchmark trials repeatedly advance the caller's input state.
  A reset-state warm call passes, while the original cold call fails an
  independent recurrent reference. Startup warmup may hide this in some paths.
- Fix: snapshot only indices used by the kernel grid, restore after each trial,
  and leave final/cache-hit execution untouched. Never zero continued state or
  clone the complete pool. Ignore unused index-buffer tails.
- Tests: eleven CPU/GPU cases pass, including cold/warm equivalence, nonzero
  state, decode, call ordering, padding and bounded allocation. Two active
  FP32 slots use about 6 MiB even in a 32-slot pool. Exception cleanup is tested
  at hook level. No full-model or cold graph-capture claim.
- Separate issue: FLA's two-warp workaround followed an SM103 reproduction.
  Our SM121 fixed-config probe passed 520 launches across 52 feasible shape/
  config combinations, including four/eight warps. No evidence here justifies
  copying that restriction; this is not proof for arbitrary long contexts.

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

## Potential MTP Prefix-Reuse Defect

- Status: source-only finding, not reproduced with a live prefix hit. Keep
  Qwen4Exp MTP3 reuse OFF; no runtime fix or speed claim in this entry.
- Trigger: V2 separate target/draft KV managers, an explicit reusable hybrid
  state snapshot, and a subsequent target prefix hit. The reuse boolean alone
  does not establish this trigger: default snapshot interval/offsets are empty.
- Trace: `tensorrt_llm/_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py`,
  `_prepare_context_impl`, advances the shared request cursor after matching
  target tokens. `_prepare_draft_resources` instead creates a fresh native
  draft cache with lookup tokens `None`, stops committing, and reserves the
  prefix capacity without restoring matching draft K/V. The native bridge is
  `_create_kv_cache`; reserving pages does not initialize valid model state.
  Native `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCacheManager.cpp:174`
  skips `matchReuse` for an empty token span; `kvCache.cpp:1863`
  `stopCommitting` finalizes commit state, not model-prefix reconstruction.
- Follow-through: `_torch/pyexecutor/model_engine.py` builds only the remaining
  context chunk. `_torch/speculative/eagle3.py`, `prepare_1st_drafter_inputs`
  and `_run_draft_forward`, passes that chunk's target hidden states into the
  MTP layer. `_torch/speculative/interface.py`,
  `prepare_attn_metadata_for_draft_replay`, swaps cache layouts/pointers, not
  missing prefix payloads. `_torch/models/modeling_qwen4_exp.py`,
  `Qwen4ExpMTP.forward`, runs the ordinary attention layer; the auxiliary PLE
  and indexer commit handlers do not rebuild draft-prefix K/V.
- Risk: draft attention may consume prefix storage that was never populated
  for this request. No initialization/recompute route was found in this bounded
  active-path audit. This does not prove target-token corruption, a particular
  acceptance effect, or an upstream-wide defect.
- Smallest proposed fix: reject this Qwen4Exp MTP + separate V2 draft-cache +
  target-reuse combination after effective cache selection, with an explicit
  message to disable reuse. Do not silently zero pages, copy target K/V into
  differently weighted draft layers, or treat pointer rebinding as restore.
  Actual support needs coordinated target/draft prefix matching and retention,
  or valid prefix reconstruction with the required target hidden states.
  Neither implementation is part of the current checkpoint.
- Evidence: sibling results `qsa-hybrid-mtp-prefix-reuse-eligibility.md` and
  `mtp-draft-prefix-initialization-followup.md`. Reuse-OFF batch-4/8 native
  allocation passes do not cover this finding; no new GPU/API test was run.
