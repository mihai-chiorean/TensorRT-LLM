<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Flash Next on DGX Spark

## Objective

Run `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` correctly in TensorRT-LLM on
Spark SM121, then compare matched workloads against vLLM. No performance win
is claimed until measured. Keep spark-094a's working deployment unchanged.

## Current Checkpoint

### Active Follow-up: September 7, 16:00 UTC

- Same-host vLLM cap4 baseline succeeded: C1 **37.2720** and C2 **59.0899**
  aggregate tok/s, one round each; C4 two-round pooled **88.0300**, C8 queued
  above active cap4 **88.9154**. All 192 fixed-output requests succeeded.
  C1/C2 second rounds were omitted to respect the guard; C4/C8 repeats were
  prioritized before round2 began. Do not describe this as a full matrix.
- Both vLLM quality phases scored 6 pass / 1 fail / 1 unscored. Corpus draft
  acceptance counters were 16604/23979 (69.244%); prefix hits stayed zero.
  These counts are not yet matched to TRT counting conventions. Tiny quality
  checks and repeat text differences do not establish general accuracy parity.
- Same hardware still does not mean identical settings: vLLM has 12.86 GiB
  allocated FP8 KV, context262144 and the original multimodal model profile;
  TRT uses a requested 2 GiB BF16 KV quota, context2048 and a text-only view.
  Both use FP32 recurrent state per source/config audit. vLLM tunes fused MoE
  GEMMs; TRT's main autotuner is disabled, although its FI MXFP8 graph warmup
  already performs separate tuning. Main autotuner ON is the next config lever.
- Cap4 parent termination completed **15:55:20 UTC**, no remaining workers;
  GPU list empty, container sleep-only and port8013 free independently checked.
  Cap8 now runs in the SAME physical `flashnext-vllm-replica-cap4` container,
  changing active capacity to8 and graph token sizes to4/8/12/16/20/24/28/32.
  Prefix: `vllm-cap8-reusecap4-20260907T155600Z`, endpoint localhost18084.
  It retains the 25-minute watchdog and 84 GiB / swap0 / 12 CPU limits.
- After cap8 cleanup: guarded actual-checkpoint non-atomic MTP comparison,
  then the reviewed M16/M32 dense B12x qualification if ready. Neither has
  executed on GPU yet. Separate autotuner and overlap trials must preserve
  CUTLASS-draft controls rather than stack unresolved changes.
- Full same-host baseline: sibling results
  `20260907-vllm-3883-cap4-1530-report.md`, with raw cells, acceptance counters,
  configuration, phase order, omitted cells, resource extrema and hashes.
  Isaac remains recreated/stopped; spark-094a remains unchanged.

### Earlier Follow-up: September 7, 15:31 UTC

- MTP-only B12x batch-8 completed: C4 rounds **54.2628 / 84.9471**,
  pooled **66.2232** aggregate tok/s; C8 **118.3147 / 120.7270**, pooled
  **119.5087**. CUTLASS-draft B8 pooled C8 was **98.6777**. This is an
  observed 21.1% deployment-profile gain, not yet a qualified keeper.
  All 128 fixed-output requests succeeded; tiny C8 quality remained 6/1/1.
  Preserve both rounds: first-pass C4 slowdown is substantial in both arms.
- Actual-checkpoint component qualification found small numerical errors
  versus independent references, but B12x M1/2/4 differed on all 42 adjacent
  repeated calls; M8 was exact on all 14. CUTLASS was exact on all 56.
  Small-M BF16 atomic accumulation is a source-backed hypothesis, not proof
  that all full-model differences are harmless. A separate non-atomic probe
  is being implemented by Ampere on **gpt-5.6-terra**, with parent integration
  and independent review before any GPU grant.
- Full-model token-ID diagnostic completed 40/40 protocol-valid requests.
  Only 3/24 sentinel comparisons were exact; all eight immediate repeats
  differed before any short/long stress phase. Keep the candidate experimental.
  Artifacts: sibling results `20260907-mtp-b12x-b8-tokenids-1522/` and
  `20260907-trt-mtp-only-b12x-b8-1513-report.md`.
- Parent deliberately stopped the completed TRT server. Guard cleanup at
  **15:29:42 UTC** reports no remaining owned processes. Same-host vLLM cap4
  is now loading in `flashnext-vllm-replica-cap4`, under 84 GiB / swap0 /
  12 CPU limits and the 25-minute guard. Log prefix in the replica's host
  `logs/` directory: `vllm-cap4-20260907T153030Z`; local API tunnel 18084.
  No simultaneous TRT model is running. Spark-094a remains unchanged.
- Same-host replica preparation finished before the TRT B8 measurement:
  exact pinned image and nine recipe patches, original checkpoint, and
  byte-verified packed PLE table. These AGPL recipe files stay outside the
  Apache-licensed TRT source. Model admission and same-host performance
  remain pending; do not substitute older different-host results.

### Resumed Checkpoint: September 7, 14:55 UTC

- User resumed the performance effort. The previous server ended on its
  25-minute watchdog at 07:46:35 UTC with no remaining owned processes or OOM.
  No model benchmark continued during the interruption. GPU is now clear;
  host available memory is about 116 GiB. Isaac remains recreated/stopped.
- Completed MTP-only B12x batch-4 matrix: two-round pooled aggregate tok/s
  C1 **32.0742**, C2 **55.9927**, C4 **84.4865**, C8 **88.1455** (queued above
  active cap4). Baseline CUTLASS draft: **28.9893 / 47.9072 / 72.8494 / 74.8724**.
  Unchanged vLLM reference: **35.6572 / 58.0639 / 77.4664 / 87.3707**.
  These are deployment-profile observations with warmup and host differences,
  not isolated causal gains or a general engine-win claim. All 256 fixed-output
  requests succeeded. Minimum host available memory was 31.8778 GiB.
- MTP-only B12x is **experimental, not a keeper yet**: C1 repeat text matched
  only 2/32 versus baseline 32/32. Tiny quality scores remain 5/2/1 at C1 and
  6/1/1 at C4 (pass/fail/unscored); coherent output does not resolve numerical
  repeatability. Paired real-checkpoint MTP expert eager/graph numerics are next.
- Full report: sibling results
  `20260907-trt-mtp-only-b12x-b4-0721-report.md`; reference matrix is complete in
  `20260907-vllm-corpus32-0636-final-report.md`. All rounds retained.
- Concurrent lanes: Schrodinger owns bounded GPU MTP component qualification;
  Hubble owns isolated same-host vLLM image/recipe/PLE preparation, no GPU launch
  yet; Archimedes audits speculative repeatability. Heavy acquisition I/O
  excludes primary performance measurements. Spark-094a stays read-only.
- W4A4 B12x wrapper capacity candidate `3483785a` is reviewed separately and
  **not integrated**. Installed FlashInfer static padding means cap512-to64
  saves only about 0.863 GiB across 48 layers, not the initially hypothesized
  18 GiB. Source proof and 199 CPU tests do not establish GPU qualification.
  Do not attempt global B12x graphs on the incorrect smaller-memory estimate.

### Earlier Overnight Sequence

- September 7 overnight follow-up: user authorizes temporarily stopping competing
  workloads on Spark-3883. Isaac Sim is the sole other GPU client, about 17 GiB
  resident and 1.6 CPU cores. Its original Docker inspect is archived as
  `20260907-isolation-isaac-before.json` in the results directory; restart policy
  is `no`. Stopped at 06:22 UTC; it had `AutoRemove=true`, so restoration requires
  recreating the container from its archived configuration/original launcher.
  Recreated, not started, as `eaf2dd88a9ef` with exact configuration and bind
  comparison; writable-layer/unsaved state was not preserved. All nine mounted
  data paths remain. Start this recreated container only after owned model
  workers are cleaned up. CVAT and idle BuildKit containers
  remain running. First repeat unchanged batch-2 MTP3/graphs, then qualify and
  measure batch/concurrency 4 and 8. Spark-094a remains unchanged.
- Isolated batch-2 pilot completed: C1 23.1439 / 22.7565 tok/s, C2 40.0270 /
  42.4992 aggregate tok/s, no API errors. Tiny quality set unchanged at 5 pass,
  2 fail, 1 unscored. Host available minimum 24.4901 GiB; no OOM; verified
  watchdog cleanup at 06:35:57 UTC. Isaac contention does not explain the whole
  performance gap. Prefix-cache/TTFT differences need separate measurement.
- Real target/draft native cache probes pass batch 4 and 8 at requested 2 GiB
  frontend quota, including all-request 2048-capacity occupancy. Allocation-only,
  not model correctness. Batch-4 full server started at 06:39 UTC under the same
  guard; prefix caching and overlap remain off for this scaling experiment.
- Isolated 32-prompt, fixed-128 results: batch 4 C1 25.8194/33.0465, C2
  45.6901/50.3504, C4 71.2175/74.5579 tok/s. Batch 8 C4 50.5436/73.9342,
  C8 95.9886/101.5218 aggregate tok/s. All requests succeeded. Reference C4
  69.4645/87.5518 and C8 88.9388 on its first pass, but reference active cap
  remains FOUR. Do not call the B8 result an equal-cap engine win. Cold/warm
  and repeat variation are material; retain both rounds. Raw artifacts use
  `20260907-trt-isolated-b4-0639`, `...b8-0700` and
  `20260907-vllm-corpus32-0636` prefixes in sibling results.
- Worker-only CPU affinity A/B/A: original 32.6969, all X925 34.2247, exact
  original masks restored 34.3395 tok/s at pilot C1. No benefit established;
  not a keeper. Twenty per-thread identities/masks were saved and restored,
  including helpers inherited on CPU 2. No global affinity/cgroup changes.
- Batch-8 watchdog cleanup completed 07:19:46 UTC; no remaining owned workers,
  no OOM, minimum host available 23.0140 GiB. Isaac remains recreated/stopped.
- Experimental default-off MTP-only B12x selector integrated as `72f24679`
  from `596d9756`; independent 44 isolated CPU tests and touched-file hooks
  pass. Only `modeling_qwen4_exp.py` was staged to the runtime, SHA256
  `7d19b8930ab0056b9ee19f374eac2fcf8baf54d3aa4ca6a4050d65e0d5f85a2b`.
  First full-model B4 graph experiment started 07:21 UTC, prefix
  `flashnext-mtp-b12x-b4-20260907-0721`. GPU numerics/performance are pending;
  this is not promoted over the working CUTLASS-MoE baseline.

### Historical Shared-Host Checkpoint (Before Isolation)

- **Working text-only port; benchmark checkpoint complete, not a vLLM win.**
  Best tested configuration is opt-in B12x dense + MTP3 + decode graphs:
  four fixed-output pairs pooled to 26.0622 tok/s at C1 and 38.9456 aggregate
  tok/s at C2, versus reference 36.0620 / 56.5436 across two pairs.
  Runs vary substantially; these are shared-host deployment comparisons with
  differing KV precision, prefix caching and serving limits, not engine isolation.
  See [bounded run guide](scripts/flashnext/RUNNING.md).
- Final test server stopped after completion at 03:01:18 UTC September 7.
  Watchdog reports no remaining owned workers; local test tunnels are closed.
  The built container/checkpoint remain available. Spark-094a was not modified.
- Final extended run minimum host memory available: 8.3429 GiB, only about
  351 MiB above the 8 GiB stop threshold. No sampled OOM events, but admission
  headroom is not comfortable with existing cotenants. Cleanup recovered
  100.81 GiB available. Keep the guard and conservative limits.
- Source head: `1175859d`; fresh native libraries are ready. B12x is opt-in,
  deployed and measured with the expanded M1/2/4/8/16 dispatch envelope.
- Verified full checkpoint and text-only view are on Spark-3883.
- All 1,690 modules loaded in the second smoke (2m32s). Initialization then
  correctly rejected an undersized cache quota: 212,893,030 bytes versus
  a 347,713,584-byte minimum. No OOM; minimum host available was 21.66 GiB.
- The third smoke loaded and warmed up successfully, but did not finish its
  first generation request. Its 4,096-token setting produced only 64 usable
  cache tokens. The parent stopped the owned guard at 01:11 UTC September 7;
  cleanup completed with no remaining descendants. No OOM occurred.
  Log/metrics prefix inside the container:
  `/tmp/flashnext-smoke-no-mtp-20260906-1801`.
- The cache-only probe reproduced capacity exhaustion at 65 tokens, after
  successful prefill. Removing `max_tokens` and setting `avg_seq_len=2048`
  passed allocation lifecycles for two 37+384-token requests, with 768-918 MiB
  actual hot storage depending on the input byte quota. These are allocation
  tests, not generated text. The configured byte quota
  is not an absolute native allocation ceiling: the C++ allocator can increase
  it to satisfy minimum slot constraints. Retain the independent host guard.
- Full-model retry succeeded at 01:37 UTC September 7 with those qualified
  settings: coherent hash-table explanation (384-token cap) and correct train
  speed answer (80 km/h, 204 output tokens). Initialization took 243.10 s;
  request rates including prefill were 11.27 and 11.49 tok/s. No graph/MTP,
  no autotuning, CUTLASS MoE/MXFP8. These are smoke rates, not TPOT.
  Container log/metrics prefix:
  `/tmp/flashnext-smoke-no-mtp-20260906-1832`.
- Baseline API pilot completed; parent stopped its guard after testing and
  confirmed no remaining workers at 01:55 UTC. Log/metrics prefix:
  `/tmp/flashnext-serve-eager-20260906-1838`. The B12x candidate `24e2d81c` was cherry-picked as
  `1cd19650`; 104 isolated CPU tests pass, eight GPU cases skip locally,
  and all touched-file pre-commit checks pass. Expansion `1175859d` adds M2/M8:
  116 CPU checks plus 36 actual-Linear changed-input GPU checks and 12 captures
  pass; paired outputs are bitwise equal to CUTLASS in that component test.
- `cea23010` adds opt-in smoke diagnostics using actual cumulative accepted/
  drafted counters, preserving unavailable values as null. Seven CPU checks
  and hooks pass. Instrumented runs are not primary performance measurements.

### First Serving Results

Eight distinct short prompts, exactly 128 output tokens each, two repetitions
per concurrency. These include reasoning tokens, prefill and batch drain.

| Deployment | C1 aggregate tok/s | C2 aggregate tok/s |
| --- | ---: | ---: |
| TensorRT eager, no MTP, CUTLASS | 10.84 / 11.50 | 21.87 / 20.91 |
| TensorRT eager, no MTP, B12x dense | 13.68 / 14.79 | 26.98 / 28.19 |
| TensorRT decode graphs, no MTP, B12x dense | 17.66 / 20.19 | 33.71 / 36.01 |
| TensorRT eager, MTP3, B12x dense | 22.56 / 25.80 | 30.77 / 35.45 |
| TensorRT decode graphs, MTP3, B12x dense | 21.91 / 30.40 | 30.70 / 44.14 |
| Unchanged vLLM reference, graphs + MTP3 | 34.99 / 37.20 | 55.04 / 58.13 |

All 32 requests per engine completed with exact lengths and no API errors.
This is not equal-settings engine isolation: TensorRT has BF16 KV, batch cap
2 and sequence limit 2048; vLLM has FP8 KV, cap 4 and limit 262144. Reference
prefix caching remains enabled and natural page/cache states are uncontrolled.
Both use local SSH tunnels; Isaac Sim remains active on the TensorRT host.
Reference counters show 2738 accepted of 4077 draft tokens (67.16%). No win yet.
The final MTP graph configuration also received two additional repeat pairs:
C1 24.53 / 29.26, C2 40.74 / 43.61 tok/s. All 64 fixed-output requests across
its four pairs succeeded. Report all four, not only the fastest pass; the third
C1 pass fell below the second, so simple warming is not an established cause.

Strict eight-case quality: TensorRT 6 passed, 1 arithmetic failure, 1 manually
reviewed correct Python function left automatically unscored. Reference was
5 passed, 2 failed, 1 unscored. Both got inventory arithmetic wrong; keep that
failure visible rather than claiming general accuracy from this small suite.
Artifacts are in sibling `flashnext-results`, prefixes `20260906-trt-eager-1838`
and `20260906-vllm-pilot`. B12x artifacts use `20260906-trt-b12x-1901`;
all eight quality and two warmup responses match the CUTLASS baseline exactly.
The B12x pilot completed all 32 fixed-output requests without errors. Its guard
was stopped deliberately, with cleanup confirmed at 02:14 UTC September 7.
Graph artifacts use `20260906-trt-graphs-1914`: all 32 fixed-output requests
succeeded. Pooled C1 throughput increased 32.54% and C2 26.30% versus B12x
eager. The eight-case strict quality score is unchanged, but six quality texts
differ. More importantly, C1 repetitions matched zero of eight texts, while
C2 repetitions matched all eight. Numerical or state-dependent causes are
under investigation; these results do not establish graph/eager equivalence.
The graph guard was stopped after loadgen finished; cleanup completed at
02:28 UTC September 7. The reference remains faster.

### MTP Results

MTP3 eager diagnostic succeeded at 02:33 UTC September 7: all 1804 modules
loaded, initialization 254.75 s, two 32-token requests completed with coherent
prefixes. Verified request counters are 22/27 and 24/30 accepted/drafted,
46/57 combined (80.70%). Instrumented short-request rates including prefill
are 16.40 and 18.37 tok/s, not primary performance measurements. Frontend mode
is MTP_EAGLE_ONE_MODEL; strict acceptance, no relaxed-thinking acceptance.
Log/metrics prefix `/tmp/flashnext-smoke-mtp3-20260906-1930`; clean exit 0,
no remaining owned processes after cleanup at 02:33:44 UTC.

MTP3 eager API validation completed all 42 requests (32 fixed-output) with no
protocol errors; cleanup confirmed at 02:44:56 UTC. Only the speculative
configuration changed from the B12x eager API baseline. The strict quality
score was 5 passed, 2 failed, 1 unscored: inventory arithmetic plus a Markdown
fence around an otherwise correct JSON-filter answer. C1 repeat texts matched
8/8, but C2 matched only 1/8; actual batch scheduling and numerical/state causes
remain unlocalized. Do not claim equivalence. Pooled C1 throughput was 24.0672,
C2 32.9467 tok/s. Container prefix:
`/tmp/flashnext-serve-mtp3-eager-20260906-1935`.

MTP3 plus decode graphs completed startup/capture and all 74 API requests,
changing only graph config to batch sizes [1, 2], padding disabled. All eight
quality texts exactly match eager MTP; strict score remains 5/2/1, not quality
parity with no-MTP. First two C1 repeats match 8/8 texts; C2 matches only 1/8.
The reference also varies (0/8 repeat texts at both concurrencies), with prefix
caching enabled; that neither explains nor clears the TRT repeatability issue.
Container log/metrics prefix `/tmp/flashnext-serve-mtp3-graphs-20260906-1945`.
Cleanup completed at 03:01:18 UTC, intentional stop exit 143, no remaining owned
workers. Source audit predicts four short-family graph keys (greedy and advanced
sampling for two batches), but capture-loop logs do not measure actual key count.
Near-limit long-family warmup may collapse to a short key and fall back to eager;
the short-prompt pilot does not qualify sparse/threshold-crossing graphs.

### Next Work, Not Running

1. Trace the working MTP path, recording actual graph replay, target verification,
   W4A16 draft MoE, dense linears, PLE faults and host scheduling. Correlate
   acceptance with net token rate; the 46/57 short smoke is not pilot acceptance.
2. Run the prepared request-order diagnostic and controlled state/logit replay
   to localize repeatability before promoting graphs as a default. API text
   differences alone do not prove state corruption or harmless rounding.
3. Benchmark isolated serving settings after coordinating shared-host access.
   Then evaluate MTP-only B12x MoE, overlap, FP8 KV or N96 caching one at a time
   if the trace supports them. No blind global CUTEDSL switch.

All findings, failed attempts, exact configs and raw results remain in sibling
`flashnext-results`; no upstream issues/PRs were opened for this fresh fork.
The final local validation repeat passed 114 benchmark/quality/checkpoint helper
tests, 116 isolated MXFP8 CPU cases (8 GPU skips), 7 stats tests and 6 diagnostic
harness tests. Real SM121 component tests and full-model evidence are separate;
this is not an upstream CI pass or clean release-wheel certification.

Source-only follow-up: global graph-enabled CUTEDSL MoE adds about 29.113 GiB
at the already-effective 512-token capacity. Do not enable it indiscriminately.
QSA FP8 KV and separate short/long graph families exist on current main, but
need full-model validation; do not classify the Python threshold branch alone
as a graph correctness bug. N96 BF16 caching remains deferred pending profiling.

## Provenance (2026-09-06)

- New fork: https://github.com/mihai-chiorean/TensorRT-LLM-FlashNext
- Upstream starting commit: `70feda6395` (1.3.0rc26 development).
- Branch: `experiment/flashnext-sm121`.
- Checkpoint revision: `925d7be6c14c6c9442ef83e8f05b5a3c39304f69`.
- Reference recipe: https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark
  at `ef1af5fa2e1e93d4bd96136568460cda843ccc29`.
- Initial isolated runtime: `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc25`,
  Torch `2.12.0a0+5aff3928d8.nv26.05`, FlashInfer `0.6.16`, transformers `5.5.4`.
  Its native library was too old for current Python sources and was replaced.
- FlashInfer was upgraded inside the isolated container to `0.6.18`, matching
  current main. Six core native artifacts built from `70feda6395` are staged;
  genuine package, `LLM`, and `PyTorchModelEngine` imports pass. Optional rc25
  extensions were retained, not rebuilt; this is not a clean release wheel.
  Upstream import-path fix `75f521dd` was cherry-picked as `34af9d83`.
- Signed fork commits: `f74e6657` (configuration/precision normalization) and
  `23872afa` (bounded eager NVFP4 PLE loading). Both passed commit hooks and
  were pushed. Neither is claimed as full-model validated yet.
- `2c844dd9` preserves GDN MXFP8 block scales during projection fusion and
  materializes lazy tensor views without copying their file-backed storage.
  Signed, hook-clean and pushed; seven isolated mapper regressions passed.
- Linear: https://linear.app/mitzoku/issue/MIT-910

## Safety

Only spark-3883 is a deployment target. Isaac Sim was stopped with explicit
authorization and its container recreated but not restarted; see the restoration
record above. CVAT and other unrelated services remain running. The experiment container
`trtllm-flashnext` has 84 GiB memory (no extra swap), 12 CPU and 4 GiB shm limits.
The checkpoint transfer is complete; all 34 shards passed SHA256 verification
against the pinned Hugging Face blob IDs. PLE storage is bounded and the first
full-model load was attempted. Reassess available host memory before every
model load. The admission watchdog preserves an 8 GiB host reserve and cleans
up only its own worker descendants.

## Findings and Work

1. Current main already implements the architecture as Qwen4Exp: GDN sigmoid
   gates, sparse QSA, hyper-connections, PLE and hybrid MTP. Reuse it; do not
   copy an entire model implementation.
2. Checkpoint registration/config compatibility is being checked separately.
   Lazy checkpoint loading is mandatory for this 105,839,538,520-byte checkpoint.
3. PLE is NVFP4, not the BF16/scaled-FP8 format supported by main. Its 128 shards
   have logical width 160, packed uint8 width 80, FP8 per-16 scale width 10,
   and a scalar global scale. Expanding the complete table to BF16 would use
   about 102 GB. Implement mmap-backed selected-row dequantization first.
4. Mixed precision has 48 NVFP4 target-expert policies, many MXFP8 linears,
   and W4A16_NVFP4 vision/MTP policies. Metadata also includes an ambiguous
   duplicated MTP layer number. The actual weight index has only MTP layer 0;
   identical policies for relative layer 0 and absolute layer 48 coalesce to
   runtime layer 48, not 96. Per-layer policy
   normalization and GDN weight/UE8M0-scale fusion have been implemented.
   Lazy SafeTensors values must become mmap-backed tensor views before the
   mapper accesses shape/dtype, with scalar-aware indexing.
5. Initial eager correctness path comes before CUDA graph or MTP optimization.
   A CPU-synchronized lookup must not be represented as graph-compatible.
6. Current main already includes B12x NVFP4/W4A16 MoE support for SM121.
   No historical cumulative patch branch has been transplanted. Explicitly
   compare CUTLASS and CUTEDSL once correct generation is established.
7. MXFP8 dense linears contain a Python/native eligibility mismatch: Python
   accepts compute-capability major >=10, including SM121, but the native
   dispatcher only serves SM100/103. Fix the gate before full-model execution.
   FlashInfer 0.6.18 CUTLASS works on the validated SM121 shapes; N=96 keeps
   per-forward BF16 dequantization. B12x is a promising separate component
   experiment, not yet the full-model default.

## Integration Checkpoint: September 6, Afternoon

- `c61c74a3`: reproducible smoke, streaming API harness, checkpoint-view
  preparation and matched benchmark plan. Fifteen helper tests pass.
- `7f4d3f0e`: retain asynchronous PLE prefetch output until its stream finishes.
  Two SM121 tests pass; removing the fix reproduces early allocator reuse and
  corruption. Independent of the new packed-table GPU path.
- `36038260`: graph-compatible packed PLE lookup through independently owned
  read-only mappings, without dense expansion or pinned host copies. Thirteen
  SM121 helper/wrapper checks pass. Tests include changed-ID graph replay,
  invalid rows, noncontiguous IDs, separate namespace mapping, and own-file
  reclaim. Full-model memory-pressure validation remains outstanding.
- `ed8858ea`: SM12x MXFP8 native gate, FlashInfer routing and backend intent:
  36 isolated dispatch/engine tests pass, including existing SM100 behavior.
  Actual edited linear methods passed three SM121 numerical probes. Full
  runtime imports now pass against the freshly built core libraries.
- Combined CPU PLE/helper/MXFP8 dispatch run: 96 passed, six GPU-only skips,
  13 subtests passed. Six additional isolated engine-warmup cases passed.
- B12x MXFP8: six GDN shapes at M=1/4/16 passed an independent FP32 reference
  and changed-input graph tests. Outputs match CUTLASS bit-for-bit. Component
  timings favor B12x, but no end-to-end speedup is claimed.
- Bug ledger: [scripts/flashnext/BUGS.md](scripts/flashnext/BUGS.md).
- Real-runtime tests: 69 config/GDN/PLE integration cases passed; a subsequent
  24-case consumption/GDN/PLE run passed with `2d95b4db`, which preserves
  incremental weight consumption through Qwen4 preprocessing. No full-load
  memory reduction has been measured yet.
- Consolidated real-runtime regression run: 177 passed, two CUDA-named cases
  deselected. Includes upstream import regressions, config/quantization,
  GDN mapping, consumption, PLE, and SM12x MXFP8 dispatch/warmup tests.
- Native QSA Top-K at K=512 passed short/empty rows, radix-dispatch boundaries,
  exact set selection and changed-input graph replay. Peak 80 MiB CUDA and
  1.70 GiB host RSS. Complete attention-layer validation is in progress.
- `ea1bdd87`: strict offline quality scorer and 109 scorer/client tests passed.
  It never executes generated code; the code case stays pending manual review.
- `6dd3a484`: smoke defaults to one loader worker and uses `max_draft_len`.
- `cdb81425`: first-call GDN chunk autotuning corrupted indexed recurrent
  state. Bounded active-slot save/restore fixes it without warm-path copies.
  Eleven focused CPU/GPU regressions pass, with independent code review.
- Complete synthetic QSA layer: prefill, decode and sparse threshold crossing
  through 2053 tokens pass an independent reference. Six GDN convolution/gate
  component checks also pass. These are not checkpoint-weight validation.
- Separate SM121 chunk-H repeatability probe: 52 feasible shape/config pairs,
  ten identical-state launches each, passed bitwise and numerical checks.
  This does not reproduce the reported SM103 warp-count bug; no blanket
  two-warp restriction was added. Longer contexts remain untested.
- Optional B12x dense branch: `experiment/flashnext-mxfp8-b12x`, commit
  `24e2d81c`. Independent review, 98 CPU tests and 48 actual-Linear changed-input
  checks passed. Defaults remain CUTLASS; do not enable before the baseline.
- Native core build completed after resolving one pinned Git LFS header.
  Artifact staging/import validation and resumable weight transfer remain
  outstanding. No full-model TensorRT generation has run. Initial smoke disables autotuning explicitly;
  enable it with `--autotune` for a separately recorded tuning experiment.

## Benchmark Gates

- Pin model revision, tokenizer/chat template, prompt token IDs, precision,
  context limit, output length, EOS policy, concurrency, MTP and cache settings.
- Establish coherent deterministic generation and task checks before speed.
- Record startup/peak memory, TTFT, inter-token latency/TPOT, output tokens,
  aggregate throughput, acceptance and tokens/step where available.
- Run warm repeats at concurrency 1, 2, 4, 8; separate prefill and decode.
- Do not equate published recipe numbers with our own matched measurements.
  Recipe reports 48.7 tok/s at C1 and 162.9 aggregate tok/s at C8 with MTP3,
  FP8 KV, BF16 recurrent state and full graphs. These are reference claims,
  not measurements of this fork.

## Results

Installed rc25 imports and CUDA visibility pass. Overlaying main does not:
`trtllm::silu_and_mul_fp8_quantize_1x128_packed_ue8m0` is missing from its
compiled library. A current-source native build has completed; no stub
operator workaround is accepted as validation.

Initial isolated PLE tests: 54 passed, one CUDA test skipped; six isolated integration
cases passed. Config/quantization harness: 51 focused plus five existing
regressions passed. Checkpoint-view and API benchmark helper tests: 15 passed.
These are not a substitute for full-runtime pytest.

Read-only API tests against existing spark-094a vLLM (same pinned checkpoint,
MTP3, max sequences 4, native 262144 context, FP8 KV) completed:

| Raw prompt | Actual output tokens | TTFT ms | Amortized decode ms/token |
| --- | ---: | ---: | ---: |
| Hash-table explanation, cap128 | 128 | 307.4 | 26.47 |
| Train-speed arithmetic, cap128 | 128 | 343.9 | 21.00 |
| Hash-table explanation, cap384 | 384 | 466.8 | 26.96 |
| Train-speed arithmetic, cap384 | 184 (EOS) | 371.3 | 24.23 |

At concurrency 2, the two cap128 requests produced 256 output tokens at
52.99 aggregate tok/s, with amortized decode estimates 35.70 and 28.71 ms/token.

Read-only inspection qualifies the actual reference paths: target MoE is
W4A4 FlashInfer CUTLASS, MTP experts are W4A16 Marlin, dense MXFP8 uses dynamic
A8 CUTLASS, and N96 uses cached BF16 emulation. Installed config/allocation
code selects FP32 recurrent state, not the recipe's advertised BF16 state.
Main KV uses fixed-scale E4M3; unit scales are consistent with initialization
and checkpoint headers, but resident scale values were not inspected. These
precision families guide the matched comparison; kernel equivalence is unproven.

The eight-case reference quality run has five automated passes, one arithmetic
failure (inventory), one JSON-format failure (correct code-trace value in a
Markdown fence), and one manually reviewed correct function. The scorer retains
its original 5/2/1 result; it never executes generated code. These are bounded
regression cases, not a general model-quality benchmark. All 40 quality/perf
prompt token arrays match between reference Transformers 5.15.1 and validation
5.5.4. Raw ChatML is valid but omits the default template's extra 40 tokens;
retain raw-completions versus default-chat distinctions in future reports.

Arithmetic answer was 80 km/h. Hash-table output was coherent but truncated.
This is an initial short-prompt reference, not a controlled throughput sweep.
Timing includes a local SSH tunnel; speculative tokens arrive in chunks, so
amortized decode is not a per-token latency distribution. The harness uses
server usage token counts, not text-event count. JSON artifacts are in the
sibling `flashnext-results` directory; reusable harness and prompts live in
`scripts/flashnext/`.

The first TensorRT-LLM full-model load failed before generation: the integrated
GPU loader left MXFP8 scales on CPU, but FlashInfer's scale interleave requires
CUDA input. The watchdog cleaned up all owned descendants, with exit code 1.
The log is `/tmp/flashnext-smoke-no-mtp-20260906-1734.log` inside the experiment
container; adjacent `.metrics.jsonl` records admission and cleanup. Fix
`c03087c7` reuses native CPU scale packing, retaining FlashInfer for CUDA
sources. All eight real-runtime CPU/CUDA destination and padding cases passed,
including 16 numerical forwards; 42 CPU checks and commit hooks passed.
Retry log: `/tmp/flashnext-smoke-no-mtp-20260906-1750.log` in the container;
watchdog confirmed launch at 00:50:47 UTC September 7. The earlier `1749`
launch lost SSH before creating a log and is not a model-test result.
That retry loaded all modules but stopped on the cache quota; see Current
Checkpoint for the next run. No successful full-model generation or matched
comparison is recorded yet.
Minimum sampled host available memory was 16.886 GiB; cgroup peaks were
26.083 GiB total and 15.918 GiB anonymous, with no OOM or limit events. These
are sampled early-failure observations, not successful full-load peaks.
GPU allocations are not adequately represented by these cgroup counters;
the host-available-memory guard remains necessary.

## Performance Hypothesis

GB10 runtime reports pageable-memory access, host-page-table access and
concurrent managed access all enabled. CUDA documents GPU access to file-backed
`mmap` on such systems:
https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html

A bounded GPU probe passed exact BF16 checks, changed-ID CUDA graph replay
across unequal shards, and own-file page eviction. Synthetic 64x2048 output
was approximately 15 microseconds warm and 3.4-4.5 milliseconds cold. These
are kernel-probe timings, not this model's width-160 or full-model timings.

Important memory finding: GPU reads through the default writable SafeTensors
mapping caused private-dirty/anonymous pages. An independently owned read-only
mapping kept zero anonymous, private-dirty and locked pages in the probe and
remained reclaimable. The runtime implementation opens fresh read-only
mappings and does not change protection on shared loader mappings. The GPU
delegate and width-160/128-shard tests are implemented; full-load pressure is
still unvalidated.

## Active Operations and Resume Points

- Deployment container: `trtllm-flashnext` on `mihai@spark-3883.local`.
- Completed native build log (inside container):
  `/opt/flashnext-build-70feda/logs/build-lfs-resume.log`. The earlier
  `build-resume.log` stopped at an unresolved Git LFS header; `build.log`
  predates both resumes. The latest build completed at eight jobs within the
  existing memory/CPU bounds; artifact staging and genuine imports passed.
- Native build source is a separate clean upstream tree; parent Python edits
  are in `/home/mihai/workspace/TensorRT-LLM-FlashNext` locally and need a final
  synchronization to the corresponding Spark workspace before validation.
- Verified weights on 3883: `/home/mihai/workspace/flashnext-weights-925d7be6`.
  Transfer completed September 6 at 17:31 PDT and the owned directory was
  moved onto the existing workspace mount. Container path:
  `/host-workspace/flashnext-weights-925d7be6`. Text-only view is ready at
  `/host-workspace/flashnext-text-925d7be6`.
- `scripts/flashnext/prepare_model.py` validates complete indexed shards,
  excludes auxiliary calibration SafeTensors, and creates a new symlinked
  view with a copied text-only configuration. It never edits source weights.
- Next validation order: full-runtime unit tests; no-MTP bounded text smoke;
  MTP; graphs; matched repeated performance measurements with quality gates.
- Retry with `TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL=1`: one worker still
  queues all modules and waits for queued work after a failure. Sequential
  loading uses the existing fail-fast path. Keep the 25-minute watchdog.
- Do not switch all MoE layers to CUTEDSL at default wrapper capacities:
  source accounting estimates about 28.125 GiB extra static activations/scales
  across target layers. MTP-only B12x W4A16 is a separate candidate. CUTLASS
  MTP allocates a 4.6875 GiB BF16 buffer per forward, but dequantizes only routed
  experts; allocation size is not DRAM traffic. Both paths need live measurement.
- Matched benchmark design and metric limitations:
  [scripts/flashnext/BENCHMARK_PLAN.md](scripts/flashnext/BENCHMARK_PLAN.md).
  The two smoke prompts cannot exercise concurrency 4/8; use a full corpus.
  The reference has an active sequence cap of four, so client concurrency eight
  includes queueing and is not eight simultaneously active model sequences.

## Decision Log

- Start from clean upstream main, not the historical cumulative fix branch.
- Preserve the working vLLM reference and unrelated Spark services.
- Avoid BF16 PLE expansion; use selective packed-row dequantization.
- Recipe is AGPL-3.0: use independently implemented integration in this
  Apache-2.0 codebase; do not transplant recipe patches without license review.
- Base-model license needs separate provenance review: the mirror advertises
  Apache-2.0, while the official base has a Qwen community license. Do not infer
  redistribution or hosted-service permission from mirror metadata alone.
- Both Spark Ethernet interfaces were down during transfer. The completed
  checkpoint copy was read-only from 094a over Wi-Fi, seeded with pinned HF
  downloads. The reference deployment was not modified.
- Verify effective cgroups, not Docker configuration alone. The declared
  84 GiB/no-extra-swap limit initially left `memory.swap.max=max`. Reapplying
  `docker update --memory 84g --memory-swap 84g trtllm-flashnext` set it to zero;
  host-cgroup readings confirm 84 GiB memory, zero swap and no OOM events.
- Header-only memory audit estimates 69.87 GiB persistent text GPU weights,
  plus 26.82 GiB reclaimable file-backed PLE. Optional MTP adds about 1.49 GiB
  weights and a 4.69 GiB CUTLASS BF16 workspace. These are estimates, not
  measured full-load peaks. Use one loader worker for initial admission.
