<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Flash Next on DGX Spark

## Objective

Run `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` correctly in TensorRT-LLM on
Spark SM121, then compare matched workloads against vLLM. No performance win
is claimed until measured. Keep spark-094a's working deployment unchanged.

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

Only spark-3883 is a deployment target. Existing Isaac Sim, CVAT and other
services are unrelated and have not been stopped. The experiment container
`trtllm-flashnext` has 84 GiB memory (no extra swap), 12 CPU and 4 GiB shm limits.
The initial four-worker HF download was stopped; a resumable read-only rsync
from the pinned checkpoint on 094a is now active. No full model load before PLE storage is
bounded. Reassess available host memory before every model load.

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

Arithmetic answer was 80 km/h. Hash-table output was coherent but truncated.
This is an initial short-prompt reference, not a controlled throughput sweep.
Timing includes a local SSH tunnel; speculative tokens arrive in chunks, so
amortized decode is not a per-token latency distribution. The harness uses
server usage token counts, not text-event count. JSON artifacts are in the
sibling `flashnext-results` directory; reusable harness and prompts live in
`scripts/flashnext/`.

No TensorRT-LLM full-model generation or matched comparison has run yet.

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
- Weight destination on 3883: `/home/mihai/models/flashnext-925d7be6`.
  Transfer log: `/tmp/flashnext-weight-copy.log`; resumable rsync is active.
  This destination is not mounted in the build container. Arrange a model
  mount or move the completed experiment directory onto its workspace mount
  before generating the text-only checkpoint view.
- `scripts/flashnext/prepare_model.py` validates complete indexed shards,
  excludes auxiliary calibration SafeTensors, and creates a new symlinked
  view with a copied text-only configuration. It never edits source weights.
- Next validation order: full-runtime unit tests; no-MTP bounded text smoke;
  MTP; graphs; matched repeated performance measurements with quality gates.
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
- Both Spark Ethernet interfaces are currently down. The checkpoint copy is
  read-only from 094a over Wi-Fi, seeded with completed pinned HF downloads.
  Dataset transfer is resumable and may take hours at current link speed.
- Verify effective cgroups, not Docker configuration alone. The declared
  84 GiB/no-extra-swap limit initially left `memory.swap.max=max`. Reapplying
  `docker update --memory 84g --memory-swap 84g trtllm-flashnext` set it to zero;
  host-cgroup readings confirm 84 GiB memory, zero swap and no OOM events.
- Header-only memory audit estimates 69.87 GiB persistent text GPU weights,
  plus 26.82 GiB reclaimable file-backed PLE. Optional MTP adds about 1.49 GiB
  weights and a 4.69 GiB CUTLASS BF16 workspace. These are estimates, not
  measured full-load peaks. Use one loader worker for initial admission.
