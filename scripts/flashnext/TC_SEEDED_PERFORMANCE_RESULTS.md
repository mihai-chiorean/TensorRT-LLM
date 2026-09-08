<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Seeded MTP TC-Decode Performance A/B

September 8, 2026, Spark3883 SM121, Qwen3.8-Flash-Next NVFP4.
**Small batching lead, not a promoted performance fix.** The non-atomic
candidate measured +1.69% at C4 and +3.51% at C8, but -1.38% at C1.
Quality scores were unchanged. Retain the existing baseline until a fresh
reverse-order comparison confirms the batching benefit. No new vLLM win is
established by this experiment.

## Controlled Profile

Same text-only checkpoint view at revision `925d7be6`, TRT attention,
CUTLASS target MoE with finalize fusion disabled, B12x MTP3, BF16 KV,
FP32 recurrent state, batch cap8, sequence limit2048 and token budget512.
Decode graphs cover batches1..8 without padding; overlap, prefix reuse and
chunked prefill are OFF. Global autotuning remains ON, with identical native
and FlashInfer seed files loaded into each fresh worker. Metrics collection
and the optional dispatch observer are OFF.

True uses `TRTLLM_QWEN4_MTP_B12X_NONATOMIC=0`, leaving the public FI wrapper's
`enable_w4a16_tc_decode=True`. False sets the environment opt-in to1 and
passes explicit False. Both arms use the same patched FI0.6.18 overlay.
This selects different GEMM/reduction paths, not an isolated atomic-only
ablation. No production source or baseline branch was changed for these runs.

Runtime provenance: tracked Python files match `a4345e6c`, with the separate
reviewed FI non-atomic overlay. Independent preflight verified1518 runtime
Python files, the helper test, three FI patch files and both seeds. The
launcher additionally checks eight critical source hashes on each launch.
An initial launcher Git-HEAD check was replaced before use: the deployment is
an archive, not a Git checkout, and failed command substitutions could compare
as equal empty strings. This was a harness defect, not an upstream runtime bug.

Both YAML files are byte-identical, SHA256
`f68a7346d7393e4ba86cba3d5184c16d49e3736a4746c5df3dc06eb118f3e392`.
Native seed SHA256
`9e1c34f5e9e6ef446ce9dc40da2b3a0026b70c70a111bdbf075f58e576540550`;
FI seed SHA256
`ac0649cc9d714c64ca67493287500d7060f3a690199b6c15d5b77b8591bb94d6`.
All four archived initialization exports match those seeds. These files are
not post-request in-memory snapshots or proof of every exercised tactic.

## Protocol

Four fresh sessions ran in order: True-low, False-low, False-batch, True-batch.
Low history: eight quality prompts at C1, then two C1 performance rounds.
Batch history: eight quality prompts at C8, then C4r1/C8r1/C4r2/C8r2.
The readiness wait targets30 seconds using whole-second scheduling. No extra
canary, token-ID probe or generation warmup was inserted. Server graph warmup
is separate from this client history.

Each performance cell uses the same32 distinct raw chat-formatted prompts,
temperature0, 128 output tokens and ignoreEOS. Each quality request allows
1024 output tokens with normal EOS. Saved prompts, scoring rules and client
hashes are pinned. All12 performance cells completed: **384/384 requests,
49,152 output tokens**, no errors, all length finishes. Quality adds32 normal
EOS responses. No requests or repeats were discarded or retried.

## Throughput

Aggregate output tokens divided by complete batch wall time, including ramp-up
and drain. Pooled values divide8192 tokens by the summed two-round wall time;
they are not arithmetic averages of rates.

| Client Concurrency | True r1 | True r2 | True Pooled | False r1 | False r2 | False Pooled | False Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 28.3974 | 36.3689 | 31.8926 | 28.5839 | 34.9581 | 31.4513 | -1.38% |
| 4 | 55.0251 | 84.0971 | 66.5235 | 55.9986 | 85.4252 | 67.6504 | +1.69% |
| 8 | 108.2806 | 120.3215 | 113.9839 | 110.5083 | 126.5388 | 117.9815 | +3.51% |

Units: output tok/s. Client concurrency is an upper bound, not a measurement
of identical internal batch or slot trajectories. The candidate leads in both
corresponding C4 and C8 rounds, but only one fresh worker per arm/block was
measured. History effects, different generated continuations and time-order
variation prevent a statistical or universal speedup claim.

At C1, pooled mean TTFT was391.05ms True versus455.48ms False. The mean
post-first-text amortized decode proxy was28.521ms versus28.457ms. Thus the
observed C1 throughput loss is not evidence of uniformly slower decode math.
This proxy is `(last_text - first_text) / (completion_tokens - 1)`; MTP streams
token groups, so it is not individually timestamped token latency or GPU TPOT.

| Concurrency | True Mean TTFT ms | False Mean TTFT ms | True Mean Decode Proxy ms | False Mean Decode Proxy ms |
| --- | ---: | ---: | ---: | ---: |
| 1 | 391.054 | 455.484 | 28.521 | 28.457 |
| 4 | 722.424 | 733.284 | 54.301 | 53.304 |
| 8 | 586.527 | 571.795 | 63.132 | 61.931 |

The False C1 repeat retains a4674ms TTFT excursion on one request; it is not
removed as an outlier. Its cause is unresolved.

## Quality and Repeatability

| Arm | C1 Pass / Fail / Unscored | C8 Pass / Fail / Unscored |
| --- | --- | --- |
| True | 5 / 2 / 1 | 6 / 1 / 1 |
| False | 5 / 2 / 1 | 6 / 1 / 1 |

All responses are coherent and terminate normally. The inventory arithmetic
failure remains; the extra C1 failure is JSON-only formatting. Generated
`stable_unique` functions are reviewed as text, never executed; the scorer's
unscored status is retained. This eight-case suite does not establish general
quality parity, calibration accuracy or a model-level accuracy improvement.

**Realized warmup histories differ:** the preceding C8 quality phase generated
817 tokens for True versus1477 for False; only1/8 complete quality texts match.
The prescribed prompts/order match, not the actual generated work before the
timing cells. This is an additional confound for the small batching gains.
C1 quality texts match8/8 across arms. Do not drop or rerun either quality phase
to manufacture a cleaner retrospective comparison.

Exact repeated text pairs: True C1/C4/C8 =29/11/4 out of32; False =32/8/3.
Text equality is not token-ID equality. Non-atomic MTP does not make the whole
concurrent model deterministic. Acceptance was intentionally not instrumented
in these timing runs; it is unavailable, not zero. Prior separate acceptance
results remain in [MTP_ACCEPTANCE_RESULTS.md](MTP_ACCEPTANCE_RESULTS.md).

## Warmup Finding

True C1 round2 reduces wall time by31.61s. About35.6% of that reduction is
TTFT and64.4% is after the first text. The trend also holds on the29 exactly
matching text pairs, so the three changed completions do not explain it.
First-pass requests improve gradually; the repeat is much flatter.

Revisiting PLE file-backed rows/address translations is plausible, not proven.
There are no clock, GPU page-fault or per-kernel traces here. Prefix reuse is
disabled and reported cached prompt tokens are zero. A fresh process does not
mean an OS-cold cache. Next diagnostic: a bounded repeated/reversed prompt
subset with precise client timestamps and low-rate clocks/fault telemetry,
without clearing caches or changing these timing runs retrospectively.

## Safety and Validation

Each worker used the unchanged1000s memory/ownership watchdog and a separate
1200s reversible Isaac pause lease. Isaac was never stopped or recreated.
All four workers exited after primary-requested guard TERM, with empty owned
PID lists; exit143 is expected cleanup, not request failure. The existing
shutdown `Failed to send object: None` messages remain in the archives.

Final cleanup14:55:30 UTC; Isaac restored14:56:20, fresh healthy check14:56:50,
same container/init/StartedAt identity, only its2140MiB GPU context remaining.
The local18086 tunnel is closed. Spark094a and unrelated services are unchanged.
Container limits remain84GiB memory, swap0 and12CPUs. Local validation:
123 client/scorer/summary tests,22 pause-helper tests and six lease-watchdog
tests passed. Independent launcher, raw-result and safety reviews accompany
the evidence. Runtime/kernel code was unchanged; no new GPU unit suite is
claimed from these API benchmarks.
Independent host samples observed at least29.04GiB available, with no OOM or
OOM-kill events. The monitor's snapshot/arm-transition attribution alerts and
late start on the final worker are retained in the safety reports; they were
checked against owned process identities, not silently whitelisted. These
samples do not establish a continuous minimum or streaming-client health.

## Decision and Next Work

1. Preserve the default-off candidate, but do not promote it or open an upstream
   performance PR yet. Keep the baseline unchanged.
2. Pre-register a new batch confirmation protocol with quality after timing,
   the same fixed-length warmup in both arms, and repeated fresh workers in
   both arm orders. Keep these current observations separate, not pooled into
   the new protocol. Confirm whether the C8 lead persists beyond worker/order
   and realized warmup variation before considering a batch-specific policy.
3. Then compare the qualified configuration against a fresh vLLM reference
   with matched request history and explicitly controlled serving settings.
   No fresh vLLM model was loaded in this experiment. The historical125.2630
   C8 result used different KV precision/budget, context, chunking and prefix
   settings; comparing it directly to this table would not be apples-to-apples.
4. Keep the PLE warmup trace as a separate experiment. Do not add the small
   measured percentages or infer that removing Python would recover them.

The pinned vLLM replica remains available on3883. Its configuration and existing
logs mount were inspected read-only for a future lease-aware launch. No rebuild
or replica recreation is established as necessary. Model-specific feasibility,
resource admission and startup time still need validation; the historical648s
load would not leave room for the full registered client timeout reserve.

Evidence is in sibling `flashnext-results/`: raw client prefix
`20260908-mtp-tc-decode-{true,false}-{low,batch}`, four
`tc-decode-*-server-20260908.tar` archives, the frozen launcher/client/protocol,
`20260908-tc-decode-speed-audit*`, `20260908-c1-round-warmup-attribution.md`,
and per-session `tc-seeded-perf-s0*-host-20260908` safety reports. MIT-923 under
MIT-912 remains open. Earlier positive and negative experiments are preserved.
