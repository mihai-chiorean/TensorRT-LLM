<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# MTP Non-Atomic Integration Experiment

September 8, 2026, Spark3883 SM121. **Do not promote this option as a speed or
stability fix.** It runs through the real model, but the component repeatability
benefit did not reproduce end to end. The baseline remains unchanged.

## Configuration and Qualification

Qwen3.8-Flash-Next NVFP4 text checkpoint view `flashnext-text-925d7be6`;
TRT attention, CUTLASS target MoE with finalize fusion disabled, B12x MTP3,
graphs for batches1..8 without padding, overlap OFF, chunking/reuse OFF,
2048 sequence limit, 512 token budget, BF16 KV and FP32 recurrent state.
Native autotuning ON with separate identical seed copies. Both processes use
the same patched FlashInfer0.6.18 overlay. Only
`TRTLLM_QWEN4_MTP_B12X_NONATOMIC` changes from0 to1, forwarding the public
wrapper's `enable_w4a16_tc_decode` option from default True to explicit False.
The False log confirms activation on MTP layer48. Installed FI and the previous
remote source tree are not modified.
TRT research checkpoint: `1a98b99c`, pushed only to the fresh fork's isolated
`experiment/flashnext-mtp-nonatomic` branch, not merged into the baseline.

The actual TRT container passed88 focused CPU tests, including the real
post-load/idempotence/shared-output lifecycle. CUDA was hidden and
`--noconftest` bypassed environment fixtures, not native package imports.
Patched FI passes capability detection; stock FI rejects explicit opt-in.
All four TRT files pass pre-commit. Earlier public-wrapper GPU qualification
passed240 eager/graph observations for the candidate; that is not full-model
qualification. No SM120 or Spark094a tests were performed.

## Measurements

Identical32-prompt fixed128/ignore-EOS corpus. Completed-output throughput is
total completion tokens divided by batch wall time, including ramp-up/drain.
Request amortized decode estimates are not measured inter-token latencies.

| Arm | C4 round1 | C4 round2 | C8 round1 | C8 round2 |
| --- | ---: | ---: | ---: | ---: |
| Default True | 61.6333 | 85.2945 | 120.7901 | 123.7546 |
| Explicit False | 62.6293 | 85.9391 | 117.2655 | Not run |

Units: aggregate output tok/s. True completed128/128 performance requests;
False96/96. Every measured cell returned4096 tokens and zero errors.
False C8 round2 was not admitted because its90s client bound no longer fit the
lease. A conditional60s recovery was also not admitted; no requests were sent.
The missing cell prevents a matched two-round C8 comparison.

C4 pooled throughput is71.5587 versus72.4556 tok/s (+1.25%), much smaller than
the within-arm round variation. True C8 pooled is122.2543; matched round1
False is2.92% slower. Neither establishes a candidate speed win. No fresh vLLM
comparison was performed; do not compare these profiles causally to older runs.

Both arms returned40/40 valid token-ID diagnostic responses. True matched
**21/24** repeated128-token sequences; False matched **0/24**. This is an
unexpected negative full-model observation, not proof of its cause or of a
general accuracy regression. All three final native caches exactly match the
seed, including20 target tactics; reloads precede final serving graph capture.
FI dense selections are unfrozen and remain a confound.

Quality pass/fail/unscored: True C1=5/2/1, C8=6/1/1; False C1=6/1/1,
C8=7/0/1. All16 responses per arm are coherent and terminate normally.
False fixes a JSON-formatting failure at C1 and the pencil arithmetic at C8
in this small sample. Its C1 arithmetic is still wrong. The coding case was
reviewed without executing generated code and remains automatically unscored.
These eight cases do not establish general quality parity or improvement.

## Limitations and Preserved Failures

- Initial True cold-warmup attempt completed only16 quality requests; startup
  consumed its later-stage budget. Its artifacts remain separate from the
  complete warm True run. No failed generation was retried or discarded.
- The client wrapper incorrectly rejected scorer exit3 (manual-only unscored)
  after False C8 scored7/0/1. The scorer contract was verified, the wrapper now
  accepts only documented exits0/1/3, and a manual recovery marker records
  validation of all16 saved responses. Original terminal exit1 remains intact.
  No quality requests were rerun; ID/performance payloads were unchanged.
- Acceptance is **unavailable, not zero**. `/metrics` returned empty iteration
  JSON; Prometheus/request counters require instrumentation not enabled here.
  The separate acceptance plan documents existing YAML support and possible
  synchronization overhead; do not add it midway through a timing comparison.
- All workers used1000s guards and1200s reversible Isaac pause leases. Final
  cleanup01:10:50 UTC left no owned PIDs; Isaac restored01:11:32 and subsequently
  verified healthy with unchanged container/init/StartedAt. Tunnel18086 closed.
  This verifies container health, not an interactive streaming-client session.

## Decision and Next Experiments

1. Keep the tested baseline and preserve the default-off adapter only as an
   isolated research checkpoint. No upstream performance/stability PR yet.
2. The source routing audit predicts int32 inputs matching the component
   fixture; the suspected int64 mismatch is not demonstrated. Pin/record FI
   dense tactics for the next control rather than changing routing on speculation.
3. Run a bounded C1 CSV-sentinel acceptance/first-divergence diagnostic. Compare
   target/draft decisions at the same prefix near the observed divergence at
   token18, not post-divergence free-running logits. Existing instrumentation
   supports request identity joins but has overhead; isolate it from speed
   measurements. Do not infer accepted drafts from output throughput.
4. Only after that isolation, measure overlap and finalize-disabled cost on a
   qualified baseline, then rerun the matched vLLM workload. Keep M32 unchanged.

Raw artifacts and independent reviews are in the sibling `flashnext-results`
directory. Prefixes: `20260908-mtp-true-b8` (partial cold attempt),
`20260908-mtp-true-warm-b8`, and `20260908-mtp-false-b8`.
Full server/cache/guard archives use corresponding `mtp-*-20260908-artifacts.tar`
names. FI dependency snapshot: `mtp-fi-nonatomic-source-20260908.tar`, SHA256
`611344e4b2419e785aa182af3d2713ec43b2a6b155655f973fc29b58a9098541`.
Tracked work: MIT-923 under MIT-912. This experiment does not close the broader
performance objective.
