<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# FI-Seeded Metrics Control

September 8, 2026. Correctness diagnostic, not a throughput benchmark.
No baseline promotion or upstream PR is implied.

## Question

The original mixed-history token-ID probe returned 0/24 exact sentinel pairs
on the non-atomic MTP candidate. A later metrics-enabled run returned 24/24,
but also had different initialization choices and preceding requests. Compare
fresh workers with identical native and FI seeds and identical request history,
changing only existing metrics collection. Do not run the dispatch observer in
these arms: it adds synchronous host work and could mask the effect being tested.

## Acquisition

- First attempt, helper `33a37a9e`: rejected the raw composite text-only config
  before model construction. Harness fix `a4345e6c` accepts the existing
  `language_model_only=true` runtime path; 67 CPU tests, independent review and
  commit hooks pass. No checkpoint weights/config were changed.
- Fresh retry uses deployed source archive `a4345e6c`, SHA256
  `1ad49e2d23f4c5f34686f6bf4dd7d8769f74d02fb2c1f2ecbd4a59b11bc1cb39`.
  Existing non-atomic MTP adapter and FI overlay are unchanged.
- Export at 05:34:02 UTC: 70 FI records; HTTP 200 readiness confirmed.
  No generation requests. Seed SHA256:
  `ac0649cc9d714c64ca67493287500d7060f3a690199b6c15d5b77b8591bb94d6`.
- Metadata: FI 0.6.18, CUDA 13.2, cuBLAS 13.4.1, cuDNN 92200,
  cuDNN frontend 1.27.0, NVIDIA GB10. Content generation digest validates.
  Loading this seed must separately pass FI's environment validation.
- Independent audit: all 70 records are MXFP8 (60 CUTLASS, 10 B12x), no
  fallback sentinels or namespaced records. All six metadata fields match
  24 archived caches. Full structured tactic inventory remains in
  `flashnext-results/20260908-fi-seed-audit.json`.
- Native output remains byte-identical to seed SHA256
  `9e1c34f5e9e6ef446ce9dc40da2b3a0026b70c70a111bdbf075f58e576540550`.
- Owned guard terminated after export; cleanup at 05:34:40 UTC reports no
  remaining PIDs. Minimum guarded host available memory: 29.79 GiB.
  Isaac restored at 05:35:17; subsequent health check passes.

Raw artifacts are in sibling `flashnext-results/`:
`fi-acquisition-server-20260908.tar`, `fi-cache-acquired-20260908.json`,
frozen source archives, launchers and host-monitor evidence. Preserve the
failed attempt separately from the successful retry.

## Comparison Protocol

Both arms retain MTP3, B8, overlap OFF, target CUTLASS finalize fusion OFF,
global autotuning ON, graphs 1..8 without padding, BF16 KV and FP32 recurrent
state. Each uses a fresh process, identical seed inputs and unique output files.
Only the ON arm adds `return_perf_metrics=true` and a unique metrics directory.

Use the original unmodified 40-request probe, SHA256
`cbb4b3264e755df9efe723ab642c63077da6fbf2f4a97c997678c71e516c7fcd`,
and original corpus SHA256
`deff19e7d82437aa4544311bef7cd8dbbf3ccd9d0eb8a8574a8a7869ae92d635`.
No preceding canaries, quality requests or warm requests. S0/S1/S2/S3 each
contain eight C1 sentinels; intervening short and long turnover phases each
contain four C4 requests. Server capacity 8 does not make those phases C8.
Use the existing lease-aware guard and independent Isaac recovery timer.

The optional bounded observer was independently reviewed, passed 98 CPU
tests and all commit hooks, and was saved as `4ae7269c` on the isolated
diagnostic branch. It is **not deployed** for these arms. Its file-key/choice
evidence would still be diagnostic: capture is not replay, a value equal to
the seed does not establish lookup provenance, and host-side work can perturb
scheduling even without reading or synchronizing GPU tensors.

Source audit of the existing metrics path confirms conditional waits for
forward/sample CUDA events, plus CPU bookkeeping and asynchronous frontend
JSONL writing. MTP already waits on its sampler event with metrics OFF;
the additional waits can be redundant. Neither source inspection nor a
passing ON run proves a race or changes an already executed atomic reduction
order. Full file/line evidence is in sibling
`flashnext-results/20260908-perf-metrics-operational-delta.md`.

## Results

OFF completed: 40/40 valid responses, 24/24 exact sentinel pairs, client exit 0.
No preceding generation. Initialization exported the byte-identical FI seed
with zero profiling records; final archived FI/native outputs match both seeds.
The FI file is a pre-request initialization snapshot, **not** an export of
post-probe in-memory state. Identical archived files therefore do not establish
that serving introduced no new choices or used every seeded entry.
Cleanup at 05:47:59 UTC reports no remaining PIDs; Isaac restored at 05:48:40
and subsequently healthy. Minimum guarded host available memory: 29.48 GiB.
Archive: `fi-seeded-metrics-off-server-20260908.tar`; client directory:
`fi-seeded-metrics-off-probe-20260908` under sibling `flashnext-results/`.

ON completed: 40/40 valid responses, 24/24 exact sentinel pairs, client exit 0.
Its initialization also exported the identical 70-record seed with zero
profiling records; archived FI/native files match the input seeds. The frontend
wrote 40 unique complete metrics records. Cleanup at 05:59:59 UTC reports no
remaining PIDs; Isaac restored at 06:00:45 and independently verified healthy
with unchanged identity at 06:01:19, with only Isaac using the GPU.
Minimum guarded host available memory: 29.34 GiB. Archive/client paths mirror
OFF with `on` in their names. The local SSH tunnel is closed.
Both logs retain `Failed to send object: None` during guard-initiated shutdown,
after requests completed. No API failures, OOMs or surviving owned PIDs were
observed; this is not being promoted as an upstream normal-shutdown defect.

| Comparison | Exact token-ID matches |
| --- | --- |
| OFF S1/S2/S3 against its S0 | 24/24 |
| ON S1/S2/S3 against its S0 | 24/24 |
| Cross-arm S0/S1/S2/S3 | 32/32 |
| Cross-arm Tshort | 4/4 |
| Cross-arm Tlong | 0/4 |

The four 384-token C4 turnover responses first differ at zero-based positions
382, 46, 362 and 7 for prompt indices 0..3. They are not repeated within an
arm, so this is cross-arm sensitivity, not a within-arm repeatability result.
Fixed client concurrency does not guarantee identical engine batch/slot
trajectories. No semantic quality score is inferred from token-ID equality.
The independent pair audit reconstructs both full schedules and raw response
bodies; detailed evidence is in sibling
`flashnext-results/20260908-fi-seeded-metrics-pair-audit.json` and its report.
Initialization-to-first-request delays differ; this was not a timing-controlled
performance benchmark.

Of the 40 complete ON metrics records, 39 expose speculative counts:
3,785 accepted / 5,586 drafted (67.7587%) for that available subset only.
The remaining record has no speculative field and is not imputed as zero.
No non-streaming client-ID/executor-ID join or per-prompt acceptance claim is
made; OFF has no corresponding metric series.

## Conclusion and Next Step

Metrics collection is **not necessary** for the observed C1 sentinel pass.
Both fresh seeded arms reproduce those outputs without the earlier prehistory.
This does not prove the cache is the sole cause, resolve the historical 0/24
failure, or establish a global numerical freeze. Cache files describe
initialization; the uninstrumented probe has no per-call selection trace.

Keep the native/FI seeds and frozen configs as diagnostic controls, not an
upstream tactic policy. No new speed keeper or model/kernel fix is promoted.
The scope-gate fix and optional observer stay on the isolated research branch.

Next, repeat the matched atomic/non-atomic performance A/B with metrics OFF,
the same audited seeds, identical prescribed request histories and the existing
semantic quality checks. Preserve the concurrent long-output differences;
if investigating them, compare exact committed prefixes and actual batch/slot
shapes before calling them a state bug. Reintroducing the older quality-request
prelude is a separate history control, not an unrecorded warmup. Do not grow a
kernel fix solely to obtain cross-batch bitwise identity.

Existing earlier results remain in `MTP_ACCEPTANCE_RESULTS.md`; this experiment
does not replace or retract them. No throughput measurement or fresh vLLM
comparison was performed in this control.
