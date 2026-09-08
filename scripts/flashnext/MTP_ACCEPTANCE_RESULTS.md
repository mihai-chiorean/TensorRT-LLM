<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# MTP Acceptance Diagnostic

September 8, 2026 UTC. Follow-up to [the non-atomic experiment](MTP_NONATOMIC_RESULTS.md).
This is an instrumented correctness/acceptance diagnostic, not a throughput benchmark.

## Status

- Control complete: one canary plus eight sequential repeats, all valid.
- Candidate complete: same canary plus eight sequential repeats, all valid.
- Every corresponding True/False repeat also has identical128 output IDs.
- Separate candidate-only mixed-history diagnostic completed:40/40 valid,
  24/24 exact repeated sequences. No first-divergence trace was triggered.
- No speed or general stability promotion. Both native caches match the seed;
  FI dense kernel selections remain unfrozen.

## Setup

Same isolated TRT adapter `1a98b99c` and frozen FI0.6.18 overlay as the previous
experiment. `enable_w4a16_tc_decode=True` is the control; False is the MTP-only
candidate. The target uses CUTLASS with finalize fusion disabled. B8, MTP3,
CUDA graphs1..8 without padding, overlap OFF, BF16 KV and FP32 recurrent state.
Global autotuning ON with independent, byte-identical native cache seeds.

Only configuration additions are `return_perf_metrics: true` and unique
`perf_metrics_output_dir` paths. No model, sampler or kernel change in this run.
Fresh worker and bounded Isaac pause lease per arm on Spark3883. Spark094a
and installed packages remain untouched.

Corpus index2 is the raw chat-formatted CSV-cleanup prompt: 78 prompt tokens,
greedy, one completion, ignore EOS, fixed128 output tokens. Streamed
`detokenize=false` provides exact token-ID deltas and terminal usage. A separate
canary must join successfully before the eight sequential repeated requests.
No failed request is retried or replaced.

Every response ID joins exactly to a unique completed server JSONL record.
Only the explicitly identified canary may be excluded from the repeat join.
Native float32 acceptance-rate rounding is allowed; integer accepted/drafted
totals are authoritative. Pooled acceptance uses summed counts, not mean rates.

## Results

| Arm | Canary accepted/drafted | Eight repeats accepted/drafted | Repeat acceptance | Exact versus first repeat |
| --- | --- | --- | --- | --- |
| True control | 88/117 | 707/945 | 74.8148% | 7/7 |
| False candidate | 88/117 | 704/936 | 75.2137% | 7/7 |

Both arms emitted the same128 IDs in all eight repeats, also equal across arms.
Control draft counters varied between88/117 and89/120; all candidate repeats
were88/117. Different draft counts can accompany identical delivered output.
The candidate difference is only **+0.3989 percentage points**, not evidence of
a meaningful general acceptance gain. Terminal clipping means accepted draft counts are not
necessarily the number of accepted draft tokens actually delivered.

## Interpretation

The earlier40-request token-ID probe interleaved eight distinct prompts and
C4 short/long drain phases. This new single-prompt C1 fixture removes that
history. It does not supersede the earlier failed repeatability observations.
Only the zero-based first unequal token position is a divergence observation;
request-level acceptance does not identify which positions were accepted.

Metrics collection can add CUDA synchronization. Client elapsed times must not
replace uninstrumented TPOT/throughput numbers. No new vLLM comparison, general
quality conclusion or performance keeper follows from this small fixture.
Identical native caches do not prove FI choices match. Separate research-only
cache persistence work is not deployed in this pair and is not itself proof of
exercised-tactic equality.

The runtime log confirms the candidate is active, not silently bypassed:
`MTP-only non-atomic B12x active: layer=48, enable_w4a16_tc_decode=False.`

## Follow-up Work

After observing the paired result, the original unchanged40-request ID probe
was authorized once on the still-loaded False worker. It uses a separate
artifact directory/label, global client lock and300-second outer timeout with
5-second kill grace, within the unchanged guard. This is an adaptive history
diagnostic, not part of the paired acceptance result or a timing comparison.
Its non-streaming UUIDs cannot be directly joined to executor request IDs.

It completed **40/40 valid requests and24/24 exact repeated sequences**:
S1/S2/S3 each8/8 versus S0. Thus mixed-prompt/drain history alone did not
reproduce the old False0/24 failure in this worker. This does not prove a race
was fixed: enabling metrics can change synchronization, FI selections were not
controlled between workers, and initialization/request histories differ.
Retain the older failure alongside this passing diagnostic.

Next: acquire a validated FI seed, verify exercised choices, then repeat the
original protocol with metrics OFF versus ON under otherwise matched settings.
Only add the bounded same-prefix sampler trace if a divergent fixture is
recovered. The source plan identifies an existing host-copy boundary so the
first trace need not add new CUDA synchronization. No kernel correction is
justified by the current evidence.

Research-only cache persistence checkpoint **`33a37a9e`** lives on
`experiment/flashnext-acceptance-diagnostic`, pushed to the fork, not enabled
here. It loads before model construction and exports after warmup, after the
worker starts but before creator return/admission. Export errors trigger normal
started-worker shutdown. Fifty-seven CPU tests, independent review and commit
hooks pass. Live cache acquisition/replay and exercised-key verification remain
untested; this is not an upstream performance fix.

## Evidence

Raw artifacts are in sibling `flashnext-results/`:

- `20260908-mtp-acceptance-protocol.md`: predeclared scope and gates.
- `acceptance-client-frozen-20260908/`: executed client, tests, launcher/configs.
- `acceptance-true-{canary,repeat}-20260908/`: SSE, IDs, usage and timestamps.
- `acceptance-true-{canary,repeat}-join-20260908.json`: validated counter joins.
- `acceptance-true-server-20260908.tar`: complete server log, guard telemetry,
  native cache and request metrics.
- Matching `acceptance-false-{canary,repeat}-20260908/` and join files contain
  the completed candidate fixture. Paired JSONL snapshots contain exactly nine
  records each and are retained separately from later history-probe records.
- `acceptance-false-history-probe-20260908/`: completed separate40-request follow-up.
- `acceptance-false-server-20260908.tar`: final complete log, guard telemetry,
  native cache and all49 request records (nine paired plus40 history).
- `20260908-mtp-acceptance-id-analysis.json`: independently derived paired ID
  and acceptance analysis, with source hashes.

Client SHA256 `44bfb91b23af5cf2e6e1cb1f6e807e669bd0095adf2eac5971a1bce2ab6e42e2`.
Fourteen CPU tests and independent review passed before requests. Review fixed
hard-deadline enforcement, exact all-response joins, terminal usage validation
and native float32 tolerance. Frozen fixtures preserve the validated version.

Native seed and both final cache SHA256
`9e1c34f5e9e6ef446ce9dc40da2b3a0026b70c70a111bdbf075f58e576540550`.
Control cleanup03:59:24 UTC left no owned PIDs; Isaac restored04:00:01 and
confirmed healthy with unchanged identity before the next pause.
Candidate cleanup04:14:32 left no owned PIDs; Isaac restored04:15:11.
Independent host inspection04:15:49 confirmed healthy/unpaused, unchanged
identity, only Isaac on GPU, TRT sleep-only and115.03GiB host availability.
Minimum guard-observed host available memory was30.19GiB True/29.41GiB False;
no cgroup OOM events. Both completed server logs contain no dropped/write-failed
performance record warnings. The private local tunnel is closed.
