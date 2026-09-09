<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Overlap Lifecycle V1

## Decision

**GO for a separately registered, bounded OFF/ON performance trial**, subject
to fresh resource admission. The amended OFF and original ON screens found no
overlap-specific discrepancy in this fixture. This is not promotion of overlap,
a general correctness or quality proof, or evidence of improved throughput or
acceptance. No earlier performance cells are pooled into this decision.

## Protocol and Results

Two fresh workers ran OFF then ON, with MTP3, TC-decode True, B8, graphs 1..8
without padding, BF16 KV, FP32 recurrent state, prefix/chunked prefill OFF,
global TRT autotuner ON, target finalize fusion disabled and metrics/observer
OFF. Both used the same private overlap allowance and source/cache pins.

The registered wire sequence was two adjacent C1 fixed32 sentinels; a C8 wave
with limits `[1,2,3,4,7,8,15,16]` and two completion-triggered fixed4
replacements; two adjacent C1 fixed32 sentinels; a normal-EOS known-answer
request capped at128; and a final C1 fixed32 sentinel. Fixed requests used
greedy raw token IDs and ignore-EOS. The EOS request used text-enabled output
and the same chat wrapper. Each worker had a 180-second client deadline,
1000-second guard and independent 1200-second Isaac recovery lease.

| Check | OFF, amended | ON |
|---|---|---|
| Successful responses | 15 original + 1 supplement | 16 original |
| Actual output tokens / requested maximum | 262 / 352 | 262 / 352 |
| Peak client outstanding requests | 8 | 8 |
| Five fixed32 sentinel vectors | All exact | All exact |
| Replacement dispatch coverage | Both pass | Both pass |
| EOS answer | Single leading reasoning block, then `7`; stop, 38 tokens | Identical full text; stop, 38 tokens |
| Original client outcome | Failed; preserved unchanged | Exit 0, no holds |

All **15 raw-ID responses match exactly across arms** by request label,
including every sentinel, wave response and replacement. EOS full text also
matches. Paired wire payloads and prompt-token counts match; response IDs are
nonempty and unique within each worker. Raw-body/parsed-response consistency,
usage accounting, fixed lengths and finish reasons pass. No generated code
was executed.

Each replacement was sent while both long-wave responses remained outstanding
on the client, verified from HTTP-send/body-completion timestamps. This does
not prove internal slot identity, simultaneous GPU execution, complete state
reuse coverage or a fixed server schedule. Exactness here does not erase
earlier instability on other prompts and request histories.

## OFF Amendment

The frozen OFF client rejected a valid single leading `<think>...</think>`
block followed by final answer `7`. It stopped new submissions, preserving
15 records and a failed summary. Its generic sentinel-divergence hold came
from the unissued final request's missing IDs, not an observed difference
among the four collected vectors.

After transparent manual adjudication, only the originally planned final32
request was issued. Send/body-completion times were 04:50:44.094119 /
04:50:45.408110 UTC, before the original 04:52:10 deadline. No request was
replayed and no new warmup was added. Its vector matches the other four.

OFF's EOS-to-final-send interval was **71.411047 seconds**, versus
**0.001222 seconds** on ON. The payload sequence matches, but timing history
does not. OFF is an amended screen, never an original-client pass.

ON used a new frozen client with the existing project scorer's strict optional
single-leading-reasoning parser, still requiring final answer exactly `7`.
Missing IDs now remain a hold without being mislabeled observed divergence.
Transport, wire payloads, request sequence and budgets were unchanged. The
same amended answer policy was applied offline to both arms; malformed,
nested, repeated or unclosed reasoning remains invalid.

## Provenance and Cleanup

Both arms pin source `1e74d8af2bcae7acd9db2fb13dec81435d6656b1` plus the same FI
overlay and private overlap allowance `1`. YAML differs only in
`disable_overlap_scheduler`; effective logs confirm True for OFF and False
for ON. Source/helper hashes match the recorded provenance.

Each server archive contains exactly 16 HTTP200 completion POSTs. Native/FI
initialization exports match their respective seeds. FI exports contain 70
ordinary records, no additional record namespaces, zero reported profiling
records and the same valid generation digest. Metadata matches: FI 0.6.18,
CUDA 13.2, cuBLAS 13.4.1, cuDNN 92200, frontend 1.27.0, GB10. This proves
initialization provenance only, not exercised-tactic or post-request cache
equality, useful overlap duration, or draft acceptance.

All timestamps below are September 9, 2026 UTC. Both archived guards have
expected parent-requested exit143, no remaining owned PIDs and zero sampled
OOM/OOM-kill counters.

| Arm | Cleanup complete | Minimum sampled host available GiB |
|---|---|---:|
| OFF | 04:51:12.615050 | 29.159 |
| ON | 05:03:00.113584 | 29.603 |

Parent reports OFF restored at04:51:50.185523 and fresh healthy before ON
launch. ON restoration was reported at05:03:21.327574, with final restored
health confirmed at05:03:51. These host attestations are separate from the
independently verified server archives. No further runtime actions were taken
by this offline audit.

## Evidence

Artifact root: `/home/mihai/workspace/flashnext-results/`.

- `20260909-overlap-lifecycle-v1-pair-audit.md`: detailed independent audit, summary and supplement hashes.
- `20260909-overlap-lifecycle-v1-amendment.md`: explicit amendment before ON.
- `overlap-lifecycle-v1-{off,on}-20260909-client/`: original manifests, raw records and summaries.
- `overlap-lifecycle-v1-off-final-supplement-20260909.json`: separately preserved final request.
- `20260909-overlap-lifecycle-v1-off-provenance.json`, `20260909-overlap-lifecycle-v1-on-reasoning-provenance.json` and both admission JSONs: identities, scope and deadlines. The unused original ON provenance is not evidence for the executed client.
- `overlap-lifecycle-v1-{off,on}-server-20260909.tar`: logs, guard metrics and initialization caches.

```text
OFF client SHA256: 620357462751e99ea0adc6c996ff8fcd6e0317150d94a308a5e1b3def3dfe3aa
ON client SHA256: 81425d449b29c18ba6c37e75cb8a7174cad98c57fe86a8ed4c6b93fcc2dc3c1f
Project scorer SHA256: 1de999a952cc46d05fa680109c7e27e373c4202560d28ec2ef33aee840dddd20
OFF archive SHA256: 0da1ae65312e2d27ff3bdc0fb69c939300b8da56448bdc037bdee4941850f626
ON archive SHA256: ad4399cb3f20ed763015d823e8300f704e6184e84c917d6a73852a76504370d3
Native seed SHA256: 9e1c34f5e9e6ef446ce9dc40da2b3a0026b70c70a111bdbf075f58e576540550
FI seed SHA256: ac0649cc9d714c64ca67493287500d7060f3a690199b6c15d5b77b8591bb94d6
```

A performance trial requires separate fresh workers, matched request history,
immutable outputs, explicit deadlines and parent admission. No timing or
performance conclusion is drawn from this correctness fixture.
