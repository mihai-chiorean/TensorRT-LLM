<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# TC Decode C8 Confirmation V2

## Outcome

All four fresh workers completed the registered sequence, in True, False,
False, True order. False throughput was higher in both prescribed comparisons:
**+4.2473% in pair 1 and +1.0798% in pair 2**. The pairs remain separate;
there is no combined winner pool or statistical significance claim from two
pairs. **No promotion:** quality failures and substantial output variation
remain. This is not a correctness, general repeatability, or vLLM superiority
result.

## Protocol and Integrity

Each worker ran the same canonical 32-prompt C8 fixed-128 warmup, followed by
two measured 32-prompt C8 fixed-128 rounds, then eight normal-EOS quality
requests at C8 (cap 1024). Warmup measurements are retained but excluded from
the primary rate. No quality-before-timing, ID probe, canary, or extra
generation request was inserted.

The offline audit verified all 16 declared phases complete with client exit 0,
canonical prompt order and identical prompt-token arrays, and no request errors.
Each fixed cell has 32 length finishes, 4096 completion tokens and 2461 prompt
tokens. All quality requests finished with `stop`; strict scoring failures
remain preserved independently of successful client completion.

- Primary: 256 requests, 32768 completion tokens.
- Excluded warmup: 128 requests, 16384 completion tokens.
- Quality: 32 requests, 4746 completion tokens.
- Server logs: exactly 104 successful completion POSTs per worker, 416 total.

Fixed profile: GB10, B8/MTP3, graphs 1..8, target finalization disabled,
overlap/prefix/chunked prefill OFF, global TRT autotuner ON, metrics and FI
observer OFF. True/False refer to `enable_w4a16_tc_decode`; the corresponding
`TRTLLM_QWEN4_MTP_B12X_NONATOMIC` values are 0/1. The same patched runtime,
configuration and native/FI initialization seeds were used in all four workers.
Isaac pause/restoration and server lifecycle were parent-owned.

## Throughput

Primary throughput is exactly `8192 / (r1_wall_seconds + r2_wall_seconds)`.
Rates below are completion tokens per second; no arithmetic averaging of
round rates is used.

| Worker | Arm | Warmup wall s (excluded) | r1 wall s | r1 tok/s | r2 wall s | r2 tok/s | Primary tok/s |
|---|---|---:|---:|---:|---:|---:|---:|
| s01 | True | 64.596 | 37.154 | 110.2445 | 33.586 | 121.9570 | 115.80533 |
| s02 | False | 66.449 | 35.737 | 114.6162 | 32.121 | 127.5192 | 120.72394 |
| s03 | False | 69.238 | 35.470 | 115.4769 | 33.149 | 123.5628 | 119.38311 |
| s04 | True | 65.637 | 35.697 | 114.7426 | 33.663 | 121.6764 | 118.10778 |

| Prescribed pair | True tok/s | False tok/s | False relative to True |
|---|---:|---:|---:|
| 1: s01 True then s02 False | 115.80533 | 120.72394 | +4.2473% |
| 2: s03 False then s04 True | 118.10778 | 119.38311 | +1.0798% |

Every worker's second measured round was faster despite the fixed warmup.
Order reversal preserves the direction here, but the magnitude varies.
Identical request history and output-token counts do not guarantee identical
generated IDs, routed experts, PLE rows, draft acceptance, or GPU work. These
measurements do not establish the cause of the timing differences. Prior V1,
C4, acceptance and vLLM results are not pooled into this experiment.

## Request Timing and Text

The following statistics pool only the 64 measured requests per worker.

| Worker | TTFT mean / median ms | Streaming decode proxy mean / median ms | r1/r2 exact text |
|---|---:|---:|---:|
| s01 | 533.964 / 506.999 | 62.561 / 62.561 | 3/32 |
| s02 | 533.036 / 524.722 | 60.434 / 60.449 | 2/32 |
| s03 | 549.536 / 535.380 | 60.749 / 60.369 | 1/32 |
| s04 | 535.324 / 503.213 | 61.774 / 61.420 | 3/32 |

TTFT ends at the first nonempty streamed text event. The decode proxy is
`(last_nonempty_text_time - first_nonempty_text_time) / 127`.
Speculative streaming can deliver several tokens per event: this is **not
true per-token arrival TPOT or isolated GPU time**. It excludes the tail after
the last text event. C8 request durations overlap and cannot be summed into an
additive wall-time decomposition.

Cross-arm exact texts in pair 1 were 2/32 for r1 and 4/32 for r2; in pair 2,
3/32 and 1/32. The client audits retain character-level first differences;
these are not token-ID divergence positions. No general determinism pass is
supported by these streaming text comparisons.

## Quality

Strict counts are pass/fail/unscored. All raw failures remain visible.

| Worker | Strict score | Observed failed cases |
|---|---|---|
| s01 True | 6/1/1 | Inventory: 13 packs plus 10 loose, expected 13 plus 4. |
| s02 False | 6/1/1 | `math_speed`: correct value 80, but Markdown-fenced JSON violates the requested format. Inventory passes. |
| s03 False | 5/2/1 | Inventory: 13 packs plus 11 loose. `code_trace`: correct `{"result":[[1,2],[3,2]]}`, but Markdown-fenced JSON violates the requested format. |
| s04 True | 6/1/1 | Inventory: 13 packs plus 10 loose, expected 13 plus 4. |

The extra s03 code-trace failure is a formatting failure, not an incorrect
trace value. All four final `stable_unique` functions were manually read as
correct for the integer-list task, using a seen set and new result list without
mutating the input. No returned code was executed; the strict unscored status
is retained. These small, differing outcomes do not establish quality parity
or superiority.

## Cache and Guard Audit

All four archives contain matching native and FI initialization snapshots.
Each FI export reports 70 records and `profiling_records=0`, the same generation
digest and metadata: FI 0.6.18, CUDA 13.2, cuBLAS 13.4.1, cuDNN 92200,
frontend 1.27.0, NVIDIA GB10. The 70 ordinary MXFP8 records contain 60 CUTLASS
and 10 B12x entries, with no additional record namespaces.

These exports precede client requests. They establish initialization equality,
**not** post-request in-memory equality, serving tactic selection, exercised-key
coverage, or identical GPU work. Global TRT tuning is ON in this protocol;
the zero-profiling observation describes the recorded FI cache export, not a
blanket claim that every runtime tuning mechanism was disabled.

All dates below are 2026-09-08 UTC. Each guard's final exit 143 is the expected
parent-requested termination, not a failed client phase. Every archived
`cleanup_complete` has `remaining_pids=[]`; sampled guard OOM/OOM-kill counters
remain zero.

| Worker | Guard start | Cleanup complete | Minimum sampled host available GiB |
|---|---|---|---:|
| s01 | 22:28:17.745853 | 22:37:20.699115 | 29.202 |
| s02 | 22:40:55.761815 | 22:50:37.865947 | 29.264 |
| s03 | 22:53:38.117816 | 23:02:34.729605 | 29.219 |
| s04 | 23:09:06.957204 | 23:17:34.381135 | 29.478 |

The parent subsequently verified final Isaac restoration at23:18:20.039770
UTC, with unchanged identity, then fresh successful health checks ending
23:18:50.064698 and23:19:20.103381 UTC. Host restoration evidence is separate
from the server archives; the read-only host monitor retains it.

## Reproducibility and Evidence

Artifact root: `/home/mihai/workspace/flashnext-results/`.

- Registered protocol: `20260908-tc-confirmation-v2-proposal.md`.
- Raw client prefixes: `20260908-tc-confirmation-v2-{s01true,s02false,s03false,s04true}-c8`. Each preserves warmup, r1, r2, quality, strict score, phase timestamps/status and `.run` provenance/terminal files.
- Server archives: `tc-confirmation-v2-{s01-true,s02-false,s03-false,s04-true}-server-20260908.tar`, each containing log, guard metrics, native cache and FI export.
- Full per-worker JSON audits: `tc-confirmation-v2-{s01true,s02false,s03false,s04true}-audit.json`, including raw and archive/member SHA256 values, timing distributions and quality findings.
- Paired calculations and hashes of the four input audits: `tc-confirmation-v2-paired-audit.json`. `combined_winner_pool` is explicitly null.
- Preserved `*-client-audit.json` files contain text comparisons and first-character differences. The CPU-only `tc-confirmation-v2-audit.py` and its eight focused tests reproduce the structural and pair checks; neither makes API/GPU calls.

Pinned identity (full deployed file hashes are in each immutable provenance):

```text
Runtime: a4345e6cec1538e7622e5539718013cfbc345542+fi-nonatomic-overlay-20260908
Checkpoint: 925d7be6c14c6c9442ef83e8f05b5a3c39304f69
Config SHA256: f68a7346d7393e4ba86cba3d5184c16d49e3736a4746c5df3dc06eb118f3e392
Native seed SHA256: 9e1c34f5e9e6ef446ce9dc40da2b3a0026b70c70a111bdbf075f58e576540550
FI seed SHA256: ac0649cc9d714c64ca67493287500d7060f3a690199b6c15d5b77b8591bb94d6
FI generation: db25c7b0a42e6af7f198c88f398f18d6490e09c4b1bd69209d8ebbec34a28f96
Canonical corpus SHA256: deff19e7d82437aa4544311bef7cd8dbbf3ccd9d0eb8a8574a8a7869ae92d635
Quality corpus SHA256: 37afa1d541d60cc4f502b0d58ddebf4b917044e3552692df449bc5550f06a59e
```

Any further quality, repeatability or generalization investigation requires a
separately authorized protocol. This audit started no services or requests and
does not authorize a follow-up benchmark.
