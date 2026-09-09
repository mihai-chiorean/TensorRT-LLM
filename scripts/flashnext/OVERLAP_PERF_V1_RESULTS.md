<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Overlap Performance V1

## Status

**Complete paired audit: do not promote overlap.** OFF achieves118.99001
tok/s versus ON113.56045 tok/s, a descriptive **-4.56304%** ON difference.
Neither measured ON round leads its corresponding OFF round. Retain OFF;
there is no positive lead supporting a configuration change in this screen.
**Configuration-policy screen, not an isolated overlap effect:** both arms
request 2 GiB KV quota, but resolved budgets differ. No promotion follows
from this single pair.
This separately registered trial follows the narrow amended-OFF/original-ON
[lifecycle screen](OVERLAP_LIFECYCLE_V1_RESULTS.md). That screen did not promote
overlap or establish general correctness. Its requests and timing are not
included here.

Registered order is OFF then ON, one fresh worker per arm. Each runs ready+30s,
canonical fixed32x128 C8 warmup, two fixed32x128 C8 measured rounds, then eight
normal-EOS quality cases at C8, cap1024. No retries or extra requests are
registered. Primary rate is `8192 / (r1_wall_seconds + r2_wall_seconds)`;
warmup and quality are excluded. Two rounds on one worker are not independent
initializations; prior experiments are not pooled into this screening pair.

## Timing

| Arm | Warmup wall s (excluded) | r1 wall s | r1 tok/s | r2 wall s | r2 tok/s | Primary tok/s |
|---|---:|---:|---:|---:|---:|---:|
| OFF | 66.53521 | 35.84373 | 114.27383 | 33.00239 | 124.11223 | 118.99001 |
| ON | 66.17808 | 36.69816 | 111.61321 | 35.43963 | 115.57683 | 113.56045 |

| Arm | Measured TTFT mean / median ms | Streaming decode proxy mean / median ms | r1/r2 exact text |
|---|---:|---:|---:|
| OFF | 562.030 / 560.688 | 61.283 / 61.267 | 3/32 |
| ON | 788.434 / 796.139 | 61.689 / 60.557 | 4/32 |

Per-request distributions below use the existing JSON fields `ttft_ms` and
`amortized_decode_ms` directly, without reconstructing token timestamps.
The latter is a **streaming amortized metric, not kernel TPOT**. Speculative
events may contain multiple tokens; C8 request durations overlap. The p95
uses linear interpolation at sorted index `0.95 * (n - 1)`. Pooled quantiles
use all 64 measured requests, not averages of round quantiles. Warmup and
quality are excluded. No acceptance counters are collected in this
metrics-OFF trial.

| Arm / round | Requests | TTFT median / p95 ms | Streaming amortized decode median / p95 ms |
|---|---:|---:|---:|
| OFF r1 | 32 | 578.947 / 737.953 | 63.075 / 72.456 |
| OFF r2 | 32 | 522.863 / 691.018 | 58.738 / 66.619 |
| OFF pooled | 64 | 560.688 / 737.022 | 61.267 / 71.265 |
| ON r1 | 32 | 796.139 / 1020.832 | 62.523 / 75.414 |
| ON r2 | 32 | 794.754 / 981.311 | 59.372 / 76.394 |
| ON pooled | 64 | 796.139 / 1020.114 | 60.557 / 75.483 |

OFF's immutable readiness artifact records target epoch1788931084, actual
epoch1788931089, lateness5s. The nominal ready+30s release therefore became
approximately35s. Preserve this timing-history difference rather than
changing the ON protocol after seeing OFF. Equal payload budgets do not
guarantee equal generated IDs, routing, PLE accesses or GPU work.
ON's artifact records target epoch1788931699, actual1788931700, lateness1s:
approximately31s rather than OFF's35s. Both exceed the registered minimum;
the immutable timing-history difference is retained, not normalized away.

### Complete Phase History

All timestamps below are UTC on2026-09-09. Each phase has exit0; terminal
records list all four planned phases validated with no omissions.

| Arm / phase | Started | Finished | Wall s | Completion tokens |
|---|---|---|---:|---:|
| OFF warmup | 05:18:09.192505 | 05:19:15.783686 | 66.53521 | 4096 |
| OFF r1 | 05:19:15.789275 | 05:19:51.683794 | 35.84373 | 4096 |
| OFF r2 | 05:19:51.690407 | 05:20:24.743788 | 33.00239 | 4096 |
| OFF quality | 05:20:24.751010 | 05:20:41.889807 | 17.08704 | 789 |
| ON warmup | 05:28:20.269863 | 05:29:26.500342 | 66.17808 | 4096 |
| ON r1 | 05:29:26.506537 | 05:30:03.254382 | 36.69816 | 4096 |
| ON r2 | 05:30:03.260644 | 05:30:38.752252 | 35.43963 | 4096 |
| ON quality | 05:30:38.761651 | 05:30:57.634730 | 18.81888 | 1036 |

Benchmark wall times are the JSON measurements, not differences of shell
timestamps. Client terminals complete at05:20:41.924770 and05:30:57.670415,
before respective cutoffs05:28:42 and05:39:24. Warmup rates61.56139/61.89360
tok/s remain excluded. Both workers improve r1-to-r2; one fixed OFF-then-ON
order cannot separate history or system variation from policy differences.

## Integrity and Quality

Each arm's four phases completed with exit0 and no request errors: 96 fixed128
responses finishing with `length`, plus eight quality responses finishing
with `stop`. Together these are192 fixed responses plus16 quality,208 total.
Every fixed cell has 4096 completion tokens and the same canonical prompt-token array
(2461 prompt tokens). Each primary pool is64 requests /8192 tokens, with
OFF wall68.84611434093677s and ON72.13779379206244s. Usage arithmetic, cached
prompt counts0, canonical prompt order, C8, payload limits, finite timing
bounds and reported aggregate rates all validate. Warmup and quality are
excluded from both primary pools.

Both strict quality scores are **6 pass / 1 fail / 1 unscored**. Inventory incorrectly
returns13 packs plus10 loose instead of13 plus4; the other six JSON cases
pass. The final `stable_unique` function is manually correct for the integer
list task, using a seen set and new result list without input mutation.
No returned code was executed; strict unscored status remains unchanged.
Both final functions are byte-identical and manually correct; their reasoning
differs. Neither was executed. All16 quality responses are coherent; these
eight cases do not establish general quality parity.

| Quality case | OFF strict outcome | ON strict outcome | Cross-arm full text / final answer |
|---|---|---|---|
| `math_speed` | Pass | Pass | Full text exact; speed80 |
| `math_inventory` | Fail: 13 packs / 10 loose, expected 13 / 4 | Same fail | Full text exact; same wrong values |
| `json_filter_orders` | Pass | Pass | Full text exact; ids a02/a04, total23 |
| `json_spanish_sort` | Pass | Pass | Reasoning differs; final text exact, values[-2,0,4,7] |
| `code_trace` | Pass | Pass | JSON whitespace differs; same result[[1,2],[3,2]] |
| `code_stable_unique` | Unscored; manually correct, not executed | Same | Reasoning differs; final function exact |
| `retrieval_positions` | Pass | Pass | Reasoning differs; final text exact, west/12/north |
| `retrieval_latest_missing` | Pass | Pass | Reasoning and JSON whitespace differ; same Dee/Ana/null |

Quality full text matches3/8; all seven parsed JSON answers match across
arms, including the wrong inventory answer. There are no final-answer
Markdown-fence failures. Full-text cross-arm matches are4/32 in warmup,
4/32 in r1 and4/32 in r2. Within-arm repeat matches are3/32 OFF and4/32 ON;
there is no repeatability promotion. Streaming text comparisons are not
token-ID comparisons or evidence identifying a numerical/state fault.

## Provenance and Resources

OFF provenance pins source `1e74d8af2bcae7acd9db2fb13dec81435d6656b1` plus the
same FI overlay, private overlap allowance1, MTP3 and TC-decode True.
Effective logs confirm overlap OFF, global autotuner ON, target finalize
disabled, metrics OFF, B8, max sequence2048, max token budget512, decode
graphs1..8 without padding and FP32 recurrent state.

The requested device KV quota is2 GiB. OFF resolves to **1,785,479,936 bytes
(1.66286 GiB)**; final target/draft quotas are1.60951/0.05335 GiB. These are
resolved quotas, not independently measured physical allocations or an
unconditional hard ceiling. ON's archive independently confirms
**2,085,442,304 bytes (1.94222 GiB)**, 299,962,368 bytes above OFF (+16.8001%).
ON target/draft quotas are1.8673797864/0.0748397093 GiB. Independent literal
comparison of the final argument strings, also present in both logs, finds
only overlap and this resolved budget differ. ON readiness is
05:27:49.740831915 UTC. Unequal final capacity prevents an isolated overlap
regression claim; it does not supply evidence of a positive overlap effect.

### Why the Requested Budget Changes

Source-only inspection uses pinned worktree commit
`1e74d8af2bcae7acd9db2fb13dec81435d6656b1`:

- `tensorrt_llm/_torch/pyexecutor/_util.py:835` computes the fraction-derived
  budget as `int((device_total - profile_peak + temporary_KV_bytes) * fraction)`.
- `_util.py:1210` samples device-wide free memory and Torch allocation stats;
  the dummy inference worker runs at1238. At1288, the peak combines Torch
  peak with `max(device_used - torch_current, 0)`. Non-Torch usage is not
  an attribution to a particular allocation or process.
- `_util.py:1399` takes the minimum of that estimate and the original explicit
  byte quota; at1412 it writes the result back to `max_gpu_total_bytes`.
  Thus the same requested 2 GiB does not promise equal final capacity.
- `py_executor.py:960` selects distinct OFF/ON executor loops, also used by
  the profiling worker. `_util.py:1012` additionally reserves another
  speculative generation-step allowance for overlap in temporary-cache
  sizing. Overlap can therefore affect estimator inputs, but neither code
  path guarantees a larger or smaller final budget.

OFF log lines311-315 show inside-Torch dynamic usage0.18 GiB, non-Torch
usage45.63 GiB, total peak119.52 GiB, device total121.69 GiB, temporary
KV1.16 GiB, fraction0.5 and resulting1.66 GiB. This confirms the profiling
estimate, not the explicit 2 GiB limit, constrains OFF. Rounded log values
are not sufficient to reconstruct exact byte arithmetic.
ON log lines305-309 show the same rounded0.18 GiB Torch dynamic usage,
45.07 GiB non-Torch usage,118.96 GiB peak,121.69 GiB total,1.16 GiB temporary
KV and0.5 fraction, yielding1.94 GiB. The rounded non-Torch bucket is0.56 GiB
lower, consistent with approximately0.28 GiB more fraction-derived KV quota.
That bucket is not a memory ownership diagnosis. The source explains the
mutation mechanism, not the cause of this pair's exact difference; profiling execution and ambient
device-memory state remain possible contributors. Do not claim that overlap
saved memory, or that extra KV capacity caused any speed change, from these
observations alone.

Each archive has exactly104 HTTP200 completion POSTs and four expected files:
server log, guard metrics, native cache and FI cache. Both initialization
exports match their seeds byte-for-byte; FI reports70 records and zero
profiling records at export (OFF05:17:32, ON05:27:47). Provenance hashes
validate; between-arm manifests differ only in arm/session, overlap flag
and corresponding config hash. Initialization equality does not prove
exercised-tactic or post-request cache equality, or absence of all tuning.

Native seed SHA256:
`9e1c34f5e9e6ef446ce9dc40da2b3a0026b70c70a111bdbf075f58e576540550`.
FI seed SHA256:
`ac0649cc9d714c64ca67493287500d7060f3a690199b6c15d5b77b8591bb94d6`.
Canonical32 corpus SHA256:
`deff19e7d82437aa4544311bef7cd8dbbf3ccd9d0eb8a8574a8a7869ae92d635`.
Quality corpus SHA256:
`37afa1d541d60cc4f502b0d58ddebf4b917044e3552692df449bc5550f06a59e`.

OFF guard caps are84 GiB, swap0 and12 CPUs. Archived cleanup at
2026-09-09 05:21:12.488219 UTC has no remaining owned PIDs and expected
parent-requested exit143. Sampled OOM counters are zero; minimum host
availability is29.291 GiB. Parent attests restoration at05:21:48.647420 and
fresh health at05:22:18.688893, before ON guard start05:22:54.177491.
ON's archive confirms identical guard limits, zero sampled OOM/OOM-kill
counters, minimum host availability29.350 GiB, cleanup at05:31:33.225467
with remaining PIDs[] and expected exit143. Both lease records identify the
same paused Isaac init2614201 and retained2140 MiB baseline. Parent attests
ON restoration05:32:08.700744 and fresh healthy05:32:38.735218928, with
identity/init/unpaused verified05:32:51. Host restoration attestations are
separate from the archive audit. No future benchmark traffic is scheduled.

## Evidence

Root: `/home/mihai/workspace/flashnext-results/`.

- Client prefixes: `20260909-overlap-perf-v1-{off,on}-c8`, preserving warmup, r1/r2, quality, scores, phase status/timestamps and `.run` admission/provenance/terminal records.
- OFF archive: `overlap-perf-v1-off-server-20260909.tar`, SHA256 `89f9309501f2035bf06e8a450212099bcdb423af3f84a1627324e44caf5c96bc`.
- ON archive: `overlap-perf-v1-on-server-20260909.tar`, SHA256 `585f7b5ff9cea985c228751a63c60caed78e3529472a2064b21d3ca4b07b1a98`.

Raw phase SHA256 values (prefixes above, in registered order):

| Arm / phase | SHA256 |
|---|---|
| OFF warmup | `775eec7286ea065e4ebdae1e3170e0e51226906a634e8224933e9b15d8909da1` |
| OFF r1 | `354644456f005dd66cd420693c706f16d9e2ad4e3c3d1b621bdbf6e253bb6860` |
| OFF r2 | `600476b828b746770bcd9ebc38a7f3257758077913dce3b9b3eb60a5de3c3540` |
| OFF quality | `fb294e0a9842b67511590bfb7f3b959fc09bbab279b6151de322066e04286feb` |
| ON warmup | `671502cd8c9f3c10d89e62a185dc5e8eb5fb19711446c4481378f57e3b6fc9e9` |
| ON r1 | `ea49492c3aa9d7380d2ad233aac3f914773fce5c30a2bad2124bfc2002790eae` |
| ON r2 | `e527ceabb21f61bf09d03c8c93bac4cf47b33050bb2de922cc2eae509f3a0854` |
| ON quality | `732ef1bdb0acd872370b3617c46b0044db1c10a76664483f1436821655b5ad28` |

This report is an offline audit only. No GPU, API, SSH or lifecycle actions
were performed by the reviewer. This single policy-screen pair does not
establish statistical significance, an isolated scheduler regression,
acceptance causality, or superiority to any historical vLLM profile. The
narrow lifecycle screen remains valid within its amendment limits; it is
not a performance promotion. The primary agent integrated this report and the final restoration evidence.
