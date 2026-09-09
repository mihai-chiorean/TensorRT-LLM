<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# MTP Length V1 Screening

## Recommendation

**Retain MTP3 for the current profile.** Its measured pooled rate was
116.84614 completion tokens/s versus MTP1's 105.68251, a **10.5634% lead**.
Both registered workers completed all client phases. This is a single-pair
screening, not a causal acceptance result, statistical significance claim,
or general quality/repeatability qualification.

MTP1 ran first and MTP3 second, once each on fresh workers. There is no
reverse-order control. Two timing rounds within one worker are not two
independent initializations. Earlier TC-decode, acceptance and vLLM results
are not pooled into this comparison.

## Matched Scope

The two YAML files differ only in `speculative_config.max_draft_len: 1` versus
`3`. Both use True TC decode (`TRTLLM_QWEN4_MTP_B12X_NONATOMIC=0`), B8/MTP,
graphs 1..8, target finalize fusion disabled, global TRT autotuner ON,
overlap/prefix/chunked prefill OFF, metrics/observer OFF, BF16 KV and FP32
recurrent state. The native cache configuration is 2 GiB; this configured
quota is not an unconditional hard allocation ceiling.

Both provenance records pin source
`ba0d02b3f9a1570b48a91ca4be28c1909f8dee4d+fi-nonatomic-overlay-20260908`.
Relative to `a4345e6c`, the source commit changes only the private FI helper's
MTP1-or-MTP3 scope guard and its focused tests. All other recorded runtime
pins are unchanged. The two provenance JSONs differ only in session,
draft length and YAML hash. Local helper/test, launcher and YAML hashes
were checked against these pins; this is not an independent live-memory
attestation of deployed code.

## Raw Integrity and Rates

Each worker ran ready+30s, canonical 32-prompt fixed128 C8 warmup, the same
32-prompt fixed128 C8 r1 and r2, then eight normal-EOS quality requests at C8,
cap1024. Warmup is retained but excluded from primary timing.

All eight client phases completed with exit 0, no request errors and canonical
prompt order. All 192 fixed requests finished at length128; all 16 quality
requests finished with `stop`. Prompt-token arrays match across fixed cells
and arms: 2461 prompt tokens per fixed cell. Measured totals are 128 requests
and 16384 completion tokens; excluded warmup totals are 64 and 8192.
Quality produced 1441 tokens on MTP1 and 1203 on MTP3, so quality durations
are not fixed-work throughput comparisons.

Primary rate is `8192 / (r1_wall_seconds + r2_wall_seconds)` per worker.

| Arm | Warmup wall s (excluded) | r1 wall s | r1 tok/s | r2 wall s | r2 tok/s | Primary tok/s |
|---|---:|---:|---:|---:|---:|---:|
| MTP1 | 70.87281 | 39.94522 | 102.54042 | 37.56997 | 109.02324 | 105.68251 |
| MTP3 | 68.01138 | 36.11085 | 113.42851 | 33.99844 | 120.47611 | 116.84614 |

Both second rounds are faster. Fixed warmup history does not remove all
history/system effects or guarantee identical routed experts, PLE rows,
generated IDs, verification iterations or GPU work.

## Request Timing and Text

Statistics below pool the 64 measured requests per worker, excluding warmup
and quality.

| Arm | TTFT mean / median ms | Streaming decode proxy mean / median ms | r1/r2 exact text |
|---|---:|---:|---:|
| MTP1 | 534.010 / 488.833 | 70.839 / 70.470 | 1/32 |
| MTP3 | 610.715 / 576.513 | 61.878 / 61.084 | 5/32 |

MTP3 has higher mean TTFT but a lower streaming decode proxy in this sample.
TTFT ends at the first nonempty text event. The proxy is
`(last_nonempty_text_time - first_nonempty_text_time) / 127`; multi-token
speculative events mean it is **not true token-arrival TPOT or isolated GPU
time**. Concurrent request timings cannot be summed into an additive wall-time
decomposition. No acceptance counters were collected; do not infer accepted
draft counts from these proxies.

Cross-arm full-text equality is 1/32 for warmup, 1/32 for r1, 0/32 for r2,
and 2/8 for quality. These streaming records do not supply token-ID traces.
Fixed128 work counts are preserved despite output differences; neither arm
passes a general text-repeatability test.

## Quality

| Arm | Strict pass/fail/unscored | Case findings |
|---|---|---|
| MTP1 | 6/1/1 | Inventory incorrectly returns 12 packs plus 10 loose, expected 13 plus 4. Other six JSON cases pass. |
| MTP3 | 7/0/1 | All seven JSON cases pass, including inventory 13 plus 4. |

There are no strict JSON-format failures in either sample. Both final
`stable_unique` functions use a seen set and a new result list, preserve
first-occurrence order and do not mutate the input. Manual reading supports
correctness for the requested integer lists; no generated code was executed,
and strict unscored status is retained. MTP3's scorer status `incomplete`
reflects that unscored case, not a missing client response. These eight-case
samples do not prove quality superiority or parity.

## Guard and Cache Evidence

Both archives independently contain exactly 104 completion POSTs each, all
HTTP200 (208 total). Effective logs confirm the respective MTP1/MTP3 lengths,
metrics OFF, finalize disabled, global tuner ON and overlap OFF. Both arms'
native/FI initialization export SHA256 values exactly match their respective
seeds. FI metadata is compatible and unchanged:
FI 0.6.18, CUDA 13.2, cuBLAS 13.4.1, cuDNN 92200, frontend 1.27.0, GB10.
The FI generation digest is valid, with 70 ordinary records, no additional
namespaces and zero reported profiling records at export.

Cache exports establish initialization provenance only, not exercised-key
coverage, identical serving tactics or post-request in-memory cache equality.
No differing FI metadata or generation digest was observed between these
initialization exports; equality alone is not a selected-key coverage claim.

MTP1 guard cleanup at **2026-09-08 23:49:51.171964 UTC** records no remaining
owned PIDs and expected parent-requested exit143. Minimum sampled host
availability was 29.105 GiB and OOM/OOM-kill counters stayed zero. Parent and
Hubble attest restoration at 23:50:26 and fresh health at 23:50:56, before
MTP3 guard start at 23:52:39.770660. This restoration attestation is distinct
from the archive audit.

MTP3 client terminal is complete at **2026-09-09 00:00:57.939149831 UTC**,
before its effective cutoff 00:09:09. Its archived guard cleanup at
**2026-09-09 00:01:34.685197 UTC** independently records no remaining owned
PIDs and expected exit143. Minimum sampled host availability was 29.038 GiB;
OOM/OOM-kill counters stayed zero. The parent subsequently verified Isaac
restoration at00:02:14.819055 UTC with unchanged identity, and fresh health
ending00:02:44.847276 UTC. Host restoration evidence is separate from the
verified server cleanup archive.

## Evidence Paths

All raw artifacts are under `/home/mihai/workspace/flashnext-results/`:

- Protocol: `20260908-mtp-length-v1-protocol.md`.
- Raw prefixes: `20260908-mtp-length-v1-{mtp1,mtp3}-true-c8`, with warmup, r1/r2, quality, strict scores and `.run` phase/provenance records preserved.
- YAML/provenance: `20260908-mtp-length-v1-{mtp1,mtp3}-true-metrics-off.yaml` and corresponding `-provenance.json` files.
- MTP1 archive: `mtp-length-v1-mtp1-true-server-20260908.tar`, SHA256 `fa4dd7dd45d8d8afe96753034c6a2823fad8ec30e2661050f18e0ed34b22eb0c`.
- MTP3 archive: `mtp-length-v1-mtp3-true-server-20260908.tar`, SHA256 `1f1bdbbeb3cf4eca73fa1ac7b1f544b34810f998572a5f0bae97258f08c28a13`.

```text
MTP1 YAML SHA256: 1e6e514d6fdf572469157149044d86e32479dda87c8680951a99ad69a4ede66e
MTP3 YAML SHA256: f68a7346d7393e4ba86cba3d5184c16d49e3736a4746c5df3dc06eb118f3e392
Native seed SHA256: 9e1c34f5e9e6ef446ce9dc40da2b3a0026b70c70a111bdbf075f58e576540550
FI seed SHA256: ac0649cc9d714c64ca67493287500d7060f3a690199b6c15d5b77b8591bb94d6
```

The offline audit reused the existing bounded `validate_cell` and
`pool_measured` calculations without modifying the audit framework or runtime.
No API, GPU, SSH or lifecycle actions were performed by this audit.
