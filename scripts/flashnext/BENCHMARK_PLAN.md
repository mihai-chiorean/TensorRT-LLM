<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# FlashNext Matched Serving Benchmark Plan

## Status and Scope

Plan and independent client review, 2026-09-06. No inference requests, service
changes, or resource probes against either Spark were made for this review.
Only this document is owned by this sidecar. `benchmark_api.py` and its tests
remain parent-owned. This is not a benchmark result or a performance claim.

Prerequisite: the parent must finish the native build, full-runtime tests,
bounded model loading, and correctness validation on spark-3883. The status
snapshot reports no TensorRT-LLM full-model generation yet. Keep spark-094a's
working vLLM deployment unchanged: no restart, patches, launch-flag changes,
profiling enablement, cache resets, clock changes, or cache eviction. Future
load generation requires an approved idle window and resource budget; even
read-only completion requests consume capacity and alter cache state.

Evidence inspected: [project status](../../FLASHNEXT_STATUS.md),
[client](benchmark_api.py), [two raw prompts](prompts.json), and the JSON files
in `/home/mihai/workspace/flashnext-results`. Documentation below is independent
of the AGPL recipe implementation; no recipe code is copied or executed.

## Client Review Findings

Review anchors are function names and expressions, not permanent line numbers.
Reviewed post-fix client SHA256:
`c854367df96fcdecba3a158446cda31268389a3575214f83a8e7a787222a7343`.

Parent follow-up: the client now requires a finish reason and verifies the
requested accepted-token count when `ignore_eos` is enabled. Failed batches
preserve successes and per-request errors, return a nonzero exit code, and
leave aggregate throughput null. The report records the effective concurrency
upper bound from the corpus size. These address the corresponding reporting
gaps below, but do not supply an observed in-flight trace or repair chunk-based
TPOT. Client and checkpoint-view helper tests now total 15 passing cases.

| Priority / state | Finding | Consequence and required treatment |
| --- | --- | --- |
| P1, remaining metric limitation | `request()` divides `last - first` by `completion_tokens - 1`. The first nonempty text event may contain several accepted MTP tokens. | The interval excludes that first batch, but the denominator excludes only one token. Do not interpret this as measured per-token decode latency or use its inverse as a clean MTP speedup. Keep it only as a clearly named conventional text-event TPOT proxy. |
| P1, remaining experiment-design trap | `main()` submits one future per prompt; the supplied file contains two prompts. | C4/C8 with that file still has at most two simultaneous requests. The reported concurrency is a configured upper bound, not observed concurrency. Use enough distinct requests and record actual in-flight occupancy. |
| P2, remaining reporting gap | A failed `future.result()` prevents the final JSON write, while other futures may continue until the pool drains. | A failed trial can lose successful responses, elapsed timing, and failure counts. Persist per-request errors/partial results or use an upstream client that records them. Never exclude failures silently from performance comparisons. |
| P2, remaining metric boundary mismatch | TTFT starts at request send and ends at first nonempty text; `elapsed_s` ends after `[DONE]` and response closure; decode timing ends at last nonempty text. | TTFT is first-visible-text latency, not a server prefill timer. Request throughput includes connection/protocol tails. Retain these boundaries explicitly rather than comparing them with another client's last-choice-event latency. |
| P2, remaining instrumentation gap | Only final text, final usage, and the number of text events are saved. | Cannot recover event-gap distributions, first-chunk token count, speculative acceptance, client queue time, or steady-state occupancy. `text_events` is neither accepted-token count nor necessarily engine-step count. |
| P2, remaining validation gap | `[DONE]`, text, and usage can pass without a `finish_reason`; `ignore_eos` is sent without verifying its effect. | Validate termination and actual lengths at the experiment layer. Fixed-output trials must return the requested accepted-token count; successful HTTP/SSE completion alone is insufficient. |
| Fixed by parent | Single-text-event responses previously yielded zero amortized decode time; invalid completion usage counts were accepted. | Current code returns `null` for a single event and rejects nonpositive/noninteger/boolean completion counts. Parent reports seven client tests passing; do not duplicate those tests here. Historical artifacts remain immutable. |

The line-oriented parser is appropriate for the currently observed one-JSON-
object-per-`data: ` line, not a general SSE parser. Multiline events, other legal
whitespace, special-token-only events, and server-specific choice conventions
need explicit compatibility checks before changing endpoints. Its 300-second
socket timeout is not a whole-trial deadline and can reject long-prefill cases.
An accepted token need not produce visible text: OSL1 can expose special-token
or byte-decoding cases that the local client's no-text guard rejects. Distinguish
first accepted-token evidence from first visible text; report the latter as
unavailable when appropriate, rather than inventing a latency or zero token count.

The finite-batch aggregate, `sum(server completion_tokens) / batch wall time`,
is meaningful **completed-output throughput for that batch** when all requests
finish. It includes pool startup, ramp-up, and drain. It is not steady-state
decode throughput, nor the sum/mean of request-local token rates. Request timing
starts inside worker execution, so waiting in the client executor is excluded
from request latency even though batch wall time includes it.

## Trusted Measurement Basis

Use **one pinned upstream vLLM serving benchmark client** against both OpenAI-
compatible servers, from the same independent client environment. NVIDIA's
[serving benchmark guide](https://github.com/NVIDIA/TensorRT-LLM/blob/70feda63959fedf0f5f12c5e8c771e5392ce2d25/docs/source/commands/trtllm-serve/run-benchmark-with-trtllm-serve.md)
also uses an OpenAI-backend concurrency sweep; it distinguishes TTFT, TPOT,
end-to-end latency, and aggregate output throughput. Do not use in-process
`trtllm-bench` numbers as one side of an HTTP comparison.

Upstream client source reviewed at
`6865e67f0be02d53694517f6f71d7fb96492792d`:
[request adapter](https://github.com/vllm-project/vllm/blob/6865e67f0be02d53694517f6f71d7fb96492792d/vllm/benchmarks/lib/endpoint_request_func.py),
[metric aggregation](https://github.com/vllm-project/vllm/blob/6865e67f0be02d53694517f6f71d7fb96492792d/vllm/benchmarks/serve.py),
[custom dataset handling](https://github.com/vllm-project/vllm/blob/6865e67f0be02d53694517f6f71d7fb96492792d/vllm/benchmarks/datasets/datasets.py).
Pin the installed client package/commit and dependency lock, not `latest`.
Keep that client separate from 094a's installed server. Validate its emitted
payloads offline before any approved live run.

Upstream does **not** solve token timing automatically: its OpenAI adapter
timestamps choice-bearing chunks, including empty text, and the aggregation
uses `(latency - ttft)/(output_len - 1)`. Its stream gaps are chunk intervals
under MTP, not individually timestamped accepted tokens. It may retokenize
output if server counts are absent; reject that fallback for primary results.
Do not pool those fields with the local client's differently bounded timings.

[AIPerf's metric reference](https://docs.nvidia.com/aiperf/reference/ai-perf-metrics-reference)
distinguishes an average ITL metric from an inter-chunk latency distribution.
An independently pinned AIPerf client is an optional cross-check, not a second
definition to average into the main result. Server-side prefill/decode timings
and speculative counters are complementary evidence where already available.
No profiler-enablement or metrics-configuration changes are allowed on 094a.

## Comparison Tracks

| Track | Fixed and variable settings | Permitted claim |
| --- | --- | --- |
| M: matched settings | Adapt TRT on 3883 to the observed reference: native context 262144, MTP3, identical weight/PLE policies, FP8 KV, recurrent-state precision, effective cache policy, and at most four active sequences. Match other controllable settings; list unresolved differences. | Matched-workload serving performance on these two hosts. If precision, MTP, or graph behavior cannot be matched, mark the row unmatched rather than claiming backend isolation. |
| D: unchanged deployments | Leave 094a exactly as found; compare the safe validated TRT launch against it with identical requests. Differences in cgroups, cache capacity, graphs, precision, active-sequence cap, and PLE storage are explicit columns. | Which of these deployed systems serves this workload better, not an intrinsic TensorRT versus vLLM algorithm win. |
| T: TRT-only tuning | On 3883, compare MTP off, then 1/2/3 speculative tokens; tune graphs, recurrent dtype, chunk size, and PLE path one factor at a time. Freeze all other settings and use held-out prompts for selection. | Within-TRT tuning deltas. A tuned TRT versus fixed 094a result belongs to D unless it also satisfies M. |

094a is reported as `max_num_seqs=4`, distinct from context length 262144.
Client C8 therefore means up to eight outstanding requests but at most four
active reference sequences, with server queueing. For M, keep TRT's corresponding
active-sequence cap at four and report C8 as **queue-inclusive C8/cap4**.
For D/T, TRT cap8 is allowed only within its approved memory budget and is an
explicit configuration advantage, not a matched eight-active-stream comparison.
A matched cap8 reference experiment is unavailable under the no-change rule.
Likewise, MTP-off versus MTP-off across both backends is unavailable here.

## Workload and Repetition Matrix

Prepare frozen JSONL corpora locally; do not generate a different random prompt
stream for each server. Use a pinned tokenizer to render prompts once, count
the resulting input IDs, and save their hashes. A row stores `prompt`,
`output_tokens`, request ID, intended input length, and prompt-token-ID hash.
Separate arbitrary token-length stress text from meaningful prose/code because
MTP acceptance depends on content. Keep the two existing prompts as smoke
inputs, not a throughput corpus.

| Workload | Actual input tokens (ISL) | Output target | Client concurrency | Purpose |
| --- | --- | --- | --- | --- |
| Prefill-dominant | 128, 1024, 8192, 32768 | 1 | 1, 2, 4, 8 | First-output service latency and effective input service rate. Decode still includes generation/verification of the first response. |
| Decode-dominant | 128, 1024 | 600 | 1, 2, 4, 8 | Accepted-output throughput and delivery gaps with enough decode work to amortize setup. |
| Balanced serving | 8192 | 256 | 1, 2, 4, 8 | Prompt/decode interference under homogeneous traffic. |
| Long-context diagnostic, gated | 65536, 131072, at most 262143 | 1 | C1 initially | Retrieval and long-prefill behavior only after memory admission. Native 262144 context includes the output budget. No 512k test. |
| Mixed-arrival diagnostic, gated | Two 600-token decode streams plus one distinct 65536-token prompt | Separate caps per request | Scripted schedule, not a homogeneous C3/C8 cell | Timestamp the prefill arrival and report decode chunk stalls inside that interval. Requires a trace-capable client. |

For each primary cell, preregister three timed repetitions and
`N = max(32, 16*C)` requests per repetition. Warm the exact shape/concurrency
with at least `max(8, 2*C)` requests from a disjoint warmup corpus, untimed.
Run a small approved pilot first to estimate total duration and confirm that
the chosen N sustains at least 16 scheduling rounds; revise the plan and budget
**before** collecting the comparison, never after seeing favorable results.
This matrix can take hours on Spark; it is not authorization to launch it now.

Use identical manifests, seeds, N, OSL, request order, and warmup counts for
both sides of each matched repetition. Alternate paired order A/B, B/A, A/B
to expose time drift, with baseline health observations around each pair.
Do not run both workloads simultaneously over a contested Wi-Fi/client link.
Use distinct early prompt content across repetitions to avoid accidental
prefix hits, but the corresponding A/B repetition uses the same prompt IDs.

Primary timing cells set `temperature=0`, `n=1`, `ignore_eos=true`, no custom
stop strings, and fixed OSL. Check actual accepted output tokens equal the
target; an ignored flag, early stop, truncation, or usage discrepancy invalidates
that cell. Quality cells instead use normal EOS and sufficient output budget.
No logprobs, streaming usage frequency changes, or verbose server logs on only
one side of a timed comparison.

### Upstream Client Recipe for a Future Approved Run

This command is a template, **not executed by this review**. Provision the
pinned client locally, check its `bench serve --help` and offline payload,
and fill all manifest-backed variables. A prepared JSONL contains at least N
distinct raw prompts; the local `prompts.json` array is not this format.

```bash
vllm bench serve \
  --backend openai --endpoint /v1/completions \
  --base-url "$BASE_URL" --model "$SERVED_MODEL_ALIAS" \
  --tokenizer "$PINNED_TOKENIZER_DIR" \
  --dataset-name custom --dataset-path "$FROZEN_DATASET_JSONL" \
  --skip-chat-template --custom-output-len "$OSL" \
  --num-prompts "$N" --no-oversample --seed "$SEED" \
  --request-rate inf --max-concurrency "$C" \
  --ignore-eos --extra-body '{"temperature":0,"n":1,"seed":17,"stream_options":{"include_usage":true}}' \
  --save-result --save-detailed \
  --result-dir "$RESULT_DIR" --result-filename "$RUN_ID.json"
```

These controls are documented in the upstream
[serving client CLI](https://docs.vllm.ai/en/latest/cli/bench/serve/).
The dataset seed controls client ordering; the explicit request seed is separate.
Verify both endpoints accept the sampling payload and preserve its semantics.
Warmup is a separate run with its own corpus and discarded timing. Account for
any automatic test/readiness request the pinned client sends: it warms the
server too. This closed-loop saturation experiment is not an open-loop QPS/SLO
load test. Do not reuse its latency percentiles as an arbitrary arrival-rate SLA.

## Metric Contract and MTP Timing

Save request enqueue/send time, first content-bearing event, every event's
monotonic receive time, last content-bearing event, terminal event, and `[DONE]`.
Save raw SSE payloads, request/response IDs, full output text, accepted token
usage, finish reason, HTTP status, and errors. Token IDs or cumulative accepted
token counts per event are preferred if already exposed by both endpoints;
record their absence explicitly. Client-side text retokenization is a cross-
check, not a substitute for accepted token IDs, especially across BPE chunks,
special tokens, reasoning, and byte decoding.

| Reported quantity | Definition / limitation |
| --- | --- |
| First-visible-text latency | First nonempty raw text event minus send time. Includes transport, server queue, tokenization, prefill, and first verification/output work; not pure prefill. |
| Request E2E | Record both last-output minus send and protocol-done minus send. Choose one fixed primary definition across backends; include client queue separately. |
| Aggregate output throughput | Sum successful server-reported accepted completion tokens divided by the same whole-trial wall interval. Also report failed/attempted counts and failure duration; a comparison gate requires zero failures. |
| Steady-window delivery throughput | Only with event traces: accepted tokens actually delivered inside a preregistered central window divided by its duration. Record in-flight occupancy and boundary treatment. Without traces, publish finite-batch throughput only. |
| Conventional TPOT proxy | `(last_output_time - first_output_time)/(Nout - 1)`; retain the exact convention but label chunk ambiguity. Null when not observable. Request-averaged percentiles are not token-level percentiles. |
| Inter-chunk latency | Consecutive content-event arrival gaps, including stalls. Report count and p50/p95/p99 only with enough events. Network buffering can merge engine emissions; these are not automatically engine-step durations. |
| Post-first-chunk average | If k1 accepted tokens in the first event and Nout covers exactly the traced tokens, use `(t_last - t_first)/(Nout - k1)` for the remaining delivery interval. This still cannot resolve timing inside a chunk. Otherwise do not compute it. |
| Effective input service rate | For OSL1 only, actual prompt tokens divided by first-output latency. Label queue-inclusive at C>1. A C1 TTFT-versus-ISL fit can estimate an effective slope within one warm/cache regime, not a GPU kernel rate. |
| Pure prefill/decode and MTP efficiency | Require server-side phase times/counter deltas with documented definitions. Save accepted draft tokens, proposed tokens, verification iterations, and per-position acceptance when available. State whether the bonus token counts toward tokens/step. No such claim from SSE event counts alone. |

Example: eight accepted tokens arrive as two four-token chunks separated by
100 ms. The current proxy is `100/7 = 14.29 ms`; the post-first-chunk delivery
average is `100/4 = 25 ms`. Neither measures the internal latency of each
token. The existing JSONs expose 128 tokens in only 38-51 text events, so
single-token-event assumptions are demonstrably unsuitable here.

Report per-run p50/p95 TTFT and E2E, throughput, actual output length, termination,
and errors. Mark p99 exploratory when sample size is small; do not manufacture
a tail guarantee from 32 requests. Across repeats publish each value, median,
range, and paired ratios; uncertainty estimates should resample runs, not
pretend correlated speculative tokens are independent requests. Define
user SLOs before calculating goodput; [AIPerf defines goodput](https://docs.nvidia.com/aiperf/dev/tutorials/metrics-analysis/benchmark-goodput-with-ai-perf)
as completed requests per second meeting specified metric constraints. Do not
invent a post-hoc latency threshold that selects a winner.

## Quality and Compatibility Gates

1. Verify the immutable checkpoint revision
   `925d7be6c14c6c9442ef83e8f05b5a3c39304f69`, tokenizer files/revision, chat
   template and its arguments, BOS/EOS/stop policy, raw prompt bytes, and input
   token IDs. Use `/v1/completions` with exactly one pre-rendered prompt; do not
   submit already rendered ChatML to a chat endpoint or apply the template twice.
2. Resolve both the served alias and actual loaded checkpoint from existing
   deployment provenance. `qwen3.8-flash-next`, the HF repository name,
   `Qwen3_8FlashNextForConditionalGeneration`, and TRT's canonical Qwen4Exp
   architecture are different identifiers. Never let a benchmark silently use
   the first `/v1/models` entry or infer tokenizer identity from a served alias.
   Pin the tokenizer path explicitly; an alias resolving to another model is a
   hard failure, not a slower/faster result.
3. Confirm the actual quantization map: routed NVFP4, MXFP8 linears,
   W4A16_NVFP4 MTP/vision, and packed NVFP4 PLE. Record any fallback or algorithm
   promotion. Match FP8 KV subtype/scales and actual recurrent-state dtype;
   neither checkpoint dtype nor recipe defaults prove deployed precision.
   The single trained MTP layer has relative weight index 0; metadata aliases
   0/48 now resolve to runtime 48. Repeated drafting is not extra trained layers.
4. Require complete, coherent outputs with no NaNs, stalls, repeated garbage,
   silent truncation, omitted final answers, or missing usage. The arithmetic
   smoke must finish with 80 km/h. A truncated hash-table explanation or correct
   answer only inside `<think>` is not a complete user-facing quality pass.
5. Freeze a scored suite of arithmetic, code with unit checks, multilingual
   instructions, short factual extraction, and deterministic long-context
   retrieval at multiple positions. Specify expected answers and tolerances
   before measuring speed. Establish the reference score, require every
   deterministic acceptance case to pass, and investigate all regressions.
   This is bounded regression evidence, not general model-quality validation.
6. On TRT, compare MTP-enabled output against its own no-MTP baseline before
   tuning. Greedy/token divergence requires investigation; floating-point and
   batch differences can break cross-backend exact text equality, so assess
   both token/logit evidence where available and task correctness. Do not claim
   lossless MTP from two plausible responses or from acceptance rate alone.
   094a remains MTP3 throughout, so this review cannot establish its no-MTP parity.

Raw completions include thinking tokens in accepted usage and throughput. Save
them, do not strip them before counting, and report final-answer completion
separately. Never count rejected draft tokens as output throughput. Fixed-OSL
stress results and normal-EOS quality results belong in separate tables.

## Resource, Cache, and Provenance Controls

Before a future run, capture read-only deployment evidence and admit the cell
against available memory. Do not start a second model on 094a or stop unrelated
services. The 3883 experiment's recorded limits are 84 GiB memory, no extra
swap, 12 CPUs, and 4 GiB shared memory. Do not remove these limits to obtain a
headline number. If reference limits differ, record them as a D-track confounder.

The run manifest must include:

- Backend git SHA, dirty diff hash, native-library/build SHA, container digest,
  Torch/Transformers/FlashInfer/CUDA/driver versions, kernel, model index and
  quantization-file hashes. Config compatibility commit `f74e6657` alone is not
  the complete runtime provenance.
- Exact server argv, YAML/environment and effective resolved settings: context
  and RoPE/YaRN, TP/EP/PP, active-sequence limit, batched-token/prefill-chunk
  budget, CUDA graph modes and captured widths, MTP token count/draft vocabulary,
  KV size/dtype/reuse, recurrent dtype, PLE placement, and quantization fallbacks.
- GPU model/SM121, power mode/clocks/temperature/throttling, CPU quotas/affinity,
  cgroup current/peak/limit/events, host available memory, RSS/PSS, file-backed
  versus anonymous memory, GPU allocation/reservation, page faults, and disk I/O.
  GB10 unified/file-backed memory is not fully characterized by Torch allocation
  counters; do not add overlapping host/device numbers as independent totals.
- Other services/users, health and error logs, power/energy if available,
  observation sampling rate and overhead, startup/ready/warmup/run timestamps,
  client version, dataset/payload hashes, run seed, endpoint alias and network
  topology. Redact secrets but retain reproducible non-secret settings.

Use the same external client host and transport topology for both backends.
The historical 094a requests traversed a local SSH tunnel; the status reports
Wi-Fi and down Ethernet links. A tunneled reference versus localhost TRT is
an end-to-end deployment comparison with network confounding, not a compute
benchmark. Observe RTT/jitter and client saturation when authorized. Do not
subtract a single RTT constant from TTFT or pretend buffering cancels in all
chunk intervals. No clock or power-policy changes on the reference.

Treat cache state as multiple independent axes:

| State | Handling |
| --- | --- |
| Process/model cold | Startup/load/ready/first generation only on approved 3883 launches. No matched cold-start claim: 094a must not restart. |
| Runtime warm | Exclude compilation, autotuning, and graph capture using disjoint warmups for each shape. Record completion of warmup, not just a fixed sleep. |
| Prefix/KV reuse | Determine effective settings without changing 094a. Main corpora have unique early content, but shared system prefixes can still hit. Record cached prompt counts where exposed. A separate deliberate repeated-prefix workload may characterize reuse; never mix it into uncached-prefill claims. |
| PLE/OS page cache | Prompt variation changes which mapped n-gram rows are touched. Report natural observed state and page faults. New prompts do not guarantee cold pages. Never drop host caches or evict the reference's files. |
| GPU/CPU storage residency | Record mapped backing, page-cache footprint, major faults, reclaim/thrashing and warm-path working set. A warm PLE microbenchmark does not establish sustained full-model behavior under memory pressure. |

The local PLE artifact used only 423,646 bytes of synthetic source data and
reports its own limitations: not full-model or memory-pressure validation,
OS-page-cache cold rather than SSD/controller cold, and timings excluding
ID/output transfers. It is correctness/storage feasibility evidence, not a
token-throughput result.

Abort a future trial on an OOM/worker restart, health regression, operator
memory-reserve breach, unacceptable impact on other users, or irrecoverable
request failure. Preserve the failure artifact and do not silently retry into
a more favorable cache regime. Predeclare timeouts appropriate to long prompts
and a whole-trial time budget. Exclude instrumentation-only runs from primary
performance tables if their overhead is not matched.

## Existing Results and External Claims

Historical files in `/home/mihai/workspace/flashnext-results`, not rerun here:

| Artifact | Client C / requests | Actual output total | Batch wall seconds | Aggregate accepted output tok/s |
| --- | --- | --- | --- | --- |
| `20260906-vllm-reference-c1.json` | 1 / 2 | 256 | 6.6816 | 38.3141 |
| `20260906-vllm-reference-c1-384.json` | 1 / 2 | 568 | 15.5999 | 36.4105 |
| `20260906-vllm-reference-c2.json` | 2 / 2 | 256 | 4.8313 | 52.9880 |

Input lengths were 37 and 48 tokens. The 384-cap arithmetic request stopped
normally at 184 output tokens; its companion reached 384. Both 128-cap
requests ended by length. These tiny, non-repeated observations establish an
initial working reference only: they are not a concurrency scaling curve,
an EOS-controlled fixed-work comparison, or a TRT/vLLM speed ratio. Historical
files omit `ignore_eos`; do not infer its value solely from today's CLI default.
They contain no per-event token/timestamp trace with which to repair MTP timing.

Reference artifact SHA256 values:

```text
c1:     2acd68ee85d52e199ea76ecbe4a8a6d52ed4c1616190a9b176bf6c8fdafe8502
c1-384: e28c28955aa09c89263b513cfc8de8d416a67b830e9303e71a55e7f4aa98a8fd
c2:     fead31c4063d770891a1fa4d64029ae004338b0e3b0bbe265650eb567328d7f3
```

The [pinned recipe README](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/ef1af5fa2e1e93d4bd96136568460cda843ccc29/README.md)
reports this 2026-09-06 prose/600-token sweep, three repetitions per level:

| Streams | Aggregate tok/s | Per-stream tok/s | ms/engine step | Tokens/step |
| --- | --- | --- | --- | --- |
| 1 | 48.7 | 48.7 | 61.5 | 3.00 |
| 2 | 74.6 | 37.3 | 74.3 | 2.83 |
| 4 | 113.7 | 28.4 | 96.2 | 2.84 |
| 8 | 162.9 | 20.4 | 131.0 | 2.81 |

Its configuration is 512k YaRN, MTP3, FP8 KV, BF16 recurrent state, 2048-token
prefill chunks, V2 runner, and full decode graphs with `MAX_NUM_SEQS=8`.
Reported warm-prefill rates include 2200 tok/s at 8k, 2257 at 64k, and 1944 at
256k. These are author-reported external measurements, not 094a's native-
262144/cap4 profile or measurements of this fork. Do not divide these rates
by the local API TPOT proxy or call their difference a backend speedup.

## Deliverable and Claim Gate

Produce one immutable directory per run containing manifest, corpus hashes,
full request/usage/error JSONL, raw timing trace, aggregate output, resource
observations, quality scores, and effective server configuration. A summary
row identifies track, host/backend, ISL/actual OSL, client C, active cap, MTP,
cache state, success count, TTFT/E2E, batch throughput, chunk gaps, memory,
and the artifact paths. Leave unobservable phase/acceptance values unavailable.

Publish a win only for passing, comparable paired cells and a named metric,
with variability and configuration differences alongside it. Prefer a
latency/throughput/resource tradeoff table to one headline number. No claim
of eight-active-stream superiority against cap4, isolated prefill from TTFT,
token-level jitter from chunks, general quality from smoke prompts, or
full-model memory safety/performance from the synthetic PLE probe is supported.
