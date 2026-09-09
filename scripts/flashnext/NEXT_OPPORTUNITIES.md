<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Next Performance Opportunities

September 8, 2026. Current model: Qwen3.8-Flash-Next NVFP4 on Spark3883.
This is an experiment queue, not new benchmark evidence. Preserve the existing
runtime and all earlier results. Spark094a remains unchanged.

## Execution Update

September9 overlap update: a strict private MTP3 allowance (`1e74d8af`) passed
88 CPU tests and independent review. The amended OFF/ON lifecycle screen
matched all15 fixed-output responses and EOS text; see
[lifecycle results](OVERLAP_LIFECYCLE_V1_RESULTS.md). Separate C8 performance
workers produced118.9900 tok/s OFF versus113.5604 ON (-4.56%); both small
quality suites scored6/1/1. Retain OFF; no keeper or new vLLM comparison.
The resolved KV allocations differed despite equal2GiB requests, so this is
a configuration-policy screen, not an isolated scheduling effect. See
[performance results](OVERLAP_PERF_V1_RESULTS.md).

Immediate next investigation is repeated N96 weight dequantization: measure
the exact deployed geometry and per-step frequency, then test a derived BF16
cache only if meaningful end-to-end time can be removed. Preserve invalidation,
refit and CUDA-graph pointer lifetime. PLE warmup remains a separate hypothesis.
Any future Nsight model capture first requires a CPU-only MPI canary. A small
later benchmark-client enhancement can preserve the existing response field
`avg_decoded_tokens_per_iter` without changing requests or runtime metrics;
this is generated tokens per request sampling iteration, not acceptance rate.

The chronology below preserves earlier proposals and their decisions; it does
not reopen completed overlap or MTP screening work.

The registered C8 confirmation is complete; the preparation text below remains
as the original protocol. Candidate gains were +4.25% and +1.08% in the two
prescribed order pairs, with differing small-suite quality outcomes. No
promotion. See [confirmation results](TC_CONFIRMATION_V2_RESULTS.md).

The subsequent bounded Nsight trace failed qualification: all eight requests
timed out and no completed report was exported. Actual submission left only
61.36s of the90s range budget. Partial-stream CPU recovery is diagnostic only;
no hotspot attribution follows yet. MTP1-versus-MTP3 preparation now has a
separate, reviewed private-cache scope allowance in research commit `ba0d02b3`.
It changes no model defaults or kernel and remains outside the stable runtime.

The MTP-length screening subsequently completed: MTP3 produced116.8461 tok/s
versus MTP1's105.6825, a10.56% lead in the registered fixed-order pair.
Retain MTP3; no acceptance or isolated-kernel causal claim follows. See
[MTP-length results](MTP_LENGTH_V1_RESULTS.md). The shorter separately registered
trace also timed out despite immediate submission; see
[short trace results](SHORT_TRACE_V2_RESULTS.md). Overlap correctness
qualification is next, with profiler diagnosis kept separate. The original
preparation sections below are historical proposals, not pending C8 work.

Overlap preparation review found a launch blocker: the pinned private
`_fi_cache_research._validate_scope` requires overlap OFF. A YAML-only ON
comparison cannot start. Any experiment must first review a narrowly scoped
research allowance and deploy it identically to both arms, retaining seeding.
The initial screen should distinguish baseline instability from ON-only
regression, use actual in-flight replacement requests rather than drained
waves, and keep text quality scoring separate from raw-ID responses.

## Immediate Control

The seeded non-atomic MTP candidate has a provisional C8 lead of 3.51%, not a
confirmed keeper. The preceding quality phase generated different token counts
between arms. See [the completed report](TC_SEEDED_PERFORMANCE_RESULTS.md).

Prepare a separate C8 confirmation: four fresh workers in True/False/False/True
order, equal fixed-length warmup requests, two timed rounds per worker, then
normal-EOS quality checks. Keep both seeds, source, graph settings and resource
limits fixed. Register exact requests and deadlines before launching. Do not
pool this new protocol with the old results or silently retry failed cells.

Equal prompts and output-token budgets do not imply identical expert routing,
PLE rows, accepted drafts or GPU work. Repeating the timed corpus measures a
warm-repeat workload, not unseen-prompt performance. A separate held-out probe
is needed before generalizing a repeat-only gain.

The independently reviewed proposal is preserved as
`flashnext-results/20260908-tc-confirmation-v2-proposal.md` in the sibling
results directory. Each fresh worker runs 32 fixed128 C8 warmup requests,
then two 32-request fixed128 C8 timing cells, then eight normal-EOS quality
requests. Caps are 90/90/90/60 seconds; the full client reserve is 385 seconds.
Before release, require the following on a consistent recorded clock basis:

```text
effective_cutoff = min(guard_start + 990, lease_deadline - 145)
effective_cutoff - max(now, ready_time + 30) > 385 + 5
```

Recheck admission before every phase. Guard+570 readiness is only a necessary
condition, not permission to run a late client. Preserve all failures and
quality results, exclude warmup from timing, and report both order pairs
separately. Same-sign tiny deltas alone do not establish a practically useful
improvement beyond variation. The V2 client and launcher were subsequently
deployed for the completed campaign recorded above.

## Ranked Investigation Queue

| Priority | Opportunity | Evidence and smallest useful test |
| --- | --- | --- |
| 1 | Explain the large warmup effect | True C1 improved from 28.40 to 36.37 output tok/s across rounds. Both TTFT and post-first-text time fell, including equal-text pairs. Run a separate repeated/reversed prompt diagnostic with precise client timestamps and low-rate clocks/fault telemetry. PLE page access is a hypothesis, not established causality. |
| 2 | Overlap host scheduling with GPU work | The current profile explicitly disables overlap. Capture CPU launch gaps and GPU idle intervals, then qualify the existing overlap option on matched fresh workers. Test early termination, turnover and state reuse before timing. Do not treat source eligibility as correctness qualification. |
| 3 | Remove repeated N96 weight dequantization | The current MXFP8 reference branch reconstructs and casts weights on every forward. A narrow derived BF16 cache for the GDN N96/K2560 projection could remove repeated work. Profile its full-model share first; the existing plan estimates 16.875 MiB for 36 buffers, excluding temporaries. Refit, graph pointer stability and cache invalidation need explicit tests. |
| 4 | Tune MTP length for the actual serving workload | First screening completed: retain MTP3,10.56% ahead of MTP1 in one fixed-order pair. A future causal investigation needs draft/verify time and workload acceptance measurements; acceptance from one sentinel is not a workload-wide estimate. |
| 5 | Longer-context KV and prefill policy | The current short-prompt, 128-output-token corpus cannot identify the best long-context policy. Compare FP8 KV and chunked prefill in separately admitted workloads, with quality and recurrent-state controls. These are workload-dependent experiments, not explanations for every short-context gap. |

These opportunities are not additive percentage promises. Measure time removed
from the end-to-end critical path: accelerating work already hidden by overlap
may not reduce latency, and eliminating a tiny operator has a limited ceiling.

### MTP Length Also Changes GEMM Dispatch

One concrete interaction is worth testing: at eight active pure-generation
sequences, MTP3 supplies 32 target token rows, whereas MTP1 supplies 16. The
current narrow dense B12x selector admits M16 but not M32 for its eligible GDN
projection geometries. Thus shorter drafting can change both verification work
and the dense kernel, not just acceptance. Source anchors: target input
construction in `model_engine.py:5657` and the B12x dispatch gate in
`linear.py:3453`, admitting M in `(1, 2, 4, 8, 16)` with the existing
device/dtype/geometry restrictions.

This is a source-based hypothesis, not a measured MTP1 win. Partial/mixed
batches and first-draft steps have different shapes; API C8 does not prove
eight active sequences. Fewer accepted tokens per verification may outweigh
a faster GEMM. Keep newly captured graph/tactic provenance and quality checks
for each draft length; report this as a configuration interaction, not an
isolated kernel speedup.

## vLLM Source Audit

Banach completed the source audit of the pinned replica on Spark3883 without
changing the working094a deployment. It compares PLE execution, dense MXFP8/N96, MTP experts,
GDN, graph capture and scheduler synchronization with the deployed TRT source.
The deliverable must separate already adopted, previously rejected and new
ideas, with source/version anchors and a minimal discriminating test.

The historical reference differs in KV precision/capacity, context length,
prefix reuse and chunked prefill. It is not a matched engine control. Use the
same client and request history for a future comparison, and report residual
differences explicitly. Recipe code is AGPL: do not transplant it into Apache
TensorRT-LLM; independently evaluate ideas and dependency licenses.

## Do Not Reopen Without New Evidence

- A tuned M32 B12x extension did not qualify as a keeper; do not simply widen
  the dense selector again.
- CPU-affinity A/B/A did not establish a benefit; retain original affinity.
- Global graph-enabled CUTEDSL has a large estimated workspace cost; MTP-only
  backend selection does not justify enabling it across all target layers.
- MTP prefix reuse has an unresolved draft-state initialization finding. Keep
  reuse OFF until that specific state lifecycle is qualified.
- Preserve existing dense B12x, graph, loader, PLE and correctness work. A new
  inconclusive experiment does not erase earlier measurements or fixes.

## Evidence and Ownership

Local raw research is preserved in the sibling `flashnext-results` directory:
`20260908-c1-round-warmup-attribution.md`, `N96-caching-plan.md`,
`20260907-overlap-trial-independent-review.md`, and
`20260908-vllm-reference-recovery.md`. Older proposals contain historical
resource/lifecycle details; use the current 1000-second lease-aware guard and
1200-second independent recovery, never their superseded launch commands.

Parent owns integration, experiment admission, independent review and service
restoration. MIT-912 tracks the overall work; MIT-923 tracks the MTP candidate.
No model was loaded and no performance default changed while preparing this
queue.
