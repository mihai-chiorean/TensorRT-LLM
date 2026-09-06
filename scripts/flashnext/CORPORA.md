<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Correctness and Throughput Corpora

`quality_cases.json` contains eight scored cases and a separate `expected_answers`
map keyed by case ID. Send only each case's `prompt`, never the answer map.
`benchmark_prompts.json` contains 32 distinct raw prompt strings accepted directly
by `benchmark_api.py --prompts`. These are authored short-to-moderate natural
prompts, not fixed-token-length workloads or measured benchmark results.

Both files retain the literal ChatML envelope used by the existing `prompts.json`.
Send the decoded strings unchanged, in the same order, to both engines via
`/v1/completions`; do not apply another chat template. This preserves matching
raw inputs, not proof of matching tokenization: verify the envelope against the
pinned tokenizer/template and freeze prompt/token-ID hashes before live runs.
No tokenizer was loaded to create these files. Record actual server-reported
`usage.prompt_tokens` and `usage.completion_tokens`; do not infer counts from
words, characters, text events, or the requested output limit.

## Quality Scoring

Use temperature zero, normal EOS, identical sufficient output budgets, and the
same sampling settings on both engines. Preserve the complete raw completion
and usage. Score only the completed final answer, not an answer inside reasoning.
An absent final answer, length-truncated completion, or protocol/usage failure
fails the quality gate. Predeclare any reasoning-section extraction rule for
both engines; do not search arbitrary output for a matching answer.

- JSON cases: after trimming surrounding whitespace, parse exactly one JSON
  object. Reject duplicate keys, NaN/Infinity, extra prose/fences, extra keys,
  wrong types, or wrong values. Object key order and JSON whitespace do not
  matter; array order does. Require integers where the prompt asks for integers;
  `speed_kmh` accepts a finite JSON number numerically equal to 80. Booleans are
  not numbers for scoring. No numerical tolerance is needed for these cases.
- Code generation: the final answer is a single Python function definition.
  Check every listed input/output pair, that the input is unchanged after each
  call, and that the return value is a distinct list. Reject imports, top-level
  calls, and I/O. Different correct implementations may pass. Run generated code
  only in a disposable restricted environment with CPU/time/memory limits and
  no network or secrets, never in the benchmark client or serving process.

The ledger queries exercise early, middle, and late positions in a short prompt;
they do not validate long-context retrieval. Eight cases are a bounded regression
gate, not general model-quality evidence. `benchmark_api.py` does not score the
quality file: a runner must extract `cases[*].prompt` and retain the case-ID map
separately. This sidecar adds data only, not an evaluator or execution sandbox.

## Throughput Boundaries

Run only after the parent's safety/quality gates and authorization. Use the same
frozen corpus, order, output budget, EOS policy, and raw inputs at client C1/2/4/8
on each engine. Fixed-output trials use `--ignore-eos` and must verify actual
accepted completion counts equal the target; keep these separate from quality
runs. Some prompts may naturally finish early or become repetitive if forced
past their answer; fixed-output text is not a quality score.

The unchanged reference has **four active sequence slots**. C8 is therefore
**queue-inclusive C8/cap4**, not eight active reference sequences. Match TRT's
active cap at four for the matched-settings track. The local client's aggregate
is finite-batch completed-output throughput, including ramp-up and drain; its
text-event timing does not resolve per-token MTP latency.

The 32-request corpus can exercise every requested client concurrency, but gives
only four nominal request waves at C8. The full [benchmark plan](BENCHMARK_PLAN.md)
requires `max(32, 16*C)` distinct timed requests: 32/32/64/128 at C1/2/4/8, plus
disjoint warmups and repetitions. Extend and freeze the corpus before full C4/C8
trials; do not duplicate these 32 strings and call the result distinct requests.
Reusing this file across runs may introduce prefix-cache hits. Shared system
prefixes also remain; distinct user openings do not guarantee cold KV or PLE
pages. Record observed cache policy and state without clearing reference caches.

The NVIDIA/Apache-2.0 notices above also apply to `benchmark_prompts.json`, whose
required array-of-strings schema cannot carry a metadata header without adding
a spurious benchmark request.
