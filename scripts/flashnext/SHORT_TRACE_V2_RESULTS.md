<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Short Trace V2 Diagnostic

## Result

The September9 UTC shorter capture also failed qualification. All eight
nonstreaming requests reached the immutable deadline without response bodies;
no finished Nsight report was produced. This is a profiling-path failure,
not a throughput measurement or proof of zero internal generation.

The earlier admission delay is eliminated: profile marker00:17:02 UTC,
first request00:17:05.012381, deadline00:18:32. Requests had approximately
87seconds, versus61.36seconds in the original failed capture. Shorter output
and a smaller capture range did not suffice. A specific profiler/runtime
fault remains unlocalized; do not claim an MPI failure or kernel hang.

## Frozen Protocol

- Original `a4345e6c` composite runtime and FI overlay, TC-decode True/MTP3.
- Same config, checkpoint, native/FI seeds and eight verified source hashes.
- First eight canonical prompts, C8 fixed64, ignore EOS, raw token IDs.
- Requested iterations0..5; CUDA software, MPI and graph-node trace options
  unchanged. No prior generation, manual readiness delay or retry.
- Automatic client staged before READY; runtime arguments and profile marker
  checked against the actual nested model log before release.
- Same90second profile-start cutoff including export,1000second guard,
  1200second independent Isaac recovery and existing memory/CPU limits.
- Independent runner review and all10 focused client CPU tests passed.

## Cleanup and Artifacts

Client exited1 and server supervisor124. Guard cleanup at00:18:34.278464 UTC
records remaining PIDs empty. Parent verified NVML contained only Isaac.
Isaac restored00:19:25.637722, fresh health ended00:19:55.662547, same identity.
No OOM was observed. Spark094a was not changed.

The6,885,772byte partial stream is preserved, not imported or treated as a
finished trace. Nsight's untimestamped connection-EOF/range-end messages appear
during shutdown; they do not establish the cause of the earlier timeout.
No third isolated importer attempt was made after the earlier two failures.

Artifacts in the sibling `flashnext-results` directory:

- `tracev2-short-true-20260908-client/`: all eight errors and requests.
- `tracev2-short-true-20260908-client-stage/`: admission and fetched runtime log.
- `tracev2-short-true-server-20260908.tar`: logs, guard metrics, initialization
  cache exports and `nsys-root/nsys-report-4c6c.qdstrm`.
- Archive SHA256: `55ad447b5f2a34f5044d1c6902b3c56875d6fb56f5b3f23a714e31a58f500788`.
- `tracev2-short-true-provenance-20260908.json`: frozen hashes and protocol.

## Decision

Do not repeat this profiling protocol or extend its timeout. Investigate the
capture mechanism independently and qualify scheduler overlap without a
profiler. N96 caching remains a source-backed hypothesis without a measured
full-model share. Neither failed trace changes the completed MTP screening
or supplies a new vLLM comparison.
