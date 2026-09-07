<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Flash Next: Existing Spark Runtime

This guide reuses the built `trtllm-flashnext` container on `spark-3883.local`.
It is not a clean-wheel installation guide. Core native libraries were built
from `70feda6395`; Python runtime changes through `1175859d` are staged in the
container's workspace mount. See [status and measurements](../../FLASHNEXT_STATUS.md)
and [the bug ledger](BUGS.md). Do not change the reference on `spark-094a`.

## Tested Limits

- Checkpoint: `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4`, revision
  `925d7be6c14c6c9442ef83e8f05b5a3c39304f69`, text-only view.
- PyTorch backend, SM121, TP1, batch cap 2, total sequence limit 2048.
- Tests used short prompts below 512 tokens; chunked prefill is disabled.
- CUTLASS MoE, opt-in B12x MXFP8 dense linears, BF16 KV, FP32 recurrent state.
- Eager no-MTP and MTP3 serving completed the quality/performance pilots.
  Some answers failed arithmetic or exact formatting; this is not a broad
  accuracy certification. Decode graphs and eager MTP at concurrency 2 have
  unresolved repeatability caveats. Eager no-MTP was repeatable in both pilots.
- The host is shared with unrelated services. Never stop those to run this.
  MTP plus graphs reached only 8.34 GiB host memory available in the extended
  pilot, close to the 8 GiB stop threshold. Do not increase workload limits or
  remove the guard based on a no-OOM result.

## Bounded Launch

First check that no experiment is running. An occupied GPU lock or port means
wait, not kill its owner. The following uses the already-staged no-MTP eager
configuration. For the faster MTP3 eager experiment, accepting its qualification
caveats, use `/host-workspace/flashnext-serve-mtp3-eager.yaml` instead.

```bash
ssh mihai@spark-3883.local 'docker exec trtllm-flashnext ps -eo pid,ppid,args'

ssh -o ServerAliveInterval=15 mihai@spark-3883.local bash -s <<'REMOTE'
set -euo pipefail
set -o noclobber
RUN=$(date -u +%Y%m%dT%H%M%SZ)
IP=$(docker inspect -f '{{.NetworkSettings.Networks.bridge.IPAddress}}' trtllm-flashnext)
test -n "$IP"
docker exec trtllm-flashnext python3 -c \
  'import socket,sys; s=socket.socket(); s.bind((sys.argv[1],18081)); s.close()' "$IP"
printf 'Run: %s; API: %s:18081; log: /tmp/flashnext-operator-%s.log\n' "$RUN" "$IP" "$RUN"
docker exec -w /host-workspace/TensorRT-LLM-FlashNext \
  -e PYTHONPATH=/host-workspace/TensorRT-LLM-FlashNext \
  -e LLM_MODELS_ROOT=/host-workspace \
  -e TLLM_LOAD_WEIGHTS_NUM_WORKERS=1 \
  -e TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL=1 \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e TRTLLM_MXFP8_GEMM_BACKEND=auto \
  -e TRTLLM_MXFP8_FLASHINFER_BACKEND=b12x \
  trtllm-flashnext flock -n -E 75 /tmp/flashnext-small-gpu-probe.lock \
  python -u /host-workspace/flashnext_admission_watchdog.py \
  --metrics "/tmp/flashnext-operator-$RUN.metrics.jsonl" -- \
  trtllm-serve /host-workspace/flashnext-text-925d7be6 \
  --backend pytorch --host "$IP" --port 18081 \
  --served_model_name flashnext-trt \
  --config /host-workspace/flashnext-serve-eager-20260906-1838.yaml \
  --num_serve_frontends 1 --num_input_processor_workers 1 \
  --num_media_load_workers 1 --num_postprocess_workers 0 \
  >"/tmp/flashnext-operator-$RUN.log" 2>&1
REMOTE
```

The printed log path is on the Spark host; metrics are inside the container.
This intentionally stops after **25 minutes including loading**. The guard also
stops on low host memory or new OOM events and cleans its owned descendants.
The 1 GiB cache quota is not a hard allocation ceiling: native slot floors can
increase it. Do not set a finite virtual-address-space limit or remove the
host-memory guard to force a run through. The deployment is not an indefinite,
auto-restarting service.

For local API access, use a free local port and discover the current bridge IP:

```bash
IP=$(ssh mihai@spark-3883.local \
  "docker inspect -f '{{.NetworkSettings.Networks.bridge.IPAddress}}' trtllm-flashnext")
test -n "$IP"
ssh -N -o ExitOnForwardFailure=yes \
  -L "127.0.0.1:18082:$IP:18081" mihai@spark-3883.local
```

Wait for `/health` and check `/v1/models`. The API has no authentication;
the loopback tunnel is not permission to expose the bridge service publicly.
Use [the benchmark plan](BENCHMARK_PLAN.md) for raw `/v1/completions` tests.
Do not compare reasoning-token throughput with answer-only token throughput.

To stop early, read `guard_pid` from this run's first metrics record, inspect
that PID's command line inside the container, then send that specific watchdog
SIGTERM. Wait for `cleanup_complete` and foreground exit. Killing only SSH
does not reliably stop remote workers. Preserve logs before another run.

Detailed launch hashes, raw results and the independent operator review are
in `/home/mihai/workspace/flashnext-results` on the development machine.
