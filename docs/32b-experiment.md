# 32B serving and admission-control experiment

The measured experiment is complete. See the [report and raw records](../results/32b-admission/report.md) for all three five-minute cases, and [project status](project-status.md) for the remaining account/billing closeout. No additional rental is required to reproduce the analysis.

This experiment uses Qwen2.5-32B-Instruct-GPTQ-Int8, revision `eddc13f573fd3648cc8a4741fdf1b70e8d6fc5c1`, on two RTX 4090 GPUs with 24 GiB each. Quantization is part of the experiment identity: do not describe the result as BF16 or claim quantization quality was evaluated. The pinned checkpoint has approximately 35.1 GB of weight files, and the serving logs showed 16.36 GiB of model weights per GPU with TP=2. No CPU-offload configuration was enabled.

The matched Triton 25.08 image and bundled vLLM 0.9.2 are pinned in `compose.gpu.yaml`. vLLM selected the GPTQ Marlin kernel. `compose.32b.yaml` sets TP=2, 32 engine sequence slots, maximum context 2048, and GPU memory utilization 0.9. The model, engine settings, and workload stay fixed when changing the gateway's admission limit.

The workload is `workloads/prompts-512.json`: eight synthetic prompts with 506-513 template input tokens, round-robin assignment, temperature zero, and at most 128 output tokens. This is a controlled synthetic workload, not a representation of production traffic or a model-quality evaluation.

## Measurement design

The fixed-arrival generator sends uniformly spaced requests independently of completions. It records planned and actual dispatch times, schedule-relative TTFT and completion latency, HTTP failures, token usage, and response text. It writes individual records as requests finish. Each measured window lasts 300 seconds; outstanding requests drain afterward and the total observation duration includes the drain. Four warmup requests are excluded.

The illustrative latency objective, selected for the 32B calibration and kept fixed for the long runs, is first content within 500 ms and completion within 10 seconds. SLO goodput counts successful requests meeting both thresholds; rejections and late successes do not qualify. It is not a production SLA.

The three long-run cases are:

| Case | Gateway active-request limit | Offered requests/s | Arrival duration |
|---|---:|---:|---:|
| Reference overload | 32 | 4 | 300 seconds |
| Protected normal load | 16 | 2 | 300 seconds |
| Protected overload | 16 | 4 | 300 seconds |

The two overload cases provide the matched comparison of admission policies. The protected overload is twice the normal offered rate; observed rejection and latency determine how the service handles it. A tighter limit can reduce admitted-request latency while rejecting more work. Report both effects and goodput across all offered requests, not only a percentile among survivors.

Example launch:

```sh
MAX_ACTIVE=16 docker compose -f compose.gpu.yaml -f compose.32b.yaml up -d --build
```

Example long run after readiness:

```sh
python3 benchmarks/run_arrival.py --model qwen --workload workloads/prompts-512.json --warmup 4 --rate 2 --duration 300 --timeout 120 --ttft-slo-ms 500 --latency-slo-ms 10000 --label protected-cap16-rate2-300s --output artifacts-32b/protected-cap16-rate2-300s.jsonl
```

Record the exact source files, hashes, driver version, resolved Compose configuration, startup logs, and offered rates alongside results. Source changes after the base Git commit must be retained in the artifact bundle. Prometheus scrapes gateway and Triton metrics, Grafana visualizes them, and independent GPU utilization/memory/power samples are recorded every five seconds. Verify actual tokenizer counts, dispatch lag, errors, and artifact readability before publication. The checkpoint was not validated with TP=1; single-card GPU residency is ruled out by the weight footprint, not by a claimed TP=1 measurement.

## Limits

The VM reports PCIe host-bridge connectivity and unavailable direct peer access for some GPU pairs. Trial order is sequential, the workload and output length are bounded, and uniform arrivals do not cover bursty or Poisson traffic. One five-minute run per case does not establish production availability or a universal capacity figure. Cost estimates require the actual provider rate; total paid cost also includes startup, downloads, storage, networking, and idle time. Reproduction on another host requires its own driver/runtime and streaming validation.

## Recompute the published result locally

From the repository root, using Python 3.10 or newer:

```sh
python -m pip install -r requirements-analysis.txt
python scripts/verify_results.py
python benchmarks/analyze_32b.py results/32b-admission
```

This verifies captured file hashes, checks raw request counts and resolved admission limits, and regenerates the analysis JSON and two figures. It requires no GPU.

## Re-run the measurement on a different host

These commands use a Linux shell and write to a fresh ignored artifact directory. Do not overwrite the published `results/` data with a new host's measurements. Allow approximately 17 minutes for the three cases and warmups after the model is ready; container/model downloads and initialization are additional and host-dependent.

### 1. Verify the host and start the model

Confirm two suitable GPUs, working container GPU access, Docker Compose, disk space, and a compatible driver before starting the full stack. The measured host used driver 580.95.05; do not upgrade a working driver during a measurement session. Run from the cloned project root:

```sh
mkdir -p artifacts/reproduction-32b
nvidia-smi > artifacts/reproduction-32b/nvidia-smi-initial.txt
nvidia-smi topo -m > artifacts/reproduction-32b/topology.txt
docker version > artifacts/reproduction-32b/docker-version.txt
docker compose version > artifacts/reproduction-32b/compose-version.txt
git rev-parse HEAD > artifacts/reproduction-32b/git-revision.txt

MAX_ACTIVE=32 docker compose -f compose.gpu.yaml -f compose.32b.yaml up -d --build
docker compose -f compose.gpu.yaml -f compose.32b.yaml logs -f backend
```

After the model and frontend are ready, stop following logs with Ctrl-C. This leaves the containers running. Confirm a streaming request completes with nonempty output and token usage:

```sh
python3 benchmarks/run_load.py --model qwen --workload workloads/prompts-512.json --warmup 1 --requests 1 --concurrency 1 --max-tokens 128 --include-usage --output artifacts/reproduction-32b/smoke.jsonl
curl -fsS http://127.0.0.1:9090/api/v1/targets > artifacts/reproduction-32b/prometheus-targets.json
```

Check the smoke record for `completed: true`, HTTP 200, and output usage. Inspect the Prometheus response for both scrape targets being up. `/healthz` alone does not prove model readiness. Keep the gateway and monitoring ports on loopback; use an SSH tunnel for remote viewing.

### 2. Record GPU samples and execute all three cases

The backend remains running with unchanged model/engine settings. Only the gateway is recreated when its admission limit changes. The process health check below follows that recreation; the excluded warmups also require a working backend.

```sh
set -euo pipefail
OUT=artifacts/reproduction-32b
nvidia-smi --query-gpu=timestamp,index,name,driver_version,memory.total,utilization.gpu,memory.used,power.draw --format=csv -l 5 > "$OUT/gpu-samples.csv" &
GPU_MONITOR_PID=$!
trap 'kill "$GPU_MONITOR_PID" 2>/dev/null || true' EXIT

run_case() {
  cap="$1"; rate="$2"; label="$3"
  MAX_ACTIVE="$cap" docker compose -f compose.gpu.yaml -f compose.32b.yaml up -d --no-deps gateway
  for attempt in $(seq 1 30); do
    if curl -fsS http://127.0.0.1:8080/healthz > /dev/null; then break; fi
    sleep 1
  done
  curl -fsS http://127.0.0.1:8080/healthz > /dev/null
  MAX_ACTIVE="$cap" docker compose -f compose.gpu.yaml -f compose.32b.yaml config > "$OUT/$label-compose.yaml"
  curl -fsS http://127.0.0.1:8080/metrics > "$OUT/$label-gateway-before.txt"
  python3 benchmarks/run_arrival.py --model qwen --workload workloads/prompts-512.json --warmup 4 --rate "$rate" --duration 300 --max-tokens 128 --timeout 120 --ttft-slo-ms 500 --latency-slo-ms 10000 --label "$label" --output "$OUT/$label.jsonl" > "$OUT/$label.console.txt"
  curl -fsS http://127.0.0.1:8080/metrics > "$OUT/$label-gateway-after.txt"
}

run_case 32 4 reference-cap32-rate4-300s
run_case 16 2 protected-cap16-rate2-300s
run_case 16 4 protected-cap16-rate4-300s

docker compose -f compose.gpu.yaml -f compose.32b.yaml logs --no-color > "$OUT/container-logs.txt"
kill "$GPU_MONITOR_PID"
trap - EXIT
```

The reported 32B experiment used this case order. For a new study intended to estimate variability, repeat and reorder cases and report each repetition; do not silently pool those with the original measurements.

### 3. Export, inspect, and stop billing

Save the resolved model configuration from the running container, the executed source, and final GPU state:

```sh
docker compose -f compose.gpu.yaml -f compose.32b.yaml cp backend:/tmp/gateway-models/qwen/1/model.json "$OUT/model.json"
cp gateway/server.py benchmarks/run_load.py benchmarks/run_arrival.py deploy/start_backend.py compose.gpu.yaml compose.32b.yaml workloads/prompts-512.json "$OUT/"
nvidia-smi > "$OUT/nvidia-smi-final.txt"
date -u > "$OUT/export-utc.txt"
tar -czf artifacts/gateway-reproduction-32b.tar.gz artifacts/reproduction-32b
```

Copy the archive to your computer and confirm that the raw JSONL files and summaries are readable. Preserve any Grafana screenshots and export Prometheus range data before tearing down monitoring; a screenshot alone cannot reconstruct client latency. Record the provider's actual rate and invoice separately.

Stop containers with `docker compose -f compose.gpu.yaml -f compose.32b.yaml down`, then terminate the rented instance through the provider and verify that no unwanted storage remains billable. Once data is local, install the analysis requirements and run:

```sh
python benchmarks/analyze_32b.py artifacts/reproduction-32b
```

The current published results already have their capture manifests and exported monitoring data. A new run should retain its own configuration, source, telemetry, and billing record.
