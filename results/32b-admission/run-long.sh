#!/bin/bash
set -euo pipefail
cd /root/llm-inference-gateway
run_case() {
 cap="$1"; rate="$2"; label="$3"
 printf '%s setting cap %s at ' "$label" "$cap" >> artifacts-32b/long-progress.txt
 date -u >> artifacts-32b/long-progress.txt
 MAX_ACTIVE="$cap" docker compose -f compose.gpu.yaml -f compose.32b.yaml up -d --no-deps gateway > "artifacts-32b/${label}-setup.log" 2>&1
 for i in $(seq 1 30); do if curl -fsS http://127.0.0.1:8080/healthz > /dev/null 2>&1; then break; fi; sleep 1; done
 curl -fsS http://127.0.0.1:8080/healthz > /dev/null
 MAX_ACTIVE="$cap" docker compose -f compose.gpu.yaml -f compose.32b.yaml config > "artifacts-32b/${label}-compose.yaml"
 curl -fsS http://127.0.0.1:8080/metrics > "artifacts-32b/${label}-gateway-before.txt"
 printf '%s measured run starting after warmup at ' "$label" >> artifacts-32b/long-progress.txt
 date -u >> artifacts-32b/long-progress.txt
 timeout 480 python3 benchmarks/run_arrival.py --model qwen --workload workloads/prompts-512.json --warmup 4 --rate "$rate" --duration 300 --timeout 120 --ttft-slo-ms 500 --latency-slo-ms 10000 --label "$label" --output "artifacts-32b/${label}.jsonl" > "artifacts-32b/${label}.console.txt"
 curl -fsS http://127.0.0.1:8080/metrics > "artifacts-32b/${label}-gateway-after.txt"
 docker logs llm-inference-gateway-gateway-1 > "artifacts-32b/${label}-gateway.log" 2>&1
 printf '%s done at ' "$label" >> artifacts-32b/long-progress.txt
 date -u >> artifacts-32b/long-progress.txt
}
run_case 32 4 reference-cap32-rate4-300s
run_case 16 2 protected-cap16-rate2-300s
run_case 16 4 protected-cap16-rate4-300s
docker logs llm-inference-gateway-backend-1 > artifacts-32b/backend.log 2>&1
