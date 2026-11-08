# GPU experiment runbook

The mock stack has been validated locally. The GPU configuration has **not** been run on a GPU; mock timings are not inference results. The [r25.08 frontend source](https://github.com/triton-inference-server/server/blob/r25.08/python/openai/openai_frontend/engine/triton_engine.py) emits SSE data events, an optional final usage event and `[DONE]`. Validate the actual image, token accounting, and TP=2 on a GPU before measuring.

## Version set

The GPU Compose file selects NVIDIA's `25.08-vllm-python-py3` image by manifest-list digest `sha256:782b60ab5c9ec57f045bfb1df91831a98cc0d094314fadaf8cf80e53581f79da` (resolved from NGC on 2026-11-08). NVIDIA's [compatibility matrix](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/introduction/compatibility.html) lists bundled vLLM `0.9.2+4ef1e343.nv25.8.post1.cu130`. The frontend and backend come from that image. Check [25.08 release notes](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-25-08.html) for driver/GPU compatibility and the TP>1 limitation with explicit model control. This deployment uses default model control.

The model is [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) at revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. The same downloaded snapshot is used for model and frontend tokenizer. This small model may not generalize to larger models; it must fit on one GPU for TP=1. The first start downloads it into a named Docker volume. Image pull, model download, startup and idle time can all incur rental charges.

## Before spending

1. Run `python -m unittest discover -v`, `docker compose up -d --build`, then `python benchmarks/run_load.py --workload workloads/prompts.json --warmup 2 --requests 16 --concurrency 4 --include-usage --output results.jsonl`. Check for zero failures, `gateway_accepted_total` in Prometheus at localhost:9090 and **Inference Gateway / Gateway overview** in Grafana at localhost:3000. Stop with `docker compose down`.
2. Choose a Linux host with two compatible NVIDIA GPUs, compatible drivers, NVIDIA Container Toolkit, Docker Compose, enough VRAM for TP=1 and disk for the image, model and artifacts. Check the actual offer, storage/network rates, billing minimums and taxes immediately before rental. Record the whole-instance hourly price and set a hard stop time under the $10–12 total budget, reserving funds for startup and retries.
3. Prepare SSH access, artifact export and provider termination. If the provider permits it without billing, pre-pull the pinned image and pre-download the model. Services have no authentication or TLS; published ports bind to loopback, and backend ports are not published. Do not expose Docker or the services publicly. Change Grafana's default credentials if retained.

As a planning reference checked 2026-11-08, [Lambda lists A6000 at $1.09 per GPU-hour](https://lambda.ai/instances), before tax, with the displayed price effective November 16, 2026 at 00:00 UTC. Lambda says to log in for current pricing. A hypothetical two-GPU rate at that future unit price is $2.18/hour, or about $8.72 for four hours before other charges; this is not a confirmed offer or reservation. Verify actual availability, configuration, final pricing and charges at launch. [Runpod lists RTX A5000 starting at $0.27/hour](https://www.runpod.io/gpu-models/rtx-a5000), but a suitable two-GPU offer and its total price have not been verified. Either candidate still requires checking the driver, interconnect and usable software environment.

## GPU session

These commands are for Linux. Save artifacts securely, keeping secrets out of files. Record provider, price, storage/network charges, start time, and offer details in `artifacts/metadata.txt`. Collect host and utilization data:

```sh
mkdir -p artifacts
nvidia-smi > artifacts/nvidia-smi.txt
nvidia-smi topo -m > artifacts/topology.txt
nvidia-smi --query-gpu=timestamp,index,name,driver_version,memory.total,utilization.gpu,memory.used,power.draw --format=csv -l 5 > artifacts/gpu-samples.csv &
GPU_MONITOR_PID=$!
docker version > artifacts/docker-version.txt
git rev-parse HEAD > artifacts/git-revision.txt
```

Start TP=1 and wait for the backend logs to show the model ready and frontend listening; a running container or gateway `/healthz` alone is insufficient. Run the smoke test, then inspect HTTP status, nonempty content, completion, usage, backend logs and GPU allocation. Stop and diagnose if streaming, usage or model loading fails. Never infer token counts from characters.

```sh
TP_SIZE=1 docker compose -f compose.gpu.yaml up -d --build
docker compose -f compose.gpu.yaml logs -f backend
# Ctrl-C stops following logs, not containers.
python benchmarks/run_load.py --model qwen --workload workloads/prompts.json --warmup 1 --requests 1 --concurrency 1 --include-usage --label smoke-tp1 --output artifacts/smoke-tp1.jsonl
```

Use the same host, visible GPUs, workload, gateway cap and settings for both modes. TP=1 is one model instance on one GPU while the second GPU remains part of the paid host; TP=2 shards one instance across both. This is not a comparison with two TP=1 replicas. The gateway cap defaults to 32 in the GPU configuration. The example matrix below has two repetitions, 64 measured requests and four sequential warmups at each concurrency; reduce it *before starting* if the estimated runtime exceeds the budget, and keep both modes matched. This is a closed-loop test, not a fixed arrival-rate test.

```sh
for C in 1 4 8 16 32; do
  for R in 1 2; do
    python benchmarks/run_load.py --model qwen --workload workloads/prompts.json --warmup 4 --requests 64 --concurrency "$C" --max-tokens 128 --timeout 120 --include-usage --label "tp1-c${C}-r${R}" --output "artifacts/tp1-c${C}-r${R}.jsonl"
  done
done
docker compose -f compose.gpu.yaml logs --no-color > artifacts/tp1-container-logs.txt
TP_SIZE=2 docker compose -f compose.gpu.yaml up -d --force-recreate backend
# Wait for readiness; repeat the smoke test with TP=2.
# Repeat the identical loop, changing only tp1 in label and output name to tp2.
python benchmarks/compare.py artifacts/tp1-c4-r1.jsonl.summary.json artifacts/tp2-c4-r1.jsonl.summary.json > artifacts/compare-c4-r1.json
```

Compare each matched pair. The comparison checks workload hash and benchmark settings but cannot verify hardware, backend version or external configuration. Inspect failures, 429s and usage completeness. Percentiles cover successful requests only, use linear interpolation, and p99 with 64 samples is weak. TTFT measures first nonempty content, which may contain multiple tokens. Output throughput divides successful completion tokens by whole measured wall time, including failures and client overhead. The workload and output lengths are limited; describe this when reporting results.

```sh
docker compose -f compose.gpu.yaml logs --no-color > artifacts/tp2-container-logs.txt
kill "$GPU_MONITOR_PID"
# Record end time, actual invoice, charges and deviations in metadata.txt.
# Export artifacts before terminating the instance.
docker compose -f compose.gpu.yaml down
```

Terminate the instance at the provider; stopping containers does not stop billing. Verify termination and clean up billable persistent resources when no longer needed. The named model cache persists until explicitly removed. Review logs and metadata for secrets before publication. Use [the results template](results-template.md); do not claim continuous batching caused an improvement based on concurrency changes alone.

## Workload token counts

`workloads/prompts.json` contains eight prompts assigned round-robin. Using Transformers 4.49.0 with the pinned Qwen tokenizer and its default user-only chat template including the generation prompt, input counts are `[38, 38, 37, 43, 40, 41, 40, 41]` in file order. The frontend may template or account differently; compare actual `usage.prompt_tokens` on the smoke run. Output-token counts must come from backend usage.
