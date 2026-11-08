# Multi-GPU LLM Inference Gateway

A work-in-progress serving gateway for a planned NVIDIA Triton Inference Server + vLLM deployment. The local prototype provides bounded admission, streaming proxying, basic metrics, and a concurrent load generator. The multi-GPU backend and performance claims have **not yet been validated**.

## Run locally

Requires Python 3.10 or later. In separate terminals:

```sh
python -m tests.mock_upstream
python -m gateway.server
python benchmarks/run_load.py --requests 20 --concurrency 4 --output results.jsonl
python -m unittest discover -v
```

The mock returns deterministic text; its timings are functional checks, not inference results. The gateway defaults to `127.0.0.1:8080` and upstream `http://127.0.0.1:9000`. Configure `UPSTREAM_URL`, `MAX_ACTIVE`, `MAX_BODY_BYTES`, `UPSTREAM_TIMEOUT_SECONDS`, `HOST`, and `PORT` through environment variables. GET `/healthz` checks the gateway process, not backend readiness. GET `/metrics` exposes simple Prometheus-format counters and an active-request gauge.

The gateway forwards JSON chat requests at `/v1/chat/completions` and streams the response; it does not implement the entire OpenAI API. At capacity it returns HTTP 429 immediately. The limit is per process and counts active requests, not queued work. The timeout is an upstream socket inactivity timeout, not an absolute request deadline. The request limit is bytes, not tokens. No authentication or TLS is provided; do not expose this prototype to the public internet. Headers such as Authorization are not forwarded.

## GPU deployment and experiment

[compose.gpu.yaml](compose.gpu.yaml) prepares the pinned Triton/vLLM image and model for TP=1 or TP=2. It has not been validated on a GPU. Follow the [Lambda rental walkthrough](docs/lambda-rental.md) for a specific provider and the [runbook](docs/runbook.md) for host requirements, smoke tests, budget controls and experiment details. The gateway dashboard covers gateway metrics; GPU utilization is collected separately with `nvidia-smi`.

The benchmark sends a fixed prompt and records TTFT to first content, completion time, response text and any usage returned by the server. It calculates latency percentiles and reports token throughput when server usage is complete. Treat incomplete or failed streams as errors before using results for a report. See [the runbook](docs/runbook.md) for the rental gate and experiment design.

NVIDIA's [vLLM backend](https://github.com/triton-inference-server/vllm_backend) handles continuous batching; this project does not claim Triton dynamic batching performs it. NVIDIA Triton Inference Server is distinct from the Triton GPU kernel language.

## Local monitoring stack

`docker compose up --build` runs a mock backend, gateway, Prometheus and Grafana. All published ports bind to localhost. Open Prometheus at http://localhost:9090 and Grafana at http://localhost:3000 (initial Grafana credentials are `admin` / `admin`; change them if retained). Prometheus and a gateway dashboard are provisioned. This is a development stack with ephemeral monitoring data; it has no backend GPU metrics configured. Stop with `docker compose down`.

Run `python benchmarks/run_load.py --workload workloads/prompts.json --warmup 2 --include-usage --output results.jsonl`. The JSONL contains individual requests and the adjacent `.summary.json` contains run parameters, workload hash, success/failure counts, p50/p90/p95/p99 for successful requests, and successful request throughput. TTFT measures first nonempty content, not necessarily a token. Output token throughput is reported only if every successful response includes a completion-token count; the mock counts are synthetic. Percentiles use linear interpolation. The run wall time includes client scheduling and failures, and warmups are sequential and excluded. See the runbook for input token counts and the need to validate backend usage before GPU measurements. A socket inactivity timeout is not a total deadline. `python benchmarks/compare.py A.summary.json B.summary.json` compares matched runs and rejects different workloads or key settings.
