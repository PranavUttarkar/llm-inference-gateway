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

## Next milestones

1. Pin and validate a compatible NVIDIA Triton vLLM container and OpenAI frontend together.
2. Smoke test streaming and GPU visibility before running an experiment.
3. Compare TP=1 and TP=2 on the same model and workload; report latency, throughput, failures and cost, with raw data.
4. Add dashboards once actual backend metric names and availability are verified.

The benchmark currently sends a fixed short prompt and records TTFT to first content, completion time, response text and any usage returned by the server. It does not yet calculate token throughput, percentiles, or handle every SSE variant. Treat incomplete or failed streams as errors before using results for a report. See [the runbook](docs/runbook.md) for the rental gate and experiment design.

NVIDIA's [vLLM backend](https://github.com/triton-inference-server/vllm_backend) handles continuous batching; this project does not claim Triton dynamic batching performs it. NVIDIA Triton Inference Server is distinct from the Triton GPU kernel language.
