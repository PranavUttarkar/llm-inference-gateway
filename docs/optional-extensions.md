# Optional extensions — November 10, 2026

Local implementation is complete for the follow-ups in `project-status.md`: reproducible arrival patterns, repeated/reordered admission trials with variability reporting, longer prompts, a two-replica deployment, and optional gateway token limits plus transport limits. These changes have local tests. They do not add GPU measurements to the published study.

## Repeated and reordered admission trials

Prepare a fresh output directory and seeded plan without contacting a server:

```sh
python benchmarks/run_suite.py --output artifacts/admission-plan --repeats 3 --seed 42
```

The planner shuffles the same three matched cases independently within each repetition: cap32/rate4, cap16/rate2, cap16/rate4. It saves the ordered trials, workload, executed source and SHA-256 hashes. Plan output is explicitly marked as unexecuted and cannot be analyzed as a measurement.

On a GPU host with the 32B model already ready, use a different fresh directory and add `--execute`:

```sh
python benchmarks/run_suite.py --output artifacts/admission-repeated --repeats 3 --seed 42 --execute
python benchmarks/analyze_suite.py artifacts/admission-repeated
```

This recreates only the gateway for each case, runs four excluded warmups, and records every request and each case's resolved Compose configuration. It stops on failure and records the failed trial in the manifest. The default nine windows take at least 45 minutes, plus warmups, recreation and drain. Estimate rental cost from the actual provider quote before execution. It neither rents nor terminates a VM. Capture GPU telemetry, logs, screenshots and provider billing using the [32B guide](32b-experiment.md), export them, and terminate through the provider afterward.

`variability.json` retains each trial and reports median/minimum/maximum across trial metrics. It never pools request percentiles or treats three repetitions as a confidence interval. Compare matched overload cases within repetitions and disclose the order and seed.

## Bursty and Poisson traffic

`run_arrival.py` accepts `--pattern uniform|burst|poisson`, `--burst-size`, and `--seed`. Uniform is the unchanged default. Bursts dispatch up to `burst-size` requests simultaneously every `burst-size/rate` seconds, keeping the same offered count as the uniform workload. The final burst may be partial. Poisson arrivals use seeded exponential gaps; rate is an expectation and the realized count varies. Both patterns retain dispatch lag and schedule-relative SLO accounting, including requests delayed by a saturated client worker pool.

```sh
python benchmarks/run_arrival.py --model qwen --workload workloads/prompts-512.json --rate 4 --duration 300 --pattern burst --burst-size 8 --seed 42 --ttft-slo-ms 500 --latency-slo-ms 10000 --label burst-cap16 --output artifacts/burst-cap16.jsonl
python benchmarks/run_suite.py --output artifacts/burst-repeated --pattern burst --burst-size 8 --repeats 3 --seed 42 --execute
```

Create the standalone arrival command's output parent first. The suite creates its own fresh directory. Run uniform and burst studies with identical model, workload, engine settings and SLOs; retain all failures and rejections. Existing uniform-arrival measurements remain a separate study.

## Longer prompts

Install `requirements-tokenizer.txt` if using generation or token admission. The checked-in `workloads/prompts-1536.json` extends the existing synthetic prompts and was counted using the pinned 7B tokenizer and its generation chat template. Its adjacent metadata records counts and tokenizer file hashes. It is synthetic repetition for a length-controlled test, not a realistic document or quality benchmark. Recount with the exact tokenizer of any different serving checkpoint before running it.

To regenerate a variant from an already downloaded local tokenizer snapshot:

```sh
python -m pip install -r requirements-tokenizer.txt
python benchmarks/make_workload.py --tokenizer /path/to/pinned/tokenizer --target-tokens 1536 --output artifacts/long-prompts.json
python benchmarks/run_suite.py --output artifacts/long-repeated --workload artifacts/long-prompts.json --repeats 3 --seed 42 --execute
```

Generation is offline and refuses to overwrite an existing output. Counts include chat-template overhead and the generation prompt; they fall within eight tokens below the requested target. The 2048-token context still needs room for the requested output (128 tokens in these commands). The generation API follows [Hugging Face's chat-template documentation](https://huggingface.co/docs/transformers/v4.49.0/chat_templating).

## Two independent TP=1 replicas versus one TP=2 instance

`compose.replicas.yaml` is a standalone stack. It pins the same 7B checkpoint and Triton image as the measured TP study. Each backend has its own container and a device reservation for GPU 0 or GPU 1, with TP=1. The gateway distributes accepted requests round-robin; there is no retry, failover, health-aware routing, or per-replica load balancing based on active work. A backend failure remains a failed request. This makes the routing policy explicit for the deployment comparison.

Use a Linux shell on the same two-GPU host. Stop one topology before starting the other:

```sh
mkdir -p artifacts/replica-comparison
docker compose -f compose.gpu.yaml -f compose.7b.yaml down
MAX_ACTIVE=32 docker compose -f compose.replicas.yaml up -d --build
docker compose -f compose.replicas.yaml logs -f backend0 backend1
```

Wait for **both** models to be ready. Verify each container sees only its assigned physical GPU, capture `nvidia-smi` and startup logs, and confirm successful requests reach both backends. `/healthz` only checks the gateway. Use the same workload, offered load, duration, warmups, global admission limit, context and memory fraction for both deployments:

```sh
python benchmarks/run_load.py --model qwen --workload workloads/prompts-512.json --warmup 4 --requests 100 --concurrency 8 --include-usage --label replicas-r1-c8 --output artifacts/replica-comparison/replicas-r1-c8.jsonl
docker compose -f compose.replicas.yaml config > artifacts/replica-comparison/replicas-compose.yaml
docker compose -f compose.replicas.yaml logs --no-color > artifacts/replica-comparison/replicas-logs.txt
docker compose -f compose.replicas.yaml down
TP_SIZE=2 MAX_ACTIVE=32 docker compose -f compose.gpu.yaml -f compose.7b.yaml up -d --build
```

After readiness, repeat the same client command with a `tp2-r1-c8` label and output filename. Extend to concurrency 1/4/8/16/32 and at least two repetitions, reversing topology order in the second repetition. Record the exact commands and resolved configuration for each run. Both deployments use and pay for the full two-GPU host. Do not substitute the old TP=2 measurements for the newly matched baseline: the gateway source and routing have changed. No replica GPU result is claimed yet.

## Gateway limits

The new serving code has the following defaults:

| Setting | Default | Behavior |
|---|---:|---|
| `MAX_CONNECTIONS` | 64 | Caps accepted handler threads before HTTP parsing. Excess TCP connections close immediately. |
| `CONNECTION_TIMEOUT_SECONDS` | 10 | Inbound socket inactivity timeout, including header/body reads and writes to slow readers. |
| `REQUEST_DEADLINE_SECONDS` | 120 | Total connection deadline, including inbound body and upstream streaming. Shuts down both sockets. |
| `MAX_ACTIVE` | 4 locally / 32 in GPU Compose | Immediate HTTP 429 when upstream admission is full. |
| `UPSTREAM_TIMEOUT_SECONDS` | 30 locally / 120 in GPU Compose | Upstream socket inactivity timeout. |
| `MAX_INPUT_TOKENS` | 0 (disabled) | Optional chat-template-aware input and requested-context limits. |
| `MAX_CONTEXT_TOKENS` | 2048 | Input plus requested `max_tokens` ceiling when token limits are enabled. |

Each HTTP connection handles one request and closes afterward. A deadline truncates an already started stream; clients must require the final `[DONE]`, as the benchmark does. Before response headers, deadline expiry may simply close the connection. Metrics include `gateway_deadline_total` and `gateway_connection_rejected_total`; transport overflow is distinct from admission HTTP 429. Handler threads are bounded; watchdog timers add at most one timer per handled connection. OS DNS resolution and CPU tokenization cannot be forcibly cancelled by a Python socket shutdown, so a worker/slot can outlive the client deadline until that operation returns. This is not a process-level hard execution limit.

Enable token limits using a matching, locally downloaded snapshot (only tokenizer files are needed):

```sh
TOKENIZER_PATH=/absolute/path/to/pinned/tokenizer MAX_INPUT_TOKENS=1792 MAX_ACTIVE=16 docker compose -f compose.gpu.yaml -f compose.32b.yaml -f compose.token-limits.yaml up -d --build
```

The overlay installs the optional pinned tokenizer dependencies and mounts the snapshot read-only. Startup fails if dependencies or local files are missing. No download or remote tokenizer code executes at gateway startup. Token mode accepts plain text role/content messages, requires explicit positive integer `max_tokens`, counts the model's chat template, and rejects unsupported request fields rather than undercounting tools or alternate templates. Over-limit input/context returns 413; unsupported shapes return 400. Keep the snapshot and template identical to the backend. This does not evaluate quality or make the proxy fully OpenAI-compatible.

Authentication, TLS, high availability, a waiting queue, prefix caching and Triton preprocessing remain separate design choices. The service stays bound to loopback externally and should sit behind a secured ingress before any public deployment. The captured published sources preserve the earlier gateway behavior; these local extensions do not retroactively change those measurements.
