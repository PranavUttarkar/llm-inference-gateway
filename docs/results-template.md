# Experiment report template

Complete only with measured values and reviewed artifacts.

- Date, code revision and raw artifact location:
- Provider, GPU type/count, VRAM, interconnect and driver:
- Actual instance rate, billable duration, storage/network charges and total paid:
- Image digest, vLLM version, model/tokenizer revision and gateway configuration:
- Workload hash, input counts, maximum output tokens, sampling, warmup, requests and repetitions:
- Deviations, failures, 429s, missing usage and limitations:

| TP | Concurrency | Repeat | Success / total | TTFT p50 / p95 / p99 | End-to-end p50 / p95 / p99 | Successful requests/s | Output tokens/s | GPU memory/utilization |
|---|---:|---:|---:|---|---|---:|---:|---|
| 1 | | | | | | | | |
| 2 | | | | | | | | |

Describe the measured comparison and uncertainty. State that both runs used the same paid two-GPU host, TP=1 is one instance, and TP=2 is one sharded instance. Link the raw data, logs and metadata after removing sensitive details.
