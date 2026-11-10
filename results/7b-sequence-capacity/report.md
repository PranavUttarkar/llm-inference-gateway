# Fixed-arrival sequence-capacity comparison

NVIDIA Triton 25.08 with vLLM 0.9.2, Qwen2.5-7B-Instruct at a09a35458c702b33eeacc393d103063234e8bc28, TP=2 on the same Vast VM with 2x RTX 4090 GPUs and driver 580.178.04. Eight synthetic prompts have 506-513 input tokens and a 128-token output limit. Temperature is zero. A 32-active-request gateway sits in front of both configurations.

The controlled parameter was vLLM max_num_seqs: serial=1, concurrent=32. The model configurations were checked to match in every other field. This changes the engine scheduler concurrency limit and can change other runtime behavior, so it is a serialized-versus-concurrent configuration comparison, not a batching-on/batching-off experiment.

At each offered rate, two independent 30-second windows of uniformly spaced requests were sent. All outstanding requests were allowed to drain; four sequential warmups preceded each window. The SLO is illustrative: schedule-to-first-content <=250 ms and schedule-to-completion <=5 s. The goodput denominator includes the full 30-second window and any drain. Mean p95 values are means of the two per-run p95 values for completed requests only; rejected requests are shown separately. Short windows, 15 requests per low-rate window, warmup, sequential configuration order, synthetic prompts, the VM and PCIe topology limit generalization.

| Mode | Offered req/s | Completed / offered | HTTP 429 | Mean p95 TTFT (s) | SLO-good / offered | Completed req/s including drain |
|---|---:|---:|---:|---:|---:|---:|
| concurrent (limit 32) | 0.5 | 30/30 | 0 | 0.071 | 100.0% | 0.50 |
| concurrent (limit 32) | 2 | 120/120 | 0 | 0.076 | 100.0% | 1.93 |
| concurrent (limit 32) | 6 | 360/360 | 0 | 0.082 | 99.7% | 5.69 |
| serial (limit 1) | 0.5 | 30/30 | 0 | 0.074 | 100.0% | 0.50 |
| serial (limit 1) | 2 | 104/120 | 16 | 44.489 | 1.7% | 0.69 |
| serial (limit 1) | 6 | 104/360 | 256 | 44.839 | 0.6% | 0.69 |

At 6 scheduled requests/s, the concurrent configuration completed 360/360 requests and the serial configuration completed 104/360; the latter returned HTTP 429 for 256 requests. The gateway cap and deliberate one-sequence scheduler limit are essential context. Under the illustrative SLO, 359/360 versus 2/360 scheduled requests met both thresholds. All successful responses are counted, including those that finished during drain.

Maximum per-run client dispatch-lag p99 was under 10 ms. The offered schedule is deterministic and uniform, not a Poisson or production traffic model. Arrival timestamps and failures are recorded in raw JSONL. Prometheus and GPU samples are included. At the low-rate point the original run summaries used actual drain time even when it was shorter than 30 seconds; the analysis intentionally normalizes the observation period to at least the full arrival window.

This does not establish the causal effect of continuous batching in isolation, prove a general production capacity, or compare against two independent single-GPU replicas. The two-GPU VM was paid for in both configurations. Review logs for sensitive information and confirm actual costs before publication.

Regenerate analysis and charts: python benchmarks/analyze_arrival.py results/7b-sequence-capacity (requires matplotlib).


## Published evidence

This directory contains the captured measurement inputs selected for publication. See [the evidence index](../README.md) for file definitions and provenance. SHA256SUMS covers captured inputs; reports, generated analysis, and figures are excluded. Historical file paths in summaries and commands refer to the measurement VM.
