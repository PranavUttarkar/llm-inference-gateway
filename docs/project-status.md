# Project status — November 9, 2026

The gateway is implemented and the GPU experiments are complete. This page records the current status against the local plan; it supersedes the original rental and smoke-test milestones.

| Planned outcome | Evidence / status |
|---|---|
| Streaming service with a pinned inference stack | Triton 25.08 and bundled vLLM 0.9.2, pinned by image digest; fixed model revisions; streaming and non-streaming paths verified. |
| Bounded admission, validation, timeout handling | Gateway implementation, local overload/recovery tests, live 400/413 validation and post-overload 200 response. Timeout is socket inactivity, not an absolute deadline. |
| TP=1 versus TP=2 under the same workload | Completed on Qwen2.5-7B, 1,000 measured requests across five concurrency levels and two repeats. One sharded instance versus one single-GPU instance, not two replicas. |
| A model that requires more than one 24 GiB GPU for resident weights | Qwen2.5-32B with 8-bit GPTQ weights, TP=2, no CPU offload; startup logs show 16.36 GiB of weights per GPU. |
| Fixed-arrival overload evaluation | Three five-minute 32B cases, 3,000 offered requests, matched admission limits 32 versus 16 at 4 req/s. |
| Serving and GPU observability | Published Grafana screenshots, provisioned dashboard, captured Prometheus data and independent GPU samples. Client records provide latency percentiles. |
| Raw data, charts, reproducible analysis | Published under `results/`, with captured source and SHA-256 manifests; verification and plot commands are in the README. |
| Resume claims grounded in measurements | 36.5% lower p95 completion under matched overload; 99.5% of 600 offered requests meeting both illustrative targets at 2 req/s. |
| Actual paid cost and rental closeout | **Unverified.** Confirm termination and provider billing. The 7B offer-based estimate is labeled; no 32B cost figure is claimed. |

## No further GPU run is required for this version

The repository now answers the core serving questions with inspectable evidence. The 7B study supports the tensor-parallelism comparison, while the 32B study demonstrates multi-GPU residency and the latency/rejection tradeoff of admission control. These studies are kept separate because model size, precision, host, and objectives differ.

The sequence-capacity experiment is also retained, but it must not be described as continuous batching enabled versus disabled. The engine's sequence limit changed; continuous batching was not isolated.

## Follow-up work, in priority order

1. Confirm the rented instances are terminated and no unwanted storage is billed. Record actual compute, storage, network charges, and total spend from the provider invoice. This requires account/billing information that is not present in the saved experiment files.
2. Review the repository changes and publish them through the normal Git workflow. This work does not change repository visibility or push changes automatically.
3. If extending the study later, repeat and reorder the admission trials to estimate run-to-run variability. Add bursty arrivals and a longer-prompt workload before making broader capacity claims.
4. Compare two independent TP=1 replicas with one TP=2 instance for the 7B model. The existing TP comparison does not answer that deployment choice.
5. Add tokenizer-aware input limits, an absolute request deadline, and transport/connection limits if expanding the gateway beyond the measured prototype. Authentication and TLS belong at a secured ingress before public exposure.

No larger model, Kubernetes deployment, or extra rental is necessary to support the current results.
