# Published results

The serving system is evaluated in three separate studies. Models, objectives, and measurement methods differ, so results should not be combined into a single speedup claim.

| Study | Question | Measured workload | Report |
|---|---|---|---|
| 32B admission control | How does lowering admitted concurrency affect latency and rejection under fixed arrivals? | Three 300 s windows; 3,000 offered requests; 32B GPTQ 8-bit, TP=2 | [Results](32b-admission/report.md) |
| 7B tensor parallelism | How does one TP=1 instance compare with one TP=2 instance? | Concurrency 1/4/8/16/32, two repeats of 50 requests per point; 1,000 offered requests | [Results](7b-tensor-parallel/report.md) |
| 7B sequence capacity | How do engine sequence limits 1 and 32 behave under the same arrivals? | Rates 0.5/2/6 req/s; two 30 s windows per point; 1,020 offered requests | [Results](7b-sequence-capacity/report.md) |

## Reading the evidence

- `*.jsonl`: individual requests, including failures and rejected requests. These contain synthetic workload responses, status, timings, and reported token usage.
- `*.jsonl.summary.json`: original client parameters, workload hash, aggregate counts, and run timing. The historical `parameters.output` path describes where the VM wrote the file; use the published sibling JSONL for analysis.
- Resolved Compose files, model configuration, host topology, and startup logs: the settings and environment used for the measurement. Captured absolute `/root/...` paths are original VM paths, not paths readers should create locally.
- Captured Python files: the source executed on the host. The current repository adds documentation and analysis; captured source remains available to audit the measurement.
- `gpu-samples.csv` and `prometheus-*.json`: independent GPU samples and exported monitoring data. GPU utilization and dashboard rolling rates do not replace raw client latency measurements.
- `SHA256SUMS`: byte-level hashes for the published captured inputs. Derived reports, analysis JSON, and plots are excluded so regenerating figures does not invalidate the capture manifests.

The publication copy preserves captured file bytes. Setup logs and redundant console dumps are omitted; provider invoices were not captured. Originals remain in the local ignored artifact archive. No provider credentials or model weights are included.

```sh
python -m pip install -r requirements-analysis.txt
python scripts/verify_results.py
python benchmarks/analyze_32b.py results/32b-admission
python benchmarks/report.py results/7b-tensor-parallel --hourly-rate 0.982 --ttft-slo-ms 250 --latency-slo-ms 5000
python benchmarks/analyze_arrival.py results/7b-sequence-capacity
```

## Definitions

**TTFT / first-content latency** means time to the first nonempty streamed content event, which can contain multiple tokens. Fixed-arrival results start the clock at scheduled dispatch; closed-loop results start it at the request.

**Completion latency percentiles** cover successful requests. Rejections are reported separately and retained in the offered-request denominator.

**Requests meeting both targets per second** counts only successful requests that meet both first-content and completion targets, divided by the full observation interval including outstanding-request drain. Code and historical plots sometimes call this SLO goodput. Thresholds are illustrative, not a production SLA.

**Output tokens/s** counts successful completion tokens over the full measured interval. It is distinct from latency-qualified request rate.

**Cost estimates** use a whole-instance hourly offer and observed throughput, excluding startup, idle time, storage, networking, and taxes. The 7B rate is an unverified historical offer; the 32B actual rate and paid total are not available in the captured data.
