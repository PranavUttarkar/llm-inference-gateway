# Inference experiments

A lab for testing ideas about LLM inference. Each published study keeps the
question, method, measured outcome and reproducible evidence together. Negative
and inconclusive outcomes stay in the record.

## Can rebuilding a compressed attention cache undo its damage?

Models keep an attention memory, called the KV cache, while processing text.
Compressing it saves storage but can change later computations. We tested whether
replaying the same text could repair that memory while keeping the original
compression parameters fixed.

Four early CPU examples looked encouraging. On **96 new GPU examples**, rebuilding
did **not demonstrate an average reduction in numerical distortion**. It helped
60 examples and hurt 36; larger regressions outweighed the more frequent gains.
The uncertainty interval permits both benefit and harm. The prespecified stop
rule blocked expansion.

This tested short reference-text prefixes, not long generated answers. It measured
next-token distributions, not answer accuracy, physical memory savings or speed.

**[Read the full technical report ?](reports/cache-repair/report.md)**
? [Frozen inputs](reports/cache-repair/specs/)
? [Raw evidence](reports/cache-repair/evidence/)
? [Figure](reports/cache-repair/figures/overview.png)

## Study record

| Study | Outcome | Report |
|---|---|---|
| Cache reconstruction | Mean benefit not demonstrated; stop expansion | [Methods, findings and limitations](reports/cache-repair/report.md) |
| Batch instability beyond logit margin | Missing matched traces; unanswered | [Evidence audit](reports/batch-instability/report.md) |
| Serving admission under overload | Tighter admission reduced p95 latency with more rejections | [32B experiment](results/32b-admission/report.md) |
| One GPU versus two | Throughput and first-token latency traded off | [7B tensor-parallel study](results/7b-tensor-parallel/report.md) |
| Engine sequence capacity | Measured fixed-arrival serving behavior | [7B capacity study](results/7b-sequence-capacity/report.md) |

## Verify the results without a GPU

```sh
python -m pip install -r requirements-research.txt
python -m research.cache_repair.reproduce
python scripts/verify_results.py
```

The first check verifies captured files, cohort membership, source and token
hashes, and recomputes the cache pilot's statistics. The second checks the earlier
serving studies. Neither downloads model weights or runs model inference.

## Run the next experiment

Follow the **[experiment workflow](docs/research-workflow.md)** for frozen inputs,
controls, bounded model runs, evidence export and publication. It also covers
replaying the existing pilot. A new idea starts as a hypothesis, not a promised
finding; a new result needs independent evidence.

| Directory | Purpose |
|---|---|
| `research/` | Inference experiments and statistical analysis |
| `reports/` | Research reports, frozen specifications and captured evidence |
| `tests/` | Engineering and measurement-validity checks |
| `gateway/`, `benchmarks/`, `deploy/` | Serving platform and load experiments |
| `results/` | Earlier serving measurements and reports |
| `artifacts/` | Ignored local outputs, weights and working notes |

The original gateway architecture, dashboard and deployment instructions remain
in the **[serving guide](docs/serving.md)**. GPU access and paid resources are
optional; no experiment launches automatically.
