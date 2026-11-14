# Batch instability beyond margin: an unanswered hypothesis

November 13, 2026 · offline evidence-availability audit

## Question

Can numerical changes caused by batch conditions reveal errors or useful branches
beyond ordinary logit margin and confidence? Near-tied tokens are already flagged
by margin, so observing flips alone cannot establish extra predictive information.

The prerequisite audit asks whether the **first** differing token across matched
conditions falls outside the lowest 25% of baseline full-trace top-two logit gaps.
Ties are retained; 10% and 50% are descriptive sensitivity sets. Later divergent
positions do not share a prefix and cannot be treated as matched numerical flips.

## Evidence and result

The existing serving captures do not contain usable matched full-vocabulary
per-token traces. The [saved audit](evidence/subset-audit.json) therefore reports
zero usable rows and `no_usable_real_traces`. This is missing data, not evidence
that instability is redundant or useful. No branch-generation or error-prediction
model result has been obtained.

`research/trace_subset.py` implements this check and rejects fixtures as model
evidence. `research/batch_probe/` contains tested collection/statistics tooling;
synthetic checks of those tools are engineering validation, not pretrained-model
findings. They have not justified a new GPU study.

## What would change the status

A real outside-set event is necessary under the current gate before designing a
branch pilot. It is insufficient for usefulness: the absolute gap may still be
tiny, the rank set uses the completed trace retrospectively, and matched-cost
sampling or forced low-margin branching may be stronger alternatives. Incremental
value needs held-out questions, strong margin/confidence baselines and charged
probe cost. The cache-reconstruction study's execution-geometry control does not
answer this question.

To run the missing-data check without a model:

```sh
python -m research.trace_subset --output artifacts/batch-instability/missing-data.json
```

Real input is JSONL with one source question per line, for example:

```json
{"id":"q1","evidence_kind":"model_trace","model_revision":"pinned SHA","prompt_token_sha256":"hash of exact prompt IDs","baseline":{"token_ids":[12,34,2],"top2_logit_gaps":[0.1,0.01,0.8],"completed":true},"variants":[{"condition":"batch8","model_revision":"pinned SHA","prompt_token_sha256":"hash of exact prompt IDs","token_ids":[12,35,2],"completed":true}]}
```

This example is a schema illustration, not model evidence. Real
data must pin the model revision, prompt token hash, complete token IDs and
full-vocabulary baseline gaps through EOS. Do not infer logits from output text.
