# Running and publishing inference experiments

Keep maintained code in `research/`, published studies in `reports/<study>/`, and
local weights, drafts and outputs in ignored `artifacts/`. `RESEARCH.md` /
`research.md` are local journals. The completed [cache pilot](../reports/cache-repair/report.md)
is the first example of this layout.

## Verify or reanalyze a published study

From the repository root, Python 3.10+:

```sh
python -m pip install -r requirements-research.txt
python -m research.cache_repair.reproduce
```

Expected: 41 captures verified, 96 GPU questions and both analyses reproduced;
decision `no_material_numerical_target_stop`. This checks saved measurements,
not whether another GPU reproduces model inference. No weights or GPU are needed.

Choose a new directory to save an analysis copy:

```sh
python -m research.cache_repair.reproduce --output artifacts/reanalysis-001
python -m research.cache_repair.plot_gpu \
  --directories reports/cache-repair/evidence/gpu0 reports/cache-repair/evidence/gpu1 \
  --analysis reports/cache-repair/evidence/analysis.json \
  --output artifacts/reanalysis-001/overview.png
```

Published evidence is immutable. Corrections get a new version and an explanation;
do not rewrite old captures to pass a check. Executed source lives beside the
measurements. Maintained source can change with tests and commits identifying it.

## Replay the model experiment, only when needed

This is optional compute work. The pilot's stop decision does not authorize a
larger study. Reusing old questions for backend replication is not independent
confirmation on new questions.

Use a fresh Linux checkout and isolated environment. The original run used two
24 GB RTX 4090s, driver 580.95.05, Python 3.10, PyTorch 2.7.1+cu128 and
Transformers 4.49.0. Its [full package list](../reports/cache-repair/evidence/environment/packages.txt)
is recorded. A working kernel check is required; a driver's CUDA version alone
does not establish that the Python build works.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r research/cache_repair/requirements-gpu.txt
python -m research.cache_repair.stage
```

Staging copies pinned specifications, metadata and licensed quantizer references
into `artifacts/cache-repair/`. It starts no download or inference and refuses an
existing destination. Use a fresh checkout/destination instead of overwriting
evidence. The exact token arrays are frozen, so replay needs no dataset download.

Once resources and budget are intentional:

```sh
bash research/cache_repair/launch_gpu.sh artifacts/cache-repair/replay-001
```

The launcher records packages/hardware, validates BF16 kernels on both devices,
downloads and hashes the checkpoint, runs the four-case smoke, and only then
runs the two fixed 48-question shards. Download and smoke each have a 10-minute
cap; workers each have a 70-minute cap. Existing output directories are refused
and failures are not retried automatically. The bounds leave setup/export room
in a two-hour allocation; they do not guarantee a new host will finish. Record
deviations instead of silently changing precision, attention backend or inputs.

`python -m research.cache_repair.transfer_bundle --help` describes packaging
staged inputs and maintained code when a repository checkout is unsuitable on
the host. The bundle excludes journals and weights. Keep SSH identities outside
the repository.

Evaluate a complete replay with:

```sh
python -m research.cache_repair.evaluate_gpu \
  --spec artifacts/cache-repair/gpu-spec-v3.json \
  --directories artifacts/cache-repair/replay-001/gpu0 artifacts/cache-repair/replay-001/gpu1 \
  --output artifacts/cache-repair/replay-001-analysis.json
```

An incomplete cohort gets an attrition/failure report, not the full-cohort gate.
Export raw manifests, source snapshots, configuration and environment logs;
verify local copies before releasing resources. Finished Python jobs do not stop
rental billing. Confirm instance termination through the provider separately.

## Design a new study

Use the runners as components, not a framework that silently searches for a
positive result. Before model scoring:

1. State a falsifiable question and check the closest primary prior work. Say what
   evidence would support or challenge it.
2. Fix one primary contrast, endpoint and independent sampling unit. Justify the
   context/concurrency regime and sample plan. Keep development and evaluation
   questions separate; acknowledge uncertainty in pilot variance.
3. Specify matched controls, strong competitors and resource accounting. Include
   repeat/path controls when numerics could explain the effect. State proxy limits:
   KL is not answer accuracy.
4. Serialize exact revisions, token/data selection, seeds, arms, budgets,
   thresholds and failure handling. Hash and commit the specification before
   scoring. Retain old versions and log amendments before dependent runs. A
   post-development design is exploratory, not confirmatory preregistration.
5. Validate cheaply, then run a bounded smoke and collect the frozen cohort once.
   Preserve failures. Extra seeds and tokens are not extra source questions.
6. Report the primary even if it disappoints, with uncertainty, all planned arms,
   heterogeneity, exclusions, costs and implementation limits. Passing a pilot
   gate authorizes the next stated step, not a broad claim.

Working notes remain local until they become a necessary protocol, completed
report or stable usage instruction. Publish:

```text
reports/<study>/
  report.md        # question, method, outcome, uncertainty, limits, decision
  specs/           # frozen machine-readable inputs
  evidence/        # raw measurements, environment, executed source, provenance
  figures/         # plots from those measurements
  SHA256SUMS       # byte inventory; excludes the editable narrative
```

Give the README a short intuitive outcome and report link. Release the evidence
needed to audit it, honoring upstream asset terms. Exclude model caches,
credentials, abandoned drafts and planning chatter. Practical benefit eventually
requires task accuracy, equal-budget competitors, real costs and independent
confirmation in the target regime.

## Validate before committing

```sh
python -m unittest discover -s tests
python -m research.cache_repair.reproduce
python scripts/verify_results.py
```

Statistical tests use `requirements-research.txt`. The small random-transformer
engineering test additionally needs PyTorch and Transformers, with no pretrained
checkpoint. CI runs that test in a separate CPU inference environment. Fixtures
and test counts validate tooling, not findings. Use focused commits for code,
frozen inputs, evidence and reports; keep unfinished notes out of publication.
