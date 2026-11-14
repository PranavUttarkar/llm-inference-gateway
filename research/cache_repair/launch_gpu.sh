#!/usr/bin/env bash
# Run from the extracted transfer root after checking the environment.
set -euo pipefail
OUT="${1:-artifacts/cache-repair/gpu-run-v3}"
if [ -e "$OUT" ]; then
  echo "Refusing to overwrite $OUT" >&2
  exit 1
fi
mkdir -p "$OUT"
date -u +%FT%TZ > "$OUT/launch-utc.txt"
python -m pip freeze > "$OUT/packages.txt"
python -m research.cache_repair.remote_preflight --output "$OUT/preflight.json"
# Keep download bounded; partial files remain resumable, but do not retry blindly.
timeout --signal=INT --kill-after=30s 600s python -u -m research.cache_repair.fetch_model > "$OUT/download.log" 2>&1
CUDA_VISIBLE_DEVICES=0 timeout --signal=INT --kill-after=30s 600s \
  python -u -m research.cache_repair.mechanistic_pilot \
  --spec artifacts/cache-repair/gpu-smoke-v3.json --output "$OUT/smoke" --device cuda \
  > "$OUT/smoke.log" 2>&1
python - "$OUT" <<'PY'
import json
import sys
from pathlib import Path
m=json.loads((Path(sys.argv[1])/'smoke/manifest.json').read_text())
assert m['status']=='complete' and len(m['results'])==4
for r in m['results']:
    assert r['frozen_scale_self_reencode_max_abs']==0
    assert abs(r['arms']['uncompressed_repeat_control']['mean_kl_nats'])<=1e-6
    assert abs(r['arms']['uncompressed_stream_rebuild_repeat']['mean_kl_nats'])<=1e-6
    assert abs(r['independent_prefill_repeat']['mean_kl_nats'])<=1e-6
    import math
    assert all(math.isfinite(x) for a in r['arms'].values() for x in a['kl_nats_per_position'])
print('CUDA model, quantizer and repeat-control smoke passed',flush=True)
PY
CUDA_VISIBLE_DEVICES=0 timeout --signal=INT --kill-after=30s 4200s \
  python -u -m research.cache_repair.mechanistic_pilot \
  --spec artifacts/cache-repair/gpu-spec-v3-gpu0.json --output "$OUT/gpu0" --device cuda \
  > "$OUT/gpu0.log" 2>&1 &
PID0=$!
CUDA_VISIBLE_DEVICES=1 timeout --signal=INT --kill-after=30s 4200s \
  python -u -m research.cache_repair.mechanistic_pilot \
  --spec artifacts/cache-repair/gpu-spec-v3-gpu1.json --output "$OUT/gpu1" --device cuda \
  > "$OUT/gpu1.log" 2>&1 &
PID1=$!
printf '%s\n%s\n' "$PID0" "$PID1" > "$OUT/worker-pids.txt"
set +e
wait "$PID0"
STATUS0=$?
wait "$PID1"
STATUS1=$?
set -e
printf 'gpu0=%s\ngpu1=%s\n' "$STATUS0" "$STATUS1" > "$OUT/exit-status.txt"
date -u +%FT%TZ > "$OUT/finish-utc.txt"
if [ "$STATUS0" -ne 0 ] || [ "$STATUS1" -ne 0 ]; then
  echo "Worker failed or timed out; preserve evidence and inspect, do not automatically restart." >&2
  exit 1
fi
