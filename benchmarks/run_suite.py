"""Plan repeated/reordered admission trials; execution must be explicitly enabled."""
import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def trial_plan(repeats, seed):
    if repeats < 2:
        raise ValueError('at least two repetitions are required')
    rng = random.Random(seed)
    trials = []
    for repeat in range(1, repeats + 1):
        cases = [(32, 4), (16, 2), (16, 4)]
        rng.shuffle(cases)
        for cap, rate in cases:
            trials.append({'repeat': repeat, 'cap': cap, 'rate': rate,
                           'label': f'r{repeat}-cap{cap}-rate{rate}'})
    return trials


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, help='fresh output directory; never results/')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--duration', type=float, default=300)
    p.add_argument('--workload', default='workloads/prompts-512.json')
    p.add_argument('--pattern', choices=['uniform', 'burst', 'poisson'], default='uniform')
    p.add_argument('--burst-size', type=int, default=8)
    p.add_argument('--execute', action='store_true', help='recreate gateway and send load to an already running GPU stack')
    args = p.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0 or args.burst_size < 1:
        p.error('duration and burst size must be positive')
    try:
        trials = trial_plan(args.repeats, args.seed)
    except ValueError as exc:
        p.error(str(exc))
    root = Path(__file__).resolve().parents[1]
    out = Path(args.output).resolve()
    if out == root / 'results' or root / 'results' in out.parents:
        p.error('published results are immutable; choose artifacts/<new-run>')
    out.mkdir(parents=True, exist_ok=False)
    workload = Path(args.workload).resolve()
    prompts = json.loads(workload.read_text(encoding='utf-8-sig'))
    if not isinstance(prompts, list) or not prompts or any(not isinstance(s, str) or not s for s in prompts):
        p.error('workload must be a nonempty array of nonempty strings')
    shutil.copyfile(workload, out / 'workload.json')
    source = out / 'source'
    for relative in ('gateway/server.py', 'benchmarks/run_suite.py', 'benchmarks/run_arrival.py',
                     'benchmarks/run_load.py', 'deploy/start_backend.py', 'compose.gpu.yaml', 'compose.32b.yaml',
                     'Dockerfile', 'requirements-tokenizer.txt'):
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    manifest = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'parameters': vars(args),
                'executed': args.execute, 'trials': trials,
                'workload_sha256': hashlib.sha256(json.dumps(prompts,ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),
                'source_sha256': {str(f.relative_to(out)): hashlib.sha256(f.read_bytes()).hexdigest()
                                  for f in out.rglob('*') if f.is_file()}}
    manifest_path = out / 'manifest.json'
    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    save()
    if not args.execute:
        print(f'Wrote {len(trials)} planned trials to {manifest_path}; no services changed or requests sent.')
        return
    import os
    compose = ['docker', 'compose', '-f', 'compose.gpu.yaml', '-f', 'compose.32b.yaml']
    try:
        for trial in trials:
            label = trial['label']
            env = dict(os.environ, MAX_ACTIVE=str(trial['cap']))
            trial['status'] = 'running'
            save()
            subprocess.run(compose + ['up', '-d', '--no-deps', '--build', 'gateway'], cwd=root, env=env, check=True)
            with (out / (label + '-compose.yaml')).open('w', encoding='utf-8') as f:
                subprocess.run(compose + ['config'], cwd=root, env=env, check=True, stdout=f)
            for attempt in range(30):
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2) as r:
                        if r.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(1)
            else:
                raise RuntimeError('gateway did not become healthy')
            command = [sys.executable, str(root / 'benchmarks/run_arrival.py'), '--workload', str(out / 'workload.json'),
                       '--model', 'qwen', '--warmup', '4', '--rate', str(trial['rate']), '--duration', str(args.duration),
                       '--timeout', '120', '--ttft-slo-ms', '500', '--latency-slo-ms', '10000',
                       '--pattern', args.pattern, '--seed', str(args.seed + trial['repeat']),
                       '--burst-size', str(args.burst_size), '--label', label, '--output', str(out / (label + '.jsonl'))]
            with (out / (label + '.console.txt')).open('w', encoding='utf-8') as f:
                subprocess.run(command, cwd=root, check=True, stdout=f, stderr=subprocess.STDOUT)
            trial['status'] = 'complete'
            save()
    except BaseException as exc:
        trial['status'] = 'failed'
        trial['error'] = str(exc)
        save()
        raise
    print(f'Completed {len(trials)} trials; export evidence and terminate the rental through the provider.')


if __name__ == '__main__':
    main()
