"""Verify published evidence and recompute the fixed numerical pilot without a GPU."""
import argparse
import copy
import hashlib
import json
import math
import statistics
from pathlib import Path

from research.batch_probe.workload import digest
from .analyze_mechanism import summarize
from .evaluate_gpu import evaluate_cohort

DEFAULT_STUDY = Path(__file__).resolve().parents[2]/'reports/cache-repair'


def verify_hashes(study):
    study = Path(study).resolve()
    seen = set()
    for line in (study/'SHA256SUMS').read_text().splitlines():
        expected, relative = line.split('  ', 1)
        path = (study/relative).resolve()
        if relative in seen or not path.is_relative_to(study):
            raise ValueError(f'duplicate or escaping capture path: {relative}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'capture hash mismatch: {relative}')
        seen.add(relative)
    actual = {p.relative_to(study).as_posix() for folder in ('evidence', 'specs', 'figures')
              for p in (study/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    if actual != seen or not seen:
        raise ValueError('capture inventory differs from SHA256SUMS')
    return len(seen)


def validate_run(study, label):
    manifest = json.loads((study/'evidence'/label/'manifest.json').read_text())
    frozen = json.loads((study/'specs'/f'{label}.json').read_text())
    spec = frozen['specification']
    if digest(spec) != frozen['sha256'] or manifest['spec_sha256'] != frozen['sha256']:
        raise ValueError(f'{label}: specification hash differs')
    if manifest['specification'] != spec or manifest['status'] != 'complete':
        raise ValueError(f'{label}: incomplete or different specification')
    expected = {q['id']: q['token_sha256'] for q in spec['questions']}
    if len(expected) != len(spec['questions']):
        raise ValueError('duplicated question in specification')
    for q in spec['questions']:
        if digest(q['token_ids']) != q['token_sha256']:
            raise ValueError('token sequence hash differs')
    rows = manifest['results']
    if len(rows) != len(expected) or {r['question_id']: r['token_sha256'] for r in rows} != expected:
        raise ValueError(f'{label}: observed cohort differs')
    source = 'source-cpu' if label == 'cpu' else 'source-gpu'
    for relative, expected_hash in manifest['source_sha256'].items():
        if Path(relative).name != relative:
            raise ValueError('source filename escapes snapshot')
        if hashlib.sha256((study/'evidence'/source/relative).read_bytes()).hexdigest() != expected_hash:
            raise ValueError('executed source hash differs')
    for relative, expected_hash in manifest['official_source_sha256'].items():
        path = (study/'evidence/upstream'/relative.replace('\\', '/')).resolve()
        if not path.is_relative_to((study/'evidence/upstream').resolve()):
            raise ValueError('upstream source path escapes snapshot')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError('upstream source hash differs')
    for r in rows:
        for a in r['arms'].values():
            values = a['kl_nats_per_position']
            if len(values) != spec['probe_tokens'] or not all(math.isfinite(x) for x in values):
                raise ValueError('invalid position-level KL')
            if not math.isclose(sum(values)/len(values), a['mean_kl_nats'], abs_tol=1e-12):
                raise ValueError('stored KL mean differs from position values')
            if len(a['per_layer_kv_errors']) != 28:
                raise ValueError('missing layer diagnostics')
    return manifest


def comparable(result):
    result = copy.deepcopy(result)
    # Capture locations change between machines; measurements and membership do not.
    for device in result.get('device_strata', []):
        device.pop('directory', None)
    return result


def reproduce(study=DEFAULT_STUDY):
    study = Path(study).resolve()
    count = verify_hashes(study)
    manifests = {label: validate_run(study, label) for label in ('cpu', 'smoke', 'gpu0', 'gpu1')}
    cpu_ids = {q['id'] for q in manifests['cpu']['specification']['questions']}
    gpu_ids = {q['id'] for label in ('gpu0', 'gpu1') for q in manifests[label]['specification']['questions']}
    if cpu_ids & gpu_ids:
        raise ValueError('development questions leaked into GPU cohort')
    global_spec = json.loads((study/'specs/gpu.json').read_text())['specification']
    for i in (0, 1):
        if manifests[f'gpu{i}']['specification']['questions'] != global_spec['questions'][i::2]:
            raise ValueError('GPU assignment is not the frozen alternating split')
    result = evaluate_cohort(study/'specs/gpu.json', [study/'evidence/gpu0', study/'evidence/gpu1'])
    published = json.loads((study/'evidence/analysis.json').read_text())
    if comparable(result) != comparable(published):
        raise ValueError('recomputed GPU analysis differs from published result')
    cpu = summarize(study/'evidence/cpu')
    if cpu != json.loads((study/'evidence/cpu-analysis.json').read_text()):
        raise ValueError('recomputed CPU analysis differs from published result')
    rows = [r for label in ('gpu0', 'gpu1') for r in manifests[label]['results']]
    descriptive = {
        'mean_kl_by_arm': {a: statistics.mean(r['arms'][a]['mean_kl_nats'] for r in rows) for a in rows[0]['arms']},
        'sd_primary_benefit_nats': statistics.stdev(-r['paired_primary_delta'] for r in rows),
        'logical_bytes_per_prefix_token': {
            a: statistics.mean(r['logical_packed_bytes'][a]/global_spec['prefix_tokens'] for r in rows)
            for a in rows[0]['logical_packed_bytes']},
        'sum_case_wall_seconds': sum(r['wall_seconds'] for r in rows),
        'sum_smoke_case_wall_seconds': sum(r['wall_seconds'] for r in manifests['smoke']['results']),
        'peak_allocated_bytes_max': max(r['diagnostic_peak_allocated_bytes'] for r in rows),
        'peak_reserved_bytes_max': max(r['diagnostic_peak_reserved_bytes'] for r in rows),
        'launch_utc': (study/'evidence/environment/launch-utc.txt').read_text().strip(),
        'finish_utc': (study/'evidence/environment/finish-utc.txt').read_text().strip(),
        'manifest_sha256': {label: hashlib.sha256((study/'evidence'/label/'manifest.json').read_bytes()).hexdigest()
                            for label in ('gpu0', 'gpu1')},
        'raw_layer0_history_error_max': max(r['raw_damaged_kv_vs_stream_reference'][0][k]
            for r in rows for k in ('k_relative_squared_l2', 'v_relative_squared_l2'))}
    if descriptive != json.loads((study/'evidence/descriptive.json').read_text()):
        raise ValueError('recomputed arm/cost summaries differ from published result')
    return {'verified_files': count, 'gpu': result, 'cpu': cpu}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=DEFAULT_STUDY)
    parser.add_argument('--output', type=Path, help='new local output directory; published data are never rewritten')
    args = parser.parse_args()
    result = reproduce(args.study)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        for label in ('gpu', 'cpu'):
            (args.output/f'{label}-analysis.json').write_text(json.dumps(result[label], indent=2)+'\n')
    print(f"Verified {result['verified_files']} captured files; all 96 GPU questions and both analyses reproduce.")
    print(f"Decision: {result['gpu']['decision']}; mean benefit = {result['gpu']['mean_benefit_nats']:.9f} nats/token")


if __name__ == '__main__':
    main()
