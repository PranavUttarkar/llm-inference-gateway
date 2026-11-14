"""Evaluate a complete frozen cohort; never promote a convenient partial sample."""
import argparse
import json
from pathlib import Path

from research.batch_probe.workload import digest
from .numerical_gate import evaluate
from .effect_reporting import describe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--directories', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    frozen = json.loads(args.spec.read_text())
    spec = frozen['specification']
    if digest(spec) != frozen['sha256']:
        raise ValueError('global specification hash mismatch')
    expected = {q['id']: q['token_sha256'] for q in spec['questions']}
    rows, devices, sources = [], [], []
    for directory in args.directories:
        m = json.loads((directory/'manifest.json').read_text())
        shard = m['specification']
        if m['status'] != 'complete' or digest(shard) != m['spec_sha256']:
            raise ValueError('incomplete or invalid shard; report attrition without full-cohort gate')
        for key, value in spec.items():
            if key != 'questions' and shard.get(key) != value:
                raise ValueError(f'shard protocol differs: {key}')
        shard_expected = {q['id']: q['token_sha256'] for q in shard['questions']}
        observed = {r['question_id']: r['token_sha256'] for r in m['results']}
        if observed != shard_expected or len(m['results']) != len(shard_expected):
            raise ValueError('shard roster differs')
        rows.extend(m['results'])
        devices.append({'directory': str(directory), 'gpu': m.get('gpu'),
                        'n': len(m['results']),
                        'mean_benefit_nats': -sum(r['paired_primary_delta'] for r in m['results'])/len(m['results'])})
        sources.append(m['source_sha256'])
    if any(s != sources[0] for s in sources):
        raise ValueError('workers used different source')
    observed = {r['question_id']: r['token_sha256'] for r in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError('global cohort incomplete or duplicated')
    result = evaluate(rows, spec['numerical_gate'])
    result['robustness_and_secondary'] = describe(rows)
    result.update(global_spec_sha256=frozen['sha256'], device_strata=devices,
                  individual_effects=[{'id': r['question_id'], 'benefit_nats': -r['paired_primary_delta']} for r in rows])
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
