"""Summarize trial variability without pooling individual request percentiles."""
import argparse
import json
import statistics
from pathlib import Path


def analyze(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if not manifest.get('executed') or any(t.get('status') != 'complete' for t in manifest['trials']):
        raise ValueError('suite must have executed and completed every planned trial')
    groups = {}
    for trial in manifest['trials']:
        summary = json.loads((directory / (trial['label'] + '.jsonl.summary.json')).read_text(encoding='utf-8'))
        params = summary['parameters']
        if (params['rate'] != trial['rate'] or params['label'] != trial['label']
                or params['pattern'] != manifest['parameters']['pattern']
                or params['duration'] != manifest['parameters']['duration']
                or summary['workload_sha256'] != manifest['workload_sha256']):
            raise ValueError('trial parameters differ from the manifest')
        n = summary['scheduled_requests']
        row = {'label': trial['label'], 'repeat': trial['repeat'],
               'success_fraction': summary['results']['successes'] / n if n else None,
               'p95_first_content_seconds': summary['schedule_to_first_content_seconds']['p95'],
               'p95_completion_seconds': summary['schedule_to_completion_seconds']['p95'],
               'slo_fraction_of_offered': summary['illustrative_slo']['successful_meeting_both'] / n if n else None,
               'slo_goodput': summary['illustrative_slo']['goodput_per_full_run_second'],
               'output_tokens_per_second': summary['results']['output_tokens_per_second']}
        groups.setdefault(f"cap{trial['cap']}-rate{trial['rate']}", []).append(row)
    report = {}
    for case, rows in groups.items():
        metrics = {}
        for metric in rows[0]:
            if metric in ('label', 'repeat'):
                continue
            values = [r[metric] for r in rows if r[metric] is not None]
            metrics[metric] = {'trials_with_value': len(values), 'median': statistics.median(values) if values else None,
                               'min': min(values) if values else None, 'max': max(values) if values else None}
        report[case] = {'trials': rows, 'across_trial_metrics': metrics}
    return {'method': 'Median and range across trials; request percentiles are not pooled. No confidence interval is inferred.',
            'cases': report}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory')
    args = p.parse_args()
    report = analyze(args.directory)
    out = Path(args.directory) / 'variability.json'
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
