"""Analyze the long 32B admission experiments, including rejected requests."""
import argparse
from collections import Counter
import json
import re
from pathlib import Path

try:
    from .run_load import percentile
except ImportError:
    from run_load import percentile


CASES = (
    ('reference-cap32-rate4-300s', 32, 4),
    ('protected-cap16-rate2-300s', 16, 2),
    ('protected-cap16-rate4-300s', 16, 4),
)


def analyze(directory):
    results = []
    workload = None
    for label, cap, rate in CASES:
        configuration = (directory / (label + '-compose.yaml')).read_text(encoding='utf-8-sig')
        cap_match = re.search(r'''^\s+MAX_ACTIVE:\s*["']?(\d+)''', configuration, re.MULTILINE)
        if not cap_match or int(cap_match.group(1)) != cap:
            raise ValueError(f'{label}: admission limit does not match resolved configuration')
        summary = json.loads((directory / (label + '.jsonl.summary.json')).read_text(encoding='utf-8-sig'))
        params = summary['parameters']
        required = {'rate': rate, 'duration': 300, 'max_tokens': 128,
                    'ttft_slo_ms': 500, 'latency_slo_ms': 10000}
        if any(params.get(k) != v for k, v in required.items()):
            raise ValueError(f'{label}: unexpected benchmark parameters')
        if workload is None:
            workload = summary['workload_sha256']
        if not workload or summary['workload_sha256'] != workload:
            raise ValueError('workload hash mismatch')
        rows = [json.loads(s) for s in (directory / (label + '.jsonl')).read_text(encoding='utf-8').splitlines()]
        if len(rows) != int(rate * 300) or len({r['id'] for r in rows}) != len(rows):
            raise ValueError(f'{label}: missing or duplicate request records')
        successful = [r for r in rows if r['completed']]
        good = [r for r in successful if r['schedule_to_first_content_seconds'] <= .5
                and r['schedule_to_completion_seconds'] <= 10]
        statuses = Counter(str(r.get('status')) for r in rows)
        if dict(statuses) != summary['statuses']:
            raise ValueError(f'{label}: status summary differs from raw rows')
        def quantiles(key, data):
            values = [r[key] for r in data if r.get(key) is not None]
            return {f'p{p}': percentile(values, p / 100) for p in (50, 95, 99)}
        wall = max(300, summary['full_drain_seconds'])
        token_counts = [r.get('usage', {}).get('completion_tokens') for r in successful]
        complete_usage = all(isinstance(n, int) and n >= 0 for n in token_counts)
        minutes = []
        for minute in range(5):
            bucket = [r for r in rows if minute * 60 <= r['scheduled_offset_seconds'] < (minute + 1) * 60]
            accepted = [r for r in bucket if r['completed']]
            within = [r for r in accepted if r['schedule_to_first_content_seconds'] <= .5
                      and r['schedule_to_completion_seconds'] <= 10]
            minutes.append({'arrival_minute': minute + 1, 'offered': len(bucket),
                            'successes': len(accepted), 'within_targets': len(within),
                            'completion_seconds': quantiles('schedule_to_completion_seconds', accepted)})
        results.append({'label': label, 'gateway_cap': cap, 'offered_rate': rate,
                        'started_at_utc': summary['started_at_utc'], 'arrival_seconds': 300,
                        'wall_seconds_including_drain': wall, 'offered': len(rows),
                        'successes': len(successful), 'statuses': dict(statuses),
                        'within_targets': len(good),
                        'fraction_of_offered_within_targets': len(good) / len(rows),
                        'fraction_of_successes_within_targets': len(good) / len(successful) if successful else None,
                        'goodput_per_full_run_second': len(good) / wall,
                        'successful_output_tokens_per_second': sum(token_counts) / wall if complete_usage and token_counts else None,
                        'completion_tokens': sum(token_counts) if complete_usage else None,
                        'schedule_to_first_content_seconds': quantiles('schedule_to_first_content_seconds', successful),
                        'schedule_to_completion_seconds': quantiles('schedule_to_completion_seconds', successful),
                        'dispatch_lag_seconds': quantiles('dispatch_lag_seconds', rows),
                        'max_dispatch_lag_seconds': max(r['dispatch_lag_seconds'] for r in rows),
                        'observed_dispatch_rate_in_window': summary['observed_dispatch_rate_in_window'],
                        'minutes': minutes})
    reference, _, protected = results
    old = reference['schedule_to_completion_seconds']['p95']
    new = protected['schedule_to_completion_seconds']['p95']
    comparison = {'completion_p95_reduction_fraction': (old - new) / old if old else None,
                  'goodput_ratio_protected_over_reference': protected['goodput_per_full_run_second'] / reference['goodput_per_full_run_second'] if reference['goodput_per_full_run_second'] else None}
    return {'workload_sha256': workload, 'targets': {'ttft_seconds': .5, 'completion_seconds': 10, 'kind': 'illustrative'},
            'note': 'Uniform scheduled arrivals. Percentiles cover successful requests, including queue time from scheduled dispatch. Rejections stay in offered-request and goodput denominators. One five-minute run per case; no production SLA or quality claim.',
            'cases': results, 'matched_overload_comparison': comparison}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    result = analyze(args.directory)
    (args.directory / 'analysis-32b.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['Limit 32\n4 req/s', 'Limit 16\n2 req/s', 'Limit 16\n4 req/s']
    for filename, title, ylabel, values in (
        ('completion-p95-32b.png', '32B admitted-request completion latency', 'p95 seconds',
         [x['schedule_to_completion_seconds']['p95'] for x in result['cases']]),
        ('goodput-32b.png', '32B requests meeting both latency targets', 'Good requests / second, including drain',
         [x['goodput_per_full_run_second'] for x in result['cases']]),
    ):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(labels, values, color=['#777777', '#3276b5', '#319267'])
        if filename.startswith('completion'):
            ax.axhline(10, linestyle='--', color='#b54a32', label='Illustrative 10 s target')
            ax.legend()
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(args.directory / filename, dpi=160)
        plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
