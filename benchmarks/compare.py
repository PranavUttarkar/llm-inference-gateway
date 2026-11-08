"""Compare two benchmark summaries only when workload settings match."""
import argparse
import json

MATCH_KEYS = ('model', 'concurrency', 'requests', 'warmup', 'max_tokens', 'include_usage', 'timeout')


def compare(a, b):
    missing = [key for key in MATCH_KEYS if key not in a.get('parameters', {}) or key not in b.get('parameters', {})]
    if missing:
        raise ValueError('missing comparison settings: ' + ', '.join(missing))
    mismatches = [key for key in MATCH_KEYS if a['parameters'].get(key) != b['parameters'].get(key)]
    if a.get('workload_sha256') is None or a.get('workload_sha256') != b.get('workload_sha256'):
        mismatches.append('workload_sha256')
    if mismatches:
        raise ValueError('incompatible runs: ' + ', '.join(mismatches))
    def ratio(key):
        x, y = a['results'].get(key), b['results'].get(key)
        return y / x if x is not None and y is not None and x != 0 else None
    return {'labels': [a['parameters'].get('label'), b['parameters'].get('label')],
            'workload_sha256': a['workload_sha256'],
            'failures': [a['results']['failures'], b['results']['failures']],
            'request_throughput_ratio_b_over_a': ratio('successful_requests_per_second'),
            'output_token_throughput_ratio_b_over_a': ratio('output_tokens_per_second'),
            'ttft_seconds': [a['results']['ttft_seconds'], b['results']['ttft_seconds']],
            'elapsed_seconds': [a['results']['elapsed_seconds'], b['results']['elapsed_seconds']]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('a')
    parser.add_argument('b')
    args = parser.parse_args()
    with open(args.a, encoding='utf-8-sig') as f:
        a = json.load(f)
    with open(args.b, encoding='utf-8-sig') as f:
        b = json.load(f)
    try:
        print(json.dumps(compare(a, b), indent=2))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
