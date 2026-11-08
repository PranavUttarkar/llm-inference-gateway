"""Measure streaming chat requests; token totals require server-provided usage."""
import argparse
import concurrent.futures
import json
import math
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    pos = (len(values) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def summarize(rows, elapsed):
    ok = [r for r in rows if r['completed']]
    counts = [r.get('usage', {}).get('completion_tokens') for r in ok]
    known = [n for n in counts if isinstance(n, int) and not isinstance(n, bool) and n >= 0]
    def latency(key):
        values = [r[key] for r in ok if r.get(key) is not None]
        return {f'p{p}': percentile(values, p / 100) for p in (50, 90, 95, 99)}
    return {'requests': len(rows), 'successes': len(ok), 'failures': len(rows) - len(ok),
            'wall_seconds': elapsed, 'successful_requests_per_second': len(ok) / elapsed if elapsed else None,
            'ttft_seconds': latency('ttft_seconds'), 'elapsed_seconds': latency('elapsed_seconds'),
            'successes_with_completion_tokens': len(known), 'completion_tokens_known': sum(known),
            'output_tokens_per_second': sum(known) / elapsed if elapsed and len(known) == len(ok) and ok else None}


def request(index, args):
    start = time.monotonic()
    first, content, done, usage = None, [], False, None
    row = {'id': index, 'concurrency': args.concurrency, 'model': args.model}
    payload = {'model': args.model, 'messages': [{'role': 'user', 'content': args.prompt}],
               'stream': True, 'max_tokens': args.max_tokens, 'temperature': 0}
    if args.include_usage:
        payload['stream_options'] = {'include_usage': True}
    req = urllib.request.Request(args.url, json.dumps(payload).encode(), {'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as response:
            row['status'] = response.status
            event_lines = []
            for raw in response:
                line = raw.decode('utf-8').rstrip('\r\n')
                if line:
                    if line.startswith('data:'):
                        event_lines.append(line[5:].lstrip(' '))
                    continue
                if not event_lines:
                    continue
                data = '\n'.join(event_lines)
                event_lines = []
                if data == '[DONE]':
                    done = True
                    break
                event = json.loads(data)
                if event.get('error'):
                    raise ValueError('stream error: ' + str(event['error']))
                choices = event.get('choices') or []
                text = (choices[0].get('delta') or {}).get('content') if choices else None
                if text:
                    if first is None:
                        first = time.monotonic() - start
                    content.append(text)
                if event.get('usage') is not None:
                    usage = event['usage']
        row['completed'] = done and first is not None
        if not row['completed']:
            row['error'] = 'stream ended without content or [DONE]'
    except (OSError, ValueError, UnicodeError, urllib.error.URLError) as exc:
        row.update(completed=False, status=getattr(exc, 'code', row.get('status')), error=str(exc))
    row.update(ttft_seconds=first, elapsed_seconds=time.monotonic()-start, output_text=''.join(content))
    if isinstance(usage, dict):
        row['usage'] = usage
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8080/v1/chat/completions')
    p.add_argument('--model', default='mock')
    p.add_argument('--concurrency', type=int, default=4)
    p.add_argument('--requests', type=int, default=20)
    p.add_argument('--warmup', type=int, default=0)
    p.add_argument('--max-tokens', type=int, default=128)
    p.add_argument('--timeout', type=float, default=60, help='socket inactivity timeout in seconds')
    p.add_argument('--prompt', default='Explain tensor parallelism briefly.')
    p.add_argument('--include-usage', action='store_true', help='request streaming usage, if supported by backend')
    p.add_argument('--output', default='results.jsonl')
    args = p.parse_args()
    if min(args.concurrency, args.requests, args.max_tokens) < 1 or args.warmup < 0 or args.timeout <= 0:
        p.error('requests, concurrency, max-tokens and timeout must be positive; warmup must be nonnegative')
    for i in range(args.warmup):
        result = request(-1-i, args)
        if not result['completed']:
            p.exit(1, 'warmup failed: ' + result.get('error', 'unknown error') + '\n')
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(lambda i: request(i, args), range(args.requests)))
    elapsed = time.monotonic() - start
    with open(args.output, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    summary = {'started_at_utc': started_at, 'parameters': vars(args), 'results': summarize(rows, elapsed)}
    with open(args.output + '.summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
        f.write('\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
