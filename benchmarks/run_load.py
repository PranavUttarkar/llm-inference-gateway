"""Run streaming requests and write per-request timings as JSONL."""
import argparse
import concurrent.futures
import json
import time
import urllib.error
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--url', default='http://127.0.0.1:8080/v1/chat/completions')
p.add_argument('--model', default='mock')
p.add_argument('--concurrency', type=int, default=4)
p.add_argument('--requests', type=int, default=20)
p.add_argument('--output', default='results.jsonl')
args = p.parse_args()
if args.concurrency < 1 or args.requests < 1:
    p.error('requests and concurrency must be positive')

def request(index):
    start = time.monotonic()
    first = None
    content = []
    done = False
    row = {'id': index, 'concurrency': args.concurrency, 'model': args.model}
    body = json.dumps({'model': args.model, 'messages': [{'role': 'user', 'content': 'Explain tensor parallelism briefly.'}], 'stream': True, 'max_tokens': 128, 'temperature': 0}).encode()
    req = urllib.request.Request(args.url, body, {'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            row['status'] = response.status
            for raw in response:
                line = raw.decode('utf-8').strip()
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    done = True
                    break
                event = json.loads(data)
                choices = event.get('choices', [])
                text = choices[0].get('delta', {}).get('content') if choices else None
                if text:
                    if first is None:
                        first = time.monotonic() - start
                    content.append(text)
                if event.get('usage'):
                    row['usage'] = event['usage']
        row['completed'] = done and first is not None
        if not row['completed']:
            row['error'] = 'stream ended without content or [DONE]'
    except (OSError, ValueError, urllib.error.URLError) as exc:
        row.update(completed=False, status=getattr(exc, 'code', None), error=str(exc))
    row.update(ttft_seconds=first, elapsed_seconds=time.monotonic()-start, output_text=''.join(content))
    return row

start = time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
    results = list(pool.map(request, range(args.requests)))
elapsed = time.monotonic() - start
with open(args.output, 'w', encoding='utf-8') as f:
    for row in results:
        f.write(json.dumps(row) + '\n')
print(json.dumps({'requests': len(results), 'successes': sum(r['completed'] for r in results), 'elapsed_seconds': elapsed, 'output': args.output}))
