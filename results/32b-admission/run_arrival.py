"""Send streaming requests on a fixed schedule and record schedule-relative latency."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from run_load import percentile, request, summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8080/v1/chat/completions')
    p.add_argument('--model', default='qwen')
    p.add_argument('--workload', required=True)
    p.add_argument('--rate', type=float, required=True, help='scheduled requests per second, evenly spaced')
    p.add_argument('--duration', type=float, default=30, help='scheduled arrival window in seconds')
    p.add_argument('--warmup', type=int, default=4)
    p.add_argument('--max-tokens', type=int, default=128)
    p.add_argument('--timeout', type=float, default=120)
    p.add_argument('--workers', type=int, default=128)
    p.add_argument('--label', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--ttft-slo-ms', type=float, default=250)
    p.add_argument('--latency-slo-ms', type=float, default=5000)
    args = p.parse_args()
    if args.rate <= 0 or args.duration <= 0 or args.warmup < 0 or min(args.max_tokens,args.workers,args.timeout,args.ttft_slo_ms,args.latency_slo_ms) <= 0:
        p.error('rate, duration, worker count, timeouts and limits must be positive')
    args.prompts=json.loads(Path(args.workload).read_text(encoding='utf-8-sig'))
    if not isinstance(args.prompts,list) or not args.prompts or any(not isinstance(s,str) or not s for s in args.prompts):
        p.error('workload must be a nonempty array of nonempty strings')
    args.include_usage=True
    args.concurrency=args.workers
    for i in range(args.warmup):
        result=request(-1-i,args)
        if not result['completed']:
            p.exit(1,'warmup failed: '+result.get('error','unknown')+'\n')
    count=math.ceil(args.duration*args.rate)
    workload_hash=hashlib.sha256(json.dumps(args.prompts,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    started_at=datetime.now(timezone.utc).isoformat()
    start=time.monotonic()
    write_lock=threading.Lock()
    def send(i, planned):
        actual=time.monotonic()
        row=request(i,args)
        lag=max(0.0, actual-planned)
        row['scheduled_offset_seconds']=planned-start
        row['dispatch_offset_seconds']=actual-start
        row['dispatch_lag_seconds']=lag
        row['schedule_to_first_content_seconds']=lag+row['ttft_seconds'] if row['ttft_seconds'] is not None else None
        row['schedule_to_completion_seconds']=lag+row['elapsed_seconds']
        row['completed_at_utc']=datetime.now(timezone.utc).isoformat()
        with write_lock:
            raw_output.write(json.dumps(row)+'\n')
            raw_output.flush()
        return row
    with open(args.output,'w',encoding='utf-8') as raw_output, concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[]
        for i in range(count):
            planned=start+i/args.rate
            delay=planned-time.monotonic()
            if delay>0: time.sleep(delay)
            futures.append(pool.submit(send,i,planned))
        rows=[f.result() for f in futures]
        # Keep the full arrival window in the denominator even if the last
        # response finishes before the window ends.
        remaining=start+args.duration-time.monotonic()
        if remaining>0: time.sleep(remaining)
    wall=time.monotonic()-start
    ok=[r for r in rows if r['completed']]
    good=[r for r in ok if r['schedule_to_first_content_seconds']*1000 <= args.ttft_slo_ms and r['schedule_to_completion_seconds']*1000 <= args.latency_slo_ms]
    def qs(key,values):
        return {f'p{n}':percentile([x[key] for x in values if x[key] is not None],n/100) for n in (50,95,99)}
    dispatch_during=sum(r['dispatch_offset_seconds'] < args.duration for r in rows)
    summary={'started_at_utc':started_at,'workload_sha256':workload_hash,'parameters':vars(args),
             'arrival_window_seconds':args.duration,'scheduled_requests':count,
             'dispatches_in_arrival_window':dispatch_during,
             'observed_dispatch_rate_in_window':dispatch_during/args.duration,
             'full_drain_seconds':wall,'statuses':dict(Counter(str(r.get('status')) for r in rows)),
             'dispatch_lag_seconds':qs('dispatch_lag_seconds',rows),
             'schedule_to_first_content_seconds':qs('schedule_to_first_content_seconds',ok),
             'schedule_to_completion_seconds':qs('schedule_to_completion_seconds',ok),
             'illustrative_slo':{'ttft_ms':args.ttft_slo_ms,'completion_ms':args.latency_slo_ms,
                                 'successful_meeting_both':len(good),'goodput_per_full_run_second':len(good)/wall},
             'results':summarize(rows,wall)}
    with open(args.output+'.summary.json','w',encoding='utf-8') as f: json.dump(summary,f,indent=2); f.write('\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
