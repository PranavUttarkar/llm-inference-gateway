"""Summarize matched TP benchmark repeats and regenerate charts.

Requires matplotlib for charts. Costs are steady-state compute estimates only.
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--hourly-rate', type=float, required=True, help='whole-instance USD/hour, excluding extras')
    p.add_argument('--ttft-slo-ms', type=float, help='optional illustrative per-request TTFT objective')
    p.add_argument('--latency-slo-ms', type=float, help='optional illustrative per-request completion objective')
    args = p.parse_args()
    if args.hourly_rate <= 0 or ((args.ttft_slo_ms is None) != (args.latency_slo_ms is None)):
        p.error('hourly rate must be positive; give both SLO thresholds or neither')
    groups = defaultdict(list)
    for path in sorted(args.directory.glob('tp[12]-c*-r*.jsonl.summary.json')):
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        label = data['parameters']['label']
        tp = int(label[2])
        c = data['parameters']['concurrency']
        groups[tp, c].append((path, data))
    if not groups:
        p.error('no labeled TP summaries found')
    match_keys = ('model', 'concurrency', 'requests', 'warmup', 'max_tokens', 'include_usage', 'timeout')
    for concurrency in sorted({c for _, c in groups}):
        left = {v['parameters']['label'].rsplit('-r', 1)[-1]: v for _, v in groups.get((1, concurrency), [])}
        right = {v['parameters']['label'].rsplit('-r', 1)[-1]: v for _, v in groups.get((2, concurrency), [])}
        if not left or left.keys() != right.keys():
            p.error(f'unmatched TP repetitions at concurrency {concurrency}')
        for repeat in left:
            a, b = left[repeat], right[repeat]
            if not a.get('workload_sha256') or a.get('workload_sha256') != b.get('workload_sha256'):
                p.error(f'workload mismatch at concurrency {concurrency}, repeat {repeat}')
            if any(k not in a['parameters'] or k not in b['parameters'] or a['parameters'][k] != b['parameters'][k] for k in match_keys):
                p.error(f'parameter mismatch at concurrency {concurrency}, repeat {repeat}')
    rows=[]
    for (tp,c), runs in sorted(groups.items()):
        throughputs = [v['results']['output_tokens_per_second'] for _,v in runs]
        rates = [v['results']['successful_requests_per_second'] for _,v in runs]
        ttfts = [v['results']['ttft_seconds']['p95'] for _,v in runs]
        elapsed = [v['results']['elapsed_seconds']['p95'] for _,v in runs]
        def mean_or_none(xs):
            return statistics.mean(xs) if xs and all(x is not None for x in xs) else None
        avg = mean_or_none(throughputs)
        row={'tp':tp,'concurrency':c,'repeats':len(runs),
             'requests':sum(v['results']['requests'] for _,v in runs),
             'successes':sum(v['results']['successes'] for _,v in runs),
             'failures':sum(v['results']['failures'] for _,v in runs),
             'output_tokens_per_second_mean':avg,'request_per_second_mean':mean_or_none(rates),
             'ttft_p95_seconds_mean':mean_or_none(ttfts),'elapsed_p95_seconds_mean':mean_or_none(elapsed),
             'estimated_compute_usd_per_million_output_tokens':args.hourly_rate*1_000_000/(3600*avg) if avg else None}
        if args.ttft_slo_ms is not None:
            goods=0; walls=0
            for path, data in runs:
                raw=Path(str(path).removesuffix('.summary.json'))
                for line in raw.read_text(encoding='utf-8').splitlines():
                    r=json.loads(line)
                    if r['completed'] and r['ttft_seconds']*1000 <= args.ttft_slo_ms and r['elapsed_seconds']*1000 <= args.latency_slo_ms:
                        goods+=1
                walls+=data['results']['wall_seconds']
            row['slo_goodput_per_second']=goods/walls if walls else None
        rows.append(row)
    out={'hourly_rate_whole_instance_usd':args.hourly_rate,
         'illustrative_slo': {'ttft_ms': args.ttft_slo_ms, 'latency_ms': args.latency_slo_ms} if args.ttft_slo_ms is not None else None,
         'note':'Means of per-run summaries; not pooled percentiles. Estimated cost excludes startup, idle, storage, bandwidth and taxes.', 'rows':rows}
    (args.directory/'analysis.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for key,title,ylabel,filename in [('output_tokens_per_second_mean','Output throughput','Output tokens / second','throughput.png'),('ttft_p95_seconds_mean','Time to first content: p95','Seconds','ttft-p95.png')]:
        fig,ax=plt.subplots(figsize=(7,4.5))
        for tp in (1,2):
            selected=[r for r in rows if r['tp']==tp and r[key] is not None]
            if selected: ax.plot([r['concurrency'] for r in selected],[r[key] for r in selected],marker='o',label=f'TP={tp}')
        ax.set_xscale('log',base=2); ax.set_xticks(sorted({r['concurrency'] for r in rows}),labels=[str(x) for x in sorted({r['concurrency'] for r in rows})])
        ax.set_xlabel('Concurrency'); ax.set_ylabel(ylabel); ax.set_title(title+' (mean of run summaries)'); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(args.directory/filename,dpi=160); plt.close(fig)
    print(json.dumps(out,indent=2))


if __name__ == '__main__':
    main()
