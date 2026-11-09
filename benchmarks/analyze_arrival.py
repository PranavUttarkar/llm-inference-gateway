"""Analyze fixed-arrival runs from matching serialized and concurrent engines."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    args=p.parse_args()
    configs={mode:json.loads((args.directory/f'model-{mode}.json').read_text()) for mode in ('serial','concurrent')}
    if configs['serial'].get('max_num_seqs') != 1 or configs['concurrent'].get('max_num_seqs') != 32:
        p.error('unexpected engine sequence limits')
    if {k:v for k,v in configs['serial'].items() if k!='max_num_seqs'} != {k:v for k,v in configs['concurrent'].items() if k!='max_num_seqs'}:
        p.error('model configurations differ beyond the sequence limit')
    groups=defaultdict(list)
    for f in args.directory.glob('*.jsonl.summary.json'):
        if not f.name.startswith(('serial-rate','concurrent-rate')): continue
        data=json.loads(f.read_text(encoding='utf-8-sig'))
        mode=f.name.split('-rate',1)[0]
        groups[mode,data['parameters']['rate']].append(data)
    if not groups: p.error('no arrival-rate summaries found')
    match_keys=('model','rate','duration','warmup','max_tokens','timeout','workers','ttft_slo_ms','latency_slo_ms')
    for rate in sorted({r for _,r in groups}):
        left={x['parameters']['label'].rsplit('-r',1)[-1]:x for x in groups['serial',rate]}
        right={x['parameters']['label'].rsplit('-r',1)[-1]:x for x in groups['concurrent',rate]}
        if not left or left.keys()!=right.keys(): p.error(f'unmatched repetitions at {rate} requests/s')
        for rep,a in left.items():
            b=right[rep]
            if a.get('workload_sha256')!=b.get('workload_sha256') or not a.get('workload_sha256'):
                p.error(f'workload mismatch at rate {rate}, repeat {rep}')
            if any(k not in a['parameters'] or k not in b['parameters'] or a['parameters'][k]!=b['parameters'][k] for k in match_keys):
                p.error(f'configuration mismatch at rate {rate}, repeat {rep}')
    thresholds = {(x['parameters']['ttft_slo_ms'], x['parameters']['latency_slo_ms']) for runs in groups.values() for x in runs}
    if len(thresholds) != 1: p.error('SLO thresholds differ across runs')
    ttft_slo, latency_slo = next(iter(thresholds))
    rows=[]
    for (mode,rate),runs in sorted(groups.items()):
        n=sum(x['scheduled_requests'] for x in runs)
        obs=sum(max(x['arrival_window_seconds'],x['full_drain_seconds']) for x in runs)
        success=sum(x['results']['successes'] for x in runs)
        good=sum(x['illustrative_slo']['successful_meeting_both'] for x in runs)
        with_tokens=sum(x['results']['successes_with_completion_tokens'] for x in runs)
        tokens=sum(x['results']['completion_tokens_known'] for x in runs)
        reject=sum(x['statuses'].get('429',0) for x in runs)
        p95=[x['schedule_to_first_content_seconds']['p95'] for x in runs]
        rows.append({'mode':mode,'max_num_seqs':configs[mode]['max_num_seqs'],'scheduled_rate_per_second':rate,
                     'repeats':len(runs),'scheduled':n,'successes':success,'http_429':reject,
                     'other_failures':n-success-reject,'observation_seconds_including_drain':obs,
                     'completed_per_observation_second':success/obs,'slo_goodput_per_observation_second':good/obs,
                     'slo_good_fraction_of_offered':good/n,'output_tokens_per_observation_second':tokens/obs if with_tokens==success and success else None,
                     'mean_per_run_schedule_to_first_content_p95_seconds':mean(p95) if all(v is not None for v in p95) else None,
                     'max_per_run_dispatch_lag_p99_seconds':max(x['dispatch_lag_seconds']['p99'] for x in runs),
                     'mean_dispatch_rate_in_arrival_window':mean(x['observed_dispatch_rate_in_window'] for x in runs)})
    result={'note':f'Uniform deterministic arrivals. Rates include drain and the full arrival window; p95 is a mean of per-run p95s. SLO {ttft_slo:g} ms TTFT and {latency_slo:g} ms completion is illustrative. max_num_seqs controls scheduling capacity; this is not an isolated batching on/off experiment.','rows':rows}
    (args.directory/'arrival-analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for key,title,label,file in [('slo_goodput_per_observation_second',f'SLO goodput ({ttft_slo:g} ms TTFT, {latency_slo:g} ms completion)','Good requests / second','arrival-goodput.png'),('slo_good_fraction_of_offered','Fraction of scheduled requests meeting SLO','Fraction','arrival-slo-fraction.png')]:
        fig,ax=plt.subplots(figsize=(7,4.5))
        for mode in ('serial','concurrent'):
            subset=sorted((r for r in rows if r['mode']==mode),key=lambda r:r['scheduled_rate_per_second'])
            ax.plot([r['scheduled_rate_per_second'] for r in subset],[r[key] for r in subset],marker='o',label=f'{mode} (max sequences {configs[mode]["max_num_seqs"]})')
        ax.set_xlabel('Scheduled requests / second'); ax.set_ylabel(label); ax.set_title(title); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(args.directory/file,dpi=160); plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
