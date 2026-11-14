"""Monte Carlo checks of the diagnostic analysis; no pretrained-model evidence.

Includes a deliberately misspecified confidence-only case where apparent added
value is not new information. Its purpose is to expose limitations, not tune a
study until positive. All scenarios/seeds are chosen before any GPU observations.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

from .analyze import predictive_analysis
from .run import source_hashes

SCENARIOS = ('confidence_null', 'nonlinear_proxy_null', 'incremental_signal')


def simulated_questions(scenario, seed, count=240):
    if scenario not in SCENARIOS or count < 40 or count % 2:
        raise ValueError('need a known scenario and an even sample size >=40')
    rng = random.Random(seed)
    questions, fs = [], []
    for i in range(count):
        confidence = rng.uniform(.35, .95)
        latent = rng.random()
        if scenario == 'confidence_null':
            instability = latent
            probability_error = 1-confidence
        elif scenario == 'nonlinear_proxy_null':
            # Instability is a deterministic transform of an observed baseline
            # covariate. It cannot add conditional information by construction.
            instability = .5 + .45 * math.sin(8 * math.pi * (confidence-.35)/.6)
            probability_error = .05 + .9 * instability
        else:
            instability = latent
            probability_error = .08 + .84 * instability
        error = rng.random() < probability_error
        questions.append({'id': f'{seed}-{i}', 'group_id': f'{seed}-{i}', 'family': 'synthetic',
                          'split': 'development' if i < count//2 else 'evaluation',
                          'correct_label': 'B' if error else 'A'})
        p = [confidence] + [(1-confidence)/3]*3
        entropy = -sum(x*math.log(x) for x in p)
        fs.append({'choice': 'A', 'confidence': confidence, 'margin': p[0]-p[1],
                   'log_margin': math.log10(p[0]-p[1]), 'entropy': entropy,
                   'analytic_sample_disagreement': 1-sum(x*x for x in p), 'log_tokens': math.log(200),
                   'choice_mass': 1.0, 'raw_top_probability': confidence,
                   'full_vocab_entropy': entropy, 'full_vocab_top2_margin': p[0]-p[1],
                   'flip_fraction': instability, 'log_max_l1': math.log10(instability),
                   'log_mean_l1': math.log10(instability/2)})
    return questions, fs


def wilson(successes, trials):
    z = 1.959963984540054
    p = successes/trials
    denominator = 1+z*z/trials
    center = (p+z*z/(2*trials))/denominator
    half = z*math.sqrt(p*(1-p)/trials+z*z/(4*trials*trials))/denominator
    return [max(0, center-half), min(1, center+half)]


def run(repetitions=64, count=240, bootstrap=200, progress=None):
    import numpy as np
    started = time.perf_counter()
    rows = []
    for scenario in SCENARIOS:
        for repetition in range(repetitions):
            questions, features = simulated_questions(scenario, 810000+repetition, count)
            report = predictive_analysis(questions, features, bootstrap)
            primary = report.get('primary_comparison', {})
            gain = primary.get('delta_log_loss', -math.inf)
            interval = primary.get('conditional_group_bootstrap_95', [-math.inf, math.inf])
            beats_sham = (report.get('confidence_plus_instability_reference', {}).get('log_loss', math.inf) <
                          report.get('confidence_plus_sham', {}).get('log_loss', -math.inf))
            rows.append({'scenario': scenario, 'seed': 810000+repetition,
                         'gain': gain, 'interval': interval, 'beats_sham': beats_sham,
                         'predictive_gate_pass': gain >= .01 and interval[0] > 0 and beats_sham})
            if progress and (repetition+1) % 8 == 0:
                progress(f'{scenario}: {repetition+1}/{repetitions}')
    summaries = {}
    for scenario in SCENARIOS:
        selected = [r for r in rows if r['scenario'] == scenario]
        passes = sum(r['predictive_gate_pass'] for r in selected)
        summaries[scenario] = {'repetitions': repetitions, 'predictive_gate_passes': passes,
                               'pass_rate': passes/repetitions,
                               'wilson_95_for_pass_rate': wilson(passes, repetitions),
                               'mean_delta_log_loss': float(np.mean([r['gain'] for r in selected]))}
    return {'purpose': 'statistical_stress_test', 'scientific_evidence': False,
            'sample_size_per_repetition': count, 'bootstrap_repetitions': bootstrap,
            'source_sha256': source_hashes(), 'summary': summaries, 'repetitions': rows,
            'wall_seconds': time.perf_counter()-started,
            'interpretation': 'Predictive-gate passes are not full pilot passes. '
                              'The nonlinear proxy null exposes calibration misspecification; '
                              'no finite baseline family proves conditional information gain.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=64)
    parser.add_argument('--questions', type=int, default=240)
    parser.add_argument('--bootstrap', type=int, default=200)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite evidence')
    if args.repetitions < 8 or args.bootstrap < 100:
        parser.error('need >=8 repetitions and >=100 bootstrap draws')
    result = run(args.repetitions, args.questions, args.bootstrap, lambda line: print(line, flush=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
