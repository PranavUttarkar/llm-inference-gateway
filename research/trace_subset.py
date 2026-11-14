"""CPU-only first-divergence audit of real, matched autoregressive traces.

No reconstruction of token margins from output text is attempted. Missing usable
traces is an inconclusive gate, never evidence that wobble is redundant.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

FRACTIONS = (.1, .25, .5)


def first_difference(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None  # A truncated/common-prefix-only pair is not a token flip.


def audit(rows):
    events, exclusions = [], []
    real = bool(rows) and all(r.get('evidence_kind') == 'model_trace' for r in rows)
    seen = set()
    for row in rows:
        if row['id'] in seen:
            raise ValueError('duplicate question ID; keep variants in one row')
        seen.add(row['id'])
        if row.get('evidence_kind') not in ('model_trace', 'fixture'):
            raise ValueError('declare model_trace or fixture provenance')
        baseline = row['baseline']
        tokens, gaps = baseline['token_ids'], baseline['top2_logit_gaps']
        if not tokens or len(gaps) != len(tokens) or any(not math.isfinite(g) or g < 0 for g in gaps):
            raise ValueError('need a finite nonnegative full-vocabulary logit gap per baseline token')
        if not baseline.get('completed'):
            exclusions.append({'id': row['id'], 'reason': 'baseline_incomplete'})
            continue
        for variant in row['variants']:
            if (variant['prompt_token_sha256'] != row['prompt_token_sha256'] or
                    variant['model_revision'] != row['model_revision']):
                raise ValueError('prompt/model changed; pair does not isolate execution context')
            if not variant.get('completed'):
                exclusions.append({'id': row['id'], 'condition': variant['condition'], 'reason': 'variant_incomplete'})
                continue
            index = first_difference(tokens, variant['token_ids'])
            if index is None:
                if tokens != variant['token_ids']:
                    exclusions.append({'id': row['id'], 'condition': variant['condition'],
                                       'reason': 'length_only_difference_without_explicit_EOS_flip'})
                continue
            # Baseline-token-only rank; neither flip positions nor answers choose the set.
            membership = {}
            for fraction in FRACTIONS:
                cutoff = sorted(gaps)[math.ceil(fraction*len(gaps))-1]
                membership[str(fraction)] = {'threshold': cutoff, 'in_low_margin_set': gaps[index] <= cutoff,
                                             'actual_set_size': sum(g <= cutoff for g in gaps)}
            events.append({'id': row['id'], 'condition': variant['condition'], 'first_difference': index,
                           'baseline_length': len(tokens), 'logit_gap': gaps[index], 'low_margin_sets': membership})
    coverage = {}
    for fraction in FRACTIONS:
        key = str(fraction)
        inside = sum(e['low_margin_sets'][key]['in_low_margin_set'] for e in events)
        outside_ids = sorted({e['id'] for e in events if not e['low_margin_sets'][key]['in_low_margin_set']})
        coverage[key] = {'first_divergence_events': len(events), 'inside': inside, 'outside': len(events)-inside,
                         'inside_fraction': inside/len(events) if events else None,
                         'distinct_questions_with_outside_flips': len(outside_ids), 'outside_question_ids': outside_ids}
    primary_outside = coverage['0.25']['distinct_questions_with_outside_flips']
    return {'purpose': 'first_divergence_vs_low_margin_subset', 'scientific_evidence': real,
            'usable_question_rows': len(rows), 'events': events, 'exclusions': exclusions, 'coverage': coverage,
            'gate': {'experiment_1_eligible_for_further_design': real and primary_outside > 0,
                     'reason': 'outside_primary_set_observed_not_proof_of_value' if real and primary_outside else
                               ('no_usable_real_traces' if not real else 'no_outside_primary_set_observed')},
            'interpretation': 'Primary set: lowest 25% baseline full-trace gaps, including ties; 10/50% are sensitivity checks. '
                              'Only first divergence shares the same generated prefix. Repeated variants are not independent questions. '
                              'Outside this rank budget does not mean outside a small-margin regime or prove usefulness. '
                              'The retrospective set is not an online branch-selection policy.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='JSONL using the schema in reports/batch-instability/report.md')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite evidence')
    rows = [json.loads(line) for line in args.input.read_text(encoding='utf-8').splitlines() if line.strip()] if args.input else []
    result = audit(rows)
    result['input_sha256'] = hashlib.sha256(args.input.read_bytes()).hexdigest() if args.input else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'usable_question_rows': result['usable_question_rows'], 'gate': result['gate']}, indent=2))


if __name__ == '__main__':
    main()
