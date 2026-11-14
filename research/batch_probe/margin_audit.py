"""Offline margin-only coverage of batch flips; never fits thresholds to flips."""
import argparse
import json
import math
from pathlib import Path

from .analyze import PERTURBED, load_run
from .run import softmax, source_hashes

# Declared before running this audit. They describe coverage, not tuned cutoffs.
ABSOLUTE_PROBABILITY_MARGINS = (.001, .005, .01, .025, .05, .1)
ABSOLUTE_LOGIT_MARGINS = (.001, .01, .05, .1, .25, .4)
DEVELOPMENT_FLAG_FRACTIONS = (.1, .25, .5)


def margin(logits, probability=False):
    values = softmax(logits) if probability else logits
    ordered = sorted(values, reverse=True)
    return ordered[0]-ordered[1]


def coverage(values, flips, threshold):
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError('need a finite nonnegative margin threshold')
    if len(values) != len(flips) or not values:
        raise ValueError('need aligned nonempty margins and flip indicators')
    flagged = [value <= threshold for value in values]
    caught = sum(flag and flip for flag, flip in zip(flagged, flips))
    total_flips = sum(flips)
    return {'threshold': threshold, 'questions': len(values), 'flipped_questions': total_flips,
            'flagged_questions': sum(flagged), 'flagged_fraction': sum(flagged)/len(values),
            'flips_covered': caught, 'flips_missed': total_flips-caught,
            'flip_recall': caught/total_flips if total_flips else None,
            'flagged_nonflips': sum(flagged)-caught,
            'flip_precision': caught/sum(flagged) if any(flagged) else None}


def development_threshold(values, fraction):
    if not values or not 0 < fraction <= 1:
        raise ValueError('need development margins and a fraction in (0,1]')
    # Empirical nearest-rank quantile; ties may exceed the nominal budget.
    return sorted(values)[max(0, math.ceil(fraction*len(values))-1)]


def bound_check(base, perturbed, tolerance=1e-12):
    winner = max(range(len(base)), key=lambda i: base[i])
    other_winner = max(range(len(perturbed)), key=lambda i: perturbed[i])
    delta = [b-a for a, b in zip(base, perturbed)]
    span = max(delta)-min(delta)
    gap = margin(base)
    flipped = winner != other_winner
    return {'flipped': flipped, 'base_logit_margin': gap, 'delta_logit_span': span,
            'bound_violated': flipped and gap > span+tolerance}


def audit(directory):
    manifest, questions, grouped = load_run(directory)
    data = []
    violations = 0
    for question in questions:
        records = grouped[question['id']]
        base = records['solo']['choice_logits']
        checks = [bound_check(base, records[key]['choice_logits']) for key in PERTURBED]
        violations += sum(check['bound_violated'] for check in checks)
        data.append({'id': question['id'], 'split': question['split'],
                     'probability_margin': margin(base, probability=True),
                     'logit_margin': margin(base),
                     'flipped': any(check['flipped'] for check in checks)})
    development = [r for r in data if r['split'] == 'development']
    evaluation = [r for r in data if r['split'] == 'evaluation']
    if not development or not evaluation:
        raise ValueError('margin audit requires both development and evaluation questions')
    flips = [r['flipped'] for r in evaluation]
    calibrated, absolute = {}, {}
    for key, fixed_thresholds in (('probability_margin', ABSOLUTE_PROBABILITY_MARGINS),
                                   ('logit_margin', ABSOLUTE_LOGIT_MARGINS)):
        values = [r[key] for r in evaluation]
        calibrated[key] = {}
        for fraction in DEVELOPMENT_FLAG_FRACTIONS:
            threshold = development_threshold([r[key] for r in development], fraction)
            calibrated[key][str(fraction)] = coverage(values, flips, threshold)
        absolute[key] = {str(t): coverage(values, flips, t) for t in fixed_thresholds}
    return {'purpose': 'margin_only_flip_coverage', 'scientific_evidence': manifest['backend']['scientific_evidence'],
            'input_purpose': manifest['purpose'],
            'development_questions': len(development), 'evaluation_questions': len(evaluation),
            'evaluation_flipped_questions': sum(flips), 'total_flipped_questions': sum(r['flipped'] for r in data),
            'threshold_fit': 'development margins only; no flip indicators, correctness labels, or evaluation data',
            'development_quantile_rules': calibrated, 'fixed_absolute_rules': absolute,
            'same_prefix_flip_bound_violations': violations,
            'analysis_source_sha256': source_hashes(), 'run_source_sha256': manifest['source_sha256'],
            'records_sha256': manifest['records_sha256'],
            'interpretation': 'Flip recall is not error recall. A probability margin is not a logit margin. '
                              'The same-prefix logit bound is an accounting identity, not a free predictor '
                              'because its perturbation scale requires another execution. '
                              'Fixture results do not estimate pretrained-model behavior.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite prior evidence')
    result = audit(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ('scientific_evidence', 'evaluation_questions',
          'evaluation_flipped_questions', 'development_quantile_rules', 'same_prefix_flip_bound_violations')}, indent=2))


if __name__ == '__main__':
    main()
