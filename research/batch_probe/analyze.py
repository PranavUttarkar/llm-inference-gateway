"""Validate complete runs; evaluate label-free features on a reserved pilot split."""
import argparse
import collections
import json
import math
import hashlib
from pathlib import Path

from .design import CONDITIONS
from .run import softmax
from .workload import LABELS, digest, validate_workload

PERTURBED = ('solo_padded', 'batch8', 'batch8_content', 'batch8_padded', 'batch8_last')


def features(records):
    """No ground-truth parameter. All comparisons refer to the solo choice."""
    base = softmax(records['solo']['choice_logits'])
    choice = max(range(4), key=lambda i: base[i])
    ordered = sorted(base, reverse=True)
    margin = ordered[0] - ordered[1]
    entropy = -sum(p * math.log(max(p, 1e-300)) for p in base)
    distributions = [softmax(records[key]['choice_logits']) for key in PERTURBED]
    flips = [max(range(4), key=lambda i: p[i]) != choice for p in distributions]
    # Continuous drift remains measurable when no argmax flips occur.
    l1 = [sum(abs(p[i] - base[i]) for i in range(4)) for p in distributions]
    repeat_l1 = sum(abs(a - b) for a, b in zip(base, softmax(records['solo_repeat']['choice_logits'])))
    return {'choice': LABELS[choice], 'confidence': ordered[0], 'margin': margin,
            'entropy': entropy, 'analytic_sample_disagreement': 1 - sum(p*p for p in base),
            'log_margin': math.log10(max(margin, 1e-12)),
            'log_tokens': math.log1p(records['solo']['target_tokens']),
            'choice_mass': records['solo']['target_choice_mass'],
            'raw_top_probability': records['solo']['raw_top_token_probability'],
            'full_vocab_entropy': records['solo']['full_vocab_entropy'],
            'full_vocab_top2_margin': records['solo']['full_vocab_top2_margin'],
            'flip_fraction': sum(flips) / len(flips),
            'log_max_l1': math.log10(max(max(l1), 1e-12)),
            'log_mean_l1': math.log10(max(sum(l1) / len(l1), 1e-12)),
            'repeat_l1': repeat_l1,
            'repeat_flip': max(range(4), key=lambda i: records['solo_repeat']['choice_logits'][i]) != choice,
            'batch_repeat_l1': sum(abs(a - b) for a, b in zip(
                softmax(records['batch8']['choice_logits']), softmax(records['batch8_repeat']['choice_logits']))) }


def load_run(directory):
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete':
        raise ValueError('refusing to analyze an incomplete or failed run')
    workload = json.loads((directory / 'workload.json').read_text(encoding='utf-8'))
    questions = workload['questions']
    validate_workload(questions)
    if digest(questions) != manifest['selected_questions_sha256'] or digest(questions) != workload['sha256']:
        raise ValueError('selected workload hash mismatch')
    jobs = json.loads((directory / 'schedule.json').read_text(encoding='utf-8'))
    if digest(jobs) != manifest['schedule_sha256']:
        raise ValueError('schedule hash mismatch')
    for relative, expected_hash in manifest['source_sha256'].items():
        archived = directory / 'source' / relative
        if hashlib.sha256(archived.read_bytes()).hexdigest() != expected_hash:
            raise ValueError('archived source hash mismatch')
    records = [json.loads(line) for line in (directory / 'records.jsonl').read_text(encoding='utf-8').splitlines()]
    expected = {(q['id'], c) for q in questions for c in CONDITIONS}
    observed = [(r['question_id'], r['condition']) for r in records]
    if len(observed) != len(expected) or set(observed) != expected or len(set(observed)) != len(observed):
        raise ValueError('missing, duplicate, or unknown question-condition records')
    if manifest['completed_jobs'] != len(records) or len(jobs) != len(records):
        raise ValueError('manifest count mismatch')
    grouped = collections.defaultdict(dict)
    for i, row in enumerate(records):
        if row['order'] != i or (row['question_id'], row['condition']) != (jobs[i]['question_id'], jobs[i]['condition']):
            raise ValueError('record order does not match frozen schedule')
        logits = row['choice_logits']
        if len(logits) != 4 or any(not math.isfinite(x) for x in logits):
            raise ValueError('invalid choice logits')
        if row['elapsed_seconds'] <= 0 or not math.isfinite(row['elapsed_seconds']):
            raise ValueError('invalid timing')
        settings = CONDITIONS[row['condition']]
        if row['batch_size'] != settings['batch'] or row['padded_length'] != row['target_tokens'] + settings['extra_padding']:
            raise ValueError('observed tensor geometry differs from design')
        grouped[row['question_id']][row['condition']] = row
    for rowset in grouped.values():
        if len({r['target_input_sha256'] for r in rowset.values()}) != 1:
            raise ValueError('target input tokens changed across conditions')
    if hashlib.sha256((directory / 'records.jsonl').read_bytes()).hexdigest() != manifest['records_sha256']:
        raise ValueError('record content hash mismatch')
    return manifest, questions, grouped


def paired_bootstrap_improvement(labels, base, augmented, groups, repetitions=2000, seed=2718):
    import numpy as np
    def losses(probabilities):
        p = np.clip(np.asarray(probabilities), 1e-12, 1-1e-12)
        y = np.asarray(labels)
        return -(y * np.log(p) + (1-y) * np.log1p(-p))
    differences = losses(base) - losses(augmented)  # positive means augmented is better
    unique = sorted(set(groups))
    by_group = {g: np.flatnonzero(np.asarray(groups) == g) for g in unique}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(repetitions):
        indices = np.concatenate([by_group[g] for g in rng.choice(unique, size=len(unique), replace=True)])
        means.append(float(differences[indices].mean()))
    return {'delta_log_loss': float(differences.mean()),
            'conditional_group_bootstrap_95': np.quantile(means, [.025, .975]).tolist(),
            'bootstrap_repetitions': repetitions,
            'scope': 'evaluation-question uncertainty conditional on fitted development models; exploratory'}


def predictive_analysis(questions, feature_rows, bootstrap=2000):
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

    confidence = ('confidence', 'margin', 'log_margin', 'entropy', 'analytic_sample_disagreement', 'log_tokens',
                  'choice_mass', 'raw_top_probability', 'full_vocab_entropy', 'full_vocab_top2_margin')
    instability = ('flip_fraction', 'log_max_l1', 'log_mean_l1')
    development = [i for i, q in enumerate(questions) if q['split'] == 'development']
    evaluation = [i for i, q in enumerate(questions) if q['split'] == 'evaluation']
    y = np.array([int(f['choice'] != q['correct_label']) for q, f in zip(questions, feature_rows)])
    if not development or not evaluation:
        return {'status': 'insufficient_splits'}
    if any(len(set(y[indices])) < 2 for indices in (development, evaluation)):
        return {'status': 'insufficient_class_variation'}

    # Deterministic pseudo-random sham features control extra model flexibility.
    enriched = []
    for q, f in zip(questions, feature_rows):
        values = dict(f)
        for i in range(3):
            values[f'sham_{i}'] = int(digest(['sham', q['id'], i])[:12], 16) / (16**12)
        enriched.append(values)

    def matrix(keys):
        return np.array([[f[key] for key in keys] + [q['family']] for q, f in zip(questions, enriched)], dtype=object)

    def fit(keys):
        x = matrix(keys)
        prep = ColumnTransformer([
            ('numeric', make_pipeline(SplineTransformer(n_knots=3, degree=2, include_bias=False),
                                      StandardScaler()), list(range(len(keys)))),
            ('family', OneHotEncoder(handle_unknown='ignore', sparse_output=False), [len(keys)])])
        model = make_pipeline(prep, LogisticRegression(C=1.0, max_iter=3000, random_state=0))
        model.fit(x[development], y[development])
        return model.predict_proba(x[evaluation])[:, 1]

    predictions = {'confidence_spline': fit(confidence),
                   'confidence_plus_instability': fit(confidence + instability),
                   'confidence_plus_sham': fit(confidence + ('sham_0', 'sham_1', 'sham_2'))}
    # Pre-GPU amendment: a single low-flexibility spline is not a sufficient
    # confidence comparator (see statistical-stress-v1). Select reference models
    # on development groups ONLY, using identical candidate families and folds.
    dev_groups = np.array([questions[i]['group_id'] for i in development])
    if min(np.bincount(y[development], minlength=2)) < 3 or len(set(dev_groups)) < 3:
        return {'status': 'insufficient_development_data_for_group_cv'}
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=671)
    folds = list(splitter.split(np.zeros(len(development)), y[development], dev_groups))
    if any(len(set(y[np.asarray(development)[train]])) < 2 for train, _ in folds):
        return {'status': 'single_class_group_cv_training_fold'}
    scorer = make_scorer(log_loss, response_method='predict_proba', labels=[0, 1], greater_is_better=False)

    def reference(keys):
        x = matrix(keys)
        def prepare():
            return ColumnTransformer([
                ('numeric', StandardScaler(), list(range(len(keys)))),
                ('family', OneHotEncoder(handle_unknown='ignore', sparse_output=False), [len(keys)])])
        candidates = {
            'knn7_distance': make_pipeline(prepare(), KNeighborsClassifier(n_neighbors=7, weights='distance')),
            'knn15_distance': make_pipeline(prepare(), KNeighborsClassifier(n_neighbors=15, weights='distance')),
            'hist_gradient_boosting': make_pipeline(prepare(), HistGradientBoostingClassifier(
                max_leaf_nodes=15, max_iter=100, min_samples_leaf=8, l2_regularization=1.0,
                early_stopping=False, random_state=0))}
        # Include the original spline in the same development-only selection.
        candidates['spline'] = make_pipeline(ColumnTransformer([
            ('numeric', make_pipeline(SplineTransformer(n_knots=3, degree=2, include_bias=False),
                                      StandardScaler()), list(range(len(keys)))),
            ('family', OneHotEncoder(handle_unknown='ignore', sparse_output=False), [len(keys)])]),
            LogisticRegression(C=1.0, max_iter=3000, random_state=0))
        cv_losses = {}
        for name, model in candidates.items():
            scores = cross_val_score(model, x[development], y[development], cv=folds,
                                     scoring=scorer, n_jobs=1, error_score='raise')
            cv_losses[name] = float(-scores.mean())
        selected = min(cv_losses, key=cv_losses.get)
        model = candidates[selected]
        model.fit(x[development], y[development])
        return model.predict_proba(x[evaluation])[:, 1], {'selected': selected, 'development_cv_log_loss': cv_losses}

    predictions['confidence_reference'], base_selection = reference(confidence)
    predictions['confidence_plus_instability_reference'], augmented_selection = reference(confidence + instability)
    scores = {
        'low_confidence': [1-feature_rows[i]['confidence'] for i in evaluation],
        'small_margin': [-feature_rows[i]['margin'] for i in evaluation],
        'entropy': [feature_rows[i]['entropy'] for i in evaluation],
        'analytic_sample_disagreement': [feature_rows[i]['analytic_sample_disagreement'] for i in evaluation],
        'batch_flip_fraction': [feature_rows[i]['flip_fraction'] for i in evaluation],
        **predictions}
    target = y[evaluation]
    report = {}
    for name, score in scores.items():
        score = np.asarray(score)
        # Group-stable tie break; a binary detector must not benefit from input order.
        ranked = sorted(range(len(evaluation)), key=lambda j: (-score[j], digest(questions[evaluation[j]]['id'])))
        risks = {}
        for fraction in (.1, .25, .5):
            flagged = max(1, math.ceil(len(ranked) * fraction))
            kept = ranked[flagged:]
            risks[str(fraction)] = {'flagged': flagged,
                                  'remaining_error': float(target[kept].mean()) if kept else None}
        report[name] = {'error_auroc': float(roc_auc_score(target, score)),
                        'error_average_precision': float(average_precision_score(target, score)),
                        'selective_risk': risks}
        if name in predictions:
            report[name]['log_loss'] = float(log_loss(target, score, labels=[0, 1]))
    report['primary_comparison'] = paired_bootstrap_improvement(
        target, predictions['confidence_reference'], predictions['confidence_plus_instability_reference'],
        [questions[i]['group_id'] for i in evaluation], bootstrap)
    report['legacy_spline_comparison'] = paired_bootstrap_improvement(
        target, predictions['confidence_spline'], predictions['confidence_plus_instability'],
        [questions[i]['group_id'] for i in evaluation], bootstrap)
    report['reference_model_selection'] = {'confidence': base_selection, 'augmented': augmented_selection,
                                          'folds': 3, 'selection_uses_evaluation_labels': False}
    report['development_questions'] = len(development)
    report['evaluation_questions'] = len(evaluation)
    report['evaluation_errors'] = int(target.sum())
    report['evaluation_groups'] = len({questions[i]['group_id'] for i in evaluation})
    report['status'] = 'exploratory_only'
    return report


def summarize(directory, bootstrap=2000):
    manifest, questions, grouped = load_run(directory)
    fs = [features(grouped[q['id']]) for q in questions]
    correct = [f['choice'] == q['correct_label'] for q, f in zip(questions, fs)]
    per_condition = {}
    for condition in CONDITIONS:
        flips = 0
        wrong_to_right = right_to_wrong = 0
        for q, f, ok in zip(questions, fs, correct):
            logits = grouped[q['id']][condition]['choice_logits']
            answer = LABELS[max(range(4), key=lambda i: logits[i])]
            flips += answer != f['choice']
            right_to_wrong += ok and answer != q['correct_label']
            wrong_to_right += not ok and answer == q['correct_label']
        per_condition[condition] = {'choice_flips': flips, 'correct_to_wrong': right_to_wrong,
                                    'wrong_to_correct': wrong_to_right,
                                    'all_batch_forward_seconds': sum(grouped[q['id']][condition]['elapsed_seconds'] for q in questions)}
    predictive = predictive_analysis(questions, fs, bootstrap)
    unstable = sum(f['flip_fraction'] > 0 for f in fs)
    repeat_flips = sum(f['repeat_flip'] for f in fs)
    families = {}
    for family in sorted({q['family'] for q in questions}):
        indices = [i for i, q in enumerate(questions) if q['family'] == family]
        families[family] = {'questions': len(indices), 'solo_errors': sum(not correct[i] for i in indices),
                            'questions_with_choice_flips': sum(fs[i]['flip_fraction'] > 0 for i in indices)}
    # These are predeclared pilot triage rules, not a significance claim.
    failures = []
    if manifest['purpose'] != 'pilot':
        failures.append('not_a_full_gpu_pilot')
    if unstable < 20:
        failures.append('fewer_than_20_questions_with_choice_flips')
    if sum(not value for value in correct) < 30:
        failures.append('fewer_than_30_solo_errors')
    if repeat_flips > .01 * len(questions):
        failures.append('solo_repeat_flip_rate_above_1_percent')
    if predictive.get('evaluation_errors', 0) < 20:
        failures.append('fewer_than_20_evaluation_errors')
    primary = predictive.get('primary_comparison', {})
    if primary.get('delta_log_loss', -1) < .01:
        failures.append('incremental_log_loss_gain_below_0.01_nats')
    if primary.get('conditional_group_bootstrap_95', [-1])[0] <= 0:
        failures.append('incremental_gain_interval_includes_zero')
    if predictive.get('confidence_plus_instability_reference', {}).get('log_loss', math.inf) >= predictive.get('confidence_plus_sham', {}).get('log_loss', -math.inf):
        failures.append('instability_does_not_beat_equal_dimension_sham_features')
    if predictive.get('confidence_plus_instability_reference', {}).get('log_loss', math.inf) >= predictive.get('confidence_spline', {}).get('log_loss', -math.inf):
        failures.append('selected_augmented_model_does_not_beat_original_confidence_spline')
    return {'purpose': manifest['purpose'], 'scientific_evidence': manifest['backend']['scientific_evidence'],
            'claim_scope': 'one-token answer-choice scoring, not free-form generation or production serving',
            'questions': len(questions), 'solo_accuracy': sum(correct) / len(correct),
            'per_family': families,
            'questions_with_choice_flips': unstable, 'solo_repeat_flips': repeat_flips,
            'mean_solo_choice_mass': sum(grouped[q['id']]['solo']['target_choice_mass'] for q in questions) / len(questions),
            'solo_unrestricted_nonchoice_top_tokens': sum(
                grouped[q['id']]['solo']['raw_top_token_text'].strip() not in LABELS or
                len(grouped[q['id']]['solo']['raw_top_token_text'].strip()) != 1 for q in questions),
            'per_condition': per_condition, 'predictive': predictive,
            'triage': {'continue_to_confirmation_candidate': not failures, 'failed_gates': failures},
            'provenance': {'workload_sha256': manifest['selected_questions_sha256'],
                           'schedule_sha256': manifest['schedule_sha256'], 'source_sha256': manifest['source_sha256']},
            'cost_warning': 'Every recorded time charges the entire batch, including companions. '
                            'Risk curves are diagnostic rankings; no verifier or useful-throughput gain measured.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error('use at least 100 bootstrap repetitions')
    result = summarize(args.directory, args.bootstrap)
    path = args.directory / 'analysis.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'purpose': result['purpose'], 'questions': result['questions'], 'triage': result['triage']}, indent=2))


if __name__ == '__main__':
    main()
