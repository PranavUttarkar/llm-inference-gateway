"""Exploratory source-question gate, not an accuracy or novelty test."""
import math


RULE = {
    'minimum_benefit_nats_per_token': .001,
    'noise_multiplier': 5,
    'repeat_tolerance_nats': 1e-6,
    'bootstrap_draws': 20000,
    'bootstrap_seed': 20261113,
    'confidence': .95,
    'unit': 'source_question',
    'scope': 'numerical mechanism only; no accuracy/deployment inference',
}


def evaluate(rows, rule=None):
    import numpy as np
    rule = RULE if rule is None else rule
    ids = [r['question_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate source question; seeds or shards cannot increase N')
    if len(rows) < 2:
        raise ValueError('need at least two independent source questions')
    benefits, repeats, geometry, keep = [], [], [], []
    for r in rows:
        a = r['arms']
        keep.append(a['keep']['mean_kl_nats'])
        benefits.append(a['keep']['mean_kl_nats']-a['state_only_frozen_scales']['mean_kl_nats'])
        repeats.append(max(abs(a['uncompressed_repeat_control']['mean_kl_nats']),
                           abs(a['uncompressed_stream_rebuild_repeat']['mean_kl_nats']),
                           abs(r['independent_prefill_repeat']['mean_kl_nats'])))
        geometry.append(abs(a['uncompressed_full_prefill_geometry']['mean_kl_nats']))
        if r['frozen_scale_self_reencode_max_abs'] != 0:
            raise ValueError('invalid frozen-parameter decomposition')
    if not all(math.isfinite(x) for xs in (benefits, repeats, geometry, keep) for x in xs):
        raise ValueError('nonfinite measurements')
    # Entire within-question arm vector stays together. Never resample tokens.
    rng = np.random.default_rng(rule['bootstrap_seed'])
    b = np.asarray(benefits)
    means = b[rng.integers(0, len(b), size=(rule['bootstrap_draws'], len(b)))].mean(axis=1)
    alpha = (1-rule['confidence'])/2
    low, high = np.quantile(means, [alpha, 1-alpha]).tolist()
    floor = max(float(np.mean(repeats)), float(np.mean(geometry)))
    threshold = max(rule['minimum_benefit_nats_per_token'], rule['noise_multiplier']*floor)
    if max(repeats) > rule['repeat_tolerance_nats']:
        decision = 'invalid_repeat_control_stop_and_investigate'
    elif float(np.mean(keep)) <= threshold:
        decision = 'no_material_numerical_target_stop'
    elif low > threshold:
        decision = 'numerical_mechanism_continue_to_design_next_study'
    elif high < rule['minimum_benefit_nats_per_token']:
        decision = 'no_material_numerical_benefit_stop'
    else:
        decision = 'inconclusive_no_automatic_expansion'
    return {'decision': decision, 'source_questions': len(rows), 'rule': rule,
            'mean_benefit_nats': float(b.mean()), 'paired_percentile_bootstrap_95_interval': [low, high],
            'mean_keep_kl_nats': float(np.mean(keep)), 'mean_noise_floor_nats': floor,
            'max_independent_or_probe_repeat_kl_abs': max(repeats),
            'operational_continue_threshold_nats': threshold,
            'warning': 'Floor is an operational safeguard, not an additive KL error bound; '
                       'interval is exploratory and does not establish task accuracy or absence of all effects.'}
