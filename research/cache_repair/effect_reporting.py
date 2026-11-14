"""Prespecified descriptive robustness and secondary paired contrasts."""
def describe(rows, draws=20000, seed=20261113):
    import numpy as np
    keep = np.array([r['arms']['keep']['mean_kl_nats'] for r in rows])
    frozen = np.array([r['arms']['state_only_frozen_scales']['mean_kl_nats'] for r in rows])
    fresh = np.array([r['arms']['state_and_scales']['mean_kl_nats'] for r in rows])
    benefit = keep-frozen
    n = len(rows)
    if n < 2 or not np.isfinite(np.concatenate((keep, frozen, fresh))).all():
        raise ValueError('need at least two finite question records')
    indices = np.random.default_rng(seed).integers(0, n, size=(draws, n))
    def interval(values):
        return np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist()
    contrast = fresh-frozen
    # Defined before GPU scoring; disclose near-zero/negative numerical clipping.
    epsilon = 1e-9
    log_ratio = np.log(np.maximum(keep, epsilon)/np.maximum(frozen, epsilon))
    result = {
        'median_primary_benefit_nats': float(np.median(benefit)),
        'primary_sign_counts': {'improved': int((benefit > 0).sum()),
                                'unchanged': int((benefit == 0).sum()), 'worsened': int((benefit < 0).sum())},
        'mean_log_keep_over_frozen_kl': float(log_ratio.mean()),
        'log_ratio_95_percentile_interval': interval(log_ratio),
        'log_ratio_floor_nats': epsilon,
        'log_ratio_clipped_questions': int(((keep < epsilon) | (frozen < epsilon)).sum()),
        'leave_one_out_primary_mean_range_nats': [float((benefit.sum()-benefit).min()/(n-1)),
                                                  float((benefit.sum()-benefit).max()/(n-1))],
        'largest_positive_question_share_of_total_benefit': float(benefit.max()/benefit.sum()) if benefit.sum() > 0 else None,
        'secondary_fresh_minus_frozen_kl_nats': {'mean': float(contrast.mean()),
                                               'paired_95_percentile_interval': interval(contrast),
                                               'positive_favors': 'frozen scales'},
        'inference_scope': 'exploratory secondary/descriptive intervals; no multiplicity-adjusted discovery claim',
    }
    anchor_name = 'higher_precision_k8v8_stream_anchor'
    if all(anchor_name in r['arms'] for r in rows):
        anchor = np.array([r['arms'][anchor_name]['mean_kl_nats'] for r in rows])
        if not np.isfinite(anchor).all():
            raise ValueError('nonfinite anchor')
        result['numerical_k8v8_anchor'] = {'mean_kl_nats': float(anchor.mean()),
            'frozen_minus_anchor_mean_kl_nats': float((frozen-anchor).mean()),
            'frozen_minus_anchor_95_interval': interval(frozen-anchor),
            'frozen_at_or_below_anchor_questions': int((frozen <= anchor).sum()),
            'warning': 'Numerical reference only; negligible accuracy cost has NOT been measured in this configuration.'}
    return result
