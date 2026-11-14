"""Paired repair bookkeeping; no fake memory savings or negligible-gap ratios."""
import math


def recovery_summary(units, minimum_gap=.05, minimum_reference_only=10):
    """One paired evaluation unit per prefix/origin/seed; no independence assumed.

    Confidence intervals must cluster/refit elsewhere. This function reports
    descriptive paired counts and the development gate, never significance.
    """
    if not units or len({u['unit_id'] for u in units}) != len(units):
        raise ValueError('need nonempty unique paired units')
    for unit in units:
        if any(type(unit[key]) is not bool for key in ('keep_correct', 'reference_correct', 'repair_correct')):
            raise ValueError('correctness must be objectively checked booleans')
    count = len(units)
    keep = sum(u['keep_correct'] for u in units)/count
    reference = sum(u['reference_correct'] for u in units)/count
    repair = sum(u['repair_correct'] for u in units)/count
    ref_only = sum(u['reference_correct'] and not u['keep_correct'] for u in units)
    reference_only_questions = {u['question_id'] for u in units if u['reference_correct'] and not u['keep_correct']}
    gate = reference-keep >= minimum_gap and len(reference_only_questions) >= minimum_reference_only
    return {'paired_units': count, 'source_questions': len({u['question_id'] for u in units}),
            'keep_accuracy': keep, 'reference_accuracy': reference, 'repair_accuracy': repair,
            'gap': reference-keep, 'absolute_repair_gain': repair-keep,
            'reference_success_keep_failure': ref_only,
            'reference_success_keep_failure_source_questions': len(reference_only_questions),
            'keep_success_reference_failure': sum(u['keep_correct'] and not u['reference_correct'] for u in units),
            'wrong_to_correct': sum(not u['keep_correct'] and u['repair_correct'] for u in units),
            'correct_to_wrong': sum(u['keep_correct'] and not u['repair_correct'] for u in units),
            'development_gap_gate_pass': gate,
            'gap_fraction_recovered': (repair-keep)/(reference-keep) if gate else None,
            'scope': 'descriptive development bookkeeping; not a CI, confirmation, or deployable selection rule'}


def deployment_classification(*, implementation, peak_reserved_bytes, memory_budget_bytes,
                              forward_seconds, quantization_seconds, transfer_seconds):
    values = (peak_reserved_bytes, memory_budget_bytes, forward_seconds, quantization_seconds, transfer_seconds)
    if implementation not in ('packed', 'numerical_emulation') or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('declare packed/emulated implementation and finite nonnegative measurements')
    if memory_budget_bytes <= 0:
        raise ValueError('declare a positive physical memory budget')
    reasons = []
    if implementation != 'packed':
        reasons.append('no_physical_low_bit_storage_evidence')
    if peak_reserved_bytes > memory_budget_bytes:
        reasons.append('transient_memory_exceeds_operating_budget')
    return {'eligible_for_deployment_comparison': not reasons, 'mechanism_only_reasons': reasons,
            'measured_intervention_seconds': forward_seconds+quantization_seconds+transfer_seconds,
            'warning': 'Eligibility is not evidence of speedup; full-task work, failures, sampling and selection must also be charged.'}
