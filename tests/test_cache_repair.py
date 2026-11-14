import unittest

from research.cache_repair.accounting import deployment_classification, recovery_summary


def units(keep, reference, repair):
    return [{'unit_id': str(i), 'question_id': str(i//2), 'keep_correct': a,
             'reference_correct': b, 'repair_correct': c}
            for i, (a, b, c) in enumerate(zip(keep, reference, repair))]


class RepairAccountingTests(unittest.TestCase):
    def test_negligible_or_negative_gap_has_no_recovery_ratio(self):
        for keep, reference in (([True, False], [True, False]), ([True, True], [True, False])):
            result = recovery_summary(units(keep, reference, [True, True]), minimum_reference_only=1)
            self.assertFalse(result['development_gap_gate_pass'])
            self.assertIsNone(result['gap_fraction_recovered'])

    def test_ratio_is_not_clipped_and_repeated_seeds_are_not_independent_questions(self):
        keep = [True, False, False, False]
        reference = [True, True, False, False]
        for repair, expected in (([True]*4, 3.0), ([False]*4, -1.0)):
            result = recovery_summary(units(keep, reference, repair), minimum_reference_only=1)
            self.assertEqual(result['gap_fraction_recovered'], expected)
            self.assertEqual(result['source_questions'], 2)
            self.assertEqual(result['paired_units'], 4)

    def test_emulation_and_transient_peak_cannot_count_as_deployable(self):
        common = dict(memory_budget_bytes=100, forward_seconds=2, quantization_seconds=1, transfer_seconds=.5)
        emulated = deployment_classification(implementation='numerical_emulation', peak_reserved_bytes=80, **common)
        too_large = deployment_classification(implementation='packed', peak_reserved_bytes=101, **common)
        packed = deployment_classification(implementation='packed', peak_reserved_bytes=80, **common)
        self.assertFalse(emulated['eligible_for_deployment_comparison'])
        self.assertFalse(too_large['eligible_for_deployment_comparison'])
        self.assertTrue(packed['eligible_for_deployment_comparison'])
        self.assertEqual(packed['measured_intervention_seconds'], 3.5)

    def test_many_seeds_of_one_question_cannot_pass_independent_gap_gate(self):
        repeated = units([False]*20, [True]*20, [True]*20)
        for unit in repeated:
            unit['question_id'] = 'one_question'
        result = recovery_summary(repeated)
        self.assertEqual(result['reference_success_keep_failure'], 20)
        self.assertEqual(result['reference_success_keep_failure_source_questions'], 1)
        self.assertFalse(result['development_gap_gate_pass'])


if __name__ == '__main__':
    unittest.main()
