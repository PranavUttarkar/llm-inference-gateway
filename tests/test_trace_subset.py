"""Common-prefix and low-margin-budget controls for the offline wobble gate."""
import copy
import unittest

from research.trace_subset import audit, first_difference


def fixture(gaps, other, evidence='fixture'):
    return {'id': 'q1', 'evidence_kind': evidence, 'model_revision': 'fixed', 'prompt_token_sha256': 'same',
            'baseline': {'token_ids': [10, 11, 12, 2], 'top2_logit_gaps': gaps, 'completed': True},
            'variants': [{'condition': 'batch8', 'model_revision': 'fixed', 'prompt_token_sha256': 'same',
                          'token_ids': other, 'completed': True}]}


class TraceSubsetTests(unittest.TestCase):
    def test_missing_data_is_not_redundancy_or_permission_to_launch(self):
        result = audit([])
        self.assertEqual(result['gate']['reason'], 'no_usable_real_traces')
        self.assertFalse(result['gate']['experiment_1_eligible_for_further_design'])
        self.assertIsNone(result['coverage']['0.25']['inside_fraction'])

    def test_only_first_divergence_tested_and_fixtures_never_enable_gate(self):
        result = audit([fixture([.5, .01, .1, .9], [10, 99, 98, 2])])
        self.assertEqual(len(result['events']), 1)
        self.assertEqual(result['events'][0]['first_difference'], 1)
        self.assertEqual(result['coverage']['0.25']['inside'], 1)
        self.assertFalse(result['gate']['experiment_1_eligible_for_further_design'])

    def test_outside_rank_set_is_only_eligibility_not_usefulness(self):
        result = audit([fixture([.5, .01, .1, .9], [99, 11, 12, 2], 'model_trace')])
        self.assertEqual(result['coverage']['0.25']['outside'], 1)
        self.assertTrue(result['gate']['experiment_1_eligible_for_further_design'])
        self.assertEqual(result['gate']['reason'], 'outside_primary_set_observed_not_proof_of_value')

    def test_ties_included_and_model_or_prompt_changes_rejected(self):
        row = fixture([.01, .01, .1, .9], [10, 99, 12, 2])
        result = audit([row])
        self.assertEqual(result['events'][0]['low_margin_sets']['0.25']['actual_set_size'], 2)
        for key in ('model_revision', 'prompt_token_sha256'):
            bad = copy.deepcopy(row)
            bad['variants'][0][key] = 'changed'
            with self.assertRaisesRegex(ValueError, 'does not isolate'):
                audit([bad])

    def test_truncation_not_counted_as_flip(self):
        self.assertIsNone(first_difference([1, 2, 3], [1, 2]))
        result = audit([fixture([.5, .01, .1, .9], [10, 11])])
        self.assertEqual(len(result['events']), 0)
        self.assertEqual(result['exclusions'][0]['reason'], 'length_only_difference_without_explicit_EOS_flip')


if __name__ == '__main__':
    unittest.main()
