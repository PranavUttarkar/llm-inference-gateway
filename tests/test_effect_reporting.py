import unittest
from research.cache_repair.effect_reporting import describe


class EffectReportingTests(unittest.TestCase):
    def test_mixed_secondary_effects_are_not_presented_as_a_directional_result(self):
        rows = [{'arms': {name: {'mean_kl_nats': value} for name, value in {
            'keep': .01, 'state_only_frozen_scales': .006,
            'state_and_scales': fresh, 'higher_precision_k8v8_stream_anchor': .0001}.items()}}
            for fresh in (.004, .008, .004, .008)]
        report = describe(rows)
        self.assertAlmostEqual(report['secondary_fresh_minus_frozen_kl_nats']['mean'], 0.)
        low, high = report['secondary_fresh_minus_frozen_kl_nats']['paired_95_percentile_interval']
        self.assertLess(low, 0.)
        self.assertGreater(high, 0.)
        self.assertEqual(report['primary_sign_counts']['improved'], 4)
        self.assertEqual(report['numerical_k8v8_anchor']['frozen_at_or_below_anchor_questions'], 0)

    def test_near_zero_kl_clipping_is_disclosed_for_relative_summary(self):
        rows = [{'arms': {'keep': {'mean_kl_nats': .01},
                           'state_only_frozen_scales': {'mean_kl_nats': value},
                           'state_and_scales': {'mean_kl_nats': .005}}} for value in (0., .005)]
        self.assertEqual(describe(rows)['log_ratio_clipped_questions'], 1)


if __name__ == '__main__':
    unittest.main()
