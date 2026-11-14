import copy
import unittest

from research.cache_repair.numerical_gate import evaluate


def row(i, benefit=.004, repeat=0., geometry=0.):
    return {'question_id': str(i), 'frozen_scale_self_reencode_max_abs': 0,
            'arms': {name: {'mean_kl_nats': value} for name, value in {
                'keep': .01, 'state_only_frozen_scales': .01-benefit,
                'uncompressed_repeat_control': 0.,
                'uncompressed_stream_rebuild_repeat': repeat,
                'uncompressed_full_prefill_geometry': geometry}.items()},
            'independent_prefill_repeat': {'mean_kl_nats': repeat}}


class NumericalGateTests(unittest.TestCase):
    def test_independent_replay_noise_can_invalidate_apparent_benefit(self):
        self.assertEqual(evaluate([row(i, repeat=1e-4) for i in range(8)])['decision'],
                         'invalid_repeat_control_stop_and_investigate')

    def test_geometry_floor_prevents_promoting_small_improvement(self):
        self.assertEqual(evaluate([row(i, geometry=.001) for i in range(8)])['decision'],
                         'inconclusive_no_automatic_expansion')

    def test_positive_and_negative_outcomes_and_no_seed_pseudoreplication(self):
        for effect, decision in ((.004, 'numerical_mechanism_continue_to_design_next_study'),
                                 (0., 'no_material_numerical_benefit_stop')):
            self.assertEqual(evaluate([row(i, benefit=effect) for i in range(8)])['decision'], decision)
        with self.assertRaises(ValueError):
            evaluate([row(0), row(0)])

    def test_real_transformer_independent_rebuilds_and_probe_inputs(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from research.cache_repair.mechanistic_pilot import teacher_force, probe, kl_positions
        torch.manual_seed(3)
        config = Qwen2Config(vocab_size=41, hidden_size=32, intermediate_size=48,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=8, attention_dropout=0.)
        config._attn_implementation = 'eager'
        model = Qwen2ForCausalLM(config).eval()
        with torch.inference_mode():
            a, _, _ = teacher_force(model, [1, 2, 3, 4, 5, 6], 2)
            b, _, _ = teacher_force(model, [1, 2, 3, 4, 5, 6], 2)
            self.assertTrue(all(torch.equal(x, y) for p, q in zip(a, b) for x, y in zip(p, q)))
            self.assertEqual(kl_positions(probe(model, a, [7, 8]), probe(model, b, [7, 8])).abs().max().item(), 0.)
            self.assertEqual(a[0][0].shape[-2], 6)  # probes must not mutate frozen prefixes


if __name__ == '__main__':
    unittest.main()
