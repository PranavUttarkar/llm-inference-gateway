import unittest
from benchmarks.compare import compare


def run(concurrency=1, workload='abc', rate=2):
    return {'workload_sha256': workload,
            'parameters': {'model': 'qwen', 'concurrency': concurrency, 'requests': 10,
                           'warmup': 2, 'max_tokens': 128, 'include_usage': True, 'timeout': 60, 'label': ''},
            'results': {'failures': 0, 'successful_requests_per_second': rate,
                        'output_tokens_per_second': None, 'ttft_seconds': {}, 'elapsed_seconds': {}}}


class ComparisonTests(unittest.TestCase):
    def test_comparison(self):
        result = compare(run(), run(rate=3))
        self.assertEqual(result['request_throughput_ratio_b_over_a'], 1.5)
        self.assertIsNone(result['output_token_throughput_ratio_b_over_a'])

    def test_mismatch_refused(self):
        with self.assertRaisesRegex(ValueError, 'concurrency'):
            compare(run(), run(concurrency=2))
        with self.assertRaisesRegex(ValueError, 'workload'):
            compare(run(), run(workload='other'))
