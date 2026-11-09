import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.analyze_32b import CASES, analyze


class LongRunAnalysisTests(unittest.TestCase):
    def write_fixture(self, directory):
        for label, cap, rate in CASES:
            (directory / (label + '-compose.yaml')).write_text(f'services:\n  gateway:\n    environment:\n      MAX_ACTIVE: "{cap}"\n')
            count = rate * 300
            rows = []
            for i in range(count):
                row = {'id': i, 'completed': i != 0, 'status': 429 if i == 0 else 200,
                       'scheduled_offset_seconds': i / rate, 'dispatch_lag_seconds': .001,
                       'schedule_to_first_content_seconds': None if i == 0 else .5,
                       'schedule_to_completion_seconds': .002 if i == 0 else 10,
                       'usage': {} if i == 0 else {'completion_tokens': 128}}
                rows.append(row)
            summary = {'parameters': {'rate': rate, 'duration': 300, 'max_tokens': 128,
                                      'ttft_slo_ms': 500, 'latency_slo_ms': 10000},
                       'workload_sha256': 'same-workload', 'started_at_utc': '2026-11-09T00:00:00+00:00',
                       'full_drain_seconds': 310, 'observed_dispatch_rate_in_window': rate,
                       'statuses': {'429': 1, '200': count - 1}}
            (directory / (label + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in rows))
            (directory / (label + '.jsonl.summary.json')).write_text(json.dumps(summary))

    def test_rejections_stay_in_offered_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.write_fixture(directory)
            case = analyze(directory)['cases'][0]
            self.assertEqual(case['within_targets'], 1199)
            self.assertAlmostEqual(case['fraction_of_offered_within_targets'], 1199 / 1200)
            self.assertEqual(case['fraction_of_successes_within_targets'], 1)
            self.assertAlmostEqual(case['goodput_per_full_run_second'], 1199 / 310)
            self.assertEqual(case['schedule_to_completion_seconds']['p95'], 10)

    def test_duplicate_requests_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.write_fixture(directory)
            path = directory / (CASES[0][0] + '.jsonl')
            rows = [json.loads(s) for s in path.read_text().splitlines()]
            rows[-1]['id'] = 0
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            with self.assertRaisesRegex(ValueError, 'missing or duplicate'):
                analyze(directory)
