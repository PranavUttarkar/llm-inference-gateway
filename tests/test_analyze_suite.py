import json
import tempfile
import unittest
from pathlib import Path
from benchmarks.analyze_suite import analyze


class SuiteAnalysisTests(unittest.TestCase):
    def test_trial_percentiles_stay_separate_and_offered_denominator_includes_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            manifest = {'executed': True, 'workload_sha256': 'same', 'parameters': {'pattern': 'burst', 'duration': 10},
                        'trials': [{'label': 'a', 'cap': 16, 'rate': 4, 'repeat': 1, 'status': 'complete'},
                                   {'label': 'b', 'cap': 16, 'rate': 4, 'repeat': 2, 'status': 'complete'}]}
            (out / 'manifest.json').write_text(json.dumps(manifest))
            for label, p95 in [('a', 1), ('b', 9)]:
                summary = {'workload_sha256': 'same', 'parameters': {'label': label, 'rate': 4, 'pattern': 'burst', 'duration': 10},
                           'scheduled_requests': 40, 'results': {'successes': 20, 'output_tokens_per_second': 12},
                           'schedule_to_first_content_seconds': {'p95': .1}, 'schedule_to_completion_seconds': {'p95': p95},
                           'illustrative_slo': {'successful_meeting_both': 10, 'goodput_per_full_run_second': 1}}
                (out / (label + '.jsonl.summary.json')).write_text(json.dumps(summary))
            metrics = analyze(out)['cases']['cap16-rate4']['across_trial_metrics']
            self.assertEqual(metrics['p95_completion_seconds'], {'trials_with_value': 2, 'median': 5, 'min': 1, 'max': 9})
            self.assertEqual(metrics['slo_fraction_of_offered']['median'], .25)
            summary['workload_sha256'] = 'different'
            (out / 'b.jsonl.summary.json').write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                analyze(out)

    def test_plan_is_not_treated_as_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'manifest.json').write_text(json.dumps({'executed': False, 'trials': []}))
            with self.assertRaises(ValueError):
                analyze(directory)
