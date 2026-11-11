import unittest
import contextlib
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
from benchmarks.run_suite import trial_plan, main


class SuiteTests(unittest.TestCase):
    def test_matched_repeated_cases_and_reproducible_order(self):
        trials = trial_plan(3, 42)
        self.assertEqual(trials, trial_plan(3, 42))
        self.assertNotEqual(trials, trial_plan(3, 43))
        self.assertEqual(len({t['label'] for t in trials}), 9)
        for repeat in (1, 2, 3):
            self.assertEqual({(t['cap'], t['rate']) for t in trials if t['repeat'] == repeat},
                             {(32, 4), (16, 2), (16, 4)})

    def test_requires_repetitions(self):
        with self.assertRaises(ValueError):
            trial_plan(1, 42)

    def test_execution_failure_preserves_manifest_and_stops_remaining_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'failed-suite'
            command = ['run_suite', '--output', str(out), '--repeats', '2', '--execute']
            with patch.object(sys, 'argv', command), patch('benchmarks.run_suite.subprocess.run',
                    side_effect=subprocess.CalledProcessError(1, ['docker'])) as run, contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(subprocess.CalledProcessError):
                    main()
            manifest = json.loads((out / 'manifest.json').read_text())
            self.assertEqual(run.call_count, 1)
            self.assertEqual(manifest['trials'][0]['status'], 'failed')
            self.assertTrue(all('status' not in t for t in manifest['trials'][1:]))
            self.assertTrue(manifest['source_sha256'])
