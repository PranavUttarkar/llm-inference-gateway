import unittest
import contextlib
import io
import json
import sys
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch
from http.server import ThreadingHTTPServer
from benchmarks.run_arrival import arrival_offsets
from benchmarks.run_arrival import main
from tests.mock_upstream import Mock


class ArrivalTests(unittest.TestCase):
    def test_uniform_preserves_original_schedule(self):
        self.assertEqual(arrival_offsets(2, 2), [0, .5, 1, 1.5])

    def test_bursts_preserve_count_and_window(self):
        offsets = arrival_offsets(4, 5, 'burst', burst_size=8)
        self.assertEqual(offsets, [0] * 8 + [2] * 8 + [4] * 4)

    def test_poisson_is_seeded_and_in_window(self):
        a = arrival_offsets(4, 10, 'poisson', seed=42)
        self.assertEqual(a, arrival_offsets(4, 10, 'poisson', seed=42))
        self.assertNotEqual(a, arrival_offsets(4, 10, 'poisson', seed=43))
        self.assertTrue(all(0 <= x < 10 for x in a))
        self.assertTrue(all(b > a for a, b in zip(a, a[1:])))

    def test_invalid_schedule(self):
        for rate, duration in [(0, 1), (1, float('nan')), (float('inf'), 1)]:
            with self.assertRaises(ValueError):
                arrival_offsets(rate, duration)

    def test_patterns_write_complete_request_records_and_summaries(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), Mock)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                prompts = Path(directory) / 'prompts.json'
                prompts.write_text(json.dumps(['hello']))
                for pattern in ('uniform', 'burst', 'poisson'):
                    output = Path(directory) / (pattern + '.jsonl')
                    command = ['run_arrival', '--url', f'http://127.0.0.1:{server.server_port}/v1/chat/completions',
                               '--workload', str(prompts), '--warmup', '0', '--rate', '10', '--duration', '.3',
                               '--pattern', pattern, '--seed', '42', '--burst-size', '2',
                               '--label', pattern, '--output', str(output)]
                    with patch.object(sys, 'argv', command), contextlib.redirect_stdout(io.StringIO()):
                        main()
                    rows = [json.loads(s) for s in output.read_text().splitlines()]
                    summary = json.loads(Path(str(output) + '.summary.json').read_text())
                    expected = arrival_offsets(10, .3, pattern, 42, 2)
                    self.assertEqual(len(rows), len(expected))
                    self.assertEqual(summary['scheduled_requests'], len(expected))
                    self.assertEqual(summary['results']['successes'], len(expected))
                    self.assertEqual(summary['results']['completion_tokens_known'], len(expected) * 2)
                    self.assertGreaterEqual(summary['full_drain_seconds'], .3)
                    ordered = sorted(rows, key=lambda r: r['id'])
                    for row, offset in zip(ordered, expected):
                        self.assertAlmostEqual(row['scheduled_offset_seconds'], offset, places=5)
        finally:
            server.shutdown()
            server.server_close()
