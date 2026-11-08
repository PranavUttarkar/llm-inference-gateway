import argparse
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from benchmarks.run_load import percentile, request, summarize
from tests.mock_upstream import Mock


class BenchmarkTests(unittest.TestCase):
    def test_summary_handles_missing_usage(self):
        rows = [{'completed': True, 'ttft_seconds': 1, 'elapsed_seconds': 2, 'usage': {'completion_tokens': 4}},
                {'completed': True, 'ttft_seconds': 3, 'elapsed_seconds': 4},
                {'completed': False, 'ttft_seconds': None, 'elapsed_seconds': 1}]
        result = summarize(rows, 2)
        self.assertEqual(percentile([1, 3], .5), 2)
        self.assertEqual(result['ttft_seconds']['p50'], 2)
        self.assertEqual(result['failures'], 1)
        self.assertIsNone(result['output_tokens_per_second'])
        rows[1]['usage'] = {'completion_tokens': 2}
        self.assertEqual(summarize(rows, 2)['output_tokens_per_second'], 3)

    def test_mock_stream_and_usage(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), Mock)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            args = argparse.Namespace(concurrency=1, model='mock', prompt='test', max_tokens=10,
                                      include_usage=True, timeout=3,
                                      url=f'http://127.0.0.1:{server.server_port}/v1/chat/completions')
            row = request(0, args)
            self.assertTrue(row['completed'])
            self.assertEqual(row['output_text'], 'Hello world')
            self.assertEqual(row['usage']['completion_tokens'], 2)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
