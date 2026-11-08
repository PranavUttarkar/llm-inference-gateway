import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from gateway import server
from tests.mock_upstream import Mock

def post(url, data):
    request = urllib.request.Request(url + "/v1/chat/completions", json.dumps(data).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()

class GatewayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.up = ThreadingHTTPServer(("127.0.0.1", 0), Mock)
        cls.proxy = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        server.UPSTREAM = "http://127.0.0.1:%d" % cls.up.server_port
        server.slots = threading.BoundedSemaphore(1)
        cls.url = "http://127.0.0.1:%d" % cls.proxy.server_port
        for s in (cls.up, cls.proxy): threading.Thread(target=s.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        for s in (cls.up, cls.proxy): s.shutdown(); s.server_close()

    def test_stream_and_validation(self):
        code, body = post(self.url, {"messages": [{"role": "user", "content": "hi"}], "stream": True})
        self.assertEqual(code, 200)
        self.assertIn(b"[DONE]", body)
        self.assertEqual(post(self.url, {"messages": []})[0], 400)

    def test_overload_and_recovery(self):
        data = {"messages": [{"role": "user", "content": "hi"}], "mock_delay": 0.4}
        thread = threading.Thread(target=post, args=(self.url, data))
        thread.start()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with server.lock: active = server.stats["active"]
            if active: break
            time.sleep(0.01)
        self.assertEqual(active, 1)
        self.assertEqual(post(self.url, data)[0], 429)
        thread.join()
        self.assertEqual(post(self.url, {"messages": [1]})[0], 200)

    def test_upstream_failure_releases_slot(self):
        old = server.UPSTREAM
        server.UPSTREAM = "http://127.0.0.1:1"
        try: self.assertEqual(post(self.url, {"messages": [1]})[0], 502)
        finally: server.UPSTREAM = old
        self.assertEqual(post(self.url, {"messages": [1]})[0], 200)

if __name__ == "__main__": unittest.main()
