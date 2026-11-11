import json
import socket
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from gateway import server
from tests.mock_upstream import Mock
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler

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
        cls.proxy = server.GatewayServer(("127.0.0.1", 0), server.Handler)
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

    def test_round_robin_reaches_both_backends(self):
        seen = []
        class Marked(Mock):
            def do_POST(self):
                seen.append(self.server.server_port)
                super().do_POST()
        other = ThreadingHTTPServer(('127.0.0.1', 0), Marked)
        threading.Thread(target=other.serve_forever, daemon=True).start()
        try:
            with patch.object(server, 'UPSTREAMS', [server.UPSTREAM, f'http://127.0.0.1:{other.server_port}']):
                for _ in range(4):
                    self.assertEqual(post(self.url, {'messages': [1]})[0], 200)
            self.assertEqual(len(seen), 2)
        finally:
            other.shutdown()
            other.server_close()

    def test_deadline_also_covers_incomplete_body(self):
        with patch.object(server, 'REQUEST_DEADLINE', .2):
            with socket.create_connection(self.proxy.server_address, timeout=2) as client:
                client.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 100\r\n\r\n{')
                # A deadline may close silently or race with a 400 for partial JSON.
                chunks = []
                while True:
                    try:
                        chunk = client.recv(4096)
                    except ConnectionResetError:
                        break
                    if not chunk:
                        break
                    chunks.append(chunk)
        self.assertEqual(post(self.url, {'messages': [1]})[0], 200)

    def test_token_limits_and_recovery(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return list(range(len(messages[0]['content']) + 3))
        with patch.object(server, 'tokenizer', Tokenizer()), patch.object(server, 'MAX_INPUT_TOKENS', 8), patch.object(server, 'MAX_CONTEXT_TOKENS', 10):
            def data(text, maximum=2):
                return {'messages': [{'role': 'user', 'content': text}], 'max_tokens': maximum}
            self.assertEqual(post(self.url, data('12345'))[0], 200)
            self.assertEqual(post(self.url, data('123456'))[0], 413)
            self.assertEqual(post(self.url, data('12345', 3))[0], 413)
            self.assertEqual(post(self.url, data('hi', True))[0], 400)
            self.assertEqual(post(self.url, dict(data('hi'), tools=[]))[0], 400)
            self.assertEqual(post(self.url, data('hi'))[0], 200)

    def test_deadline_stops_dripping_stream_and_releases_capacity(self):
        class Drip(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                self.send_response(200)
                self.end_headers()
                try:
                    for _ in range(100):
                        self.wfile.write(b'data: {}\n\n')
                        self.wfile.flush()
                        time.sleep(.03)
                except OSError:
                    pass
        upstream = ThreadingHTTPServer(('127.0.0.1', 0), Drip)
        upstream.daemon_threads = True
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        try:
            before = server.stats['deadline_total']
            with patch.object(server, 'REQUEST_DEADLINE', .2), patch.object(server, 'UPSTREAM', f'http://127.0.0.1:{upstream.server_port}'):
                start = time.monotonic()
                code, body = post(self.url, {'messages': [1]})
                self.assertEqual(code, 200)
                self.assertIn(b'data:', body)
                self.assertLess(time.monotonic() - start, 1.5)
            limit = time.monotonic() + 1
            while server.stats['active'] and time.monotonic() < limit:
                time.sleep(.01)
            self.assertEqual(server.stats['active'], 0)
            self.assertEqual(server.stats['deadline_total'], before + 1)
            self.assertEqual(post(self.url, {'messages': [1]})[0], 200)
        finally:
            upstream.shutdown()
            upstream.server_close()

    def test_connection_limit_closes_overflow_and_recovers(self):
        with patch.object(server, 'MAX_CONNECTIONS', 1), patch.object(server, 'REQUEST_DEADLINE', 3):
            proxy = server.GatewayServer(('127.0.0.1', 0), server.Handler)
            threading.Thread(target=proxy.serve_forever, daemon=True).start()
            first = socket.create_connection(proxy.server_address, timeout=2)
            try:
                first.sendall(b'POST /v1/chat/completions HTTP/1.1\r\n')
                limit = time.monotonic() + 1
                while time.monotonic() < limit:
                    if not proxy.connections.acquire(blocking=False):
                        break
                    proxy.connections.release()
                    time.sleep(.01)
                with socket.create_connection(proxy.server_address, timeout=2) as second:
                    try:
                        self.assertEqual(second.recv(1), b'')
                    except ConnectionResetError:
                        pass
                first.close()
                limit = time.monotonic() + 1
                while time.monotonic() < limit:
                    if proxy.connections.acquire(blocking=False):
                        proxy.connections.release()
                        break
                    time.sleep(.01)
                # Use a normal deadline for the recovery request, independent of
                # the deliberately short deadline of the stalled connection.
                with patch.object(server, 'REQUEST_DEADLINE', 3):
                    with urllib.request.urlopen(f'http://127.0.0.1:{proxy.server_port}/healthz', timeout=2) as response:
                        self.assertEqual(response.status, 200)
            finally:
                first.close()
                proxy.shutdown()
                proxy.server_close()

if __name__ == "__main__": unittest.main()
