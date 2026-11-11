"""Streaming chat proxy; optional local tokenizer support, otherwise standard library."""
import json
import http.client
import math
import os
import socket
import threading
import time
from urllib.parse import urlsplit
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.getenv("UPSTREAM_URL", "http://127.0.0.1:9000").rstrip("/")
MAX_ACTIVE = int(os.getenv("MAX_ACTIVE", "4"))
MAX_BODY = int(os.getenv("MAX_BODY_BYTES", "65536"))
TIMEOUT = float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "30"))
CONNECTION_TIMEOUT = float(os.getenv("CONNECTION_TIMEOUT_SECONDS", "10"))
REQUEST_DEADLINE = float(os.getenv("REQUEST_DEADLINE_SECONDS", "120"))
MAX_CONNECTIONS = int(os.getenv("MAX_CONNECTIONS", "64"))
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "0"))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "2048"))
UPSTREAMS = [u.strip().rstrip('/') for u in os.getenv('UPSTREAM_URLS', '').split(',') if u.strip()]
tokenizer = None
upstream_index = 0
slots = threading.BoundedSemaphore(MAX_ACTIVE)
lock = threading.Lock()
stats = {"active": 0, "accepted_total": 0, "rejected_total": 0, "upstream_errors_total": 0,
         "deadline_total": 0, "connection_rejected_total": 0}


def validate_tokens(obj):
    if tokenizer is None:
        return
    allowed = {'model', 'messages', 'stream', 'max_tokens', 'temperature', 'top_p',
               'presence_penalty', 'frequency_penalty', 'stop', 'seed', 'stream_options'}
    if set(obj) - allowed:
        raise ValueError('token limits support plain text chat only')
    messages = obj['messages']
    if any(not isinstance(m, dict) or m.get('role') not in ('system', 'user', 'assistant')
           or not isinstance(m.get('content'), str) or set(m) != {'role', 'content'} for m in messages):
        raise ValueError('token limits require plain text role/content messages')
    maximum = obj.get('max_tokens')
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ValueError('token limits require a positive integer max_tokens')
    count = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
    if count > MAX_INPUT_TOKENS or count + maximum > MAX_CONTEXT_TOKENS:
        raise OverflowError('input or requested context token limit exceeded')


class GatewayServer(ThreadingHTTPServer):
    """Bound handler threads before parsing HTTP; overflow closes the TCP socket."""
    daemon_threads = True

    def __init__(self, *args, **kwargs):
        self.connections = threading.BoundedSemaphore(MAX_CONNECTIONS)
        super().__init__(*args, **kwargs)

    def process_request(self, request, address):
        if not self.connections.acquire(blocking=False):
            with lock: stats['connection_rejected_total'] += 1
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except BaseException:
            self.connections.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.connections.release()

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def handle(self):
        self.connection.settimeout(CONNECTION_TIMEOUT)
        self.upstream_socket = None
        self.expired = threading.Event()
        self.deadline = time.monotonic() + REQUEST_DEADLINE
        timer = threading.Timer(REQUEST_DEADLINE, self.expire)
        timer.daemon = True
        timer.start()
        try:
            super().handle()
        except OSError:
            # Deadlines and peer disconnects close partially parsed requests.
            self.close_connection = True
        finally:
            timer.cancel()

    def expire(self):
        self.expired.set()
        with lock: stats['deadline_total'] += 1
        for sock in (self.connection, self.upstream_socket):
            if sock is not None:
                try: sock.shutdown(socket.SHUT_RDWR)
                except OSError: pass

    def end_headers(self):
        self.close_connection = True
        self.send_header('Connection', 'close')
        super().end_headers()

    def reply(self, code, value):
        data = json.dumps(value).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/healthz":
            return self.reply(200, {"status": "ok"})
        if self.path == "/metrics":
            with lock:
                data = "".join(f"gateway_{k} {v}\n" for k, v in stats.items()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.reply(404, {"error": "not found"})

    def do_POST(self):
        if self.headers.get('Transfer-Encoding') or len(self.headers.get_all('Content-Length', [])) != 1:
            return self.reply(400, {'error': 'one Content-Length and no Transfer-Encoding required'})
        if self.path != "/v1/chat/completions":
            self.close_connection = True
            return self.reply(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            length = -1
        if length < 1 or length > MAX_BODY:
            self.close_connection = True
            return self.reply(413, {"error": "invalid or oversized request body"})
        try:
            body = self.rfile.read(length)
            obj = json.loads(body)
            if not isinstance(obj, dict) or not isinstance(obj.get("messages"), list) or not obj["messages"]:
                raise ValueError()
        except (ValueError, UnicodeDecodeError):
            return self.reply(400, {"error": "expected JSON with nonempty messages array"})
        if not slots.acquire(blocking=False):
            with lock: stats["rejected_total"] += 1
            return self.reply(429, {"error": "gateway capacity exceeded"})
        with lock:
            stats["active"] += 1
            stats["accepted_total"] += 1
        try:
            try:
                validate_tokens(obj)
            except OverflowError as exc:
                return self.reply(413, {'error': str(exc)})
            except (ValueError, TypeError) as exc:
                return self.reply(400, {'error': str(exc)})
            global upstream_index
            with lock:
                target = UPSTREAMS[upstream_index % len(UPSTREAMS)] if UPSTREAMS else UPSTREAM
                upstream_index += 1
            parsed = urlsplit(target)
            if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.query or parsed.fragment:
                raise ValueError('invalid upstream URL')
            remaining = self.deadline - time.monotonic()
            if self.expired.is_set() or remaining <= 0:
                raise TimeoutError()
            connection_type = http.client.HTTPSConnection if parsed.scheme == 'https' else http.client.HTTPConnection
            conn = connection_type(parsed.hostname, parsed.port, timeout=min(TIMEOUT, remaining))
            try:
                conn.connect()
                self.upstream_socket = conn.sock
                if self.expired.is_set():
                    raise TimeoutError()
                conn.request('POST', parsed.path.rstrip('/') + self.path, body, {'Content-Type': 'application/json'})
                response = conn.getresponse()
                self.forward(response)
            finally:
                conn.close()
                self.upstream_socket = None
        except (OSError, TimeoutError, ValueError, http.client.HTTPException):
            with lock: stats["upstream_errors_total"] += 1
            self.close_connection = True
            if not getattr(self, 'response_started', False) and not self.expired.is_set():
                try: self.reply(502, {"error": "upstream unavailable or timed out"})
                except OSError: pass
        finally:
            with lock: stats["active"] -= 1
            slots.release()

    def forward(self, response):
        with response:
            self.send_response(response.status)
            self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
            self.end_headers()
            self.response_started = True
            while True:
                chunk = response.read1(4096)
                if not chunk: break
                self.wfile.write(chunk)
                self.wfile.flush()

if __name__ == "__main__":
    if (min(MAX_ACTIVE, MAX_BODY, MAX_CONNECTIONS, MAX_CONTEXT_TOKENS) <= 0 or MAX_INPUT_TOKENS < 0
            or not all(math.isfinite(v) and v > 0 for v in (TIMEOUT, CONNECTION_TIMEOUT, REQUEST_DEADLINE))):
        raise ValueError('gateway limits must be positive (MAX_INPUT_TOKENS may be zero to disable)')
    if MAX_INPUT_TOKENS:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(os.environ['TOKENIZER_PATH'], local_files_only=True, trust_remote_code=False)
    GatewayServer((os.getenv("HOST", "127.0.0.1"), int(os.getenv("PORT", "8080"))), Handler).serve_forever()
