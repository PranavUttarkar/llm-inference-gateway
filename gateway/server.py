"""Small streaming chat proxy with bounded admission; standard library only."""
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.getenv("UPSTREAM_URL", "http://127.0.0.1:9000").rstrip("/")
MAX_ACTIVE = int(os.getenv("MAX_ACTIVE", "4"))
MAX_BODY = int(os.getenv("MAX_BODY_BYTES", "65536"))
TIMEOUT = float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "30"))
slots = threading.BoundedSemaphore(MAX_ACTIVE)
lock = threading.Lock()
stats = {"active": 0, "accepted_total": 0, "rejected_total": 0, "upstream_errors_total": 0}

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

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
        started_response = False
        try:
            req = urllib.request.Request(UPSTREAM + self.path, body, {"Content-Type": "application/json"}, method="POST")
            try:
                response = urllib.request.urlopen(req, timeout=TIMEOUT)
            except urllib.error.HTTPError as exc:
                response = exc
            with response:
                self.send_response(response.status)
                self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
                self.send_header("Connection", "close")
                self.end_headers()
                started_response = True
                self.close_connection = True
                while True:
                    chunk = response.read1(4096)
                    if not chunk: break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except (OSError, TimeoutError, urllib.error.URLError):
            with lock: stats["upstream_errors_total"] += 1
            self.close_connection = True
            if not started_response:
                try: self.reply(502, {"error": "upstream unavailable or timed out"})
                except OSError: pass
        finally:
            with lock: stats["active"] -= 1
            slots.release()

if __name__ == "__main__":
    ThreadingHTTPServer((os.getenv("HOST", "127.0.0.1"), int(os.getenv("PORT", "8080"))), Handler).serve_forever()
