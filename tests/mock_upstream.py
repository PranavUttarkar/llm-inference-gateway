"""Local deterministic streaming upstream; never use as a model benchmark."""
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

class Mock(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        time.sleep(float(body.get("mock_delay", 0)))
        for word in ("Hello", " world"):
            event = {"choices": [{"delta": {"content": word}}]}
            self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
            self.wfile.flush()
        if body.get("stream_options", {}).get("include_usage"):
            usage = {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}}
            self.wfile.write(("data: " + json.dumps(usage) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

if __name__ == "__main__":
    ThreadingHTTPServer((os.getenv("HOST", "127.0.0.1"), 9000), Mock).serve_forever()
