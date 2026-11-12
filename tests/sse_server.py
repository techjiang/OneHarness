"""A tiny HTTP server that replays canned SSE chunks, shared by tests and demos."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class SSERecorder:
    def __init__(self, events, status=200):
        self.events = events
        self.status = status
        self.requests = []
        recorder = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                recorder.requests.append({"path": self.path, "body": body})
                self.send_response(recorder.status)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                if recorder.status != 200:
                    self.wfile.write(json.dumps({"error": "boom"}).encode())
                    return
                for event, data in recorder.events:
                    frame = ""
                    if event:
                        frame += f"event: {event}\n"
                    frame += f"data: {json.dumps(data)}\n\n"
                    self.wfile.write(frame.encode())
                self.wfile.flush()

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
