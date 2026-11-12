import contextlib
import io
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from oneharness.errors import ProviderError
from oneharness.transport import _read_events, post_chunks, post_json


class FlakyServer:
    """Returns the scripted statuses in order, then 200 with `{"ok": true}`."""

    def __init__(self, statuses, body=b'{"ok": true}', delay=0.0):
        self.statuses = list(statuses)
        self.body = body
        self.delay = delay
        self.hits = 0
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("content-length", 0)))
                if server.delay:
                    time.sleep(server.delay)
                server.hits += 1
                status = server.statuses.pop(0) if server.statuses else 200
                payload = server.body if status == 200 else b'{"error":"boom"}'
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                # The client may already have given up; that is the point of the
                # timeout test, not a server bug.
                with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                    self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1/chat"

    def __enter__(self):
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


class TransportTests(unittest.TestCase):
    def test_success(self):
        with FlakyServer([]) as server:
            self.assertEqual(post_json(server.url, {"a": 1}), {"ok": True})
        self.assertEqual(server.hits, 1)

    def test_retries_then_succeeds(self):
        with FlakyServer([500, 429]) as server:
            self.assertEqual(post_json(server.url, {}, retries=2, backoff=0.01), {"ok": True})
        self.assertEqual(server.hits, 3)

    def test_gives_up_after_retries(self):
        with FlakyServer([503, 503, 503]) as server, self.assertRaises(ProviderError) as ctx:
            post_json(server.url, {}, retries=2, backoff=0.01)
        self.assertEqual(ctx.exception.status, 503)
        self.assertEqual(server.hits, 3)

    def test_client_error_is_not_retried(self):
        with FlakyServer([400]) as server, self.assertRaises(ProviderError) as ctx:
            post_json(server.url, {}, retries=3, backoff=0.01)
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(server.hits, 1)

    def test_connection_refused(self):
        with self.assertRaises(ProviderError) as ctx:
            post_json("http://127.0.0.1:1/v1", {}, retries=0)
        self.assertIn("cannot reach", str(ctx.exception))

    def test_non_json_body(self):
        with (
            FlakyServer([], body=b"<html>nope</html>") as server,
            self.assertRaises(ProviderError) as ctx,
        ):
            post_json(server.url, {})
        self.assertIn("not JSON", str(ctx.exception))

    def test_json_array_rejected(self):
        with FlakyServer([], body=b"[1,2]") as server, self.assertRaises(ProviderError) as ctx:
            post_json(server.url, {})
        self.assertIn("JSON object", str(ctx.exception))

    def test_timeout_is_reported(self):
        with FlakyServer([], delay=0.5) as server, self.assertRaises(ProviderError) as ctx:
            post_json(server.url, {}, timeout=0.1, retries=0)
        self.assertIn("timed out", str(ctx.exception))


class SseParsingTests(unittest.TestCase):
    def parse(self, text):
        stream = io.BytesIO(text.encode("utf-8"))
        return list(_read_events(stream))

    def test_comments_and_blank_lines_are_skipped(self):
        events = self.parse(': keepalive\n\ndata: {"a":1}\n\n')
        self.assertEqual(events, [(None, {"a": 1})])

    def test_named_event(self):
        events = self.parse('event: ping\ndata: {"x":2}\n\n')
        self.assertEqual(events[0][0], "ping")

    def test_done_sentinel(self):
        events = self.parse('data: {"a":1}\n\ndata: [DONE]\n\n')
        self.assertEqual(events[-1], ("done", {}))

    def test_multiline_data_is_joined(self):
        events = self.parse('data: {"a":\ndata: 1}\n\n')
        self.assertEqual(events[0][1], {"a": 1})

    def test_trailing_frame_without_blank_line(self):
        events = self.parse('data: {"a":3}')
        self.assertEqual(events, [(None, {"a": 3})])

    def test_bad_json_raises(self):
        with self.assertRaises(ProviderError):
            self.parse("data: not json\n\n")


if __name__ == "__main__":
    unittest.main()


class StreamChunkTests(unittest.TestCase):
    """`post_chunks` must hand bytes over as they arrive, not at the end."""

    def _server(self, pieces, delay=0.05):
        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                self.rfile.read(int(self.headers.get("content-length", 0)))
                total = sum(len(piece) for piece in pieces)
                self.send_response(200)
                self.send_header("content-type", "application/octet-stream")
                self.send_header("content-length", str(total))
                self.end_headers()
                for piece in pieces:
                    self.wfile.write(piece)
                    self.wfile.flush()
                    time.sleep(delay)

            def log_message(self, *args):
                pass

        httpd = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        return f"http://127.0.0.1:{httpd.server_port}/stream"

    def test_chunks_arrive_before_the_stream_ends(self):
        # 3 x 40 bytes at 50ms apart. A buffered read would return all of them
        # after the last one; streaming means the first arrives while the server
        # is still working.
        url = self._server([b"x" * 40] * 3)
        seen = []
        for chunk in post_chunks(url, {}):
            seen.append((len(chunk), time.monotonic()))
        self.assertEqual(sum(size for size, _ in seen), 120)
        self.assertGreater(len(seen), 1, "the whole body came back in one read")

    def test_signer_runs_again_on_retry(self):
        # A signature has a validity window; replaying the one from the first
        # attempt is how a retry fails for a reason unrelated to the outage.
        attempts = []

        def signer(url, body, headers):
            signed = {**headers, "x-attempt": str(len(attempts) + 1)}
            attempts.append(signed)
            return signed

        url = self._flaky_then_stream()
        list(post_chunks(url, {}, sign=signer, retries=1, backoff=0.01))
        self.assertEqual([attempt["x-attempt"] for attempt in attempts], ["1", "2"])

    def _flaky_then_stream(self):
        remaining = {"n": 1}

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                self.rfile.read(int(self.headers.get("content-length", 0)))
                if remaining["n"]:
                    remaining["n"] = 0
                    body = b'{"error":"throttled"}'
                    self.send_response(429)
                    self.send_header("content-length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(200)
                self.send_header("content-length", "2")
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, *args):
                pass

        httpd = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        return f"http://127.0.0.1:{httpd.server_port}/stream"
