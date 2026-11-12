import contextlib
import io
import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from oneharness.errors import ProviderError
from oneharness.transport import _read_events, post_chunks, post_json, post_sse


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


class RawServer:
    """A socket that answers with whatever raw bytes it is told, then closes.

    `reset=True` sets SO_LINGER to 0 so `close()` sends a TCP RST instead of a
    FIN. That difference is not cosmetic: an orderly FIN ends a chunked body and
    the parser shrugs, while an RST raises `ConnectionResetError` out of
    `socket.recv_into`. The bug this file now guards against only showed up with
    the reset.
    """

    def __init__(self, respond, *, reset=True):
        import socket

        self.socket = socket
        self.respond = respond
        self.reset = reset
        self.hits = 0
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(4)
        self.url = f"http://127.0.0.1:{self.listener.getsockname()[1]}/v1/chat"

    def __enter__(self):
        threading.Thread(target=self._serve, daemon=True).start()
        return self

    def __exit__(self, *exc):
        with contextlib.suppress(OSError):
            self.listener.close()

    def _serve(self):
        while True:
            try:
                conn, _ = self.listener.accept()
            except OSError:
                return
            self.hits += 1
            with contextlib.suppress(OSError):
                conn.recv(65536)
                self.respond(conn)
                if self.reset:
                    conn.setsockopt(
                        self.socket.SOL_SOCKET,
                        self.socket.SO_LINGER,
                        b"\x01\x00\x00\x00\x00\x00\x00\x00",
                    )
                conn.close()


def chunk(payload):
    return b"%x\r\n%s\r\n" % (len(payload), payload)


def openai_delta(text):
    return chunk(json.dumps({"choices": [{"index": 0, "delta": {"content": text}}]}).encode())


class ConnectionLossTests(unittest.TestCase):
    """A dying socket must never reach the caller as a socket error.

    Every layer above the transport catches `OneHarnessError`. `ConnectionReset
    Error` inherits from OSError, so it sails past all of them: the CLI printed a
    traceback, `doctor` never classified it, and the half answer that had already
    arrived was thrown away -- which is precisely the promise v0.8 made.
    """

    def test_json_before_any_answer_is_a_provider_error(self):
        with RawServer(lambda conn: None) as server, self.assertRaises(ProviderError) as ctx:
            post_json(server.url, {}, retries=0)
        self.assertIsInstance(ctx.exception, ProviderError)
        self.assertIn("lost the connection", str(ctx.exception))
        self.assertIsNone(ctx.exception.status)  # not a rejection the provider sent

    def test_json_body_cut_short_says_so(self):
        def respond(conn):
            conn.sendall(
                b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 5000\r\n\r\n"
            )
            conn.sendall(b'{"partial": "yes"}')

        # Closed with a FIN, not an RST. A reset is genuinely racy here: the
        # kernel discards whatever is still buffered, so if the RST wins the
        # race the client never sees the headers at all and honestly reports
        # "before any answer". This test is about the short body, so it needs a
        # close the client is guaranteed to observe the headers through.
        with RawServer(respond, reset=False) as server, self.assertRaises(ProviderError) as ctx:
            post_json(server.url, {}, retries=0)
        self.assertIn("while reading the response", str(ctx.exception))

    def test_sse_reset_before_any_event_is_a_provider_error(self):
        with RawServer(lambda conn: None) as server, self.assertRaises(ProviderError) as ctx:
            list(post_sse(server.url, {}, retries=0))
        self.assertIn("before any answer", str(ctx.exception))

    def test_chunks_body_cut_short_is_a_provider_error(self):
        def respond(conn):
            conn.sendall(
                b"HTTP/1.1 200 OK\r\ncontent-type: application/octet-stream\r\n"
                b"transfer-encoding: chunked\r\n\r\n"
            )
            conn.sendall(chunk(b"0123456789"))

        with RawServer(respond, reset=False) as server, self.assertRaises(ProviderError) as ctx:
            list(post_chunks(server.url, {}, retries=0))
        self.assertIn("while reading the response", str(ctx.exception))

    def test_a_lost_connection_is_retried(self):
        # Losing a connection before the answer is transient; treating it as
        # fatal turns one reset into a failed run.
        def respond(conn):
            body = b'{"ok": true}'
            conn.sendall(
                b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n"
                b"content-length: %d\r\n\r\n%s" % (len(body), body)
            )

        with RawServer(respond) as server:
            served = {"n": 0}

            def flaky(conn):
                served["n"] += 1
                if served["n"] == 1:
                    return  # hang up with nothing
                respond(conn)

            server.respond = flaky
            self.assertEqual(post_json(server.url, {}, retries=1, backoff=0.01), {"ok": True})
        self.assertEqual(server.hits, 2)


class RetryAfterTests(unittest.TestCase):
    """When a provider says how long to wait, waiting is not optional."""

    def _server(self, header_value):
        served = {"n": 0}

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                self.rfile.read(int(self.headers.get("content-length", 0)))
                served["n"] += 1
                if served["n"] == 1:
                    body = b'{"error":"slow down"}'
                    self.send_response(429)
                    if header_value is not None:
                        self.send_header("retry-after", header_value)
                    self.send_header("content-length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                body = b'{"ok": true}'
                self.send_response(200)
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        httpd = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        return f"http://127.0.0.1:{httpd.server_port}/x", served

    def test_seconds_form_is_honoured(self):
        url, served = self._server("1")
        started = time.monotonic()
        self.assertEqual(post_json(url, {}, retries=1, backoff=0.01), {"ok": True})
        self.assertGreaterEqual(time.monotonic() - started, 0.95)
        self.assertEqual(served["n"], 2)

    def test_http_date_form_is_honoured(self):
        # The form people forget: read as a number it would retry immediately.
        from email.utils import formatdate

        # Two seconds out, not one: `formatdate` truncates to whole seconds, so a
        # date built from `time.time() + 1` can round down to a wait of nearly
        # zero. That is a flaky test, not a broken parser.
        url, _ = self._server(formatdate(time.time() + 2, usegmt=True))
        started = time.monotonic()
        self.assertEqual(post_json(url, {}, retries=1, backoff=0.01), {"ok": True})
        # Well above the 0.01s backoff, and below the 2s asked for: proof the
        # header was read as a date rather than parsed as a number or ignored.
        waited = time.monotonic() - started
        self.assertGreater(waited, 0.9)
        self.assertLess(waited, 2.5)

    def test_absurd_wait_is_capped(self):
        # "come back in an hour" must not look like a frozen CLI.
        from oneharness.transport import MAX_RETRY_DELAY, _retry_after

        headers = {"retry-after": "3600"}
        self.assertEqual(min(_retry_after(headers), MAX_RETRY_DELAY), MAX_RETRY_DELAY)

    def test_no_header_falls_back_to_backoff(self):
        from oneharness.transport import _retry_after

        self.assertIsNone(_retry_after({}))
        self.assertIsNone(_retry_after(None))
        self.assertIsNone(_retry_after({"retry-after": "not-a-number"}))


class RetryNoticeTests(unittest.TestCase):
    """A wait that is not announced is indistinguishable from a hang."""

    def tearDown(self):
        from oneharness.transport import watch_retries

        watch_retries(None)

    def test_the_wait_is_announced_before_it_is_taken(self):
        from oneharness.transport import watch_retries

        seen = []
        watch_retries(seen.append)
        with FlakyServer([503]) as server:
            post_json(server.url, {}, retries=1, backoff=0.05)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["type"], "provider_retry")
        self.assertEqual(seen[0]["status"], 503)
        self.assertEqual(seen[0]["attempt"], 1)
        self.assertEqual(seen[0]["retries"], 1)

    def test_retry_after_is_named_as_the_source(self):
        from oneharness.transport import watch_retries

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            hits = 0

            def do_POST(self):
                self.rfile.read(int(self.headers.get("content-length", 0)))
                Handler.hits += 1
                status = 429 if Handler.hits == 1 else 200
                body = b'{"ok": true}' if status == 200 else b'{"error":"later"}'
                self.send_response(status)
                if status == 429:
                    self.send_header("retry-after", "0")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        httpd = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)

        seen = []
        watch_retries(seen.append)
        post_json(f"http://127.0.0.1:{httpd.server_port}/x", {}, retries=1, backoff=5)
        self.assertEqual(seen[0]["source"], "Retry-After")
        self.assertEqual(seen[0]["delay"], 0.0)
