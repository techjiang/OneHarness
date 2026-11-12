import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from oneharness import doctor
from oneharness.errors import ProviderError
from oneharness.providers.openai import OpenAIProvider
from oneharness.tape import Tape, TapeError
from oneharness.types import Message
from tests.sse_server import SSERecorder

TAPES = Path(__file__).parent / "fixtures" / "tapes"


class JSONServer:
    """Answers every POST with one fixed JSON body, the way `complete()` expects."""

    def __init__(self, body, status=200):
        payload = json.dumps(body).encode()
        server = self
        self.requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("content-length", 0))
                server.requests.append(json.loads(self.rfile.read(length) or b"{}"))
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

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


def reply(text, finish="stop"):
    return {
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        "model": "gpt",
    }


def openai_chunk(content=None, finish=None):
    delta = {"content": content} if content is not None else {}
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


class RecordTests(unittest.TestCase):
    def test_records_a_successful_request_without_headers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.json"
            with JSONServer(reply("hi")) as server, Tape.record(path) as tape:
                provider = OpenAIProvider("gpt", api_key="sk-secret", base_url=server.url)
                provider.complete([Message.user("ping")])
            self.assertEqual(tape.recorded, 1)
            text = path.read_text()
            self.assertNotIn("sk-secret", text, "a tape must never contain the API key")
            document = json.loads(text)
            self.assertEqual(document["version"], 1)
            self.assertEqual(document["entries"][0]["response"]["status"], 200)

    def test_records_the_raw_bytes_not_the_parsed_events(self):
        # The point of a tape is to re-exercise the parser, so what is stored
        # has to be the wire bytes rather than decoded JSON.
        events = [(None, openai_chunk("he")), (None, openai_chunk("llo", "stop"))]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.json"
            with SSERecorder(events) as server, Tape.record(path) as tape:
                provider = OpenAIProvider("gpt", api_key="sk", base_url=server.url)
                list(provider.stream([Message.user("ping")]))
            self.assertEqual(tape.recorded, 1)

            import base64

            body = base64.b64decode(tape.entries[0]["response"]["body_b64"])
            self.assertTrue(body.startswith(b"data: "), f"expected SSE text, got {body[:40]!r}")

    def test_records_a_rejection_with_its_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.json"
            with SSERecorder([], status=401) as server, Tape.record(path) as tape:
                provider = OpenAIProvider("gpt", api_key="sk", base_url=server.url)
                with self.assertRaises(ProviderError):
                    provider.complete(
                        [Message.user("ping")],
                    )
            self.assertEqual(tape.entries[0]["response"]["status"], 401)


class ReplayTests(unittest.TestCase):
    def test_replay_answers_without_a_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.json"
            with JSONServer(reply("hello")) as server, Tape.record(path):
                OpenAIProvider("gpt", api_key="sk", base_url=server.url).complete(
                    [Message.user("ping")]
                )

            # The server is gone; only the tape is left, and it still answers.
            with Tape.replay(path):
                provider = OpenAIProvider("gpt", api_key="sk", base_url=server.url)
                completion = provider.complete([Message.user("ping")])
        self.assertEqual(completion.text, "hello")

    def test_replay_rebuilds_streamed_text(self):
        events = [(None, openai_chunk("he")), (None, openai_chunk("llo", "stop"))]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.json"
            with SSERecorder(events) as server, Tape.record(path):
                provider = OpenAIProvider("gpt", api_key="sk", base_url=server.url)
                list(provider.stream([Message.user("ping")]))

            with Tape.replay(path):
                provider = OpenAIProvider("gpt", api_key="sk", base_url=server.url)
                out = list(provider.stream([Message.user("ping")]))
        self.assertEqual("".join(event.text for event in out if event.kind == "text"), "hello")

    def test_replayed_rejection_keeps_status_and_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.json"
            with SSERecorder([], status=401) as server, Tape.record(path):
                provider = OpenAIProvider("gpt", api_key="sk", base_url=server.url)
                with self.assertRaises(ProviderError):
                    provider.complete([Message.user("ping")])

            with Tape.replay(path), self.assertRaises(ProviderError) as ctx:
                OpenAIProvider("gpt", api_key="sk", base_url=server.url).complete(
                    [Message.user("ping")]
                )
        self.assertEqual(ctx.exception.status, 401)
        self.assertIn("boom", ctx.exception.body or "")

    def test_a_changed_request_is_a_loud_failure(self):
        # A permissive tape would keep every wire-format regression green, which
        # defeats the reason for having a fixture at all.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.json"
            with JSONServer(reply("hi")) as server, Tape.record(path):
                OpenAIProvider("gpt", api_key="sk", base_url=server.url).complete(
                    [Message.user("ping")]
                )

            with Tape.replay(path), self.assertRaises(TapeError) as ctx:
                OpenAIProvider("gpt", api_key="sk", base_url=server.url).complete(
                    [Message.user("a different prompt")]
                )
        message = str(ctx.exception)
        self.assertIn("messages[0].content", message)
        self.assertIn("re-record", message)

    def test_a_changed_url_is_a_loud_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.json"
            with JSONServer(reply("hi")) as server, Tape.record(path):
                OpenAIProvider("gpt", api_key="sk", base_url=server.url).complete(
                    [Message.user("ping")]
                )

            with Tape.replay(path), self.assertRaises(TapeError) as ctx:
                OpenAIProvider("gpt", api_key="sk", base_url=server.url + "/v2").complete(
                    [Message.user("ping")]
                )
        self.assertIn("recorded POST", str(ctx.exception))

    def test_running_out_of_entries_says_so(self):
        with Tape.replay(TAPES / "openai.json") as tape:
            provider = OpenAIProvider(
                "gpt-4o-mini", api_key="sk", base_url="https://api.openai.com/v1"
            )
            with self.assertRaises(ProviderError):
                provider.complete([Message.user("ping")])
            with self.assertRaises(TapeError) as ctx:
                provider.complete([Message.user("ping")])
        self.assertEqual(tape.matched, 1)
        self.assertIn("ran out", str(ctx.exception))

    def test_key_order_is_not_a_difference(self):
        tape = Tape(replaying=True)
        tape.entries = [{"url": "u", "request": {"a": 1, "b": 2}, "response": {"body_b64": ""}}]
        self.assertIsNotNone(tape.exchange("u", {"b": 2, "a": 1}))

    def test_missing_and_broken_files_are_reported(self):
        with self.assertRaises(TapeError):
            Tape.replay("/nonexistent/tape.json")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{not json")
            with self.assertRaises(TapeError):
                Tape.replay(path)
            path.write_text('{"version": 99, "entries": []}')
            with self.assertRaises(TapeError) as ctx:
                Tape.replay(path)
        self.assertIn("version", str(ctx.exception))


class RealTapeTests(unittest.TestCase):
    """Replays of bytes captured from the real endpoints, not from a fake server.

    Each file holds the response the provider actually sent to an invalid key.
    An `auth` verdict proves the request body was shaped the way that provider
    wants: it got past the fields and stopped at the credential. That is the
    property none of the fake-server tests can establish.
    """

    def _replay(self, name: str, preset: str):
        with Tape.replay(TAPES / f"{name}.json"):
            self.assertEqual(doctor.check(preset).status, doctor.AUTH)

    def test_openai(self):
        self._replay("openai", "openai")

    def test_anthropic(self):
        self._replay("anthropic", "anthropic")

    def test_gemini(self):
        self._replay("gemini", "gemini")

    def test_deepseek(self):
        self._replay("deepseek", "deepseek")

    def test_groq(self):
        self._replay("groq", "groq")

    def test_bedrock_signs_and_is_refused_on_the_credential(self):
        # The 403 comes from a request AWS parsed, signature and all; a signing
        # bug would surface as 400 `InvalidSignatureException` instead.
        with Tape.replay(TAPES / "bedrock.json"):
            probe = doctor.check("bedrock")
        self.assertEqual(probe.status, doctor.AUTH)
        self.assertIn("security token", probe.detail)

    def test_a_wrong_wire_format_is_not_blamed_on_the_key(self):
        """The captured body is what a real Gemini says to an OpenAI-shaped request.

        It offers a field-name complaint and a credential-shaped 400 from the same
        endpoint, which is exactly the confusion `doctor` exists to prevent.
        """
        from oneharness.transport import post_json

        with Tape.replay(TAPES / "protocol-bug.json"), self.assertRaises(ProviderError) as ctx:
            post_json(
                "https://generativelanguage.googleapis.com/v1beta/models/"
                "gemini-2.0-flash:generateContent",
                {
                    "model": "gemini-2.0-flash",
                    "messages": [{"role": "user", "content": "ping"}],
                    "temperature": 0.0,
                },
                {"x-goog-api-key": "invalid-oneharness-key"},
                retries=0,
            )
        status, detail = doctor.classify(ctx.exception)
        self.assertEqual(status, doctor.PROTOCOL)
        self.assertIn("Unknown name", detail)
        # The honest comparison: the correct body got `auth` from the same host.
        with Tape.replay(TAPES / "gemini.json"):
            self.assertEqual(doctor.check("gemini").status, doctor.AUTH)

    def test_replay_needs_no_credentials_at_all(self):
        saved = {
            key: os.environ.pop(key)
            for key in list(os.environ)
            if key.startswith(("OPENAI_", "ANTHROPIC_", "GEMINI_", "AWS_"))
        }
        try:
            with Tape.replay(TAPES / "openai.json"):
                self.assertEqual(doctor.check("openai").status, doctor.AUTH)
            with Tape.replay(TAPES / "bedrock.json"):
                self.assertEqual(doctor.check("bedrock").status, doctor.AUTH)
        finally:
            os.environ.update(saved)


if __name__ == "__main__":
    unittest.main()
