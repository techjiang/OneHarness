import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from oneharness.errors import ProviderError
from oneharness.providers import create_provider
from oneharness.providers.gemini import GeminiProvider
from oneharness.types import Message, ToolCall


class Recorder:
    def __init__(self, reply, status=200):
        self.reply = reply
        self.status = status
        self.requests = []
        recorder = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("content-length", 0))
                recorder.requests.append(
                    {
                        "path": self.path,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "body": json.loads(self.rfile.read(length) or b"{}"),
                    }
                )
                payload = json.dumps(recorder.reply).encode()
                self.send_response(recorder.status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def reply(parts, finish="STOP"):
    return {
        "modelVersion": "gemini-test",
        "candidates": [{"content": {"parts": parts}, "finishReason": finish}],
        "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 4},
    }


class GeminiProviderTests(unittest.TestCase):
    def test_system_hoisted_and_text(self):
        with Recorder(reply([{"text": "hey"}])) as server:
            provider = GeminiProvider("gemini-test", api_key="k", base_url=server.url)
            completion = provider.complete([Message.system("be terse"), Message.user("hi")])

        self.assertEqual(completion.text, "hey")
        self.assertEqual(completion.usage.total_tokens, 12)
        self.assertEqual(completion.stop_reason, "STOP")
        request = server.requests[0]
        self.assertEqual(request["path"], "/models/gemini-test:generateContent")
        self.assertEqual(request["headers"]["x-goog-api-key"], "k")
        self.assertEqual(request["body"]["systemInstruction"]["parts"][0]["text"], "be terse")
        self.assertEqual(request["body"]["contents"], [{"role": "user", "parts": [{"text": "hi"}]}])

    def test_function_call_decoded_without_ids(self):
        with Recorder(
            reply([{"functionCall": {"name": "add", "args": {"a": 1, "b": 2}}}], "STOP")
        ) as server:
            provider = GeminiProvider("gemini-test", api_key="k", base_url=server.url)
            completion = provider.complete(
                [Message.user("sum")],
                [{"name": "add", "description": "d", "parameters": {"type": "object"}}],
            )
        call = completion.tool_calls[0]
        self.assertEqual(call.name, "add")
        self.assertEqual(call.arguments, {"a": 1, "b": 2})
        self.assertEqual(
            server.requests[0]["body"]["tools"][0]["functionDeclarations"][0]["name"], "add"
        )

    def test_turns_encode_tool_round_trip(self):
        provider = GeminiProvider("m", api_key="k", base_url="http://x.test")
        call = ToolCall.from_raw(call_id="c1", name="add", arguments={"a": 1})
        turns = provider._encode_turns(
            [
                Message.user("sum"),
                Message.assistant("ok", [call]),
                Message.tool("c1", "3", "add"),
            ]
        )
        self.assertEqual([t["role"] for t in turns], ["user", "model", "user"])
        self.assertEqual(turns[1]["parts"][0]["text"], "ok")
        self.assertEqual(turns[1]["parts"][1]["functionCall"]["name"], "add")
        self.assertEqual(
            turns[2]["parts"][0]["functionResponse"], {"name": "add", "response": {"result": "3"}}
        )

    def test_consecutive_same_role_turns_merge(self):
        provider = GeminiProvider("m", api_key="k", base_url="http://x.test")
        turns = provider._encode_turns([Message.user("a"), Message.user("b")])
        self.assertEqual(len(turns), 1)
        self.assertEqual(len(turns[0]["parts"]), 2)

    def test_empty_candidates_raises(self):
        with Recorder({"candidates": []}) as server:
            provider = GeminiProvider("m", api_key="k", base_url=server.url)
            with self.assertRaises(ProviderError):
                provider.complete([Message.user("hi")])

    def test_preset_builds_gemini_provider(self):
        provider = create_provider("gemini", api_key="k", model="gemini-2.0-flash")
        self.assertIsInstance(provider, GeminiProvider)
        self.assertEqual(provider.model, "gemini-2.0-flash")


if __name__ == "__main__":
    unittest.main()
