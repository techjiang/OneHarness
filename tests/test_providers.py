import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from oneharness.errors import ConfigError, ProviderError
from oneharness.providers import create_provider, list_presets
from oneharness.providers.anthropic import AnthropicProvider
from oneharness.providers.openai import OpenAIProvider
from oneharness.types import Message


class Recorder:
    """A tiny HTTP server that records the request and replays a canned reply."""

    def __init__(self, reply, status=200):
        self.reply = reply
        self.status = status
        self.requests = []
        recorder = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                headers = {k.lower(): v for k, v in self.headers.items()}
                recorder.requests.append({"path": self.path, "headers": headers, "body": body})
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


class CreateProviderTests(unittest.TestCase):
    def test_unknown_preset(self):
        with self.assertRaises(ConfigError):
            create_provider("nope")

    def test_presets_exposed(self):
        self.assertIn("anthropic", list_presets())
        self.assertIn("openai", list_presets())

    def test_missing_api_key(self):
        with self.assertRaises(ConfigError) as ctx:
            create_provider("openai", api_key=None)
        self.assertIn("API key", str(ctx.exception))

    def test_explicit_key_and_overrides(self):
        provider = create_provider(
            "openai", api_key="sk-test", model="gpt-x", base_url="http://example.test/v1/"
        )
        self.assertEqual(provider.model, "gpt-x")
        self.assertEqual(provider.base_url, "http://example.test/v1")

    def test_local_preset_needs_no_key(self):
        self.assertEqual(create_provider("ollama").api_key, "not-needed")

    def test_key_from_environment(self):
        import os

        os.environ["DEEPSEEK_API_KEY"] = "sk-env"
        try:
            self.assertEqual(create_provider("deepseek").api_key, "sk-env")
        finally:
            del os.environ["DEEPSEEK_API_KEY"]


class OpenAIProviderTests(unittest.TestCase):
    def reply(self, **overrides):
        base = {
            "model": "gpt-test",
            "choices": [
                {"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3},
        }
        base.update(overrides)
        return base

    def test_plain_text(self):
        with Recorder(self.reply()) as server:
            provider = OpenAIProvider("gpt-test", api_key="sk", base_url=server.url)
            completion = provider.complete([Message.user("hello")])
        self.assertEqual(completion.text, "hi")
        self.assertEqual(completion.usage.input_tokens, 11)
        self.assertEqual(completion.stop_reason, "stop")
        request = server.requests[0]
        self.assertEqual(request["path"], "/chat/completions")
        self.assertEqual(request["headers"]["authorization"], "Bearer sk")
        self.assertEqual(request["body"]["messages"], [{"role": "user", "content": "hello"}])

    def test_tool_calls_round_trip(self):
        reply = self.reply(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "add", "arguments": '{"a":1,"b":2}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        )
        tools = [{"name": "add", "description": "add", "parameters": {"type": "object"}}]
        with Recorder(reply) as server:
            provider = OpenAIProvider("gpt-test", api_key="sk", base_url=server.url)
            completion = provider.complete([Message.user("sum")], tools)

            self.assertEqual(len(completion.tool_calls), 1)
            self.assertEqual(completion.tool_calls[0].arguments, {"a": 1, "b": 2})
            self.assertEqual(server.requests[0]["body"]["tools"][0]["type"], "function")

            # the assistant turn plus a tool result must encode cleanly
            encoded = provider._encode(Message.assistant("", completion.tool_calls))
            self.assertIsNone(encoded["content"])
            self.assertEqual(encoded["tool_calls"][0]["function"]["arguments"], '{"a":1,"b":2}')

            tool_message = provider._encode(Message.tool("call_1", "3", "add"))
            self.assertEqual(tool_message["tool_call_id"], "call_1")

    def test_empty_choices_raises(self):
        with Recorder({"choices": []}) as server:
            provider = OpenAIProvider("m", api_key="sk", base_url=server.url)
            with self.assertRaises(ProviderError):
                provider.complete([Message.user("hi")])

    def test_error_status_surfaces_body(self):
        with Recorder({"error": "nope"}, status=400) as server:
            provider = OpenAIProvider("m", api_key="sk", base_url=server.url)
            with self.assertRaises(ProviderError) as ctx:
                provider.complete([Message.user("hi")])
        self.assertEqual(ctx.exception.status, 400)
        self.assertIn("nope", str(ctx.exception))


class AnthropicProviderTests(unittest.TestCase):
    def reply(self, blocks, stop_reason="end_turn"):
        return {
            "model": "claude-test",
            "content": blocks,
            "stop_reason": stop_reason,
            "usage": {"input_tokens": 7, "output_tokens": 5},
        }

    def test_system_hoisted_and_text(self):
        with Recorder(self.reply([{"type": "text", "text": "hey"}])) as server:
            provider = AnthropicProvider("claude-test", api_key="k", base_url=server.url)
            completion = provider.complete([Message.system("be terse"), Message.user("hello")])
        self.assertEqual(completion.text, "hey")
        self.assertEqual(completion.usage.input_tokens, 7)
        body = server.requests[0]["body"]
        self.assertEqual(body["system"], "be terse")
        self.assertEqual(body["messages"], [{"role": "user", "content": "hello"}])
        self.assertEqual(server.requests[0]["headers"]["x-api-key"], "k")
        self.assertEqual(server.requests[0]["headers"]["anthropic-version"], "2023-06-01")

    def test_tool_use_and_tool_result_folding(self):
        reply = self.reply(
            [
                {"type": "text", "text": "let me check"},
                {"type": "tool_use", "id": "tu_1", "name": "add", "input": {"a": 1, "b": 2}},
            ],
            stop_reason="tool_use",
        )
        with Recorder(reply) as server:
            provider = AnthropicProvider("c", api_key="k", base_url=server.url)
            completion = provider.complete(
                [Message.user("sum")],
                [{"name": "add", "description": "d", "parameters": {"type": "object"}}],
            )
        self.assertEqual(completion.text, "let me check")
        self.assertEqual(completion.tool_calls[0].arguments, {"a": 1, "b": 2})
        self.assertEqual(server.requests[0]["body"]["tools"][0]["input_schema"], {"type": "object"})

        _, turns = provider._split(
            [
                Message.user("sum"),
                Message.assistant("", completion.tool_calls),
                Message.tool("tu_1", "3", "add"),
                Message.tool("tu_2", "4", "add"),
            ]
        )
        self.assertEqual(
            turns[1]["content"],
            [{"type": "tool_use", "id": "tu_1", "name": "add", "input": {"a": 1, "b": 2}}],
        )
        # consecutive tool results fold into a single user turn
        self.assertEqual(turns[2]["role"], "user")
        self.assertEqual(len(turns[2]["content"]), 2)
        self.assertEqual(len(turns), 3)


if __name__ == "__main__":
    unittest.main()
