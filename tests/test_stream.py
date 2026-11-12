import unittest

from oneharness.errors import ProviderError
from oneharness.providers.anthropic import AnthropicProvider
from oneharness.providers.gemini import GeminiProvider
from oneharness.providers.openai import OpenAIProvider
from oneharness.types import Message
from tests.sse_server import SSERecorder


def openai_chunk(content=None, calls=None, finish=None, usage=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    if calls is not None:
        delta["tool_calls"] = calls
    chunk = {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage:
        chunk["usage"] = usage
    return chunk


class OpenAIStreamTests(unittest.TestCase):
    def test_text_deltas_and_usage(self):
        events = [
            (None, openai_chunk("he")),
            (None, openai_chunk("llo")),
            (None, openai_chunk(finish="stop")),
            (None, {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 2}}),
            (None, {"done": True}),
        ]
        with SSERecorder(events) as server:
            provider = OpenAIProvider("gpt", api_key="sk", base_url=server.url)
            out = list(provider.stream([Message.user("hi")]))

        self.assertEqual("".join(e.text for e in out if e.kind == "text"), "hello")
        done = out[-1]
        self.assertEqual(done.kind, "done")
        self.assertEqual(done.completion.text, "hello")
        self.assertEqual(done.completion.usage.input_tokens, 9)
        self.assertEqual(done.completion.stop_reason, "stop")
        self.assertTrue(server.requests[0]["body"]["stream"])

    def test_tool_call_arguments_are_stitched(self):
        events = [
            (
                None,
                openai_chunk(
                    calls=[
                        {
                            "index": 0,
                            "id": "call_a",
                            "function": {"name": "read_file", "arguments": '{"pa'},
                        }
                    ]
                ),
            ),
            (None, openai_chunk(calls=[{"index": 0, "function": {"arguments": 'th":"a.txt"}'}}])),
            (None, openai_chunk(finish="tool_calls")),
        ]
        with SSERecorder(events) as server:
            provider = OpenAIProvider("gpt", api_key="sk", base_url=server.url)
            done = list(provider.stream([Message.user("hi")]))[-1]

        call = done.completion.tool_calls[0]
        self.assertEqual(call.id, "call_a")
        self.assertEqual(call.name, "read_file")
        self.assertEqual(call.arguments, {"path": "a.txt"})

    def test_http_error_before_first_token_raises(self):
        with SSERecorder([], status=401) as server:
            provider = OpenAIProvider("gpt", api_key="sk", base_url=server.url)
            with self.assertRaises(ProviderError) as ctx:
                list(provider.stream([Message.user("hi")]))
        self.assertEqual(ctx.exception.status, 401)


class AnthropicStreamTests(unittest.TestCase):
    def test_text_and_tool_use(self):
        events = [
            ("message_start", {"type": "message_start", "message": {"usage": {"input_tokens": 5}}}),
            (
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "lo"},
                },
            ),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "tool_use", "id": "tu_1", "name": "add", "input": {}},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": '{"a":1,'},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": '"b":2}'},
                },
            ),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use"},
                    "usage": {"output_tokens": 12},
                },
            ),
        ]
        with SSERecorder(events) as server:
            provider = AnthropicProvider("c", api_key="k", base_url=server.url)
            out = list(provider.stream([Message.user("hi")]))

        self.assertEqual([e.text for e in out if e.kind == "text"], ["lo"])
        done = out[-1].completion
        self.assertEqual(done.text, "lo")
        self.assertEqual(done.usage.input_tokens, 5)
        self.assertEqual(done.usage.output_tokens, 12)
        self.assertEqual(done.stop_reason, "tool_use")
        self.assertEqual(done.tool_calls[0].arguments, {"a": 1, "b": 2})


class GeminiStreamTests(unittest.TestCase):
    def chunk(self, parts, finish=None):
        return {
            "candidates": [{"content": {"parts": parts}, "finishReason": finish}],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 4},
        }

    def test_text_stream(self):
        events = [
            (None, self.chunk([{"text": "he"}])),
            (None, self.chunk([{"text": "y"}], finish="STOP")),
        ]
        with SSERecorder(events) as server:
            provider = GeminiProvider("gemini-x", api_key="k", base_url=server.url)
            out = list(provider.stream([Message.user("hi")]))

        self.assertEqual("".join(e.text for e in out if e.kind == "text"), "hey")
        done = out[-1].completion
        self.assertEqual(done.text, "hey")
        self.assertEqual(done.usage.input_tokens, 3)
        self.assertEqual(done.stop_reason, "STOP")

    def test_function_call_parts_become_tool_calls(self):
        events = [(None, self.chunk([{"functionCall": {"name": "add", "args": {"a": 1, "b": 2}}}]))]
        with SSERecorder(events) as server:
            provider = GeminiProvider("gemini-x", api_key="k", base_url=server.url)
            done = list(provider.stream([Message.user("hi")]))[-1].completion

        self.assertEqual(done.tool_calls[0].name, "add")
        self.assertEqual(done.tool_calls[0].arguments, {"a": 1, "b": 2})


if __name__ == "__main__":
    unittest.main()
