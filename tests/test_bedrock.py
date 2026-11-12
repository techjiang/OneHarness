import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from oneharness.agent import Agent
from oneharness.errors import ConfigError, ProviderError
from oneharness.eventstream import frame_for
from oneharness.providers import create_provider, list_presets
from oneharness.providers.bedrock import BedrockProvider
from oneharness.sigv4 import Credentials, sign_request
from oneharness.tools import Registry, tool
from oneharness.types import Message, ToolCall

CREDENTIALS = Credentials("AKIDEXAMPLE", "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY")

CONVERSE_REPLY = {
    "output": {
        "message": {
            "role": "assistant",
            "content": [
                {"text": "let me look"},
                {"toolUse": {"toolUseId": "tu-1", "name": "read_file", "input": {"path": "a"}}},
            ],
        }
    },
    "stopReason": "tool_use",
    "usage": {"inputTokens": 12, "outputTokens": 5, "totalTokens": 17},
    "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
}


def stream_frames(pieces):
    """The events Bedrock really sends, in order, as one binary event stream."""
    frames = [frame_for({"role": "assistant"}, "messageStart")]
    frames.append(
        frame_for(
            {
                "contentBlockIndex": 0,
                "start": {"toolUse": {"toolUseId": "tu-1", "name": "read_file"}},
            },
            "contentBlockStart",
        )
    )
    for piece in pieces:
        frames.append(
            frame_for({"contentBlockIndex": 0, "delta": {"text": piece}}, "contentBlockDelta")
        )
    frames.append(
        frame_for(
            {"contentBlockIndex": 1, "start": {"toolUse": {"toolUseId": "tu-2", "name": "add"}}},
            "contentBlockStart",
        )
    )
    for fragment in ('{"a":', " 1, ", '"b": 2}'):
        frames.append(
            frame_for(
                {"contentBlockIndex": 1, "delta": {"toolUse": {"input": fragment}}},
                "contentBlockDelta",
            )
        )
    frames.append(frame_for({"stopReason": "tool_use"}, "messageStop"))
    frames.append(
        frame_for({"usage": {"inputTokens": 40, "outputTokens": 9}, "metrics": {}}, "metadata")
    )
    return b"".join(frames)


class BedrockRecorder:
    """A fake Bedrock endpoint that checks the request the way AWS would."""

    def __init__(self, reply=None, frames=b"", status=200):
        self.reply = reply or CONVERSE_REPLY
        self.frames = frames
        self.status = status
        self.requests = []
        recorder = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("content-length", 0)))
                headers = {k.lower(): v for k, v in self.headers.items()}
                recorder.requests.append({"path": self.path, "headers": headers, "raw": raw})
                if recorder.status != 200:
                    payload = b'{"message":"denied"}'
                    self.send_response(recorder.status)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                if recorder.frames:
                    self.send_response(200)
                    self.send_header("content-type", "application/vnd.amazon.eventstream")
                    self.send_header("content-length", str(len(recorder.frames)))
                    self.end_headers()
                    self.wfile.write(recorder.frames)
                    return
                payload = json.dumps(recorder.reply).encode()
                self.send_response(200)
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


def provider_for(recorder, **options):
    return BedrockProvider(
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
        credentials=CREDENTIALS,
        base_url=recorder.url,
        **options,
    )


class SigningTests(unittest.TestCase):
    def test_signature_covers_the_bytes_that_were_sent(self):
        # The strongest check available without AWS: recompute the signature from
        # what the server actually received. If signing and sending ever use two
        # different serialisations of the same payload, this fails and real AWS
        # would answer SignatureDoesNotMatch.
        with BedrockRecorder() as recorder:
            provider_for(recorder).complete([Message.user("hi")])
            request = recorder.requests[0]
            expected = sign_request(
                method="POST",
                url=recorder.url + request["path"],
                body=request["raw"],
                headers={
                    "content-type": "application/json",
                    "accept": "application/json",
                    "host": request["headers"]["host"],
                },
                credentials=CREDENTIALS,
                service="bedrock",
                region="us-east-1",
                timestamp=request["headers"]["x-amz-date"],
            )
            self.assertEqual(request["headers"]["authorization"], expected["authorization"])

    def test_scope_names_the_bedrock_service_and_region(self):
        with BedrockRecorder() as recorder:
            provider_for(recorder, region="eu-central-1").complete([Message.user("hi")])
            self.assertIn(
                "/eu-central-1/bedrock/aws4_request",
                recorder.requests[0]["headers"]["authorization"],
            )

    def test_model_id_is_percent_encoded_in_the_path(self):
        # Cross-region inference profile ids and any versioned model id are full
        # of colons; an unencoded colon in a signed path is a signature mismatch.
        with BedrockRecorder() as recorder:
            provider_for(recorder).complete([Message.user("hi")])
            self.assertIn(
                "/model/anthropic.claude-3-5-sonnet-20241022-v2%3A0/converse",
                recorder.requests[0]["path"],
            )


class ConversePayloadTests(unittest.TestCase):
    def test_system_prompt_is_a_block_list_not_a_message(self):
        with BedrockRecorder() as recorder:
            provider_for(recorder).complete([Message.system("be terse"), Message.user("hi")])
            body = json.loads(recorder.requests[0]["raw"])
            self.assertEqual(body["system"], [{"text": "be terse"}])
            self.assertEqual(body["messages"], [{"role": "user", "content": [{"text": "hi"}]}])

    def test_tool_schema_is_nested_under_json(self):
        @tool
        def add(a: int, b: int) -> int:
            """Add two integers."""
            return a + b

        with BedrockRecorder() as recorder:
            registry = Registry([add])
            provider_for(recorder).complete([Message.user("2+2")], registry.specs())
            spec = json.loads(recorder.requests[0]["raw"])["toolConfig"]["tools"][0]["toolSpec"]
            self.assertEqual(spec["name"], "add")
            self.assertEqual(spec["inputSchema"]["json"]["properties"]["a"]["type"], "integer")

    def test_tool_results_merge_into_one_user_turn(self):
        # Converse rejects two consecutive turns with the same role, and every
        # tool result is a user turn: two calls in one turn must be merged.
        messages = [
            Message.user("do two things"),
            Message.assistant("", [ToolCall("a", "f"), ToolCall("b", "g")]),
            Message.tool("a", "1"),
            Message.tool("b", "2"),
        ]
        with BedrockRecorder() as recorder:
            provider_for(recorder).complete(messages)
            turns = json.loads(recorder.requests[0]["raw"])["messages"]
            self.assertEqual([t["role"] for t in turns], ["user", "assistant", "user"])
            self.assertEqual(len(turns[2]["content"]), 2)

    def test_failed_tool_result_is_marked_as_an_error(self):
        messages = [
            Message.user("x"),
            Message.assistant("", [ToolCall("a", "f")]),
            Message.tool("a", "boom", is_error=True),
        ]
        with BedrockRecorder() as recorder:
            provider_for(recorder).complete(messages)
            block = json.loads(recorder.requests[0]["raw"])["messages"][2]["content"][0]
            self.assertEqual(block["toolResult"]["status"], "error")

    def test_empty_assistant_turn_is_not_sent_empty(self):
        messages = [Message.user("x"), Message.assistant("")]
        with BedrockRecorder() as recorder:
            provider_for(recorder).complete(messages)
            turn = json.loads(recorder.requests[0]["raw"])["messages"][1]
            self.assertTrue(turn["content"])


class ConverseResponseTests(unittest.TestCase):
    def test_parses_text_tools_and_usage(self):
        with BedrockRecorder() as recorder:
            completion = provider_for(recorder).complete([Message.user("hi")])
        self.assertEqual(completion.text, "let me look")
        self.assertEqual(completion.tool_calls[0].name, "read_file")
        self.assertEqual(completion.tool_calls[0].arguments, {"path": "a"})
        self.assertEqual(completion.usage.total_tokens, 17)
        self.assertEqual(completion.stop_reason, "tool_use")

    def test_missing_output_is_a_provider_error(self):
        with BedrockRecorder(reply={"stopReason": "end_turn"}) as recorder:
            self.assertRaises(ProviderError, provider_for(recorder).complete, [Message.user("hi")])


class StreamTests(unittest.TestCase):
    def test_text_and_tool_arguments_arrive_across_events(self):
        with BedrockRecorder(frames=stream_frames(["Hel", "lo"])) as recorder:
            events = list(provider_for(recorder).stream([Message.user("hi")]))
        text = "".join(event.text for event in events if event.kind == "text")
        self.assertEqual(text, "Hello")
        done = events[-1].completion
        self.assertEqual([c.name for c in done.tool_calls], ["read_file", "add"])
        # The arguments were split across three events and reassembled.
        self.assertEqual(done.tool_calls[1].arguments, {"a": 1, "b": 2})
        self.assertEqual(done.usage.total_tokens, 49)

    def test_exception_frame_raises_with_the_server_message(self):
        body = frame_for({"message": "Input is too long"}, "ValidationException", exception=True)
        with (
            BedrockRecorder(frames=body) as recorder,
            self.assertRaises(ProviderError) as caught,
        ):
            list(provider_for(recorder).stream([Message.user("hi")]))
        self.assertIn("ValidationException", str(caught.exception))
        self.assertIn("Input is too long", str(caught.exception))

    def test_agent_loop_runs_on_a_bedrock_stream(self):
        # The whole point: the loop must not care that this stream was binary.
        @tool
        def add(a: int, b: int) -> int:
            """Add two integers."""
            return a + b

        with BedrockRecorder(frames=stream_frames(["Computing."])) as recorder:
            agent = Agent(provider_for(recorder), Registry([add]), max_steps=1)
            result = agent.run("2+2", stream=True)
        self.assertEqual(result.text, "Computing.")
        self.assertEqual(result.steps[0].completion.tool_calls[1].arguments, {"a": 1, "b": 2})
        self.assertEqual(result.stopped_reason, "max_steps")


class PresetTests(unittest.TestCase):
    def test_bedrock_preset_exists(self):
        self.assertIn("bedrock", list_presets())

    def test_missing_aws_credentials_is_a_config_error(self):
        import os

        saved = {k: os.environ.pop(k, None) for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")}
        try:
            with self.assertRaises(ConfigError):
                create_provider("bedrock")
        finally:
            for key, value in saved.items():
                if value is not None:
                    os.environ[key] = value

    def test_region_comes_from_the_environment(self):
        import os

        os.environ["AWS_ACCESS_KEY_ID"] = "AK"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "SK"
        os.environ["AWS_REGION"] = "ap-southeast-2"
        try:
            provider = create_provider("bedrock")
        finally:
            os.environ.pop("AWS_ACCESS_KEY_ID")
            os.environ.pop("AWS_SECRET_ACCESS_KEY")
            os.environ.pop("AWS_REGION")
        self.assertEqual(provider.region, "ap-southeast-2")
        self.assertIn("ap-southeast-2", provider.base_url)


if __name__ == "__main__":
    unittest.main()
