import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from oneharness import cli
from oneharness.types import Completion, Message, ToolCall, Usage


class FakeProvider:
    """Stands in for a real provider inside the CLI tests."""

    name = "fake"
    model = "fake-1"

    def __init__(self, script):
        self.script = list(script)

    def complete(self, messages, tools=None):
        return self.script.pop(0)


def run_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


class CliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        # a stray key in the environment would make these tests lie
        self._env = mock.patch.dict("os.environ", {}, clear=True)
        self._env.start()

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()

    def patch_agent(self, script):
        provider = FakeProvider(script)
        return mock.patch.object(cli, "create_provider", return_value=provider)

    def test_run_prints_text(self):
        with self.patch_agent([Completion(message=Message.assistant("pong"))]):
            code, out, _ = run_cli(["run", "ping", "--workspace", str(self.root)])
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "pong")

    def test_run_json_output(self):
        script = [
            Completion(
                message=Message.assistant(
                    "", [ToolCall.from_raw(call_id="c1", name="list_dir", arguments={})]
                ),
                usage=Usage(3, 1),
            ),
            Completion(message=Message.assistant("done"), usage=Usage(4, 2)),
        ]
        with self.patch_agent(script):
            code, out, _ = run_cli(["run", "go", "--json", "--workspace", str(self.root)])
        payload = json.loads(out)
        self.assertEqual(code, 0)
        self.assertEqual(payload["text"], "done")
        self.assertEqual(payload["usage"]["total_tokens"], 10)

    def test_tools_command_lists_builtins(self):
        code, out, _ = run_cli(["tools"])
        self.assertEqual(code, 0)
        for name in ("read_file", "write_file", "list_dir", "run_shell"):
            self.assertIn(name, out)

    def test_config_command_is_json(self):
        code, out, _ = run_cli(["config"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["provider"]["preset"], "openai")

    def test_missing_api_key_exits_nonzero(self):
        code, _, err = run_cli(["run", "hi", "--provider", "openai"])
        self.assertEqual(code, 1)
        self.assertIn("API key", err)

    def test_missing_config_file_exits_nonzero(self):
        code, _, err = run_cli(["run", "hi", "--config", str(self.root / "nope.toml")])
        self.assertEqual(code, 1)
        self.assertIn("not found", err)

    def test_max_steps_exit_code(self):
        call = ToolCall.from_raw(call_id="c1", name="list_dir", arguments={})
        script = [Completion(message=Message.assistant("", [call])) for _ in range(2)]
        with self.patch_agent(script):
            code, _, err = run_cli(
                ["run", "go", "--max-steps", "2", "--workspace", str(self.root), "--quiet"]
            )
        self.assertEqual(code, 1)
        self.assertIn("without a final answer", err)

    def test_presets_command_lists_providers(self):
        code, out, _ = run_cli(["presets"])
        self.assertEqual(code, 0)
        self.assertIn("gemini", out)
        self.assertIn("openrouter", out)

    def test_stream_flag_exposes_text_deltas(self):
        """`--stream` must print the final text once, not twice."""

        class StreamingProvider:
            name = "fake"
            model = "fake-1"

            def complete(self, messages, tools=None):
                raise AssertionError("stream mode must not call complete()")

            def stream(self, messages, tools=None):
                from oneharness.types import StreamEvent

                yield StreamEvent(kind="text", text="po")
                yield StreamEvent(kind="text", text="ng")
                yield StreamEvent(
                    kind="done", completion=Completion(message=Message.assistant("pong"))
                )

        with mock.patch.object(cli, "create_provider", return_value=StreamingProvider()):
            code, out, _ = run_cli(["run", "ping", "--stream", "--workspace", str(self.root)])
        self.assertEqual(code, 0)
        self.assertEqual(out.count("pong"), 1)

    def test_mcp_server_receives_tools(self):
        from tests.test_mcp import SERVER

        seen = {}

        class RecordingProvider:
            name = "fake"
            model = "fake-1"

            def complete(self, messages, tools=None):
                seen["tools"] = [spec["name"] for spec in tools or ()]
                return Completion(message=Message.assistant("ok"))

        with mock.patch.object(cli, "create_provider", return_value=RecordingProvider()):
            code, out, _ = run_cli(
                ["run", "go", "--quiet", "--workspace", str(self.root), "--mcp", " ".join(SERVER)]
            )
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "ok")
        self.assertIn("echo", seen["tools"])

    def test_broken_mcp_command_exits_nonzero(self):
        code, _, err = run_cli(
            ["run", "go", "--quiet", "--workspace", str(self.root), "--mcp", "no-such-binary-xyz"]
        )
        self.assertEqual(code, 1)
        self.assertIn("MCP", err)

    def test_delegate_flag_exposes_the_tool(self):
        seen = {}

        class TwoTurn:
            name = "fake"
            model = "fake-1"

            def __init__(self):
                self.turn = 0

            def complete(self, messages, tools=None):
                self.turn += 1
                seen.setdefault("tools", [spec["name"] for spec in tools or ()])
                if self.turn == 1:
                    call = ToolCall.from_raw(
                        call_id="c1", name="delegate", arguments={"task": "count files"}
                    )
                    return Completion(message=Message.assistant("", [call]))
                return Completion(message=Message.assistant("1 file"))

        with mock.patch.object(cli, "create_provider", return_value=TwoTurn()):
            code, out, err = run_cli(["run", "go", "--delegate", "--workspace", str(self.root)])
        self.assertEqual(code, 0)
        self.assertIn("delegate", seen["tools"])
        self.assertIn("delegate_many", seen["tools"])
        self.assertIn("[sub-1]", err)
        self.assertIn("delegated 1 task", err)
        self.assertEqual(out.strip(), "1 file")

    def test_parallel_flag_reaches_the_agent(self):
        captured = {}

        class OneTurn:
            name = "fake"
            model = "fake-1"

            def complete(self, messages, tools=None):
                return Completion(message=Message.assistant("ok"))

        original = cli.Agent

        class Spy(original):
            def __init__(self, *args, **kwargs):
                captured["max_parallel"] = kwargs.get("max_parallel")
                super().__init__(*args, **kwargs)

        with (
            mock.patch.object(cli, "create_provider", return_value=OneTurn()),
            mock.patch.object(cli, "Agent", Spy),
        ):
            code, _, _ = run_cli(["run", "go", "--parallel", "4", "--workspace", str(self.root)])
        self.assertEqual(code, 0)
        self.assertEqual(captured["max_parallel"], 4)

    def test_printer_ignores_sub_agent_tokens(self):
        """A nested run's tokens are not the answer; they must not hit stdout."""
        out, err = io.StringIO(), io.StringIO()
        observe = cli._printer(quiet=False, stream=True)
        with redirect_stdout(out), redirect_stderr(err):
            observe({"type": "text_delta", "text": "spliced", "depth": 1})
            observe({"type": "subagent_start", "task": "go"})
            observe({"type": "text_delta", "text": "answer"})
        self.assertNotIn("spliced", out.getvalue())
        self.assertIn("answer", out.getvalue())
        self.assertIn("⇢ go", err.getvalue())

    def test_stream_with_delegate_prints_the_answer_once(self):
        class TwoTurn:
            name = "fake"
            model = "fake-1"

            def __init__(self):
                self.turn = 0

            def complete(self, messages, tools=None):

                self.turn += 1
                if self.turn == 1:  # the sub agent, which never streams
                    return Completion(message=Message.assistant("sub says hi"))
                raise AssertionError("the parent should stream, not complete")

            def stream(self, messages, tools=None):
                from oneharness.types import StreamEvent

                self.turn += 1
                if self.turn == 1:
                    call = ToolCall.from_raw(
                        call_id="c1", name="delegate", arguments={"task": "say hi"}
                    )
                    done = Completion(Message.assistant("", [call]))
                    yield StreamEvent(kind="done", completion=done)
                    return
                for piece in ("parent ", "answer"):
                    yield StreamEvent(kind="text", text=piece)
                done = Completion(Message.assistant("parent answer"))
                yield StreamEvent(kind="done", completion=done)

        with mock.patch.object(cli, "create_provider", return_value=TwoTurn()):
            code, out, err = run_cli(
                ["run", "go", "--stream", "--delegate", "--workspace", str(self.root)]
            )
        self.assertEqual(code, 0)
        self.assertEqual(out.count("parent answer"), 1)
        self.assertIn("⇢ say hi", err)

    def test_parallel_rejects_zero(self):
        code, _, err = run_cli(["run", "go", "--parallel", "0"])
        self.assertEqual(code, 1)
        self.assertIn("at least 1", err)

    def test_delegation_numbers_land_in_json(self):
        class TwoTurn:
            name = "fake"
            model = "fake-1"

            def __init__(self):
                self.turn = 0

            def complete(self, messages, tools=None):
                self.turn += 1
                if self.turn == 1:
                    call = ToolCall.from_raw(
                        call_id="c1", name="delegate", arguments={"task": "look"}
                    )
                    return Completion(message=Message.assistant("", [call]))
                return Completion(message=Message.assistant("done"), usage=Usage(7, 3))

        with mock.patch.object(cli, "create_provider", return_value=TwoTurn()):
            code, out, err = run_cli(
                ["run", "go", "--delegate", "--json", "--workspace", str(self.root)]
            )
        payload = json.loads(out)  # stdout must stay parseable
        self.assertEqual(code, 0)
        self.assertEqual(payload["text"], "done")
        self.assertEqual(payload["delegation"]["runs"], 1)
        self.assertEqual(payload["delegation"]["usage"]["input_tokens"], 7)
        self.assertIn("delegated 1 task", err)

    def test_max_steps_zero_is_rejected_not_ignored(self):
        code, _, err = run_cli(["run", "go", "--max-steps", "0"])
        self.assertEqual(code, 1)
        self.assertIn("at least 1", err)

    def test_parse_error_for_unknown_flag(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as ctx:
            cli.main(["run", "hi", "--nope"])
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
