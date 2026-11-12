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

    def test_parse_error_for_unknown_flag(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as ctx:
            cli.main(["run", "hi", "--nope"])
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
