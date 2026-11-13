import resource
import tempfile
import time
import unittest
from pathlib import Path
from typing import Literal

from oneharness.builtin import default_tools, make_shell_tool
from oneharness.errors import ToolError
from oneharness.tools import Registry, tool
from oneharness.types import ToolCall


def make_call(name, **arguments):
    return ToolCall.from_raw(call_id="c1", name=name, arguments=arguments)


@tool
def takes_int(a: int) -> str:
    """Turn an int into a string."""
    return str(a)


class SchemaTests(unittest.TestCase):
    def test_primitives_and_required(self):
        @tool
        def add(a: int, b: float, label: str) -> str:
            """Add numbers."""
            return ""

        schema = add.parameters
        self.assertEqual(schema["properties"]["a"], {"type": "integer"})
        self.assertEqual(schema["properties"]["b"], {"type": "number"})
        self.assertEqual(schema["properties"]["label"], {"type": "string"})
        self.assertEqual(schema["required"], ["a", "b", "label"])

    def test_defaults_and_docstring(self):
        @tool
        def greet(name: str, times: int = 2) -> str:
            """Say hello.

            Extra detail that should not leak into the description.
            """
            return ""

        self.assertEqual(greet.description, "Say hello.")
        self.assertEqual(greet.parameters["properties"]["times"]["default"], 2)
        self.assertEqual(greet.parameters["required"], ["name"])

    def test_containers_and_literals(self):
        @tool
        def pick(mode: Literal["fast", "slow"], tags: list[str], opts: dict) -> str:
            return ""

        props = pick.parameters["properties"]
        self.assertEqual(props["mode"], {"enum": ["fast", "slow"], "type": "string"})
        self.assertEqual(props["tags"], {"type": "array", "items": {"type": "string"}})
        self.assertEqual(props["opts"], {"type": "object"})

    def test_optional_unwraps(self):
        @tool
        def maybe(count: int | None = None) -> str:
            return ""

        self.assertEqual(maybe.parameters["properties"]["count"], {"type": "integer"})

    def test_custom_name_and_description(self):
        @tool(name="ping", description="Ping it")
        def whatever() -> str:
            return "pong"

        self.assertEqual(whatever.name, "ping")
        self.assertEqual(whatever.description, "Ping it")

    def test_invalid_name_rejected(self):
        with self.assertRaises(ToolError):

            @tool(name="bad name!")
            def broken() -> str:
                return ""


class RegistryTests(unittest.TestCase):
    def test_duplicate_names_rejected(self):
        @tool
        def a() -> str:
            return "a"

        registry = Registry([a])
        with self.assertRaises(ToolError):
            registry.add(a)

    def test_specs_shape(self):
        @tool
        def echo(text: str) -> str:
            """Echo text."""
            return text

        spec = Registry([echo]).specs()[0]
        self.assertEqual(spec["name"], "echo")
        self.assertEqual(spec["description"], "Echo text.")
        self.assertEqual(spec["parameters"]["type"], "object")

    def test_executes_and_serialises(self):
        @tool
        def info() -> dict:
            return {"ok": True}

        result = Registry([info]).execute(make_call("info"))
        self.assertEqual(result.content, '{"ok":true}')
        self.assertFalse(result.is_error)
        self.assertEqual(result.tool_call_id, "c1")

    def test_unknown_tool_is_an_error_result(self):
        result = Registry([]).execute(make_call("ghost"))
        self.assertTrue(result.is_error)
        self.assertIn("unknown tool", result.content)

    def test_raising_tool_does_not_escape(self):
        @tool
        def boom() -> str:
            raise RuntimeError("kaboom")

        result = Registry([boom]).execute(make_call("boom"))
        self.assertTrue(result.is_error)
        self.assertIn("kaboom", result.content)

    def test_bad_arguments_are_reported(self):
        result = Registry([takes_int]).execute(make_call("takes_int", a=1, unexpected=True))
        self.assertTrue(result.is_error)
        self.assertIn("bad arguments", result.content)

    def test_invalid_json_arguments_are_reported(self):
        call = ToolCall.from_raw(call_id="c1", name="takes_int", arguments="{nope")
        result = Registry([takes_int]).execute(call)
        self.assertTrue(result.is_error)
        self.assertIn("not valid JSON", result.content)


class BuiltinToolTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.tools = {t.name: t for t in default_tools(self.root, shell=True)}

    def tearDown(self):
        self._tmp.cleanup()

    def test_write_then_read(self):
        self.tools["write_file"]("notes/a.txt", "hello")
        self.assertEqual((self.root / "notes/a.txt").read_text(), "hello")
        self.assertEqual(self.tools["read_file"]("notes/a.txt"), "hello")

    def test_cannot_escape_workspace(self):
        with self.assertRaises(ToolError):
            self.tools["read_file"]("../outside.txt")

    def test_absolute_path_outside_is_refused(self):
        with self.assertRaises(ToolError):
            self.tools["write_file"]("/etc/passwd-x", "x")

    def test_list_dir_marks_directories(self):
        (self.root / "sub").mkdir()
        (self.root / "file.txt").write_text("x")
        listing = self.tools["list_dir"](".")
        self.assertIn("sub/", listing)
        self.assertIn("file.txt", listing)

    def test_shell_reports_exit_code(self):
        output = self.tools["run_shell"]("exit 3")
        self.assertIn("exit_code=3", output)

    def test_shell_returns_stdout(self):
        self.assertIn("hi", self.tools["run_shell"]("echo hi"))

    def test_shell_timeout(self):
        tool_obj = make_shell_tool(self.root, timeout=0.2)
        with self.assertRaises(ToolError):
            tool_obj("sleep 5")


class ShellHonestyTests(unittest.TestCase):
    """What the shell tool says has to match what the shell did.

    Three claims it used to make that were not true: that a timeout means
    nothing is known (the output up to the stall is known), that a command
    which printed a line ended cleanly (that was only true for UTF-8), and that
    a killed command is gone (the grandchildren were not).
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_timeout_keeps_what_the_command_printed(self):
        # The most valuable part of a timed-out answer: how far it got. A test
        # suite that printed forty failures before hanging has said everything.
        shell = make_shell_tool(self.root, timeout=0.4)
        with self.assertRaises(ToolError) as ctx:
            shell("echo first-line; echo second-line; sleep 5")
        message = str(ctx.exception)
        self.assertIn("timed out", message)
        self.assertIn("first-line", message)
        self.assertIn("second-line", message)
        self.assertNotIn("after-the-stall", message)

    def test_a_timeout_kills_the_whole_group_not_just_the_shell(self):
        # `sh -c '... &'` leaves a grandchild that survives a kill aimed at the
        # shell. The old implementation reported "killed" and the grandchild
        # then wrote its file two seconds later -- verified, not imagined.
        marker = self.root / "leaked.txt"
        shell = make_shell_tool(self.root, timeout=0.4)
        with self.assertRaises(ToolError):
            shell(f"sh -c 'sleep 0.8; echo leaked > {marker}' & sleep 5")
        time.sleep(1.2)
        self.assertFalse(marker.exists(), "a process the tool killed was still alive")

    def test_stderr_is_interleaved_not_appended(self):
        # Two captured buffers concatenated afterwards reorder the transcript:
        # the warning that came first lands last, and the cause reads after the
        # effect. One pipe keeps the order the command chose.
        shell = make_shell_tool(self.root)
        output = shell("echo one; echo two >&2; echo three")
        self.assertLess(output.index("one"), output.index("two"))
        self.assertLess(output.index("two"), output.index("three"))

    def test_binary_output_is_not_an_exception(self):
        # `head -c 200 /dev/urandom` is a legitimate command. Answering it with
        # `UnicodeDecodeError` tells the model nothing about what it ran -- and
        # the decode happens across read boundaries, so the decoder has to be
        # incremental or it mangles the characters it straddles.
        shell = make_shell_tool(self.root)
        output = shell("head -c 4000 /dev/urandom")
        self.assertIn("exit_code=0", output)
        self.assertGreater(len(output), 100)

    def test_multibyte_characters_survive_chunk_boundaries(self):
        # 中文 is three bytes per character; a 64 KB read ends mid-character
        # fairly often at this size, and a stateless decoder replaces both
        # halves with two replacement characters.
        shell = make_shell_tool(self.root)
        output = shell("python3 -c \"print('中文测试' * 30000, end='')\"")
        self.assertIn("中文测试", output)
        self.assertNotIn("\ufffd", output)

    def test_a_wall_of_output_keeps_both_ends(self):
        # The beginning says what it started doing, the end says why it stopped.
        # Keeping only the head keeps the passes and loses the failures.
        shell = make_shell_tool(self.root)
        output = shell("seq 1 40000")
        self.assertIn("dropped from the middle", output)
        self.assertIn("\n1\n", output)
        self.assertTrue(output.rstrip().endswith("40000"), output[-80:])
        self.assertLess(len(output), 25_000)

    def test_output_is_capped_while_reading_not_after(self):
        # The cap has to be enforced during the read: `yes` writes 60 MB in
        # 0.14s and `capture_output` holds every byte of it before anyone can
        # truncate (measured 182 MB peak RSS for a one-line answer).
        shell = make_shell_tool(self.root)
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        output = shell("yes aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa | head -c 40000000")
        growth = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before) / 1024
        self.assertLess(len(output), 25_000)
        self.assertLess(growth, 60, f"held {growth:.0f} MB of a 40 MB stream")

    def test_a_command_that_ignores_sigterm_still_ends(self):
        shell = make_shell_tool(self.root, timeout=0.3)
        started = time.monotonic()
        with self.assertRaises(ToolError):
            shell("trap '' TERM; sleep 30")
        self.assertLess(time.monotonic() - started, 5.0)

    def test_a_talkative_grandchild_does_not_hang_the_call(self):
        # A grandchild that inherited stdout keeps the pipe open after the shell
        # exits; joining the reader without a grace period is a tool call that
        # never returns for a command that already finished.
        shell = make_shell_tool(self.root, timeout=5)
        started = time.monotonic()
        output = shell("nohup sleep 30 & echo done")
        self.assertIn("done", output)
        self.assertLess(time.monotonic() - started, 3.0)


if __name__ == "__main__":
    unittest.main()
