import tempfile
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


if __name__ == "__main__":
    unittest.main()
