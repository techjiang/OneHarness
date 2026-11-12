import sys
import unittest
from pathlib import Path

from oneharness.agent import Agent
from oneharness.errors import ToolError
from oneharness.mcp import McpClient, McpError, connect_mcp
from oneharness.providers.base import Provider
from oneharness.tools import Registry, Tool
from oneharness.types import Completion, Message, ToolCall

SERVER = [sys.executable, str(Path(__file__).parent / "fixtures" / "fake_mcp_server.py")]


class ScriptedProvider(Provider):
    name = "scripted"

    def __init__(self, *completions):
        super().__init__(model="scripted")
        self.script = list(completions)

    def complete(self, messages, tools=None):
        self.calls_seen = list(tools or ())
        return self.script.pop(0)


class McpClientTests(unittest.TestCase):
    def setUp(self):
        self.client = McpClient(SERVER, timeout=10)
        self.client.start()

    def tearDown(self):
        self.client.close()

    def test_handshake_records_server_info(self):
        self.assertEqual(self.client.server_info["name"], "fake")

    def test_tools_listed_with_schema(self):
        specs = self.client.list_tools()
        self.assertEqual([spec["name"] for spec in specs], ["echo", "boom"])

    def test_tool_call_returns_text(self):
        self.assertEqual(self.client.call_tool("echo", {"text": "hi"}), "echo:hi")

    def test_tool_error_becomes_exception(self):
        with self.assertRaises(McpError) as ctx:
            self.client.call_tool("boom", {})
        self.assertIn("boom happened", str(ctx.exception))

    def test_tools_are_namespaced(self):
        tools = self.client.tools(prefix="fake.")
        self.assertEqual([t.name for t in tools], ["fake.echo", "fake.boom"])
        self.assertIn("text", tools[0].parameters["properties"])

    def test_unknown_method_raises(self):
        with self.assertRaises(McpError):
            self.client._request("nope/nothing", {})

    def test_command_that_does_not_exist(self):
        client = McpClient(["definitely-not-a-real-binary-xyz"])
        with self.assertRaises(McpError):
            client.start()

    def test_timeout_when_server_silent(self):
        client = McpClient([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.3)
        try:
            with self.assertRaises(McpError) as ctx:
                client.start()
            self.assertIn("did not answer initialize", str(ctx.exception))
        finally:
            client.close()

    def test_connect_helper_roundtrip(self):
        with connect_mcp(SERVER, prefix="m.") as client:
            self.assertEqual(len(client.tools()), 2)


class McpToolIntegrationTests(unittest.TestCase):
    def test_unknown_mcp_tool_error_reaches_the_model(self):
        """A failing MCP tool must not kill the loop: it comes back as content."""
        client = McpClient(SERVER, timeout=10)
        client.start()
        try:
            registry = Registry(client.tools(prefix="fake."))
            call = ToolCall.from_raw(call_id="c1", name="fake.boom", arguments={})
            result = registry.execute(call)
            self.assertTrue(result.is_error)
            self.assertIn("boom happened", result.content)

            call = ToolCall.from_raw(call_id="c2", name="fake.echo", arguments={"text": "yo"})
            ok = registry.execute(call)
            self.assertFalse(ok.is_error)
            self.assertEqual(ok.content, "echo:yo")
        finally:
            client.close()

    def test_agent_loop_uses_mcp_tool(self):
        client = McpClient(SERVER, timeout=10)
        client.start()
        try:
            call = ToolCall.from_raw(call_id="c1", name="fake.echo", arguments={"text": "ping"})
            provider = ScriptedProvider(
                Completion(message=Message.assistant("", [call])),
                Completion(message=Message.assistant("done")),
            )
            agent = Agent(provider, Registry(client.tools(prefix="fake.")))
            result = agent.run("say ping")

            self.assertEqual(result.text, "done")
            self.assertEqual(result.steps[0].tool_results[0].content, "echo:ping")
            names = [spec["name"] for spec in provider.calls_seen]
            self.assertEqual(names, ["fake.echo", "fake.boom"])
        finally:
            client.close()

    def test_tool_object_is_plain_tool(self):
        client = McpClient(SERVER, timeout=10)
        client.start()
        try:
            self.assertIsInstance(client.tools(prefix="f.")[0], Tool)
        finally:
            client.close()


class ToolErrorHierarchyTests(unittest.TestCase):
    def test_mcp_error_is_a_tool_error(self):
        self.assertTrue(issubclass(McpError, ToolError))


if __name__ == "__main__":
    unittest.main()
