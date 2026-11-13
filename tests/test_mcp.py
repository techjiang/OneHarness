import sys
import tempfile
import time
import unittest
from pathlib import Path

from oneharness.agent import Agent
from oneharness.errors import ToolError
from oneharness.mcp import McpClient, McpError, McpServerGone, connect_mcp
from oneharness.providers.base import Provider
from oneharness.tools import Registry, Tool
from oneharness.types import Completion, Message, ToolCall

FIXTURES = Path(__file__).parent / "fixtures"
SERVER = [sys.executable, str(FIXTURES / "fake_mcp_server.py")]
BROKEN = [sys.executable, str(FIXTURES / "broken_mcp_server.py")]


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


class McpServerLifetimeTests(unittest.TestCase):
    """A dead server is not a slow server, and a wrapper's children are not ours.

    Both of these used to end the same way: the client waited out its full
    timeout and reported "did not answer", which points at the network, the
    server's event loop, or the timeout value -- anywhere except at the process
    that had already exited with a traceback naming the real problem.
    """

    def test_crash_before_handshake_is_reported_at_once(self):
        client = McpClient(BROKEN + ["crash-at-start"], timeout=30)
        started = time.monotonic()
        try:
            with self.assertRaises(McpServerGone) as ctx:
                client.start()
        finally:
            client.close()
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 2.0, "waited for a 30s timeout instead of noticing the exit")
        self.assertIn("exited with code 3", str(ctx.exception))
        self.assertIn("MISSING_TOKEN", str(ctx.exception))

    def test_crash_mid_session_reports_the_stderr_tail(self):
        client = McpClient(BROKEN + ["crash-on-list"], timeout=30)
        client.start()
        started = time.monotonic()
        try:
            with self.assertRaises(McpServerGone) as ctx:
                client.list_tools()
        finally:
            client.close()
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertIn("before answering tools/list", str(ctx.exception))
        self.assertIn("database_url", str(ctx.exception))

    def test_server_initiated_ping_is_answered(self):
        """The protocol runs both ways; a server may ask and then wait for us."""
        client = McpClient(BROKEN + ["pings-after-init"], timeout=3)
        client.start()
        try:
            self.assertEqual(client.list_tools(), [])
        finally:
            client.close()

    def test_close_kills_what_the_server_wrapped(self):
        with tempfile.TemporaryDirectory() as tmp:
            pidfile = Path(tmp) / "child.pid"
            client = McpClient(BROKEN + ["wraps-child", str(pidfile)], timeout=10)
            client.start()
            for _ in range(100):
                if pidfile.exists() and pidfile.read_text():
                    break
                time.sleep(0.02)
            grandchild = int(pidfile.read_text())
            started = time.monotonic()
            client.close()
            elapsed = time.monotonic() - started
        self.assertLess(elapsed, 2.0, "close() blocked on a pipe the grandchild inherited")
        # The grandchild inherited the server's stdout, so a close() that only
        # kills the parent leaves both the process and the pipe alive.
        self.assertFalse(_alive(grandchild), "close() left the server's own child running")

    def test_dead_server_reaches_the_model_as_a_tool_result(self):
        """A server that dies mid-run is a tool result, not the end of the run."""
        client = McpClient(BROKEN + ["crash-on-call"], timeout=10)
        client.start()
        try:
            registry = Registry(client.tools(prefix="b."))
            call = ToolCall.from_raw(call_id="c1", name="b.echo", arguments={"text": "hi"})
            result = registry.execute(call)
        finally:
            client.close()
        self.assertTrue(result.is_error)
        self.assertIn("exited with code 3", result.content)
        self.assertIn("database went away", result.content)

    def test_mcp_server_gone_is_still_an_mcp_error(self):
        self.assertTrue(issubclass(McpServerGone, McpError))
        self.assertTrue(issubclass(McpServerGone, ToolError))


def _alive(pid: int) -> bool:
    """Alive *and* not a zombie: a killed child is reaped by its parent here."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
    except (FileNotFoundError, IndexError):
        return False
    return state != "Z"


class ToolErrorHierarchyTests(unittest.TestCase):
    def test_mcp_error_is_a_tool_error(self):
        self.assertTrue(issubclass(McpError, ToolError))


if __name__ == "__main__":
    unittest.main()
