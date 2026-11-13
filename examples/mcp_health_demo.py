"""What an MCP client says when the server is not there.

    python3 examples/mcp_health_demo.py

v1.4's MCP client had one answer for every failure: *did not answer in 30s*.
A server that exits with `KeyError: 'database_url'` on stderr never says a word,
so the client waited out the timeout and then blamed the clock. The traceback --
the only thing that says what to fix -- was collected into a list and never read.

Four ways that shows up, each run against v1.4 and the current client in one
process, with the same server script:

1. **A server that dies during the handshake.** 30s of silence, then "did not
   answer initialize". The process had exited 0.05s in, with its reason on stderr.
2. **A server that dies mid-session.** Same wait, same sentence, and the model's
   tool call comes back as "did not answer tools/call in 10s" -- a timeout story
   about a crash.
3. **A server that pings us.** The protocol runs both ways: a server may send a
   request and wait for the reply. Ignoring it (v1.4) means two processes each
   waiting for the other, and the client calls that a timeout.
4. **`close()` on a server that wrapped another process.** The wrapper's child
   inherits stdout, so the reader thread stays blocked in `read()`, and `close()`
   waits for it. Measured below: the child's lifetime becomes the shutdown time.

The fifth part goes through the real CLI, so the last block is what a user
actually sees rather than what this demo prints.

No network, no API key.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oneharness.mcp import McpClient  # noqa: E402

BROKEN = [sys.executable, str(ROOT / "tests" / "fixtures" / "broken_mcp_server.py")]
FAST = 3.0  # what "the timeout" is set to in every comparison below
# How long the wrapped child lives. The old client's shutdown waits for it, so
# this is both the size of the bug and the cost of measuring it.
CHILD_LIFE = 10


def banner(text: str) -> None:
    print(f"\n=== {text} ===")


class OldClient:
    """v1.4's client, verbatim in the parts that matter here.

    Same request/response over stdio, same `_read_stdout` that drops anything
    without one of our ids, same close() that waits on the process and then
    closes the pipes from another thread.
    """

    def __init__(self, command, timeout=FAST):
        self.command = list(command)
        self.timeout = timeout
        self._next_id = 0
        self._lock = threading.Lock()
        self._replies = {}
        self._stderr = []
        self._proc = None

    def start(self):
        self._proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        self._request(
            "initialize",
            {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "old"}},
        )

    def close(self):
        proc, self._proc = self._proc, None
        if proc is None:
            return
        with contextlib.suppress(OSError):
            proc.stdin.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        for stream in (proc.stdout, proc.stderr):
            with contextlib.suppress(OSError):
                stream.close()

    def list_tools(self):
        return self._request("tools/list", {}).get("tools") or []

    def _request(self, method, params):
        with self._lock:
            self._next_id += 1
            request_id = self._next_id
            waiting = queue.Queue(maxsize=1)
            self._replies[request_id] = waiting
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()
        try:
            reply = waiting.get(timeout=self.timeout)
        except queue.Empty:
            self._replies.pop(request_id, None)
            raise TimeoutError(f"MCP server did not answer {method} in {self.timeout}s") from None
        self._replies.pop(request_id, None)
        return reply.get("result") or {}

    def _read_stdout(self):
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            waiting = self._replies.get(message.get("id"))
            if waiting is not None:
                waiting.put(message)

    def _read_stderr(self):
        for line in self._proc.stderr:
            self._stderr.append(line.rstrip())
            del self._stderr[:-20]


def report(label: str, call) -> None:
    started = time.monotonic()
    try:
        outcome = call()
    except Exception as exc:  # noqa: BLE001 - the point is what the text says
        outcome = f"{type(exc).__name__}: {exc}"
    print(f"  {label:<6} {time.monotonic() - started:5.2f}s  {outcome}")


def part_one() -> None:
    banner("1. a server that dies during the handshake")
    print("  It prints a traceback naming the missing setting and exits 3.")
    print(f"  The client was told to wait {FAST:.0f}s.\n")

    def old():
        client = OldClient(BROKEN + ["crash-at-start"])
        try:
            client.start()
            return "handshake returned"
        finally:
            client.close()

    def new():
        client = McpClient(BROKEN + ["crash-at-start"], timeout=FAST)
        try:
            client.start()
            return "handshake returned"
        finally:
            client.close()

    report("v1.4", old)
    report("v1.5", new)


def part_two() -> None:
    banner("2. a server that dies mid-session, during a tool call")
    print("  Same shape, except the run is already going: the model asked for a")
    print("  tool and the server exited instead of answering.\n")

    def old():
        client = OldClient(BROKEN + ["crash-on-call"])
        client.start()
        try:
            return client._request("tools/call", {"name": "echo", "arguments": {}})
        finally:
            client.close()

    def new():
        client = McpClient(BROKEN + ["crash-on-call"], timeout=FAST)
        client.start()
        try:
            return client.call_tool("echo", {})
        finally:
            client.close()

    report("v1.4", old)
    report("v1.5", new)


def part_three() -> None:
    banner("3. a server that pings the client after the handshake")
    print("  Legal in the spec, and the server will not answer `tools/list` until")
    print("  it gets the reply. Two processes, each waiting for the other.\n")

    def old():
        client = OldClient(BROKEN + ["pings-after-init"])
        client.start()
        try:
            return f"{len(client.list_tools())} tool(s)"
        finally:
            client.close()

    def new():
        client = McpClient(BROKEN + ["pings-after-init"], timeout=FAST)
        client.start()
        try:
            return f"{len(client.list_tools())} tool(s)"
        finally:
            client.close()

    report("v1.4", old)
    report("v1.5", new)


def part_four() -> None:
    banner("4. close() when the server wrapped another process")
    print("  The server's own child inherits its stdout. Killing the server leaves")
    print("  the child, the child keeps the pipe open, the reader thread keeps")
    print("  blocking in read(), and close() waits for that thread.\n")
    for label, factory in (("v1.4", OldClient), ("v1.5", McpClient)):
        with tempfile.TemporaryDirectory() as tmp:
            pidfile = Path(tmp) / "child.pid"
            client = factory(BROKEN + ["wraps-child", str(pidfile), str(CHILD_LIFE)], timeout=FAST)
            client.start()
            for _ in range(200):
                if pidfile.exists() and pidfile.read_text():
                    break
                time.sleep(0.01)
            child = int(pidfile.read_text())
            started = time.monotonic()
            client.close()
            elapsed = time.monotonic() - started
            # The child sleeps `CHILD_LIFE` seconds, so "still alive" here means
            # the shutdown returned without dealing with it.
            alive = Path(f"/proc/{child}").exists()
            if alive:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(child, 9)
            if elapsed >= CHILD_LIFE * 0.9:
                verdict = f"close() waited out the child's whole {CHILD_LIFE}s life"
            elif alive:
                verdict = "close() returned but the child is still running"
            else:
                verdict = "close() returned at once, and the child is gone"
            print(f"  {label:<6} {elapsed:5.2f}s  {verdict}")


def part_five() -> None:
    banner("5. what a user sees, through the real CLI")
    print("  A fake provider, one MCP server that crashes at startup.\n")
    code, output = _through_cli()
    for line in output.splitlines()[-6:]:
        print(f"    {line}")
    print(f"  exit code {code}")


def _through_cli() -> tuple[int, str]:
    """Run `oneharness run --mcp <broken server>` and return what it printed."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length", 0)) or 0)
            payload = json.dumps(
                {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "unreachable"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 1},
                }
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH": str(ROOT),
        "ONEHARNESS_PROVIDER": "openai",
        "ONEHARNESS_MODEL": "gpt-demo",
        "ONEHARNESS_API_KEY": "sk-demo",
        "ONEHARNESS_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
    }
    command = [
        sys.executable,
        "-m",
        "oneharness",
        "run",
        "list the files",
        "--mcp",
        f"{sys.executable} {BROKEN[1]} crash-at-start",
        "--quiet",
    ]
    try:
        done = subprocess.run(
            command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60
        )
    finally:
        server.shutdown()
        server.server_close()
    return done.returncode, (done.stdout + done.stderr)


def main() -> None:
    part_one()
    part_two()
    part_three()
    part_four()
    part_five()
    print("\nParts 1-4 ran one server script twice in this process, with only the")
    print("client swapped. Part 5 went through the CLI, so the exit code and the")
    print("message are the ones a user gets.")


if __name__ == "__main__":
    main()
