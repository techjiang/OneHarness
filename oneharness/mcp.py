"""A minimal MCP (Model Context Protocol) client over stdio.

MCP servers are just subprocesses speaking newline-delimited JSON-RPC on
stdin/stdout. That is small enough to implement with the standard library,
which keeps the "no dependencies" promise intact while making every MCP tool
available to the agent as an ordinary `Tool`.

Only the tools surface is implemented: `initialize`, `tools/list`,
`tools/call`. Resources and prompts exist in the protocol but the agent loop
has no use for them yet, and unused code is worse than missing code.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import signal
import subprocess
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from oneharness.errors import ToolError
from oneharness.tools import Tool
from oneharness.types import dumps

PROTOCOL_VERSION = "2024-11-05"
DEFAULT_TIMEOUT = 30.0
# How often a waiting request looks up to see whether the server died. A death
# is a fact that arrives on a pipe, not a deadline: noticing it takes as long as
# the next poll, not as long as the timeout.
POLL = 0.05


class McpError(ToolError):
    """The server refused, crashed, or answered something unusable."""


class McpServerGone(McpError):
    """The server process is not there any more.

    A separate class because it is the one MCP failure that is not about time:
    the same call seconds later will fail identically, and the answer is "look
    at why the server died", which is not what "did not answer in 30s" says.
    """


class McpClient:
    """One MCP server subprocess, spoken to over stdin/stdout."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: Path | str | None = None,
        env: dict[str, str] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        name: str | None = None,
    ) -> None:
        if not command:
            raise McpError("MCP server command must not be empty")
        self.command = list(command)
        self.cwd = str(cwd) if cwd else None
        self.env = {**os.environ, **env} if env else None
        self.timeout = timeout
        self.name = name or Path(self.command[0]).name
        self.server_info: dict[str, Any] = {}
        self._next_id = 0
        self._lock = threading.Lock()
        # Writes come from two threads: the caller sending a request, and the
        # reader answering a request the *server* sent us. Interleaved writes to
        # one pipe would produce two half-lines and a protocol error that looks
        # like the server's fault.
        self._write_lock = threading.Lock()
        self._replies: dict[int, queue.Queue[dict[str, Any]]] = {}
        self._stderr: list[str] = []
        self._proc: subprocess.Popen[str] | None = None
        # Set the instant the server's stdout closes: that is the earliest
        # honest signal that nothing will ever answer.
        self._gone = threading.Event()
        self.exit_code: int | None = None
        # Distinguishes "we stopped it" from "it died": `SIGTERM` in the message
        # would read as a crash to anyone who did not write the close() call.
        self.closed = False

    def __enter__(self) -> McpClient:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def start(self) -> None:
        if self._proc is not None:
            return
        try:
            self._proc = subprocess.Popen(
                self.command,
                cwd=self.cwd,
                env=self.env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                # Its own session, for the same reason the shell tool takes one:
                # a server that wraps another process leaves grandchildren that
                # survive a kill aimed at the parent, and an inherited pipe keeps
                # `close()` waiting on a process that is already gone.
                start_new_session=True,
            )
        except OSError as exc:
            raise McpError(f"cannot start MCP server {self.command!r}: {exc}") from exc
        threading.Thread(target=self._read_stdout, args=(self._proc,), daemon=True).start()
        threading.Thread(target=self._read_stderr, args=(self._proc,), daemon=True).start()

        from oneharness import __version__  # imported late: avoids a cycle

        self.server_info = self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "oneharness", "version": __version__},
            },
        ).get("serverInfo", {})
        self._notify("notifications/initialized", {})

    def close(self) -> None:
        """Stop the server and everything it started, without waiting on pipes.

        Three ways the old version lied or stalled, all of them here:

        * It closed `stdout` from this thread while the reader thread was still
          blocked reading it. `close()` waits for that lock, so a server whose
          grandchild inherited the pipe made `close()` block for as long as the
          grandchild lived (measured: 114s).
        * It waited on the process before killing it, so a server that ignores
          stdin ran the full 5s grace on every shutdown.
        * It killed only the server, leaving anything it spawned running.

        The reader thread owns its streams and closes them itself, so nothing
        here touches `proc.stdout`.
        """
        proc, self._proc = self._proc, None
        if proc is None:
            return
        self.closed = True
        self._gone.set()
        with contextlib.suppress(OSError, ValueError):
            if proc.stdin is not None:
                proc.stdin.close()
        self._terminate(proc)
        self.exit_code = proc.returncode

    @staticmethod
    def _terminate(proc: subprocess.Popen[str], grace: float = 0.5) -> None:
        """SIGTERM the whole group, then SIGKILL it. Never wait forever."""
        if proc.poll() is not None:
            return
        try:
            group = os.getpgid(proc.pid)
        except (ProcessLookupError, PermissionError):
            group = None
        for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 2.0)):
            if proc.poll() is not None:
                return
            try:
                if group is not None:
                    os.killpg(group, sig)
                else:
                    proc.kill()
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=wait)
            except subprocess.TimeoutExpired:
                continue

    # -- MCP methods -----------------------------------------------------

    def list_tools(self) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            reply = self._request("tools/list", {"cursor": cursor} if cursor else {})
            found.extend(reply.get("tools") or ())
            cursor = reply.get("nextCursor")
            if not cursor:
                return found

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        reply = self._request("tools/call", {"name": name, "arguments": arguments})
        if reply.get("isError"):
            raise McpError(_flatten(reply.get("content")))
        return _flatten(reply.get("content"))

    def tools(self, prefix: str = "") -> list[Tool]:
        """Expose the server's tools as local Tools, namespaced by server name."""
        exposed: list[Tool] = []
        for spec in self.list_tools():
            server_tool = spec.get("name")
            if not server_tool:
                continue
            exposed.append(self._as_tool(spec, f"{prefix}{server_tool}"))
        return exposed

    def _as_tool(self, spec: dict[str, Any], name: str) -> Tool:
        def invoke(**kwargs: Any) -> str:
            return self.call_tool(spec["name"], kwargs)

        return Tool(
            name=name,
            description=(spec.get("description") or spec["name"]).strip(),
            parameters=spec.get("inputSchema") or {"type": "object", "properties": {}},
            fn=invoke,
        )

    # -- JSON-RPC plumbing ------------------------------------------------

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._next_id += 1
            request_id = self._next_id
            waiting: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
            self._replies[request_id] = waiting
        try:
            self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
            reply = self._await_reply(waiting, method)
        finally:
            self._replies.pop(request_id, None)

        if "error" in reply:
            error = reply["error"] or {}
            raise McpError(f"{method} failed: {error.get('message') or error}")
        result = reply.get("result")
        return result if isinstance(result, dict) else {}

    def _await_reply(self, waiting: queue.Queue[dict[str, Any]], method: str) -> dict[str, Any]:
        """Wait for a reply, but stop the moment the server is known to be gone.

        A crashed server is not a slow server. Waiting out the full timeout for
        a process that has already exited turns "your server died, here is its
        traceback" into "did not answer in 30s", which sends the reader looking
        for a network or a hang that does not exist.
        """
        deadline = time.monotonic() + self.timeout
        while True:
            # The reply first, then the death: a server that answers and *then*
            # exits has answered, and reading the EOF as a failure would throw
            # away a fact in hand.
            try:
                return waiting.get_nowait()
            except queue.Empty:
                pass
            if self._gone.is_set():
                raise self._gone_error(method)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                message = f"MCP server {self.name!r} did not answer {method} in {self.timeout}s"
                raise McpError(f"{message}{self._tail()}")
            try:
                return waiting.get(timeout=min(POLL, remaining))
            except queue.Empty:
                continue

    def _gone_error(self, method: str) -> McpServerGone:
        message = f"MCP server {self.name!r} {self._state()} before answering {method}"
        return McpServerGone(f"{message}{self._tail()}")

    def _tail(self) -> str:
        """The server's last words, because they are usually the whole answer."""
        lines = [line for line in self._stderr if line.strip()]
        return f"; last stderr: {' | '.join(lines[-3:])}" if lines else ""

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _send(self, payload: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.poll() is not None:
            raise McpServerGone(f"MCP server {self.name!r} {self._state()}{self._tail()}")
        with self._write_lock:
            try:
                proc.stdin.write(json.dumps(payload) + "\n")
                proc.stdin.flush()
            except (BrokenPipeError, ValueError) as exc:
                raise McpServerGone(
                    f"MCP server {self.name!r} closed its stdin: {exc}{self._tail()}"
                ) from exc

    def _state(self) -> str:
        if self.closed:
            return "was closed by the client"
        code = self.exit_code if self._proc is None else self._proc.poll()
        return "is not running" if code is None else f"exited with code {code}"

    def _read_stdout(self, proc: subprocess.Popen[str]) -> None:
        stream = proc.stdout
        if stream is None:
            self._gone.set()
            return
        try:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue  # servers sometimes log to stdout; ignore noise
                if not isinstance(message, dict):
                    continue
                self._route(message)
        except (OSError, ValueError):
            pass  # the pipe died under us; `_gone` below says so
        finally:
            # EOF on stdout is the server's answer to everything from now on.
            self._gone.set()
            with contextlib.suppress(OSError, ValueError):
                stream.close()

    def _route(self, message: dict[str, Any]) -> None:
        """Deliver a reply, or answer a request the server sent us.

        The protocol runs both ways: a server may send `ping` or its own
        requests and wait for the reply. The old version dropped anything
        without one of our ids on the floor, so a server that pings after
        `initialize` -- which the spec encourages -- waited forever while the
        client waited for it, and the client blamed the timeout.
        """
        if "method" in message:
            if "id" not in message:
                return  # a notification; we advertise no capabilities to notify
            self._reply_to_server(message)
            return
        waiting = self._replies.get(message.get("id"))
        if waiting is not None:
            waiting.put(message)

    def _reply_to_server(self, request: dict[str, Any]) -> None:
        method = request.get("method")
        reply: dict[str, Any] = {"jsonrpc": "2.0", "id": request.get("id")}
        if method == "ping":
            reply["result"] = {}  # the one request we can actually satisfy
        else:
            reply["error"] = {"code": -32601, "message": f"method not supported: {method}"}
        # Nobody left to tell if this fails; `_gone` already covers the caller.
        with contextlib.suppress(McpError):
            self._send(reply)

    def _read_stderr(self, proc: subprocess.Popen[str]) -> None:
        stream = proc.stderr
        if stream is None:
            return
        try:
            for line in stream:
                self._stderr.append(line.rstrip())
                del self._stderr[:-20]
        except (OSError, ValueError):
            pass
        finally:
            with contextlib.suppress(OSError, ValueError):
                stream.close()


def _flatten(content: Any) -> str:
    """Turn MCP's content blocks into the plain text a model expects."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return dumps(content)
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
        elif block.get("type") == "text":
            parts.append(block.get("text") or "")
        elif block.get("type") == "resource":
            resource = block.get("resource") or {}
            parts.append(resource.get("text") or resource.get("uri") or "")
        else:
            parts.append(dumps(block))
    return "\n".join(part for part in parts if part)


def connect_mcp(
    command: Sequence[str],
    *,
    prefix: str = "",
    cwd: Path | str | None = None,
    env: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> McpClient:
    """Start a server and return a connected client (caller owns close())."""
    client = McpClient(command, cwd=cwd, env=env, timeout=timeout)
    client.start()
    client.list_tools()  # fail here, at startup, instead of mid-run
    return client
