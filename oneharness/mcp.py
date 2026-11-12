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

import json
import os
import queue
import subprocess
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from oneharness.errors import ToolError
from oneharness.tools import Tool
from oneharness.types import dumps

PROTOCOL_VERSION = "2024-11-05"
DEFAULT_TIMEOUT = 30.0


class McpError(ToolError):
    """The server refused, crashed, or answered something unusable."""


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
        self._replies: dict[int, queue.Queue[dict[str, Any]]] = {}
        self._stderr: list[str] = []
        self._proc: subprocess.Popen[str] | None = None

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
            )
        except OSError as exc:
            raise McpError(f"cannot start MCP server {self.command!r}: {exc}") from exc
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

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
        proc, self._proc = self._proc, None
        if proc is None:
            return
        for stream in (proc.stdin,):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass

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
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})

        try:
            reply = waiting.get(timeout=self.timeout)
        except queue.Empty:
            self._replies.pop(request_id, None)
            message = f"MCP server {self.name!r} did not answer {method} in {self.timeout}s"
            raise McpError(message) from None
        self._replies.pop(request_id, None)

        if "error" in reply:
            error = reply["error"] or {}
            raise McpError(f"{method} failed: {error.get('message') or error}")
        result = reply.get("result")
        return result if isinstance(result, dict) else {}

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _send(self, payload: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.poll() is not None:
            detail = "; ".join(self._stderr[-3:])
            message = f"MCP server {self.name!r} is not running"
            raise McpError(f"{message}: {detail}" if detail else message)
        try:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            raise McpError(f"MCP server {self.name!r} closed its stdin: {exc}") from exc

    def _read_stdout(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue  # servers sometimes log to stdout; ignore noise
            if not isinstance(message, dict):
                continue
            waiting = self._replies.get(message.get("id"))
            if waiting is not None:
                waiting.put(message)
            # Server-initiated requests are ignored: we advertise no capabilities.

    def _read_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            self._stderr.append(line.rstrip())
            del self._stderr[:-20]


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
