"""MCP servers that misbehave in the three ways a client has to survive.

    python3 broken_mcp_server.py <mode>

`crash-at-start`   exits before the handshake, with a traceback on stderr
`crash-on-list`    handshakes, then dies when asked for its tools
`crash-on-call`    handshakes and lists a tool, then dies when it is called
`pings-after-init` handshakes, then sends the client a `ping` request and will
                   not answer `tools/list` until that ping is replied to
`wraps-child <pidfile> [seconds]`
                   handshakes while a grandchild of its own inherits the pipe,
                   so the pipe outlives the server unless the group is killed
"""

from __future__ import annotations

import json
import subprocess
import sys

TOOLS = [
    {
        "name": "echo",
        "description": "Echo the given text back.",
        "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}},
    }
]

HANDSHAKE = {
    "protocolVersion": "2024-11-05",
    "capabilities": {"tools": {}},
    "serverInfo": {"name": "broken", "version": "0"},
}


def out(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def die(reason: str) -> None:
    trace = f'Traceback (most recent call last):\n  File "server.py", line 12\n{reason}\n'
    sys.stderr.write(trace)
    sys.stderr.flush()
    sys.exit(3)


def main(mode: str) -> None:
    if mode == "crash-at-start":
        die("ValueError: MISSING_TOKEN is not set")
        return

    child_pid_file = None
    if mode == "wraps-child":
        child_pid_file = sys.argv[2] if len(sys.argv) > 2 else None
        seconds = sys.argv[3] if len(sys.argv) > 3 else "120"
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import os, sys, time\n"
                "open(sys.argv[1], 'w').write(str(os.getpid()))\n"
                "time.sleep(float(sys.argv[2]))\n",
                child_pid_file,
                seconds,
            ]
        )
        assert child.poll() is None

    pending: object = None
    pinged = False
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        method = request.get("method")
        if method == "initialize":
            out({"jsonrpc": "2.0", "id": request["id"], "result": HANDSHAKE})
        elif method == "tools/list" and not pinged and mode == "pings-after-init":
            pinged = True
            pending = request["id"]
            out({"jsonrpc": "2.0", "id": "srv-ping-1", "method": "ping", "params": {}})
        elif method == "tools/list":
            if mode == "crash-on-list":
                die("KeyError: 'database_url'")
            out({"jsonrpc": "2.0", "id": request["id"], "result": {"tools": TOOLS}})
        elif method == "tools/call":
            if mode == "crash-on-call":
                die("ConnectionResetError: the database went away")
            out({"jsonrpc": "2.0", "id": request["id"], "result": {"content": []}})
        elif "id" in request and method is None and pending is not None:
            # The client answered our ping; only now does it get its tools.
            out({"jsonrpc": "2.0", "id": pending, "result": {"tools": []}})
            pending = None


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "crash-at-start")
