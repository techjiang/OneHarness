"""Drive the CLI end to end with a real MCP server subprocess.

    python3 examples/mcp_demo.py

The model is a scripted fake served over local HTTP; the MCP server is a real
child process that the CLI spawns via `--mcp`. What this proves is the wiring:
tools discovered over JSON-RPC show up in the model's tool list, and their
results come back into the loop as ordinary tool messages.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "tests" / "fixtures" / "fake_mcp_server.py"

TURNS = [
    {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "I will ask the MCP server to echo that.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": json.dumps({"text": "hello from the agent"}),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 88, "completion_tokens": 16},
    },
    {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "The MCP server echoed: hello from the agent.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 130, "completion_tokens": 12},
    },
]

STATE = {"turn": 0, "tools": []}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        # OpenAI-shaped payload: {"type": "function", "function": {...}}
        STATE["tools"] = [spec["function"]["name"] for spec in body.get("tools") or ()]
        reply = TURNS[min(STATE["turn"], len(TURNS) - 1)]
        STATE["turn"] += 1
        payload = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def main() -> None:
    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{server.server_port}/v1"

    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "ONEHARNESS_PROVIDER": "openai",
        "ONEHARNESS_MODEL": "gpt-demo",
        "ONEHARNESS_API_KEY": "sk-demo",
        "ONEHARNESS_BASE_URL": base_url,
        "PYTHONPATH": str(ROOT),
    }
    command = [
        sys.executable,
        "-m",
        "oneharness",
        "run",
        "echo something through the MCP server",
        "--workspace",
        str(ROOT),
        "--mcp",
        f"{sys.executable} {SERVER}",
    ]
    print("$ " + " ".join(command[-6:]) + "\n")
    completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    server.shutdown()
    server.server_close()

    sys.stdout.write(completed.stdout)
    sys.stdout.write(completed.stderr)
    print("\n--- what the model was offered ---")
    print("tools:", STATE["tools"])
    print("exit code:", completed.returncode)


if __name__ == "__main__":
    main()
