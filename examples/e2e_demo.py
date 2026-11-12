"""End-to-end demo against a local fake OpenAI-compatible server.

No network and no API key required:

    python3 examples/e2e_demo.py
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oneharness.agent import Agent
from oneharness.builtin import default_tools
from oneharness.providers.openai import OpenAIProvider
from oneharness.tools import Registry

SCRIPT = [
    # turn 1: the model asks to read a file
    {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Let me look at the config first.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": json.dumps({"path": "app.conf"}),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 120, "completion_tokens": 18},
    },
    # turn 2: it writes the fix
    {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": json.dumps(
                                    {"path": "app.conf", "content": "timeout=30\n"}
                                ),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 190, "completion_tokens": 31},
    },
    # turn 3: final answer
    {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Found it: timeout was 5, bumped it to 30 in app.conf.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 240, "completion_tokens": 20},
    },
]


class Handler(BaseHTTPRequestHandler):
    turn = 0
    seen = []

    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(length))
        Handler.seen.append(body)
        reply = SCRIPT[min(Handler.turn, len(SCRIPT) - 1)]
        Handler.turn += 1
        payload = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def main() -> None:
    workspace = Path(__file__).parent / ".demo-workspace"
    workspace.mkdir(exist_ok=True)
    (workspace / "app.conf").write_text("timeout=5\n")

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{server.server_port}/v1"

    provider = OpenAIProvider("gpt-demo", api_key="sk-demo", base_url=base_url)
    agent = Agent(
        provider,
        Registry(default_tools(workspace)),
        system="You are a careful config fixer.",
        on_event=lambda e: print(f"  [{e['type']}] {e.get('name', '')}".rstrip()),
    )

    try:
        result = agent.run("app.conf has a too-small timeout, please fix it.")
    finally:
        server.shutdown()
        server.server_close()

    print("\n--- result ---")
    print("text:", result.text)
    print("stopped_reason:", result.stopped_reason)
    print("usage:", result.usage.as_dict())
    print("app.conf now:", (workspace / "app.conf").read_text().strip())
    print("requests sent:", len(Handler.seen))
    print("tool messages model saw:", [m["role"] for m in Handler.seen[1]["messages"]])


if __name__ == "__main__":
    main()
