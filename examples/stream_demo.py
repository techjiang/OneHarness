"""Streaming demo against a local fake SSE provider. No network, no API key.

    python3 examples/stream_demo.py

Shows the two things streaming has to get right: tokens appear as they are
produced, and a tool call that spans several chunks is reassembled before the
agent loop ever sees it.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oneharness.agent import Agent
from oneharness.builtin import default_tools
from oneharness.providers.openai import OpenAIProvider
from oneharness.tools import Registry

# Chunks as a real provider sends them: answer text first, then a tool call
# whose JSON arguments are split across two frames.
CHUNKS: list[dict] = [
    {"choices": [{"index": 0, "delta": {"content": "Let me "}}]},
    {"choices": [{"index": 0, "delta": {"content": "check the conf"}}]},
    {"choices": [{"index": 0, "delta": {"content": "ig file."}}]},
    {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "function": {"name": "read_file", "arguments": '{"path":'},
                        }
                    ]
                },
            }
        ]
    },
    {
        "choices": [
            {
                "index": 0,
                "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"app.conf"}'}}]},
            }
        ]
    },
    {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    {"choices": [], "usage": {"prompt_tokens": 140, "completion_tokens": 22}},
]

FINAL_TEXT = [
    "Found it. ",
    "timeout was 5 seconds, ",
    "which is far too short for a real request. ",
    "I raised it to 30 in app.conf.",
]

TURN = {"n": 0}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", 0)))
        turn = TURN["n"]
        TURN["n"] += 1
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()

        if turn == 0:
            for chunk in CHUNKS:
                self._send(chunk)
        else:
            for piece in FINAL_TEXT:
                self._send({"choices": [{"index": 0, "delta": {"content": piece}}]})
                time.sleep(0.05)  # pretend the model is thinking
            self._send({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            self._send({"choices": [], "usage": {"prompt_tokens": 210, "completion_tokens": 34}})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _send(self, chunk):
        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.flush()

    def log_message(self, *args):
        pass


def main() -> None:
    workspace = Path(__file__).parent / ".demo-workspace"
    workspace.mkdir(exist_ok=True)
    (workspace / "app.conf").write_text("timeout=5\n")

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    provider = OpenAIProvider(
        "gpt-demo", api_key="sk-demo", base_url=f"http://127.0.0.1:{server.server_port}/v1"
    )

    def observe(event):
        if event["type"] == "text_delta":
            print(event["text"], end="", flush=True)
        elif event["type"] == "tool_start":
            print(f"\n  → {event['name']}({json.dumps(event['arguments'], ensure_ascii=False)})")
        elif event["type"] == "tool_end":
            preview = str(event["content"]).replace("\n", " ")[:60]
            print(f"  ✓ {event['name']}: {preview}")

    agent = Agent(provider, Registry(default_tools(workspace)), on_event=observe)
    try:
        result = agent.run("app.conf 的 timeout 是不是太小了？", stream=True)
    finally:
        server.shutdown()
        server.server_close()

    print("\n\n--- result ---")
    print("stopped_reason:", result.stopped_reason)
    print("usage:", result.usage.as_dict())
    print("requests sent:", TURN["n"])


if __name__ == "__main__":
    main()
