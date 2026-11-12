"""Fan-out: three sub agents in one turn, driven through the real CLI.

    python3 examples/fanout_demo.py

The parent asks for three independent things in a single turn. With
`--parallel 1` the sub agents are spawned one after another; with
`--parallel 3` they overlap. Each sub agent itself runs a shell tool that
sleeps, so the wall clock is the honest measurement -- and the per-sub-agent
ledger shows who spent what, which is the part that used to be lost.

The model is a scripted fake served over local HTTP; the CLI, the agent loop,
the tool executor and the sub agents are all the real code.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path(__file__).parent / ".demo-workspace"

TOPICS = [
    ("alpha", "app.conf 里的 timeout 是多少？"),
    ("beta", "notes.md 里的结论是什么？"),
    ("gamma", "README 有没有提到 0.4？"),
]
SLEEP = "sleep 0.7 && cat {}"


def reply(message: dict, finish: str = "stop", usage=(120, 18)) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", **message}, "finish_reason": finish}],
        "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]},
    }


def call(name: str, arguments: dict, call_id: str) -> dict:
    return reply(
        {
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        },
        finish="tool_calls",
    )


class Handler(BaseHTTPRequestHandler):
    """One script for the parent, one per sub agent, keyed off the message list."""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        payload = self._answer(body)
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _answer(self, body: dict) -> dict:
        offered = [spec["function"]["name"] for spec in body.get("tools") or ()]
        messages = body.get("messages") or []
        turns = sum(1 for m in messages if m["role"] == "assistant") + 1

        # The parent is the only one offered the fan-out tool.
        if "delegate_many" in offered:
            if turns == 1:
                return call("delegate_many", {"tasks": [q for _, q in TOPICS]}, "call_fan")
            results = [m["content"] for m in messages if m["role"] == "tool"]
            return reply({"content": "三件事都查完了：" + "；".join(results)}, usage=(480, 96))

        # A sub agent. It never sees the parent's prompt: the only user message
        # in its own list is the single task it was handed.
        task = next((m["content"] for m in messages if m["role"] == "user"), "") or ""
        topic = next((name for name, question in TOPICS if question in task), "unknown")
        file_for = {"alpha": "app.conf", "beta": "notes.md", "gamma": "README.md"}[topic]
        if turns == 1:
            return call("run_shell", {"command": SLEEP.format(file_for)}, f"call_{topic}")
        return reply({"content": f"{topic}: 读完了 {file_for}"}, usage=(300, 40))

    def log_message(self, *args):
        pass


def start_server() -> HTTPServer:
    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def run_cli(port: int, *extra: str) -> tuple[str, dict, float, int]:
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "ONEHARNESS_PROVIDER": "openai",
        "ONEHARNESS_MODEL": "gpt-demo",
        "ONEHARNESS_API_KEY": "sk-demo",
        "ONEHARNESS_BASE_URL": f"http://127.0.0.1:{port}/v1",
        "PYTHONPATH": str(ROOT),
    }
    command = [
        sys.executable,
        "-m",
        "oneharness",
        "run",
        "同时查这三件事，最后汇总",
        "--workspace",
        str(WORKSPACE),
        "--shell",
        "--delegate",
        "--json",
        *extra,
    ]
    started = time.monotonic()
    done = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    elapsed = time.monotonic() - started
    try:
        payload = json.loads(done.stdout)
    except json.JSONDecodeError:
        payload = {"text": done.stdout.strip()}
    return done.stderr, payload, elapsed, done.returncode


def main() -> None:
    WORKSPACE.mkdir(exist_ok=True)
    (WORKSPACE / "app.conf").write_text("timeout=5\n")
    (WORKSPACE / "notes.md").write_text("- 5 秒对真实请求太短\n")
    (WORKSPACE / "README.md").write_text("# demo\n\n版本 0.3.0\n")

    server = start_server()
    port = server.server_port
    try:
        for label, flag in (
            ("逐个 spawn  --parallel=1", "--parallel=1"),
            ("同时 spawn  --parallel=3", "--parallel=3"),
        ):
            print(f"\n=== {label} ===")
            events, payload, elapsed, code = run_cli(port, flag)
            sys.stdout.write(events)
            print(f"  {payload.get('text', '').strip()}")
            delegation = payload.get("delegation") or {}
            spent = delegation.get("usage", {})
            # The per-sub-agent ledger is what `--json` now carries; read it back
            # from the payload so this demo proves the numbers rather than prints them.
            labels = sorted(delegation.get("by_label") or {})
            print(
                f"[wall {elapsed:.2f}s | {delegation.get('runs', 0)} sub agent(s) | "
                f"{spent.get('input_tokens', 0)} in / {spent.get('output_tokens', 0)} out | "
                f"ledger {', '.join(labels)} | exit {code}]"
            )
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
