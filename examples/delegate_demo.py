"""Sub agents and parallel tools, driven through the real CLI.

    python3 examples/delegate_demo.py

Two things get measured here rather than described:

1. the same three shell tool calls run sequentially and then in parallel;
2. a `delegate` tool call, where the sub agent reads two files and the parent
   sees one line of answer instead of the whole transcript.

The model is a scripted fake served over local HTTP; everything else (the CLI,
the agent loop, the tool executor, the sub agent) is the real code.
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

# Staggered on purpose: run in parallel the answers come back in reverse, which
# is the visible proof that they overlapped rather than queued.
SHELL_CALLS = [
    "sleep 0.9 && echo alpha",
    "sleep 0.5 && echo beta",
    "sleep 0.2 && echo gamma",
]


def reply(message: dict, finish: str = "stop", usage=(120, 18)) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", **message}, "finish_reason": finish}],
        "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]},
    }


def tool_call(name: str, arguments: dict, call_id: str) -> dict:
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
    """Two scripts, chosen by URL prefix: /parallel/... and /delegate/..."""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        payload = self._answer(self.path, body)
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _answer(self, path: str, body: dict) -> dict:
        offered = [spec["function"]["name"] for spec in body.get("tools") or ()]
        messages = body.get("messages") or []
        turn = sum(1 for m in messages if m["role"] == "assistant") + 1

        if path.startswith("/parallel/"):
            return self._parallel_script(turn)
        if "delegate" in offered:  # the parent agent in delegate mode
            if turn == 1:
                return tool_call(
                    "delegate",
                    {"task": "读取 app.conf 和 notes.md，告诉我超时配了多少"},
                    "call_sub",
                )
            answer = "子 agent 说 app.conf 里 timeout 是 5，notes.md 标注这个值太小。"
            return reply({"content": answer})

        # no `delegate` offered -> this request is from the sub agent
        if turn == 1:
            return tool_call("read_file", {"path": "app.conf"}, "call_r1")
        if turn == 2:
            return tool_call("read_file", {"path": "notes.md"}, "call_r2")
        answer = "app.conf: timeout=5；notes.md 写着 5 秒对真实请求太短。结论：需要调大。"
        return reply({"content": answer}, usage=(340, 44))

    def _parallel_script(self, turn: int) -> dict:
        if turn == 1:
            calls = [
                {
                    "id": f"call_{index}",
                    "type": "function",
                    "function": {"name": "run_shell", "arguments": json.dumps({"command": cmd})},
                }
                for index, cmd in enumerate(SHELL_CALLS)
            ]
            return reply({"content": None, "tool_calls": calls}, finish="tool_calls")
        return reply({"content": "三条命令都跑完了：alpha / beta / gamma。"})

    def log_message(self, *args):
        pass


def start_server() -> HTTPServer:
    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def run_cli(port: int, mode: str, *extra: str) -> tuple[str, str, float, float, int]:
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "ONEHARNESS_PROVIDER": "openai",
        "ONEHARNESS_MODEL": "gpt-demo",
        "ONEHARNESS_API_KEY": "sk-demo",
        "ONEHARNESS_BASE_URL": f"http://127.0.0.1:{port}/{mode}/v1",
        "PYTHONPATH": str(ROOT),
    }
    command = [
        sys.executable,
        "-m",
        "oneharness",
        "run",
        "把这三件事的都跑一遍，然后简短汇报",
        "--workspace",
        str(WORKSPACE),
        "--shell",
        "--json",  # stderr keeps the progress lines, stdout gives the numbers
        *extra,
    ]
    started = time.monotonic()
    done = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    elapsed = time.monotonic() - started
    try:
        payload = json.loads(done.stdout)
        tool_step = payload["steps"][0]["seconds"]
        answer = payload["text"]
    except (json.JSONDecodeError, KeyError, IndexError):
        tool_step, answer = -1.0, done.stdout.strip()
    return done.stderr, answer, elapsed, tool_step, done.returncode


def main() -> None:
    WORKSPACE.mkdir(exist_ok=True)
    (WORKSPACE / "app.conf").write_text("timeout=5\n")
    (WORKSPACE / "notes.md").write_text("- 5 秒对真实请求太短\n- 目标是 30 秒\n")

    server = start_server()
    port = server.server_port
    try:
        for label, flag in (
            ("顺序执行  --parallel=1", "--parallel=1"),
            ("并发执行  --parallel=3", "--parallel=3"),
        ):
            print(f"\n=== {label} ===")
            events, answer, elapsed, step, code = run_cli(port, "parallel", flag)
            sys.stdout.write(events)
            print(f"  {answer.strip()}")
            print(f"[the tool turn took {step:.2f}s | process {elapsed:.2f}s | exit {code}]")

        print("\n=== 子 agent 委派  --delegate ===")
        events, answer, elapsed, _step, code = run_cli(port, "delegate", "--delegate")
        sys.stdout.write(events)
        print(f"  {answer.strip()}")
        print(f"[process {elapsed:.2f}s | exit {code}]")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
