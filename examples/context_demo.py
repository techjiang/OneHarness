"""Context budget: what happens when a tool result is bigger than the window.

    python3 examples/context_demo.py

Three runs against a local fake OpenAI-compatible server that behaves like a
real endpoint in the one way that matters here: it refuses any request whose
prompt is over its window, and it says so in the provider's own words
(`maximum context length is N tokens`). The tool result is a 40 KB log, so the
second request of every run is about 10,000 tokens.

The three runs are not variations of a setting; they are the three answers a
harness can give:

1. `--context-budget 1200` -- trim before sending. The old tool result is
   elided, the model is told it was, and the run finishes.
2. no budget -- the first request goes out whole and is refused. The refusal is
   read as a budget problem rather than a wire-format bug, the budget is
   derived from what was just sent, and the retry succeeds. This is the case
   that used to be classified `protocol`, i.e. "this is a OneHarness bug".
3. no budget, and nothing to elide -- the prompt itself is over the window.
   Dropping it to fit would answer a different question, so the run stops and
   reports the provider's own sentence instead of looping.

The CLI, the agent loop, the trimmer and the retry are all the real code.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oneharness import doctor  # noqa: E402
from oneharness.errors import ProviderError  # noqa: E402

WORKSPACE = Path(__file__).parent / ".demo-workspace"
LIMIT = 2000  # the fake window, in tokens
LOG_BYTES = 40_000


def estimate(payload: dict) -> int:
    """A crude token count of the request body, the way a provider counts it."""
    chars = 0
    for message in payload.get("messages") or ():
        chars += len(message.get("content") or "")
        for call in message.get("tool_calls") or ():
            chars += len(call["function"]["arguments"])
    return chars // 4


class Handler(BaseHTTPRequestHandler):
    """Refuses oversized prompts, then plays a two-turn tool loop."""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        size = estimate(body)
        if size > LIMIT:
            self._send(
                400,
                {
                    "error": {
                        "message": (
                            f"This model's maximum context length is {LIMIT} tokens. "
                            f"However, your messages resulted in {size} tokens."
                        ),
                        "type": "invalid_request_error",
                        "code": "context_length_exceeded",
                    }
                },
            )
            return
        self._send(200, self._answer(body))

    def _answer(self, body: dict) -> dict:
        turns = sum(1 for m in body["messages"] if m["role"] == "assistant") + 1
        if turns == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "先看看这份日志。",
                            "tool_calls": [
                                {
                                    "id": "call_big",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": json.dumps({"path": "big.log"}),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 120, "completion_tokens": 18},
            }
        results = [m.get("content") or "" for m in body["messages"] if m["role"] == "tool"]
        elided = "elided" in results[-1] if results else False
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": (
                            f"big.log 一共 {LOG_BYTES} 字节，约 {LOG_BYTES // 4} tokens，"
                            "比这个窗口大。"
                            + ("上一次的结果被裁掉了，我手上只剩占位符。" if elided else "")
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 32},
        }

    def _send(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


def run_cli(port: int, prompt: str, *extra: str) -> tuple[str, dict, int]:
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
        prompt,
        "--workspace",
        str(WORKSPACE),
        "--json",
        *extra,
    ]
    done = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    try:
        payload = json.loads(done.stdout)
    except json.JSONDecodeError:
        payload = {"text": done.stdout.strip()}
    return done.stderr, payload, done.returncode


def show(label: str, stderr: str, payload: dict, code: int) -> None:
    print(f"\n=== {label} ===")
    lines = [line for line in stderr.strip().splitlines() if line.strip()]
    # The long `body=` tail is the provider talking; it is worth reading, so it
    # gets its own wrapped line instead of being cut off by the terminal.
    for line in lines:
        if " | body=" in line:
            head, _, body = line.partition(" | body=")
            print(f"  {head.strip()}")
            try:
                said = json.loads(body)["error"]["message"]
            except (json.JSONDecodeError, KeyError, TypeError):
                said = body
            print(f"  provider said: {said}")
        else:
            print(line)
    text = payload.get("text", "").strip()
    print(f"  {text}" if text else "  (no answer: the run stopped)")
    trim = payload.get("trim") or {}
    if trim:
        print(
            f"[budget {trim['budget']} | {trim['tokens_before']} → {trim['tokens_after']} tokens "
            f"| elided {trim['elided']} | dropped {trim['dropped']} | exit {code}]"
        )
    else:
        print(f"[nothing trimmed | exit {code}]")


# The one comparison this whole file exists for: two 400s, same status, and
# `doctor` must not report them as the same kind of problem.
SAME_STATUS = [
    (
        "gemini, OpenAI-shaped body",
        '{"error":{"message":"Unknown name \\"messages\\": Cannot find field."}}',
    ),
    (
        "openai, oversized prompt",
        '{"error":{"message":"maximum context length is 128000 tokens, '
        'your messages resulted in 208310 tokens","code":"context_length_exceeded"}}',
    ),
]


def verdicts() -> None:
    print("\n=== 4. 同一个 400，两个相反的结论 ===")
    for label, body in SAME_STATUS:
        exc = ProviderError("failed", status=400, body=body)
        status, detail = doctor.classify(exc)
        print(f"  HTTP {exc.status}  {label:<28} -> {status:<9} {detail}")
    print(f"  {'':<9}  {'':<28}    ^ protocol 要改代码，context 要改请求大小")


def main() -> None:
    WORKSPACE.mkdir(exist_ok=True)
    (WORKSPACE / "big.log").write_text("2026-11-12 INFO request done\n" * 1300)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_port
    try:
        print(f"fake window: {LIMIT} tokens · tool result: {LOG_BYTES} bytes (~{LOG_BYTES // 4})")

        show(
            "1. --context-budget 1200：发之前就裁",
            *run_cli(port, "看看这份日志有多大", "--context-budget", "1200"),
        )
        show("2. 不给预算：被拒一次，读原文，重试", *run_cli(port, "看看这份日志有多大"))
        # 20000 chars of nothing to elide: the prompt itself is over the window.
        show(
            "3. 没东西可裁：停手，报出 provider 的原话",
            *run_cli(port, "重复这段：" + "日志" * 10_000),
        )
    finally:
        server.shutdown()
        server.server_close()

    verdicts()


if __name__ == "__main__":
    main()
