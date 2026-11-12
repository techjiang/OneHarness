"""Streamed tool calls: know what is about to run while it is still being written.

    python3 examples/stream_tools_demo.py

Before v0.8 a streamed run said a tool was coming only when the whole turn had
finished: the `done` event carried the arguments, the agent ran the tool, and
only then did anything appear. For a `write_file` with a two-kilobyte body that
means the tool is announced after the model has already written all of it --
which is exactly when the announcement stops being useful.

This demo shows the three things that came with the fix, all through the real
CLI and the real agent loop, against a local SSE server that sends one frame
every 200 ms:

1. **The announcement comes first.** A timeline of client-side arrival times:
   the tool name lands while the arguments are still arriving, second by
   second, not after them.
2. **Two calls assemble at once.** Fragments are interleaved by `index`, so a
   watcher has to be told which call each one belongs to.
3. **A dying stream keeps its half-answer.** The server cuts the connection
   mid-sentence; the text that already arrived is kept and the run says so,
   instead of throwing away the only thing that came back.

No network and no API key: the server is local and its events are canned.
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
sys.path.insert(0, str(ROOT))

from oneharness.agent import Agent  # noqa: E402
from oneharness.builtin import default_tools  # noqa: E402
from oneharness.providers.openai import OpenAIProvider  # noqa: E402
from oneharness.tools import Registry  # noqa: E402

GAP = 0.2  # seconds between frames, so the timeline measures something


# Read left to right: the name arrives with the first fragment of the body and
# every later frame adds to it. Both calls interleave on `index`.
def call(index: int, call_id: str | None = None, name: str | None = None, args: str | None = None):
    """One `tool_calls` delta, with only the fields a real frame would carry."""
    slot: dict = {"index": index}
    if call_id:
        slot["id"] = call_id
    function: dict = {}
    if name:
        function["name"] = name
    if args is not None:
        function["arguments"] = args
    if function:
        slot["function"] = function
    return {"choices": [{"index": 0, "delta": {"tool_calls": [slot]}}]}


def text(piece: str):
    return {"choices": [{"index": 0, "delta": {"content": piece}}]}


# Read in order: the name arrives with the first fragment of the body, and every
# later frame appends to it. Both calls interleave, told apart by `index`.
FIRST_TURN = [
    text("Two files to fix. "),
    call(0, "call_a", "write_file", '{"path":'),
    call(1, "call_b", "write_file"),
    call(0, args=' "a.conf",'),
    call(1, args='{"path":'),
    # `\\n` inside a JSON string: the two characters a real frame would carry.
    call(0, args=' "content": "timeout=30\\n"}'),
    call(1, args=' "b.conf", "content": "retries=3\\n"}'),
    {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
]

SECOND_TURN = ["Both written. ", "a.conf gets timeout=30, ", "b.conf gets retries=3."]


def sse(chunk: dict) -> bytes:
    return f"data: {json.dumps(chunk)}\n\n".encode()


class Handler(BaseHTTPRequestHandler):
    """Turn one plays the tool-call frames; turn two answers or dies."""

    die_at: int | None = None  # cut the connection after this many text pieces
    turns = {"n": 0}
    requests: list[dict] = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        Handler.requests.append(body)
        turn = Handler.turns["n"]
        Handler.turns["n"] += 1

        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        try:
            if turn == 0:
                for chunk in FIRST_TURN:
                    self.wfile.write(sse(chunk))
                    self.wfile.flush()
                    time.sleep(GAP)
            else:
                for index, piece in enumerate(SECOND_TURN):
                    if Handler.die_at is not None and index >= Handler.die_at:
                        # No [DONE], no finish_reason: the connection just ends.
                        self.close_connection = True
                        return
                    self.wfile.write(sse({"choices": [{"index": 0, "delta": {"content": piece}}]}))
                    self.wfile.flush()
                    time.sleep(GAP)
                done_chunk = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                self.wfile.write(sse(done_chunk))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the client stopped listening; that is the point of run 3

    def log_message(self, *args):
        pass


def reset(die_at: int | None = None) -> None:
    Handler.die_at = die_at
    Handler.turns["n"] = 0
    Handler.requests.clear()


def banner(text: str) -> None:
    print(f"\n=== {text} ===")


def run_cli(port: int, *extra: str) -> subprocess.CompletedProcess:
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH": str(ROOT),
        "ONEHARNESS_PROVIDER": "openai",
        "ONEHARNESS_MODEL": "gpt-demo",
        "ONEHARNESS_API_KEY": "sk-demo",
        "ONEHARNESS_BASE_URL": f"http://127.0.0.1:{port}/v1",
    }
    command = [
        sys.executable,
        "-m",
        "oneharness",
        "run",
        "把两个配置文件都改一下",
        "--workspace",
        str(WORKSPACE),
        "--stream",
        *extra,
    ]
    return subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)


def timeline(port: int) -> None:
    """Client-side arrival times: the only honest evidence of "before"."""
    provider = OpenAIProvider("gpt-demo", api_key="sk-demo", base_url=f"http://127.0.0.1:{port}/v1")
    marks: list[tuple[float, str]] = []

    def observe(event):
        kind = event["type"]
        if kind == "tool_call_delta":
            what = f"tool_call_delta   {event['name'] or '?':<10} fragment={event['fragment']!r}"
        elif kind == "tool_start":
            what = f"tool_start        {event['name']}"
        elif kind == "tool_end":
            what = f"tool_end          {str(event['content'])[:28]}"
        elif kind == "text_delta":
            what = f"text_delta        {event['text']!r}"
        else:
            return
        marks.append((time.monotonic(), what))

    agent = Agent(provider, Registry(default_tools(WORKSPACE)), on_event=observe)
    agent.run("把两个配置文件都改一下", stream=True)

    print("client-side arrival time of each event (frames are sent 200 ms apart):")
    base = marks[0][0]
    for moment, what in marks:
        print(f"  +{moment - base:5.2f}s  {what}")


def main() -> None:
    WORKSPACE.mkdir(exist_ok=True)
    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_port
    try:
        banner("1. what the terminal shows")
        reset()
        done = run_cli(port)
        print("$ oneharness run '把两个配置文件都改一下' --stream")
        for line in done.stderr.rstrip().splitlines():
            print(f"  {line}")
        print(f"{done.stdout.rstrip()}")
        print(f"[exit {done.returncode}]")

        banner("2. when each event actually arrived")
        reset()
        timeline(port)
        print("  the first fragment already carries the name: the tool is known")
        print("  before the model has finished writing its arguments.")

        banner("3. the stream dies mid-sentence")
        reset(die_at=2)
        done = run_cli(port)
        print("$ oneharness run '把两个配置文件都改一下' --stream   (server hangs up)")
        for line in done.stderr.rstrip().splitlines():
            print(f"  {line}")
        print(f"{done.stdout.rstrip()}")
        print(f"[exit {done.returncode}]  the two pieces that arrived are still there")

        banner("4. the same failure, without --stream")
        reset(die_at=2)
        done = run_cli(port, "--json")
        print("$ oneharness run '把两个配置文件都改一下' --json   (server hangs up)")
        for line in (done.stderr or done.stdout).rstrip().splitlines()[:3]:
            print(f"  {line}")
        print(f"[exit {done.returncode}]  nothing to keep: the body never parsed, so")
        print("  there is no partial answer to preserve. Streaming is what makes a")
        print("  half-answer exist in the first place.")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
