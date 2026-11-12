"""Where did the time go: model, tools, or the pipe?

    python3 examples/timing_demo.py

A run reports one number, "it took four seconds", and that number hides two
completely different problems. Waiting for the model and waiting for a tool
have opposite fixes -- a shorter context or a faster model on one side, a
narrower tool or more parallelism on the other -- so `--timing` splits every
step into the part spent in the provider and the part spent in tools.

Three measurements, all through the real CLI against a local server:

1. **Same total, different culprits.** Two runs both take about a second. One
   spends it thinking, the other spends it in a shell. The totals agree and the
   breakdowns disagree, which is the only reason to have a breakdown at all.
2. **Parallel tools.** Three `sleep` calls run one by one and then side by
   side. The sum of the calls is the same both times; the wall clock is not,
   and that gap is exactly what `--parallel` bought.
3. **Time to first token.** A provider that thinks for 400 ms and then writes
   fast is not the same as one that starts at once and dribbles -- to a person
   staring at the terminal they feel identical. Only the first one has a long
   "first token" and a short "generation".

No network, no API key: the server is local and its timings are scripted.
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


def sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def text_chunk(piece: str) -> dict:
    return {"choices": [{"index": 0, "delta": {"content": piece}}]}


def tool_chunk(index: int, call_id: str | None = None, name: str | None = None, args=None) -> dict:
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


def finish(reason: str) -> dict:
    return {"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]}


class Script:
    """What the fake provider does on one run: think, then stream some frames."""

    def __init__(self, turns, think=0.0, gap=0.0):
        self.turns = turns  # list[list[dict]] -- one list of frames per request
        self.think = think  # seconds of silence before the first byte
        self.gap = gap  # seconds between frames
        self.turn = 0
        self.requests = 0


def assemble(frames: list[dict]) -> dict:
    """Fold streamed deltas into the one message a non-streaming request gets."""
    text: list[str] = []
    calls: dict[int, dict] = {}
    reason = "stop"
    for frame in frames:
        choice = (frame.get("choices") or [{}])[0]
        if choice.get("finish_reason"):
            reason = choice["finish_reason"]
        delta = choice.get("delta") or {}
        if delta.get("content"):
            text.append(delta["content"])
        for raw in delta.get("tool_calls") or ():
            slot = calls.setdefault(raw["index"], {"id": "", "name": "", "arguments": ""})
            slot["id"] = raw.get("id") or slot["id"]
            function = raw.get("function") or {}
            slot["name"] = function.get("name") or slot["name"]
            slot["arguments"] += function.get("arguments") or ""
    message: dict = {"role": "assistant", "content": "".join(text) or None}
    if calls:
        message["tool_calls"] = [
            {
                "id": slot["id"],
                "type": "function",
                "function": {"name": slot["name"], "arguments": slot["arguments"]},
            }
            for _, slot in sorted(calls.items())
        ]
    return {
        "choices": [{"index": 0, "message": message, "finish_reason": reason}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 18},
    }


class Handler(BaseHTTPRequestHandler):
    script = Script([])

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        Handler.script.requests += 1
        frames = Handler.script.turns[min(Handler.script.turn, len(Handler.script.turns) - 1)]
        Handler.script.turn += 1
        streaming = bool(body.get("stream"))

        if not streaming:
            # A non-streaming run has no "first token" to observe, and this
            # demo must not pretend otherwise: the same scripted delay applies,
            # the answer is delivered in one piece.
            time.sleep(Handler.script.think + Handler.script.gap * len(frames))
            payload = json.dumps(assemble(frames)).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        # Silence before the first byte: this is what a slow model feels like
        # and it is invisible in a total that also includes the writing.
        time.sleep(Handler.script.think)
        try:
            for frame in frames:
                self.wfile.write(sse(frame))
                self.wfile.flush()
                if Handler.script.gap:
                    time.sleep(Handler.script.gap)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


def run(port: int, prompt: str, *extra: str) -> subprocess.CompletedProcess:
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
        prompt,
        "--workspace",
        str(WORKSPACE),
        "--shell",
        *extra,
    ]
    return subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)


def show(done: subprocess.CompletedProcess, command: str) -> None:
    print(f"$ {command}")
    for line in (done.stderr or "").rstrip().splitlines():
        print(f"  {line}")
    print(f"  {done.stdout.strip()}")
    print(f"  [exit {done.returncode}]\n")


def delegate_turns(task: str) -> list[list[dict]]:
    """Parent asks for a delegation; the sub agent then thinks and answers."""
    call = tool_chunk(0, "call_d", "delegate", json.dumps({"task": task}))
    return [
        [call, finish("tool_calls")],
        [text_chunk("Sub agent "), text_chunk("finished."), finish("stop")],
        [text_chunk("Parent "), text_chunk("done."), finish("stop")],
    ]


def shell_turns(command: str, calls: int = 1) -> list[list[dict]]:
    """One turn asking for `calls` identical shell calls, then one that answers.

    The arguments are split across two frames on purpose: that is what a real
    stream looks like, and it is why tool calls can be announced before they are
    complete.
    """
    frames: list[dict] = []
    for index in range(calls):
        arguments = json.dumps({"command": command})
        frames.append(tool_chunk(index, f"call_{index}", "run_shell", arguments[:8]))
        frames.append(tool_chunk(index, args=arguments[8:]))
    frames.append(finish("tool_calls"))
    return [frames, [text_chunk("Done. "), text_chunk("The shell said so."), finish("stop")]]


def main() -> None:
    WORKSPACE.mkdir(exist_ok=True)
    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_port
    try:
        print("=== 1. the same second, spent in two different places ===")
        print("Both runs take about a second. One waits for the model, the other")
        print("waits for a shell. The total is the same number in both cases.\n")

        Handler.script = Script(
            [[text_chunk("Thinking "), text_chunk("it over."), finish("stop")]], think=0.9
        )
        show(
            run(port, "say something", "--timing", "--stream"),
            "oneharness run 'say something' --stream --timing   (provider thinks 900ms)",
        )

        Handler.script = Script(shell_turns("sleep 0.9 && echo slow", calls=1))
        show(
            run(port, "run the slow command", "--timing", "--stream"),
            "oneharness run 'run the slow command' --stream --timing   (shell sleeps 900ms)",
        )
        print("  model 0.9s in the first row, tools 0.9s in the second: same wall clock,")
        print("  different file to open.\n")

        print("=== 2. parallel tools: the sum of the calls is not the wall clock ===")
        print("Three `sleep 0.3` calls. Serially the calls sum to the wall clock;")
        print("in parallel they sum to roughly three times it.\n")

        Handler.script = Script(shell_turns("sleep 0.3 && echo hi", calls=3))
        show(
            run(port, "run three commands", "--timing"),
            "oneharness run 'run three commands' --timing   (sequential)",
        )
        Handler.script = Script(shell_turns("sleep 0.3 && echo hi", calls=3))
        show(
            run(port, "run three commands", "--timing", "--parallel", "3"),
            "oneharness run 'run three commands' --timing --parallel 3",
        )

        print("=== 3. time to first token vs time spent writing ===")
        print("Two runs of about the same length. The first thinks 700ms and then")
        print("writes fast; the second starts instantly and dribbles. Same total,")
        print("opposite diagnosis -- and no way to tell them apart without the split.\n")

        fast = [text_chunk("One. "), text_chunk("Two. "), text_chunk("Three."), finish("stop")]
        Handler.script = Script([fast], think=0.7)
        show(
            run(port, "count to three", "--timing", "--stream"),
            "oneharness run 'count to three' --stream --timing   (thinks, then bursts)",
        )
        Handler.script = Script([fast], think=0.0, gap=0.23)
        show(
            run(port, "count to three", "--timing", "--stream"),
            "oneharness run 'count to three' --stream --timing   (starts at once, dribbles)",
        )
        print("  first token 700ms in the first run, 3ms in the second, with a similar")
        print("  total. The first wants a smaller prompt; the second wants fewer tokens.\n")

        print("=== 4. delegated time is not tool time ===")
        print("A `delegate` call is a tool call, so its wait lands in tools. But what")
        print("it waits for is another agent thinking -- reporting that as a slow tool")
        print("is a wrong diagnosis with a confident number attached.\n")

        Handler.script = Script(delegate_turns("look at the config"), think=0.3)
        show(
            run(port, "delegate the config check", "--timing", "--delegate"),
            "oneharness run 'delegate the config check' --timing --delegate",
        )
        print("  the `tool time was sub agents thinking` line is the one that says")
        print("  the 300ms under tools was not a slow shell.\n")

        print("=== 5. the same numbers, in --json ===")
        Handler.script = Script(shell_turns("sleep 0.2 && echo hi", calls=2))
        done = run(port, "run two commands", "--json", "--parallel", "2")
        payload = json.loads(done.stdout)
        print("$ oneharness run 'run two commands' --json --parallel 2")
        print(json.dumps(payload["timing"], indent=2, ensure_ascii=False))
        print("\n  `timing.tool_call_seconds` above `timing.tool_seconds` is the")
        print("  parallelism, measured rather than claimed. `unaccounted_seconds`")
        print("  is what the loop did not measure -- it is printed even when small,")
        print("  because a breakdown that always adds up is inventing its numbers.")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
