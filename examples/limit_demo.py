"""When the model runs out of output, nothing looks broken -- and nothing said so.

    python3 examples/limit_demo.py

Every provider answers "I was cut off at my output limit" in one word, and each
one spells it differently: OpenAI-shaped services say `finish_reason: length`,
Anthropic says `stop_reason: max_tokens`, Gemini says `finishReason:
MAX_TOKENS`, Bedrock says `stopReason: max_tokens`. Until v1.7 this harness
stored that word and never read it.

Which made it the quietest failure here. Compare it with the ones the previous
versions fixed:

  * a stream that dies loses bytes      -> `truncated`, and v0.8 kept the half
  * a network drop raises                -> `NetworkError` with a kind
  * a 400 about size                     -> `context overflow`, retried
  * output limit                         -> HTTP 200, valid JSON, terminal
                                           marker present, `stopped_reason:
                                           done`, exit code 0

The first three are loud in comparison. The fourth is a complete response to a
question the model had not finished answering, and every layer above the
adapter -- the loop, `--json`, the exit code -- was entitled to believe it was.

Part 1 runs the real CLI against a local SSE server that stops one frame short
of the answer and says why, the way OpenAI does.
Part 2 does the same for a cut-off tool call, where half the JSON reaches a
tool.
Part 3 shows the four dialects being read.

No network, no API key.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oneharness.agent import Agent  # noqa: E402
from oneharness.builtin import default_tools  # noqa: E402
from oneharness.tools import Registry  # noqa: E402
from oneharness.types import (  # noqa: E402
    Completion,
    Message,
    ToolCall,
    cut_off_reason,
    mark_incomplete_arguments,
)

GAP = 0.05
# The answer a model gives when its budget runs out in the middle of a
# sentence. This one is cut where a real one usually is: after the read.
CUT_SENTENCE = (
    "The timeout is 5 seconds, which is probably why the client gave up after three tries. "
)

# A `write_file` whose body never finished. `args` is a JSON *string*, as it
# arrives on the wire: `read1` splits it where the model stopped writing.
CUT_ARGS = '{"path": "app.conf", "content": "timeout=30\\nretries='


def frame(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def text_chunk(piece: str) -> dict:
    return {"choices": [{"index": 0, "delta": {"content": piece}}]}


def call_chunk(args: str | None = None, *, first: bool = False) -> dict:
    slot: dict = {"index": 0}
    function: dict = {}
    if first:
        slot["id"] = "call_cut"
        function["name"] = "write_file"
    if args is not None:
        function["arguments"] = args
    if function:
        slot["function"] = function
    return {"choices": [{"index": 0, "delta": {"tool_calls": [slot]}}]}


def finish(reason: str) -> dict:
    return {"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]}


class Handler(BaseHTTPRequestHandler):
    """Plays one scripted turn per request; the script is set per part."""

    script: list[dict] = []
    # `--stream` reads SSE; a blocking request reads JSON. Both are just the
    # same frames, so the only thing that changes here is how they are framed.
    streaming = True

    def do_POST(self):
        body_in = self.rfile.read(int(self.headers.get("content-length", 0)))
        streaming = json.loads(body_in or b"{}").get("stream", False)
        # A blocking request gets one JSON object: the same frames, read all at
        # once, with the last chunk being the one that carries the reason.
        body = (
            b"".join(frame(chunk) for chunk in Handler.script) + b"data: [DONE]\n\n"
            if streaming
            else json.dumps(Handler.script[-1]).encode()
        )
        self.send_response(200)
        self.send_header("content-type", "text/event-stream" if streaming else "application/json")
        self.send_header("content-length", str(len(body)))
        self.send_header("connection", "close")
        self.end_headers()
        try:
            if not streaming:
                self.wfile.write(body)
            else:
                frames = [frame(chunk) for chunk in Handler.script]
                for one in frames:
                    self.wfile.write(one)
                    self.wfile.flush()
                    time.sleep(GAP)
                self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True

    def log_message(self, *args):
        pass


def banner(text: str) -> None:
    print(f"\n=== {text} ===")


def env(port: int) -> dict[str, str]:
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH": str(ROOT),
        "ONEHARNESS_PROVIDER": "openai",
        "ONEHARNESS_MODEL": "gpt-demo",
        "ONEHARNESS_API_KEY": "sk-demo",
        "ONEHARNESS_BASE_URL": f"http://127.0.0.1:{port}/v1",
    }


def through_cli(
    port: int, workspace: Path, *extra: str, stream: bool = True
) -> subprocess.CompletedProcess:
    Handler.streaming = stream
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "oneharness",
            "run",
            "为什么客户端三秒就放弃了？",
            "--workspace",
            str(workspace),
            *(["--stream"] if stream else []),
            *extra,
        ],
        cwd=ROOT,
        env=env(port),
        capture_output=True,
        text=True,
    )


def show_result(label: str, done: subprocess.CompletedProcess) -> None:
    print(f"  {label}")
    print(f"    exit code      {done.returncode}")
    print(f"    stdout         {_short(done.stdout.strip())!r}")
    for line in done.stderr.strip().splitlines()[-2:]:
        print(f"    stderr         {line}")


def _short(text: str, limit: int = 100) -> str:
    """One line of the answer, because these are screenshots first."""
    return text if len(text) <= limit else text[:limit] + "..."


def part_one(port: int, workspace: Path) -> None:
    banner("1. an answer that stops at the limit, through the real CLI")
    Handler.script = [text_chunk(CUT_SENTENCE), finish("length")]
    done = through_cli(port, workspace)
    show_result("openai, finish_reason=length", done)
    print()
    print("  Before v1.7 the exit code was 0 and stderr was empty: `--json` said")
    print('  "stopped_reason": "done", which is the same word a finished run gets.')
    print("  Now the verdict is `truncated` -- the text is still the only copy of")
    print("  it, so it is printed, and the run does not claim to have succeeded.")

    done = through_cli(port, workspace, "--json", stream=False)
    Handler.script = [text_chunk(CUT_SENTENCE), finish("length")]
    try:
        payload = json.loads(done.stdout)
    except json.JSONDecodeError:
        print("\n  (unexpected: --json did not produce a document)")
        return
    print()
    print(f"  --json: stopped_reason={payload['stopped_reason']!r}")
    print(f"          cut_off={_short(payload.get('cut_off', '(absent)'), 60)!r}")
    print(f"          interrupted={payload.get('interrupted', '(absent)')}")
    print()
    print("  `cut_off` and `interrupted` are separate keys on purpose: an")
    print("  interrupted answer lost bytes, this one lost nothing. It stopped.")
    print()
    print("  And a normal stop stays what it was:")
    Handler.script = [text_chunk("The timeout is 5 seconds."), finish("stop")]
    done = through_cli(port, workspace, "--quiet")
    show_result("openai, finish_reason=stop", done)


def part_two(port: int, workspace: Path) -> None:
    banner("2. an answer that stops in the middle of a tool call")
    # Two frames of arguments, then the budget runs out. The JSON never closes.
    half = CUT_ARGS[: len(CUT_ARGS) // 2]
    rest = CUT_ARGS[len(CUT_ARGS) // 2 :]
    Handler.script = [call_chunk(half, first=True), call_chunk(rest), finish("length")]
    done = through_cli(port, workspace, "--shell")
    show_result("openai, finish_reason=length, mid-tool-call", done)
    print()
    print("  The call is not dropped. An assistant turn that declares a call and")
    print("  has no `tool` message behind it is rejected by every provider, so")
    print("  dropping it would throw the request away too. Instead the call is")
    print("  marked: the adapter says the arguments never finished. (Marked, not")
    print('  guessed -- `str.cut` above is mid-string on purpose, and `{"path":')
    print('  "x"` would be one closing brace away from parsing.)')

    # What the registry does with that payload. It never reaches `write_file`:
    # a JSON object that does not parse is refused before the tool runs, so the
    # error names the actual problem instead of a missing argument.

    call = ToolCall.from_raw(call_id="call_cut", name="write_file", arguments=half + rest)
    print()
    print("  as the adapter would hand it over after a dying stream:")
    mark_incomplete_arguments([call])
    print(f"    arguments_error = {call.arguments_error!r}")
    registry = Registry(default_tools(workspace, shell=True))
    result = registry.execute(call)
    print()
    print("  the tool result the model would read next:")
    print(f"    is_error = {result.is_error}")
    print(f"    content  = {result.content}")


def part_three(port: int, workspace: Path) -> None:
    banner("3. four providers, four words for the same fact")

    print("  every adapter now closes with `Provider._finish(completion, reason)`")
    print()
    rows = [
        ("openai", "length"),
        ("anthropic", "max_tokens"),
        ("gemini", "MAX_TOKENS"),
        ("bedrock", "max_tokens"),
    ]
    for provider, reason in rows:
        sentence = cut_off_reason(provider, reason) or "(not a cut-off)"
        print(f"  {provider:<10} {reason:<12} -> {sentence.split(':')[0]}")
    print()
    print("  and the reasons that mean 'the model chose to stop' stay quiet:")
    quiet = ["openai/stop", "openai/tool_calls", "anthropic/end_turn", "gemini/STOP"]
    print(f"    {', '.join(quiet)} -> {cut_off_reason('x', 'end_turn') or 'no cut_off'}")

    # The end-to-end shape, without a server: what the loop records.
    from oneharness.providers.base import Provider

    class OneShot(Provider):
        name = "scripted"

        def complete(self, messages, tools=None):
            return self._finish(
                Completion(message=Message.assistant(CUT_SENTENCE.strip())), "length"
            )

    events: list[dict] = []
    result = Agent(OneShot("m"), on_event=events.append).run("why?")
    print()
    print(f"  stopped_reason {result.stopped_reason!r}")
    print(f"  ok             {result.ok}")
    print(f"  cut_off event  {[e['type'] for e in events if e['type'] == 'cut_off']}")

    # The mark only survives a cut-off: a stream that died on a complete call
    # has nothing to declare, so a parsed call is left alone.
    print()
    parsed = ToolCall.from_raw(call_id="c2", name="write_file", arguments={"path": "x"})
    mark_incomplete_arguments([parsed])
    print(f"  a call whose JSON parsed stays clean: arguments_error={parsed.arguments_error}")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            part_one(server.server_port, workspace)
            part_two(server.server_port, workspace)
            part_three(server.server_port, workspace)
        finally:
            server.shutdown()
            server.server_close()
    print("\nA cut-off turn used to be the one failure that reported success.")
    print("Now it says so, in the exit code, in `--json`, and on stderr.")


if __name__ == "__main__":
    main()
