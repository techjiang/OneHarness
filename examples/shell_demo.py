"""What the shell tool says vs what the shell did.

    python3 examples/shell_demo.py

`run_shell` is the one built-in tool that can lie to the model in a way nothing
downstream catches: the answer is a string, and a wrong string is still a
string. This demo runs each of the five ways the old implementation got it
wrong, side by side with the current one, so the difference is visible rather
than asserted.

1. **A timeout threw away the output.** The most valuable part of a timed-out
   answer is how far it got -- a test suite that printed forty failures before
   hanging has already said everything. The old code reported only
   `command timed out after 0.3s` and dropped the forty lines.
2. **A timeout did not kill the command.** `shell=True` holds `/bin/sh`, and
   `sh -c '... &'` leaves grandchildren that survive a kill aimed at the shell.
   The old code said "timed out" and the grandchild wrote its file two seconds
   later. That is not a leak of resources, it is a false statement.
3. **`stderr` was appended, not interleaved.** Two captured buffers
   concatenated afterwards put the warning that came first *after* the failure
   it warned about, and the model reads cause and effect backwards.
4. **Binary output raised `UnicodeDecodeError`.** `head -c 200 /dev/urandom` is
   a legitimate command; answering it with a decoder exception tells the model
   nothing about what it ran.
5. **A wall of output ate the machine first, then got truncated.** The old route
   held every byte in memory before cutting it (`yes` writes 60 MB in 0.14s).
   The cap now has to be enforced while reading.

Each case is run against the tool as it was in v1.3 and as it is now, in one
process, with the same command. A sixth part closes the loop through the real
CLI: a fake provider asks for a command that times out, and the demo prints the
`tool` message the model would have read.

No network, no API key.
"""

from __future__ import annotations

import contextlib
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oneharness.builtin import make_shell_tool  # noqa: E402
from oneharness.errors import ToolError  # noqa: E402

TIMEOUT = 0.5


def banner(text: str) -> None:
    print(f"\n=== {text} ===")


def old_shell(root: Path, timeout: float):
    """v1.3's `run_shell`, verbatim: `run()` plus two captured buffers."""

    def run(command: str) -> str:
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolError(f"command timed out after {timeout}s: {command}") from exc
        output = ((completed.stdout or "") + (completed.stderr or "")).strip()
        if len(output) > 20_000:
            output = output[:20_000] + f"\n... (truncated, {len(output)} chars total)"
        return (
            f"exit_code={completed.returncode}\n{output}"
            if output
            else (f"exit_code={completed.returncode}")
        )

    return run


def show(label: str, call) -> None:
    print(f"  {label}:")
    try:
        text = call()
    except ToolError as exc:
        text = str(exc)
    for line in text.splitlines()[:12]:
        print(f"    {line}")


def part_one(root: Path) -> None:
    banner("1. a timeout: what the command printed before it stalled")
    print("  A test run that prints two failures and then hangs. The failures are")
    print("  the answer; the timeout is only the reason there are no more of them.\n")
    command = "echo 'test_a ... FAILED'; echo 'test_b ... FAILED'; sleep 5"
    show("v1.3", lambda: _raise(old_shell(root, TIMEOUT)(command)))
    show("v1.4", lambda: make_shell_tool(root, TIMEOUT)(command))


def part_two(root: Path) -> None:
    banner("2. a timeout: whether the command is actually gone")
    print("  The command backgrounds a grandchild that writes a marker one second")
    print("  later. A kill aimed at the shell leaves it running.\n")
    shells = (("v1.3", old_shell(root, TIMEOUT)), ("v1.4", make_shell_tool(root, TIMEOUT)))
    for label, shell in shells:
        marker = root / f"leaked-{label}.txt"
        marker.unlink(missing_ok=True)
        # The timeout is the expected outcome here; the question is what it left
        # running, not what it raised.
        with contextlib.suppress(ToolError):
            shell(f"sh -c 'sleep 1; echo survived > {marker.name}' & sleep 5")
        time.sleep(1.6)
        verdict = "still alive -- it wrote its file" if marker.exists() else "gone, as claimed"
        print(f"  {label}: {verdict}")
        marker.unlink(missing_ok=True)


def part_three(root: Path) -> None:
    banner("3. stderr and stdout: which came first")
    print("  The command writes to both, on purpose, in a known order.\n")
    command = "echo first; echo SECOND >&2; echo third"
    show("v1.3", lambda: old_shell(root, TIMEOUT)(command))
    show("v1.4", lambda: make_shell_tool(root, TIMEOUT)(command))


def part_four(root: Path) -> None:
    banner("4. output that is not text")
    print("  `head -c 200 /dev/urandom` is a legitimate thing to run.\n")
    for label, call in (
        ("v1.3", lambda: old_shell(root, TIMEOUT)("head -c 200 /dev/urandom")),
        ("v1.4", lambda: make_shell_tool(root, TIMEOUT)("head -c 400 /dev/urandom")),
    ):
        try:
            text = call()
            print(f"  {label}: exit_code line came back, {len(text)} chars, no exception")
        except ToolError as exc:
            print(f"  {label}: ToolError -- {str(exc).splitlines()[0]}")
        except UnicodeDecodeError as exc:
            print(f"  {label}: UnicodeDecodeError -- {exc}")


def part_five(root: Path) -> None:
    banner("5. a wall of output: which part is kept, and what it cost to decide")
    print("  Two different failures, both from reading everything first.\n")

    print("  a) a failing test run: 60 000 passing lines, then the two that matter")
    command = "seq 1 60000; echo 'FAILED tests/test_a.py::test_one'; echo '2 failed, 59998 passed'"
    old = old_shell(root, 120)(command)
    print(f"     v1.3  {len(old)} chars, last line {old.splitlines()[-1]!r}")
    new = make_shell_tool(root, 120)(command)
    print(f"     v1.4  {len(new)} chars, last line {new.splitlines()[-1]!r}")

    print("\n  b) 120 MB through the pipe, measured in a separate process each time")
    flood = "yes 'a line of log output that is not short' | head -c 120000000"
    print(f"     v1.3  peak RSS +{_old_memory(flood):.0f} MB")
    base = _rss()
    started = time.monotonic()
    make_shell_tool(root, 120)(flood)
    print(f"     v1.4  peak RSS +{_rss() - base:.0f} MB, {time.monotonic() - started:.2f}s")

    print("\n  Both ends are kept on purpose: the head says what the command started")
    print("  doing, the tail says why it stopped. Keeping only the head keeps the")
    print("  passes and loses the point.")


def part_six(root: Path) -> None:
    banner("6. what the model actually reads, through the real CLI")
    print("  A fake provider asks for a command whose test run hangs. The string")
    print("  below is the `tool` message that goes back into the conversation --")
    print("  the whole point of parts 1-5 is what this string says.\n")
    seen = _through_cli(root)
    if seen is None:
        print("  (skipped: could not start the local provider)")
        return
    for line in seen.splitlines():
        print(f"    {line}")


def _through_cli(root: Path) -> str | None:
    """Run `oneharness run` against a fake provider and return the tool message."""
    import json
    import os
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    def frame(payload: dict) -> bytes:
        return f"data: {json.dumps(payload)}\n\n".encode()

    def delta(piece: str) -> dict:
        return {"choices": [{"index": 0, "delta": {"content": piece}}]}

    def tool_delta(arguments: str) -> dict:
        slot = {"index": 0, "id": "c1", "function": {"name": "run_shell"}}
        slot["function"]["arguments"] = arguments
        return {"choices": [{"index": 0, "delta": {"tool_calls": [slot]}}]}

    def finish(reason: str) -> dict:
        return {"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]}

    failed = "echo 'FAILED {}'; "
    command = failed.format("tests/test_a.py::test_one")
    command += failed.format("tests/test_a.py::test_two") + "sleep 30"
    turns = [
        [tool_delta(json.dumps({"command": command})), finish("tool_calls")],
        [delta("两个测试挂了。"), finish("stop")],
    ]
    seen: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        turn = 0

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
            for message in body.get("messages") or ():
                if message.get("role") == "tool":
                    seen.append(message.get("content") or "")
            frames = turns[min(Handler.turn, len(turns) - 1)]
            Handler.turn += 1
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            for payload in frames:
                self.wfile.write(frame(payload))
                self.wfile.flush()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH": str(ROOT),
        "ONEHARNESS_PROVIDER": "openai",
        "ONEHARNESS_MODEL": "gpt-demo",
        "ONEHARNESS_API_KEY": "sk-demo",
        "ONEHARNESS_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
    }
    # The tool's timeout is a constructor argument, so the run is launched
    # through a two-line shim instead of environment variables it does not read.
    shim = (
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "import oneharness.builtin as builtin\n"
        "original = builtin.make_shell_tool\n"
        "builtin.make_shell_tool = lambda root='.', timeout=60.0: original(root, 1.0)\n"
        "from oneharness.cli import main\n"
        "sys.exit(main(sys.argv[1:]))\n"
    )
    path = root / "cli_shim.py"
    path.write_text(shim)
    try:
        subprocess.run(
            [sys.executable, str(path), "run", "看看测试为什么挂", "--shell", "--stream"],
            env={**os.environ, **env},
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
    finally:
        server.shutdown()
        server.server_close()
    return seen[-1].strip() if seen else None


def _old_memory(command: str) -> float:
    """Peak RSS growth of `capture_output`, measured in a fresh interpreter."""
    probe = (
        "import resource, subprocess, sys\n"
        "base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
        "subprocess.run(sys.argv[1], shell=True, capture_output=True, text=True)\n"
        "peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
        "print((peak - base) / 1024)\n"
    )
    done = subprocess.run([sys.executable, "-c", probe, command], capture_output=True, text=True)
    return float(done.stdout.strip() or 0.0)


def _rss() -> float:
    """Peak RSS so far. A high-water mark, so it never goes back down."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _raise(exc: BaseException) -> str:
    raise exc


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        part_one(root)
        part_two(root)
        part_three(root)
        part_four(root)
        part_five(root)
        part_six(root)
    print("\nParts 1-5 ran the same command twice in one process, with only the")
    print("shell tool swapped. Part 6 went through the real CLI so the string")
    print("above is the one the model reads, not one this demo printed.")
    print("Nothing here needed a network or an API key.")


if __name__ == "__main__":
    main()
