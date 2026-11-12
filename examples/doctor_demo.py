"""Diagnose real endpoints offline: `doctor` over tapes recorded from them.

    python3 examples/doctor_demo.py

There is no key in this demo and no network call. Every verdict below comes from
a response a real provider sent, captured in `tests/fixtures/tapes/`: OpenAI,
Anthropic, Gemini, DeepSeek, Groq and Bedrock, each answering an invalid
credential.

That is the whole point of the fixture. A fake server I write myself can only
confirm my own reading of a spec. These bytes came from somewhere else, so when
`doctor` says `auth` it means the request got past the fields and stopped at the
credential -- a statement no fake server can make.

The second half records a fresh tape against a local server and then replays it
with the server shut down, which is the loop the CLI offers: `--record` once,
`--replay` forever, on a machine with no keys.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAPES = ROOT / "tests" / "fixtures" / "tapes"

sys.path.insert(0, str(ROOT))

from oneharness import doctor  # noqa: E402
from oneharness.errors import ProviderError  # noqa: E402
from oneharness.tape import Tape  # noqa: E402
from oneharness.transport import post_json  # noqa: E402

CASES = ["openai", "anthropic", "gemini", "deepseek", "groq", "bedrock"]

# Not a provider case: the OpenAI request body sent to Gemini. Gemini complains
# about the field names, which is the one 400 that means "the harness is wrong".
BUG_TAPE = TAPES / "protocol-bug.json"
BUG_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
BUG_BODY = {
    "model": "gemini-2.0-flash",
    "messages": [{"role": "user", "content": "ping"}],
    "temperature": 0.0,
}


def banner(text: str) -> None:
    print(f"\n=== {text} ===")


def diagnose_offline() -> None:
    banner("1. real responses, replayed with no key and no network")
    # Strip anything that looks like a credential, so this demo cannot pass by
    # accidentally signing a real request.
    saved = {key: os.environ.pop(key) for key in list(os.environ) if "API_KEY" in key}
    saved.update({key: os.environ.pop(key) for key in list(os.environ) if key.startswith("AWS_")})
    try:
        width = max(len(name) for name in CASES)
        for name in CASES:
            tape_path = TAPES / f"{name}.json"
            with Tape.replay(tape_path):
                probe = doctor.check(name)
            entry = json.loads(tape_path.read_text())["entries"][0]
            status = entry["response"]["status"]
            print(
                f"  {probe.preset:<{width}}  HTTP {status}  ->  {probe.status:<10} {probe.detail}"
            )
        print(f"  {'':<{width}}  {'':<9}      the request shape was accepted every time")

        print()
        print("  the same host, given an OpenAI-shaped body:")
        with Tape.replay(BUG_TAPE):
            try:
                post_json(BUG_URL, BUG_BODY, {"x-goog-api-key": "invalid"}, retries=0)
            except ProviderError as exc:
                verdict, detail = doctor.classify(exc)
                code = exc.status
        print(f"  gemini(wrong body)  HTTP {code}  ->  {verdict:<10} {detail[:68]}")
        print(f"  {'':<{width}}  {'':<9}      that one is a OneHarness bug, not a rotated key")
    finally:
        os.environ.update(saved)


def round_trip() -> None:
    banner("2. record once, replay with the server gone")
    replies = {"n": 0}
    payload = json.dumps(
        {
            "choices": [
                {"message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 1},
        }
    ).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length", 0)))
            replies["n"] += 1
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_port
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH": str(ROOT),
        "ONEHARNESS_PROVIDER": "openai",
        "ONEHARNESS_MODEL": "gpt-demo",
        "ONEHARNESS_API_KEY": "sk-demo",
        "ONEHARNESS_BASE_URL": f"http://127.0.0.1:{port}/v1",
    }

    with tempfile.TemporaryDirectory() as tmp:
        tape_path = Path(tmp) / "round-trip.json"
        record = subprocess.run(
            [sys.executable, "-m", "oneharness", "run", "ping", "--record", str(tape_path)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        print(f"  record: {record.stdout.strip()}")
        print(f"          {record.stderr.strip().splitlines()[-1]}")
        print(f"          the server was hit {replies['n']} time(s)")

        server.shutdown()
        server.server_close()
        print("  server is down; only the tape is left")

        replay = subprocess.run(
            [sys.executable, "-m", "oneharness", "run", "ping", "--replay", str(tape_path)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        print(f"  replay: {replay.stdout.strip()}")
        print(f"          {replay.stderr.strip().splitlines()[-1]}")
        print(f"          the server was still hit {replies['n']} time(s)")

        changed = subprocess.run(
            [
                sys.executable,
                "-m",
                "oneharness",
                "run",
                "a different prompt",
                "--replay",
                str(tape_path),
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        first = changed.stderr.strip().splitlines()[0]
        print(f"  a different request: exit {changed.returncode}, {first.split(': ', 2)[-1]}")


def main() -> None:
    diagnose_offline()
    round_trip()
    banner("done")
    print("  `auth` means: fields accepted, credential refused. A field-name 400 would")
    print("  have been reported as `protocol` -- a bug in this harness, not in the key.")


if __name__ == "__main__":
    main()
