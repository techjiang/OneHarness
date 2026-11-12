"""Bedrock over a local fake endpoint: real signing, real binary event stream.

    python3 examples/bedrock_demo.py

Nothing here needs AWS. A local HTTP server plays Bedrock: it recomputes the
SigV4 signature from the bytes it actually received (so a signature over
anything other than the sent body shows up as a mismatch), and it answers
`converse-stream` with a genuine `application/vnd.amazon.eventstream` body,
one frame at a time with pauses in between.

The pauses are the point. If the transport buffered the response, all the
frames would arrive at the end and the timestamps would say nothing; staggered
arrival proves the decode path is streaming rather than batching.
"""

from __future__ import annotations

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
from oneharness.eventstream import frame_for  # noqa: E402
from oneharness.providers.bedrock import BedrockProvider  # noqa: E402
from oneharness.sigv4 import Credentials, sign_request  # noqa: E402
from oneharness.tools import Registry  # noqa: E402

MODEL = "anthropic.claude-3-5-sonnet-20241022-v2:0"
CREDENTIALS = Credentials("AKIDEXAMPLE", "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY")
GAP = 0.25  # seconds between frames, so the timeline means something
STARTED = time.monotonic()
SERVER_LOG: list[str] = []


def text_frames(pieces):
    frames = [frame_for({"role": "assistant"}, "messageStart")]
    for piece in pieces:
        frames.append(
            frame_for({"contentBlockIndex": 0, "delta": {"text": piece}}, "contentBlockDelta")
        )
    return frames


def tool_frames(call_id, name, fragments):
    frames = [
        frame_for(
            {"contentBlockIndex": 1, "start": {"toolUse": {"toolUseId": call_id, "name": name}}},
            "contentBlockStart",
        )
    ]
    for fragment in fragments:
        frames.append(
            frame_for(
                {"contentBlockIndex": 1, "delta": {"toolUse": {"input": fragment}}},
                "contentBlockDelta",
            )
        )
    return frames


def finish_frames(reason="end_turn", tokens=(60, 20)):
    return [
        frame_for({"stopReason": reason}, "messageStop"),
        frame_for({"usage": {"inputTokens": tokens[0], "outputTokens": tokens[1]}}, "metadata"),
    ]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("content-length", 0)))
        headers = {key.lower(): value for key, value in self.headers.items()}
        signed_ok = self.verify(body, headers)
        SERVER_LOG.append(
            f"[server +{time.monotonic() - STARTED:5.2f}s] {self.path[:70]}\n"
            f"         body {len(body)} bytes, sha256 {self.hash(body)[:16]}…\n"
            f"         signature recomputed from received bytes: "
            f"{'MATCH' if signed_ok else 'MISMATCH'}"
        )
        if not signed_ok:
            self.send_error(403, "SignatureDoesNotMatch")
            return

        first_turn = b'"toolResult"' not in body
        if first_turn:
            frames = (
                text_frames(["Let me ", "check ", "the config."])
                + tool_frames("tu-1", "read_file", ['{"path":', ' "app.conf"}'])
                + finish_frames("tool_use", (48, 26))
            )
        else:
            frames = text_frames(
                ["timeout was ", "5 seconds; ", "I raised it ", "to 30."]
            ) + finish_frames("end_turn", (78, 14))
        self.stream(frames)

    def stream(self, frames):
        total = sum(len(frame) for frame in frames)
        self.send_response(200)
        self.send_header("content-type", "application/vnd.amazon.eventstream")
        self.send_header("content-length", str(total))
        self.end_headers()
        for frame in frames:
            self.wfile.write(frame)
            self.wfile.flush()
            time.sleep(GAP)

    @staticmethod
    def hash(body):
        import hashlib

        return hashlib.sha256(body).hexdigest()

    def verify(self, body, headers):
        """Recompute the signature from what arrived, and compare."""
        expected = sign_request(
            method="POST",
            url=f"http://{headers['host']}{self.path}",
            body=body,
            headers={
                "accept": headers.get("accept", ""),
                "content-type": headers.get("content-type", ""),
                "host": headers["host"],
            },
            credentials=CREDENTIALS,
            service="bedrock",
            region="us-east-1",
            timestamp=headers.get("x-amz-date", ""),
        )
        return expected["authorization"] == headers.get("authorization")

    def log_message(self, *args):
        pass


def run_cli(port):
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "ONEHARNESS_PROVIDER": "bedrock",
        "ONEHARNESS_MODEL": MODEL,
        "ONEHARNESS_BASE_URL": f"http://127.0.0.1:{port}",
        "AWS_ACCESS_KEY_ID": CREDENTIALS.access_key,
        "AWS_SECRET_ACCESS_KEY": CREDENTIALS.secret_key,
        "PYTHONPATH": str(ROOT),
    }
    command = [
        sys.executable,
        "-m",
        "oneharness",
        "run",
        "看看 app.conf 的 timeout 是不是太小了",
        "--workspace",
        str(WORKSPACE),
        "--stream",
    ]
    done = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    print("\n$ oneharness run '看看 app.conf 的 timeout 是不是太小了' --stream")
    # Interleaved the way a terminal would show it: progress on stderr, the
    # streamed answer on stdout, tool events before the sentence they enabled.
    for line in done.stderr.rstrip().splitlines():
        print(f"  {line}")
    print(done.stdout.rstrip())
    return done.returncode


def timeline(port):
    """Client-side arrival times: the only honest evidence of streaming."""
    provider = BedrockProvider(
        MODEL, credentials=CREDENTIALS, base_url=f"http://127.0.0.1:{port}", region="us-east-1"
    )
    marks = []

    def observe(event):
        if event["type"] in ("text_delta", "tool_start", "tool_end"):
            marks.append((time.monotonic(), event))

    agent = Agent(provider, Registry(default_tools(WORKSPACE)), on_event=observe)
    agent.run("看看 app.conf 的 timeout 是不是太小了", stream=True)

    print("\nclient-side arrival time of each event (frames are sent 0.25s apart):")
    base = marks[0][0]
    for moment, event in marks:
        if event["type"] == "text_delta":
            what = f"text {event['text']!r}"
        elif event["type"] == "tool_start":
            what = f"tool call {event['name']}"
        else:
            what = f"tool result {event['content'][:20]!r}"
        print(f"  +{moment - base:5.2f}s  {what}")


def main():
    WORKSPACE.mkdir(exist_ok=True)
    (WORKSPACE / "app.conf").write_text("timeout=5\n")

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        code = run_cli(server.server_port)
        timeline(server.server_port)
        print("\nevery request re-verified by the server from the bytes it received:")
        for entry in SERVER_LOG:
            print(f"  {entry}")
    finally:
        server.shutdown()
        server.server_close()
    print(
        f"\n[exit {code} | {len(SERVER_LOG)} signed requests across two runs | "
        f"streams decoded frame by frame, both CRCs verified]"
    )


if __name__ == "__main__":
    main()
