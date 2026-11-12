"""When the network is the thing that broke, not the model.

    python3 examples/network_demo.py

Everything else in this repo is verified against a server that answers politely.
Real endpoints do four rude things, and this demo reproduces all four with real
sockets rather than with a provider that pretends:

1. **The connection resets mid-answer.** Not a `ProviderError` — a raw
   `ConnectionResetError` out of `socket.recv_into`. Before this version that
   sailed past every `except OneHarnessError` in the codebase: the CLI printed a
   traceback, and the half answer that had already arrived was thrown away. That
   mattered because v0.8 promised exactly the opposite, and its tests agreed
   with the docs — both were wrong about the network.
2. **The connection dies before any answer at all.** urllib reports this as a
   `URLError`, the same face it wears for DNS and a refused port. "Cannot reach"
   and "we got there and it hung up" are different facts, and one of them is
   worth retrying.
3. **The body is shorter than the server said.** A `Content-Length: 5000`
   fulfilled with 20 bytes.
4. **A 429 that says `Retry-After: 1`.** The header used to be ignored, so three
   requests went out in 0.15s against a service that had just asked for room.

The servers here are raw sockets, not fake providers: the point is the socket
behaviour itself, and a fake provider cannot produce a TCP reset.

No network, no API key.
"""

from __future__ import annotations

import contextlib
import json
import socket
import struct
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oneharness.errors import ProviderError  # noqa: E402
from oneharness.transport import post_chunks, post_json, post_sse, watch_retries  # noqa: E402

SSE_HEAD = (
    b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n"
)


def banner(text: str) -> None:
    print(f"\n=== {text} ===")


def chunk(payload: bytes) -> bytes:
    return b"%x\r\n%s\r\n" % (len(payload), payload)


def delta(text: str) -> bytes:
    body = {"choices": [{"index": 0, "delta": {"content": text}}]}
    return chunk(f"data: {json.dumps(body)}\n\n".encode())


class Raw:
    """A socket that answers with what it is told, then hangs up with an RST."""

    def __init__(self, respond: Callable[[socket.socket], None], *, reset: bool = True):
        self.respond = respond
        self.reset = reset
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(4)
        self.url = f"http://127.0.0.1:{self.listener.getsockname()[1]}/v1/chat"

    def __enter__(self) -> Raw:
        threading.Thread(target=self._serve, daemon=True).start()
        return self

    def __exit__(self, *exc) -> None:
        with contextlib.suppress(OSError):
            self.listener.close()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self.listener.accept()
            except OSError:
                return
            with contextlib.suppress(OSError):
                conn.recv(65536)
                self.respond(conn)
                if self.reset:
                    # SO_LINGER(1, 0): close() sends RST, not FIN. This is the
                    # whole trick -- a FIN looks like a body that simply ended,
                    # which the parser accepts without complaint.
                    conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                conn.close()


def describe(exc: BaseException) -> str:
    return f"{type(exc).__module__}.{type(exc).__name__}"


def sentence(exc: BaseException) -> str:
    """The transport's own words, without the ` | body=...` suffix `__str__` adds."""
    return str(exc).split(" | ", 1)[0]


def reset_mid_answer() -> None:
    banner("1. the connection resets half-way through the answer")

    def respond(conn: socket.socket) -> None:
        conn.sendall(SSE_HEAD)
        conn.sendall(delta("The timeout is "))
        time.sleep(0.05)
        conn.sendall(delta("5 seconds"))

    with Raw(respond) as server:
        events = []
        try:
            for event in post_sse(server.url, {}, retries=0):
                events.append(event)
        except BaseException as exc:
            caught = exc
        else:  # pragma: no cover - the demo depends on the failure
            caught = None

    text = "".join(
        str(((data.get("choices") or [{}])[0].get("delta") or {}).get("content") or "")
        for _, data in events
    )
    print(f"  events that arrived before the reset: {len(events)}")
    print(f"  what the caller gets: {describe(caught)}")
    print(f"  isinstance(exc, OneHarnessError): {isinstance(caught, ProviderError)}")
    print(f"  the text that came through: {text!r}")
    print("  ^ the type matters: every layer above catches OneHarnessError, so a raw")
    print("    ConnectionResetError used to skip them all and reach the terminal as a")
    print("    traceback -- losing the half answer v0.8 had promised to keep.")


def reset_before_answering() -> None:
    banner("2. the connection dies before any answer")

    with Raw(lambda conn: None) as server:  # accepts, reads, then RSTs
        try:
            # `post_sse` is a generator: the request only happens on the first
            # `next()`, so calling it and walking away would prove nothing.
            list(post_sse(server.url, {}, retries=0))
        except BaseException as exc:
            caught = exc
        else:  # pragma: no cover
            caught = None

    print(f"  what the caller gets: {describe(caught)}")
    print(f"  the sentence: {sentence(caught)}")
    print("  urllib hands this over as a URLError, exactly like a DNS failure or a")
    print("  refused port. But the server did accept the connection -- so this is")
    print("  not 'cannot reach', and a retry here is not pointless.")


def body_cut_short() -> None:
    banner("3. a stream cut in the middle of a frame")

    def respond(conn: socket.socket) -> None:
        conn.sendall(
            b"HTTP/1.1 200 OK\r\ncontent-type: application/octet-stream\r\n"
            b"transfer-encoding: chunked\r\n\r\n"
        )
        conn.sendall(chunk(b"half of a frame"))

    with Raw(respond) as server:
        seen: list[bytes] = []
        try:
            for piece in post_chunks(server.url, {}, retries=0):
                seen.append(piece)
        except BaseException as exc:
            caught = exc
        else:  # pragma: no cover
            caught = None

    print(f"  bytes that arrived: {sum(len(piece) for piece in seen)}")
    print(f"  what the caller gets: {describe(caught)}")
    print(f"  the sentence: {sentence(caught)}")
    print("  a truncated frame must never be read as a short frame: the decoder sees")
    print("  an error, not a valid event stream that happened to be tiny.")


def retry_after() -> None:
    banner("4. a 429 that asks for room: Retry-After is honoured, and announced")

    hits = {"n": 0}
    waits: list[float] = []

    def respond(conn: socket.socket) -> None:
        hits["n"] += 1
        if hits["n"] <= 2:
            body = b'{"error":"rate limited"}'
            conn.sendall(
                b"HTTP/1.1 429 Too Many Requests\r\ncontent-type: application/json\r\n"
                b"retry-after: 1\r\ncontent-length: %d\r\n\r\n%s" % (len(body), body)
            )
            return
        body = b'{"choices":[{"message":{"content":"pong"}}]}'
        conn.sendall(
            b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n"
            b"content-length: %d\r\n\r\n%s" % (len(body), body)
        )

    def notice(event: dict) -> None:
        if event["type"] != "provider_retry":
            return
        waits.append(event["delay"])
        why = f"HTTP {event['status']}" if event["status"] else "connection lost"
        print(
            f"  announced: {why}, waiting {event['delay']:.2f}s ({event['source']}) "
            f"[{event['attempt']}/{event['retries']}]"
        )

    watch_retries(notice)
    try:
        with Raw(respond, reset=False) as server:
            started = time.monotonic()
            reply = post_json(server.url, {}, retries=2, backoff=0.01)
            elapsed = time.monotonic() - started
    finally:
        watch_retries(None)

    answer = reply["choices"][0]["message"]["content"]
    print("  the server asked for 1s of room, twice")
    print(f"  requests: {hits['n']}   elapsed: {elapsed:.2f}s   answer: {answer!r}")
    print("  without the header this was 3 requests in 0.15s -- a retry storm, and")
    print("  the kind of failure that leaves no trace beyond a rate-limit ban.")


def main() -> None:
    reset_mid_answer()
    reset_before_answering()
    body_cut_short()
    retry_after()
    banner("done")
    print("  All four are failures the network produces and a fake provider cannot.")
    print("  The rule they share: nothing above the transport should ever have to")
    print("  know what a socket is.")


if __name__ == "__main__":
    main()
