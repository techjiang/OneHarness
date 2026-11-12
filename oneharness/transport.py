"""Minimal JSON-over-HTTP client built on urllib, so OneHarness needs no deps.

The network is not the fake servers this repo tests against. A socket can reset
in the middle of a body, TLS can fail, a server can promise 5000 bytes and send
20. None of those arrive as a `ProviderError`, so each one has to be turned into
one here: otherwise the exception leaks past every `except OneHarnessError`, the
CLI shows a traceback where it owes a sentence, and `doctor` never gets to say
whose fault it was. A failure *during* a response additionally has to carry the
bytes that already arrived -- that is the case the agent loop is written to
handle, and it can only handle what has the shape it expects.

Retries are paced by the provider when it says how: a 429 often carries
`Retry-After`, and ignoring it means hammering a service that just asked for
room. A wait is announced before it is taken, because three silent seconds in a
terminal are indistinguishable from a hang.
"""

from __future__ import annotations

import http.client
import io
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

from oneharness import tape as tape_module
from oneharness.errors import ProviderError

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}

# A provider may ask for any wait it likes; a CLI should not honour "come back
# in an hour" and look frozen. Past this, give up and let the caller decide.
MAX_RETRY_DELAY = 30.0

# A signer sees the exact bytes that are about to go on the wire and returns the
# headers to send with them. It must sign the body it is given, not the payload
# it came from: re-serialising JSON for signing is how a body ends up signed but
# not sent (or the reverse), and AWS answers that with a bare
# `SignatureDoesNotMatch`. It runs again on every retry so a signature is never
# replayed outside its validity window.
Signer = Callable[[str, bytes, dict[str, str]], dict[str, str]]

# What a lost connection looks like once it is not a raw socket error any more.
# `http.client` raises these from `readline`/`read` when the peer goes away, and
# they inherit from OSError/HTTPException rather than from anything of ours.
_LOST = (OSError, http.client.HTTPException)

Notice = Callable[[dict[str, Any]], None]
_NOTICE: Notice | None = None


def watch_retries(callback: Notice | None) -> None:
    """Make `callback` hear about every wait before it is taken.

    Process-wide for the same reason the tape is: a retry happens deep inside a
    provider call, and sub agents retry on worker threads that nothing hands a
    context down to. It is a diagnostic switch, not state the library reads.
    """
    global _NOTICE
    _NOTICE = callback


def post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    *,
    timeout: float = 120.0,
    retries: int = 2,
    backoff: float = 1.5,
    sign: Signer | None = None,
) -> dict[str, Any]:
    """POST JSON and return the decoded response, retrying transient failures."""
    tape = tape_module.active()
    if tape is not None and tape.replaying:
        return _decode(_replayed(tape, url, payload))

    body = json.dumps(payload).encode("utf-8")
    base_headers = {"content-type": "application/json", "accept": "application/json"}
    base_headers.update(headers or {})

    attempt = 0
    while True:
        headers_to_send = sign(url, body, base_headers) if sign else base_headers
        request = urllib.request.Request(url, data=body, headers=headers_to_send, method="POST")
        error: ProviderError | None = None
        retry_after: float | None = None
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = _read_body(url, response, timeout=timeout)
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", "replace")
            error = ProviderError(f"request to {url} failed", status=exc.code, body=text)
            if exc.code not in RETRY_STATUS or attempt == retries:
                _record(tape, url, payload, exc.code, text.encode("utf-8"))
                raise error from None
            retry_after = _retry_after(exc.headers)
        except urllib.error.URLError as exc:
            error = _unreachable(url, exc.reason)
            if attempt == retries:
                _record(tape, url, payload, 0, str(error).encode("utf-8"))
                raise error from None
        except TimeoutError:
            error = ProviderError(f"request to {url} timed out after {timeout}s")
            if attempt == retries:
                _record(tape, url, payload, 0, str(error).encode("utf-8"))
                raise error from None
        except _LOST as exc:
            # Reset, refused mid-handshake, or a body shorter than its own
            # Content-Length. Transient, so it is retried -- but retried as our
            # own error type, so nothing upstream has to know about sockets.
            error = _lost(url, exc, mid=False)
            if attempt == retries:
                _record(tape, url, payload, 0, str(error).encode("utf-8"))
                raise error from None
        else:
            _record(tape, url, payload, 200, raw)
            return _decode(raw)
        attempt += 1
        delay, source = _delay(backoff, retry_after, attempt)
        _announce(error, delay, attempt, retries, source)
        time.sleep(delay)


def _read_body(url: str, response: Any, *, timeout: float = 120.0) -> bytes:
    """The whole body, with a dead or stalled read turned into a sentence.

    Not streaming, so there is no half answer to salvage here -- but a body cut
    short by a reset must not surface as `ConnectionResetError`, or the CLI shows
    a traceback where it owes a line and `doctor` never classifies it.
    """
    try:
        return response.read()
    except TimeoutError:
        raise ProviderError(f"reading the response from {url} stalled for {timeout}s") from None
    except _LOST as exc:
        raise _lost(url, exc, mid=True) from None


def _decode(raw: bytes) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"response was not JSON: {raw[:200]!r}") from exc
    if not isinstance(data, dict):
        raise ProviderError(f"expected a JSON object, got {type(data).__name__}")
    return data


def post_sse(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    *,
    timeout: float = 120.0,
    retries: int = 2,
    backoff: float = 1.5,
    sign: Signer | None = None,
) -> Iterator[tuple[str | None, dict[str, Any]]]:
    """POST JSON and yield server-sent events as `(event_name, data)` pairs.

    Retries only apply while nothing has been yielded yet: once the caller has
    seen a token, replaying the request would duplicate output. A failure after
    that point is raised with the raw bytes kept on the tape first, because the
    stream that died is exactly the one worth being able to replay.
    """
    tape = tape_module.active()
    if tape is not None and tape.replaying:
        yield from _read_events(io.BytesIO(_replayed(tape, url, payload)))
        return

    body = json.dumps(payload).encode("utf-8")
    base_headers = {"content-type": "application/json", "accept": "text/event-stream"}
    base_headers.update(headers or {})

    attempt = 0
    while True:
        headers_to_send = sign(url, body, base_headers) if sign else base_headers
        request = urllib.request.Request(url, data=body, headers=headers_to_send, method="POST")
        error: ProviderError | None = None
        retry_after: float | None = None
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", "replace")
            error = ProviderError(f"request to {url} failed", status=exc.code, body=text)
            if exc.code not in RETRY_STATUS or attempt == retries:
                _record(tape, url, payload, exc.code, text.encode("utf-8"))
                raise error from None
            retry_after = _retry_after(exc.headers)
        except urllib.error.URLError as exc:
            error = _unreachable(url, exc.reason)
            if attempt == retries:
                _record(tape, url, payload, 0, str(error).encode("utf-8"))
                raise error from None
        except TimeoutError:
            error = ProviderError(f"request to {url} timed out after {timeout}s")
            if attempt == retries:
                _record(tape, url, payload, 0, str(error).encode("utf-8"))
                raise error from None
        except _LOST as exc:
            error = _lost(url, exc, mid=False)
            if attempt == retries:
                _record(tape, url, payload, 0, str(error).encode("utf-8"))
                raise error from None
        else:
            # The raw bytes are collected on the way through: a tape has to
            # hold what the server sent, not what this parser made of it.
            sink: list[bytes] = []
            with response:
                try:
                    yield from _read_events(_tee(response, sink))
                except _LOST as exc:
                    # Whatever arrived before the reset is what a replay needs:
                    # pushing it back through this same parser reproduces the
                    # truncation, which no locally invented failure can. A stall
                    # is a different sentence from a reset, and both are the
                    # provider's connection ending, not a rejection.
                    _record(tape, url, payload, 200, b"".join(sink))
                    raise _stream_broke(url, exc, timeout) from None
            _record(tape, url, payload, 200, b"".join(sink))
            return
        attempt += 1
        delay, source = _delay(backoff, retry_after, attempt)
        _announce(error, delay, attempt, retries, source)
        time.sleep(delay)


def post_chunks(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    *,
    accept: str = "application/octet-stream",
    timeout: float = 120.0,
    retries: int = 2,
    backoff: float = 1.5,
    sign: Signer | None = None,
) -> Iterator[bytes]:
    """POST JSON and yield the response body as raw byte chunks.

    Not every stream is SSE. Bedrock answers with a binary event stream, which
    has no line structure at all, so the transport must be able to hand over
    bytes and let the format decide how to frame them.
    """
    tape = tape_module.active()
    if tape is not None and tape.replaying:
        body = _replayed(tape, url, payload)
        # Sliced, not handed over whole: a chunk consumer that only works when
        # everything arrives at once is exactly what this path must catch.
        for start in range(0, len(body), 4096):
            yield body[start : start + 4096]
        return

    body = json.dumps(payload).encode("utf-8")
    base_headers = {"content-type": "application/json", "accept": accept}
    base_headers.update(headers or {})

    attempt = 0
    while True:
        headers_to_send = sign(url, body, base_headers) if sign else base_headers
        request = urllib.request.Request(url, data=body, headers=headers_to_send, method="POST")
        error: ProviderError | None = None
        retry_after: float | None = None
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", "replace")
            error = ProviderError(f"request to {url} failed", status=exc.code, body=text)
            if exc.code not in RETRY_STATUS or attempt == retries:
                _record(tape, url, payload, exc.code, text.encode("utf-8"))
                raise error from None
            retry_after = _retry_after(exc.headers)
        except urllib.error.URLError as exc:
            error = _unreachable(url, exc.reason)
            if attempt == retries:
                _record(tape, url, payload, 0, str(error).encode("utf-8"))
                raise error from None
        except TimeoutError:
            error = ProviderError(f"request to {url} timed out after {timeout}s")
            if attempt == retries:
                _record(tape, url, payload, 0, str(error).encode("utf-8"))
                raise error from None
        except _LOST as exc:
            error = _lost(url, exc, mid=False)
            if attempt == retries:
                _record(tape, url, payload, 0, str(error).encode("utf-8"))
                raise error from None
        else:
            sink: list[bytes] = []
            with response:
                # `read(4096)` blocks until the buffer is full or the body ends,
                # so it would hand a whole stream over at once at the end --
                # turning every stream into a batch. `read1` returns whatever
                # has arrived after one system call, which is what streaming
                # from a socket actually means.
                reader = getattr(response, "read1", None) or response.read
                while True:
                    try:
                        chunk = reader(4096)
                    except _LOST as exc:
                        if sink:  # a partial stream is still the only copy there is
                            _record(tape, url, payload, 200, b"".join(sink))
                        raise _stream_broke(url, exc, timeout) from None
                    if not chunk:
                        _record(tape, url, payload, 200, b"".join(sink))
                        return
                    sink.append(chunk)
                    yield chunk
        attempt += 1
        delay, source = _delay(backoff, retry_after, attempt)
        _announce(error, delay, attempt, retries, source)
        time.sleep(delay)


def _unreachable(url: str, reason: Any) -> ProviderError:
    """No answer arrived. Two different situations wear that same face.

    DNS, TLS, a proxy and a refused port all mean the request never got
    anywhere, so retrying is pointless but harmless. A reset *before* the first
    byte means the connection was made and then dropped -- urllib reports it
    here too, wrapped as a reason, and saying "cannot reach" about it would hide
    that the server did accept the connection.
    """
    if isinstance(reason, OSError) and not isinstance(reason, urllib.error.URLError):
        loss = ("ConnectionResetError", "ConnectionAbortedError", "BrokenPipeError")
        if type(reason).__name__ in loss:
            return _lost(url, reason, mid=False)
    return ProviderError(f"cannot reach {url}: {reason}")


def _stream_broke(url: str, exc: BaseException, timeout: float) -> ProviderError:
    """A stream that stopped arriving, told apart from one that was cut.

    `TimeoutError` is an `OSError`, so it lands in the same handler as a reset;
    saying "lost the connection" about a stall would send the reader to look at
    the network for a server that is simply slow to finish.
    """
    if isinstance(exc, TimeoutError):
        return ProviderError(f"stream from {url} stalled for {timeout}s mid-response")
    return _lost(url, exc, mid=True)


def _lost(url: str, exc: BaseException, *, mid: bool) -> ProviderError:
    """A connection that died, described by where it died.

    `mid` matters to the caller: before the first byte there is nothing to
    salvage, while after it there is a half answer worth keeping, and those two
    need different handling in the agent loop even though both are one sentence
    here. `status=None` keeps `is_overflow` and `classify` from reading this as
    a rejection the provider chose to send.
    """
    detail = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
    where = "while reading the response" if mid else "before any answer"
    return ProviderError(f"request to {url} lost the connection {where}: {detail}")


def _retry_after(headers: Any) -> float | None:
    """The wait the provider asked for, from either form the header allows.

    `Retry-After` is either a number of seconds or an HTTP date. The date form
    is the one people forget, and reading it as a number would retry instantly
    against a service that just asked for room.
    """
    raw = headers.get("retry-after") if headers is not None else None
    if not raw:
        return None
    text = str(raw).strip()
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


def _delay(backoff: float, retry_after: float | None, attempt: int) -> tuple[float, str]:
    """How long the next attempt waits, and which of the two rules said so.

    Reached only when there is another attempt to make: each caller gives up
    (`record`, then `raise`) *before* asking, so nothing here has to decide
    whether the retry budget is spent.
    """
    if retry_after is not None:
        return min(retry_after, MAX_RETRY_DELAY), "Retry-After"
    return backoff * (2 ** (attempt - 1)), "backoff"


def _announce(
    error: ProviderError | None,
    delay: float,
    attempt: int,
    retries: int,
    source: str,
) -> None:
    """Say what is about to happen before the terminal goes quiet.

    A silent three-second pause reads as a hang, and a person who thinks the
    tool is stuck starts punishing it with Ctrl-C -- which throws away the run
    the wait was protecting.
    """
    notice = _NOTICE
    if notice is None:
        return
    notice(
        {
            "type": "provider_retry",
            "status": error.status if error is not None else None,
            "delay": round(delay, 3),
            "attempt": attempt,
            "retries": retries,
            "source": source,
            "reason": str(error) if error is not None else "",
        }
    )


def _read_events(stream: Any) -> Iterator[tuple[str | None, dict[str, Any]]]:
    """Parse an SSE body into (event, data) pairs, tolerating keepalive comments."""
    event_name: str | None = None
    data_lines: list[str] = []
    for raw_line in stream:
        line = raw_line.decode("utf-8", "replace").rstrip("\r\n")
        if not line:
            if data_lines:
                payload = "\n".join(data_lines)
                data_lines, name = [], event_name
                event_name = None
                if payload.strip() == "[DONE]":
                    yield ("done", {})
                    continue
                try:
                    decoded = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise ProviderError(f"stream chunk was not JSON: {payload[:200]!r}") from exc
                if isinstance(decoded, dict):
                    yield (name, decoded)
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "event":
            event_name = value
        elif field == "data":
            data_lines.append(value)

    if data_lines:  # a stream that ends without a trailing blank line
        payload = "\n".join(data_lines)
        if payload.strip() != "[DONE]":
            try:
                decoded = json.loads(payload)
            except json.JSONDecodeError:
                return
            if isinstance(decoded, dict):
                yield (event_name, decoded)


def _tee(stream: Any, sink: list[bytes]) -> Iterator[bytes]:
    """Iterate `stream` while remembering every raw line for the tape."""
    for raw in stream:
        sink.append(raw)
        yield raw


def _record(
    tape: tape_module.Tape | None,
    url: str,
    payload: dict[str, Any],
    status: int,
    body: bytes,
) -> None:
    if tape is not None:
        tape.store(url, payload, status, body)


def _replayed(tape: tape_module.Tape, url: str, payload: dict[str, Any]) -> bytes:
    """The recorded body for this request, or the recorded failure re-raised.

    A replayed error keeps its status and body so a test can assert on what the
    provider actually said, not on a paraphrase of it.
    """
    status, body = tape.exchange(url, payload)
    if status == 0:
        raise ProviderError(body.decode("utf-8", "replace"))
    if status != 200:
        raise ProviderError(
            f"request to {url} failed", status=status, body=body.decode("utf-8", "replace")
        )
    return body
