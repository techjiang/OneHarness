"""Minimal JSON-over-HTTP client built on urllib, so OneHarness needs no deps."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

from oneharness.errors import ProviderError

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


def post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    *,
    timeout: float = 120.0,
    retries: int = 2,
    backoff: float = 1.5,
) -> dict[str, Any]:
    """POST JSON and return the decoded response, retrying transient failures."""
    body = json.dumps(payload).encode("utf-8")
    base_headers = {"content-type": "application/json", "accept": "application/json"}
    base_headers.update(headers or {})

    last_error: ProviderError | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, data=body, headers=base_headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return _decode(response.read())
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", "replace")
            last_error = ProviderError(f"request to {url} failed", status=exc.code, body=text)
            if exc.code not in RETRY_STATUS or attempt == retries:
                raise last_error from None
        except urllib.error.URLError as exc:
            last_error = ProviderError(f"cannot reach {url}: {exc.reason}")
            if attempt == retries:
                raise last_error from None
        except TimeoutError:
            last_error = ProviderError(f"request to {url} timed out after {timeout}s")
            if attempt == retries:
                raise last_error from None
        time.sleep(backoff * (2**attempt))

    raise last_error or ProviderError(f"request to {url} failed")


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
) -> Iterator[tuple[str | None, dict[str, Any]]]:
    """POST JSON and yield server-sent events as `(event_name, data)` pairs.

    Retries only apply while nothing has been yielded yet: once the caller has
    seen a token, replaying the request would duplicate output.
    """
    body = json.dumps(payload).encode("utf-8")
    base_headers = {"content-type": "application/json", "accept": "text/event-stream"}
    base_headers.update(headers or {})

    attempt = 0
    while True:
        request = urllib.request.Request(url, data=body, headers=base_headers, method="POST")
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", "replace")
            error = ProviderError(f"request to {url} failed", status=exc.code, body=text)
            if exc.code not in RETRY_STATUS or attempt == retries:
                raise error from None
        except urllib.error.URLError as exc:
            error = ProviderError(f"cannot reach {url}: {exc.reason}")
            if attempt == retries:
                raise error from None
        except TimeoutError:
            error = ProviderError(f"request to {url} timed out after {timeout}s")
            if attempt == retries:
                raise error from None
        else:
            with response:
                yield from _read_events(response)
            return
        attempt += 1
        time.sleep(backoff * (2**attempt))


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
