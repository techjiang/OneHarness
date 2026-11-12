"""Minimal JSON-over-HTTP client built on urllib, so OneHarness needs no deps."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
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
