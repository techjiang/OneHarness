"""`oneharness doctor`: ask a real provider one minimal question, then classify the failure.

Everything else in this repo is verified against a server I wrote from my own
reading of each spec. That is a closed loop: misread the spec and the fake
server misreads it identically, so every test stays green. `doctor` is the way
out -- it sends the smallest possible real request and then decides **whose
fault** a failure is:

    ok          the answer came back; the pipe is real
    auth        the endpoint understood the request and rejected the key
    model       the key worked, the model/deployment name did not
    endpoint    the URL is wrong, or something between here and there is
    quota       rate limited or out of credit
    protocol    a 400 that complains about *our* field names -- a OneHarness bug
    network     no answer at all: DNS, TLS, proxy, timeout
    unconfigured  no credentials in the environment; not a failure

The distinction that earns its keep is `auth` vs `protocol`. A 401 saying
"invalid x-api-key" proves the request shaped up fine and only the credential is
wrong; a 400 saying "unknown name messages" means this harness is sending the
wrong body and no amount of correct keys will fix it. Those two look equally red
in a log and mean opposite things.

`--record` writes the exchange to a tape so the same classified answer can be
replayed offline later (see `oneharness/tape.py`).
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from oneharness.errors import ConfigError, OneHarnessError, ProviderError
from oneharness.providers import PRESETS, create_provider, list_presets
from oneharness.types import Message

PING = "ping"
# One token, no tools: a doctor run must be cheap enough to do on every new
# machine and small enough that a success proves nothing but the pipe.
MAX_TOKENS = 1

OK = "ok"
AUTH = "auth"
MODEL = "model"
ENDPOINT = "endpoint"
QUOTA = "quota"
PROTOCOL = "protocol"
NETWORK = "network"
UNCONFIGURED = "unconfigured"

# Bodies that blame our request rather than the credential. These are the ones
# worth a bug report; the rest are configuration.
_FIELD_WORDS = (
    "unknown name",
    "cannot find field",
    "unrecognized",
    "unexpected keyword",
    "extra fields",
    "not allowed",
    "invalid json payload",
    "is not supported",
)
_AUTH_WORDS = (
    "api key",
    "api-key",
    "apikey",
    "x-api-key",
    "authentication",
    "unauthorized",
    "credential",
    "token",
    "forbidden",
    "permission",
)
_MODEL_WORDS = ("model", "deployment", "engine")


@dataclass(slots=True)
class Probe:
    """One provider, asked once, with the verdict spelled out."""

    preset: str
    status: str
    detail: str = ""
    hint: str = ""
    model: str | None = None
    base_url: str | None = None
    seconds: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == OK

    @property
    def failed(self) -> bool:
        """`unconfigured` is an absence, not a failure: most machines carry one key."""
        return self.status not in (OK, UNCONFIGURED)

    def as_dict(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "status": self.status,
            "detail": self.detail,
            "hint": self.hint,
            "model": self.model,
            "base_url": self.base_url,
            "seconds": round(self.seconds, 3),
            "usage": self.usage,
        }


def check(preset: str, *, timeout: float = 20.0, **overrides: Any) -> Probe:
    """Send one tiny request to `preset` and classify whatever comes back."""
    try:
        provider = create_provider(preset, timeout=timeout, **overrides)
    except ConfigError as exc:
        return Probe(
            preset=preset, status=UNCONFIGURED, detail=str(exc), hint=_next_step(UNCONFIGURED)
        )

    probe = Probe(
        preset=preset,
        status=OK,
        model=getattr(provider, "model", None),
        base_url=getattr(provider, "base_url", None),
    )
    started = time.monotonic()
    try:
        completion = provider.complete([Message.user(PING)])
    except ProviderError as exc:
        probe.status, probe.detail = classify(exc)
    except OneHarnessError as exc:
        probe.status, probe.detail = NETWORK, str(exc)
    except OSError as exc:  # DNS, TLS, proxy: not a provider error, still a verdict
        probe.status, probe.detail = NETWORK, f"{type(exc).__name__}: {exc}"
    else:
        probe.usage = completion.usage.as_dict()
        probe.detail = completion.text.strip()[:60] or "(empty reply, but a real one)"
    probe.seconds = time.monotonic() - started
    probe.hint = _next_step(probe.status)
    probe.detail = _strip_key(probe.detail, provider)
    return probe


def classify(exc: ProviderError) -> tuple[str, str]:
    """Turn a provider rejection into a verdict plus the sentence it came from."""
    body = (exc.body or "").strip()
    lowered = body.lower()
    status = exc.status

    detail = _first_sentence(body) or str(exc)

    if status is None:
        return NETWORK, detail
    if status in (401, 403):
        return AUTH, detail
    if status == 404:
        if any(word in lowered for word in _MODEL_WORDS):
            return MODEL, detail
        return ENDPOINT, detail
    if status == 429:
        return QUOTA, detail
    if status == 400:
        # Order matters: a body can mention both. Blaming our fields is the
        # claim that deserves evidence, so it is checked before the catch-all.
        if any(word in lowered for word in _FIELD_WORDS):
            return PROTOCOL, detail
        if any(word in lowered for word in _AUTH_WORDS):
            return AUTH, detail
        if any(word in lowered for word in _MODEL_WORDS):
            return MODEL, detail
        return PROTOCOL, detail
    if status >= 500:
        return NETWORK, detail
    return PROTOCOL, detail


def _next_step(status: str) -> str:
    return {
        OK: "the pipe works end to end",
        AUTH: "the request shape was accepted and the key was not: rotate the key",
        MODEL: "the key worked and the name did not: fix model / deployment",
        ENDPOINT: "nothing answered at that URL: check base_url and region",
        QUOTA: "rate limited or out of credit: wait, or use another key",
        PROTOCOL: "the provider rejected the request body itself -- this is a OneHarness bug",
        NETWORK: "no answer: DNS, TLS, proxy or timeout, not a credential problem",
        UNCONFIGURED: "not configured on this machine; set the key to test it",
    }[status]


def _first_sentence(body: str) -> str:
    """Pull the provider's own words out, however it nested them."""
    if not body:
        return ""
    text = body
    if text.lstrip().startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return text.replace("\n", " ")[:200]
        error = data.get("error") if isinstance(data, dict) else None
        if isinstance(error, dict):
            text = str(error.get("message") or error.get("Message") or error)
        elif isinstance(error, str):
            text = error
        else:
            text = str(data.get("message") or data)
    return " ".join(text.split())[:200]


def _strip_key(detail: str, provider: Any) -> str:
    """A doctor report gets pasted into issues; the key must already be gone."""
    key = getattr(provider, "api_key", None)
    if not key or key == "not-needed" or len(key) < 8:
        return detail
    return detail.replace(key, f"{key[:4]}…{key[-2:]}")


def check_many(
    presets: list[str] | None = None,
    *,
    timeout: float = 20.0,
    workers: int = 6,
    **overrides: Any,
) -> list[Probe]:
    """Probe several providers side by side: they are independent network waits."""
    names = presets or list_presets()

    def one(name: str) -> Probe:
        return check(name, timeout=timeout, **overrides)

    if len(names) == 1:
        return [one(names[0])]
    with ThreadPoolExecutor(max_workers=min(workers, len(names))) as pool:
        return list(pool.map(one, names))


def configured_presets() -> list[str]:
    """Presets whose credentials are present, so `doctor` does not report noise."""
    names: list[str] = []
    for name, spec in PRESETS.items():
        env = spec.get("env") or ()
        if not env or any(os.environ.get(item) for item in env):
            if spec.get("auth") == "aws" and not os.environ.get("AWS_ACCESS_KEY_ID"):
                continue
            names.append(name)
    return sorted(names)
