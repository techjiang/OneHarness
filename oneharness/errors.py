"""Exception hierarchy. Everything raised by OneHarness derives from OneHarnessError."""

from __future__ import annotations

from typing import Any

# Why a request never produced an answer. The distinctions are not cosmetic:
# each one is fixed in a different place, and two of them look identical at the
# terminal (a red line, no answer).
#
#   dns      the name does not resolve        -> fix the host / your resolver
#   refused  the port is closed               -> fix the port / the service
#   tls      the certificate was rejected     -> fix trust, or the URL
#   proxy    a proxy demanded credentials     -> fix proxy config / auth
#   timeout  nothing came back in time        -> slow network or slow service
#   lost     the connection died mid-flight   -> usually worth a retry
#
# `retryable` is a separate axis from the kind: most of these are a bet that the
# world changed (a slow service, a restarted one), and that bet is worth taking.
# Two are not bets at all -- a certificate nobody trusts and a proxy asking for
# a password will say the same thing on the second attempt, so retrying only
# buys silence and then the identical sentence.
#
# `tls` and `proxy` are not network reachability problems, and that is the whole
# point of naming them: "check your network" is the wrong instruction for a
# self-signed certificate and a completely wrong one for a proxy that answered
# 407 and asked for a password.
NETWORK_KINDS = ("dns", "refused", "tls", "proxy", "timeout", "lost")

_HINTS = {
    "dns": "the host name did not resolve: check base_url and your DNS",
    "refused": "nothing is listening on that port: check the port and that the service is up",
    "tls": "the certificate was rejected: untrusted, expired, or issued for another host",
    "proxy": "a proxy refused the tunnel and asked for credentials: check HTTPS_PROXY / NO_PROXY",
    "timeout": "nothing came back in time: the path or the service is slow, not the key",
    "lost": "the connection was accepted and then dropped: usually transient, worth a retry",
}


class OneHarnessError(Exception):
    """Base class for all OneHarness failures."""


class ConfigError(OneHarnessError):
    """Invalid or incomplete configuration."""


class ProviderError(OneHarnessError):
    """A provider rejected the request or returned something unusable."""

    def __init__(self, message: str, *, status: int | None = None, body: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.body = body

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.status is not None:
            parts.append(f"status={self.status}")
        if self.body:
            parts.append(f"body={self.body[:500]}")
        return " | ".join(parts)


class NetworkError(ProviderError):
    """No answer arrived, and the reason names what to go and fix.

    A connection reset, an unresolvable name, an untrusted certificate and a
    proxy demanding a password are four different facts. Collapsing them into
    "network" makes `doctor` tell you to check the Wi-Fi about an expired cert,
    which is a wrong instruction dressed up as a diagnosis.

    It still *is* a `ProviderError`, because the one case that carries bytes --
    a connection lost while reading the response -- has to keep reaching the
    agent loop's "there is half an answer, keep it" branch. That branch catches
    `ProviderError`; a sibling class would have slipped past it and thrown the
    half answer away, which is the exact bug v1.0 set out to fix. `status`
    stays `None` so nothing reads this as a rejection the provider chose.
    """

    def __init__(self, message: str, *, kind: str, retryable: bool = True) -> None:
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable

    @property
    def hint(self) -> str:
        return _HINTS.get(self.kind, "")

    def __str__(self) -> str:
        return f"[{self.kind}] {super().__str__()}"


class ToolError(OneHarnessError):
    """A tool failed in a way the model is allowed to see and react to."""


def describe(exc: BaseException) -> dict[str, Any]:
    """The shape `--json` and `doctor` both want: what failed, and why."""
    if isinstance(exc, NetworkError):
        return {"kind": "network", "detail": exc.kind, "hint": exc.hint}
    return {"kind": "other", "detail": type(exc).__name__, "hint": ""}


__all__ = [
    "NETWORK_KINDS",
    "ConfigError",
    "NetworkError",
    "OneHarnessError",
    "ProviderError",
    "ToolError",
    "describe",
]
