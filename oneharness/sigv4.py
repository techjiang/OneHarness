"""AWS Signature Version 4, in about eighty lines of standard library.

Bedrock's hard part is not its wire format but its signing: every request
carries a canonical hash of itself in the `Authorization` header, so a body
that is signed but not sent (or sent but not signed) fails with a
`SignatureDoesNotMatch` that tells you nothing about which byte was wrong.
Signing is a pure function of the request, so it is worth pinning down with
the vectors AWS publishes for exactly this purpose.
"""

from __future__ import annotations

import hashlib
import hmac
import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

ALGORITHM = "AWS4-HMAC-SHA256"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


@dataclass(frozen=True, slots=True)
class Credentials:
    access_key: str
    secret_key: str
    session_token: str | None = None


def sign_request(
    *,
    method: str,
    url: str,
    body: bytes,
    headers: Mapping[str, str],
    credentials: Credentials,
    service: str,
    region: str,
    timestamp: str | None = None,
) -> dict[str, str]:
    """Return the headers a signed request needs, `Authorization` included.

    Everything in `headers` gets signed, so callers must put the final values
    in before calling — a `host` that is added afterwards would be invisible
    to the signature and the request would fail on the server side.
    """
    parts = urllib.parse.urlsplit(url)
    stamp = timestamp or now()
    date = stamp[:8]

    signed = {key.lower(): str(value).strip() for key, value in headers.items()}
    signed.setdefault("host", parts.netloc)
    signed["x-amz-date"] = stamp
    if credentials.session_token:
        signed["x-amz-security-token"] = credentials.session_token

    payload_hash = hashlib.sha256(body).hexdigest()
    canonical = "\n".join(
        [
            method.upper(),
            _canonical_uri(parts.path),
            _canonical_query(parts.query),
            _canonical_headers(signed),
            ";".join(sorted(signed)),
            payload_hash,
        ]
    )
    scope = f"{date}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join(
        [ALGORITHM, stamp, scope, hashlib.sha256(canonical.encode("utf-8")).hexdigest()]
    )
    key = signing_key(credentials.secret_key, date, region, service)
    signature = hmac.new(key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    result = dict(headers)
    result["x-amz-date"] = stamp
    if credentials.session_token:
        result["x-amz-security-token"] = credentials.session_token
    result["authorization"] = (
        f"{ALGORITHM} Credential={credentials.access_key}/{scope}, "
        f"SignedHeaders={';'.join(sorted(signed))}, Signature={signature}"
    )
    return result


def signing_key(secret_key: str, date: str, region: str, service: str) -> bytes:
    """The four-step HMAC ladder AWS calls a signing key (it is never one secret)."""
    key = _hmac(("AWS4" + secret_key).encode("utf-8"), date)
    key = _hmac(key, region)
    key = _hmac(key, service)
    return _hmac(key, "aws4_request")


def _hmac(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()


def now() -> str:
    """`YYYYMMDDTHHMMSSZ` — the only timestamp format `x-amz-date` accepts."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _canonical_uri(path: str) -> str:
    if not path:
        return "/"
    # Segments are quoted one by one so the separating slashes survive: AWS
    # compares this string byte for byte and normalises nothing for us.
    return "/".join(urllib.parse.quote(segment, safe="-_.~") for segment in path.split("/"))


def _canonical_query(query: str) -> str:
    if not query:
        return ""
    pairs = urllib.parse.parse_qsl(query, keep_blank_values=True)
    encoded = [
        (urllib.parse.quote(key, safe="-_.~"), urllib.parse.quote(value, safe="-_.~"))
        for key, value in pairs
    ]
    # Sorted after encoding, which is not the same as sorting before it.
    return "&".join(f"{key}={value}" for key, value in sorted(encoded))


def _canonical_headers(headers: Mapping[str, str]) -> str:
    return "".join(f"{key}:{' '.join(value.split())}\n" for key, value in sorted(headers.items()))
