"""The `application/vnd.amazon.eventstream` binary framing that Bedrock streams in.

Every other provider on the market streams newline-delimited SSE: `data: {...}`,
one JSON object per line, and a generator over lines is the whole parser. Bedrock
streams *binary* frames, each one delimited by a length in its own prelude and
guarded by two CRC32 checkstops — one over the prelude, one over the whole
message. So you cannot slice the stream on newlines, and you cannot trust the
length until you have verified it.

The layout of one frame:

    [0:4]   total length          uint32, big endian, includes all four parts
    [4:8]   headers length        uint32
    [8:12]  prelude CRC           CRC32 of bytes [0:8]
    [12:..] headers               repeated: name len, name, type, value
    [..:-4] payload               usually a JSON document
    [-4:]   message CRC           CRC32 of everything before it

The prelude CRC exists so a corrupted length cannot make you read past the
buffer: you validate the number that tells you how much to read, before you
read it. Both checksums are verified here; a mismatch raises rather than
returning half a turn.
"""

from __future__ import annotations

import json
import struct
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from oneharness.errors import ProviderError

PRELUDE = 12
CHECKSUM = 4
MIN_FRAME = PRELUDE + CHECKSUM  # an empty frame with no headers and no payload

# Only the types Bedrock actually emits. The protocol allows more, but a decoder
# that guesses at types nobody sends is a decoder whose bugs nobody notices.
_VARIANTS: dict[int, tuple[str, str]] = {
    0: ("bool_true", ">?"),
    1: ("bool_false", ">?"),
    2: ("byte", ">b"),
    3: ("int16", ">h"),
    4: ("int32", ">i"),
    5: ("int64", ">q"),
    6: ("bytes", ""),
    7: ("str", ""),
    8: ("timestamp", ">q"),
    9: ("uuid", ""),
}


@dataclass(slots=True)
class Frame:
    """One decoded event: a header map (always strings for Bedrock) plus JSON."""

    headers: dict[str, Any]
    payload: bytes

    def json(self) -> Any:
        try:
            return json.loads(self.payload.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ProviderError(f"event stream payload was not JSON: {exc}") from None

    @property
    def message_type(self) -> str:
        return str(self.headers.get(":message-type") or "")

    @property
    def event(self) -> str:
        return str(self.headers.get(":event-type") or self.headers.get(":exception-type") or "")


def encode_frame(payload: bytes, headers: dict[str, str] | None = None) -> bytes:
    """Build one frame. Used by tests and by the offline demo's fake server.

    The same code that decodes frames should be able to produce them: an encoder
    written from the spec and a decoder written from the spec can be wrong in
    the same direction, but an encoder and decoder that round-trip against real
    captured bytes cannot both be wrong about the layout.
    """
    encoded_headers = b""
    for name, value in (headers or {}).items():
        raw_name = name.encode("utf-8")
        raw_value = value.encode("utf-8")
        encoded_headers += bytes([len(raw_name)]) + raw_name
        encoded_headers += bytes([7])  # string
        encoded_headers += struct.pack(">H", len(raw_value)) + raw_value

    total = PRELUDE + len(encoded_headers) + len(payload) + CHECKSUM
    prelude = struct.pack(">II", total, len(encoded_headers))
    with_prelude_crc = prelude + struct.pack(">I", zlib.crc32(prelude) & 0xFFFFFFFF)
    body = with_prelude_crc + encoded_headers + payload
    return body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def decode_frames(chunks: Iterator[bytes] | bytes) -> Iterator[Frame]:
    """Yield frames from a byte stream, verifying both CRCs as it goes."""
    buffer = bytearray()

    def feed(source: Iterator[bytes] | bytes) -> Iterator[bytes]:
        if isinstance(source, bytes):
            yield source
        else:
            yield from source

    for chunk in feed(chunks):
        buffer.extend(chunk)
        while True:
            frame, consumed = _take(bytes(buffer))
            if frame is None and not consumed:
                break
            del buffer[:consumed]
            if frame is not None:
                yield frame

    if buffer:
        raise ProviderError(
            f"event stream ended mid-frame: {len(buffer)} trailing byte(s) "
            "could not form a complete message"
        )


def _take(buffer: bytes) -> tuple[Frame | None, int]:
    """Slice one frame off the front of `buffer`; (None, 0) means "need more"."""
    if len(buffer) < PRELUDE:
        return None, 0
    total, header_length = struct.unpack(">II", buffer[:8])
    (prelude_crc,) = struct.unpack(">I", buffer[8:PRELUDE])
    if zlib.crc32(buffer[:8]) & 0xFFFFFFFF != prelude_crc:
        raise ProviderError("event stream prelude checksum mismatch")
    if total < MIN_FRAME or total > len(buffer):
        if total < MIN_FRAME:
            raise ProviderError(f"event stream frame length {total} is impossible")
        return None, 0  # wait for the rest of it

    (message_crc,) = struct.unpack(">I", buffer[total - CHECKSUM : total])
    if zlib.crc32(buffer[: total - CHECKSUM]) & 0xFFFFFFFF != message_crc:
        raise ProviderError("event stream message checksum mismatch")

    headers_start = PRELUDE
    headers_end = PRELUDE + header_length
    if headers_end > total - CHECKSUM:
        raise ProviderError("event stream header block runs past the frame")
    headers = _decode_headers(buffer[headers_start:headers_end])
    payload = buffer[headers_end : total - CHECKSUM]
    return Frame(headers=headers, payload=payload), total


def _decode_headers(block: bytes) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    offset = 0
    while offset < len(block):
        name_length = block[offset]
        offset += 1
        name = block[offset : offset + name_length].decode("utf-8")
        offset += name_length
        if offset >= len(block):
            raise ProviderError("event stream header is missing its type")
        kind = block[offset]
        offset += 1
        spec = _VARIANTS.get(kind)
        if spec is None:
            raise ProviderError(f"event stream header {name!r} has unknown type {kind}")

        label, fmt = spec
        if label.startswith("bool"):
            headers[name] = label == "bool_true"
            continue
        if label in ("bytes", "str", "uuid"):
            (length,) = struct.unpack(">H", block[offset : offset + 2])
            offset += 2
            raw = block[offset : offset + length]
            offset += length
            headers[name] = raw.decode("utf-8") if label == "str" else raw
            continue
        size = struct.calcsize(fmt)
        headers[name] = struct.unpack(fmt, block[offset : offset + size])[0]
        offset += size
    return headers


def frame_for(payload: dict[str, Any], event: str, *, exception: bool = False) -> bytes:
    """A JSON event with the wire headers set, for demos and fake servers."""
    headers = {":content-type": "application/json"}
    if exception:
        headers[":message-type"] = "exception"
        headers[":exception-type"] = event
    else:
        headers[":message-type"] = "event"
        headers[":event-type"] = event
    return encode_frame(json.dumps(payload).encode("utf-8"), headers)
