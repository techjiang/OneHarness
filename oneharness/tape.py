"""Record real provider traffic to a file, replay it later without a network.

Every provider test in this repo talks to a fake server I wrote from my own
reading of each spec. That is circular: if I misread the spec, the fake server
misreads it the same way and everything stays green. A tape breaks the circle by
holding the bytes a *real* provider sent, and pushing them back through the same
decode path.

Two rules make a tape worth having:

* **Bytes in, bytes out.** A tape stores the raw response body, not the parsed
  events. Replay therefore still exercises the SSE line parser, the Bedrock
  frame splitter and each provider's decoder -- the parts that can be wrong. A
  tape of decoded objects would only test the decoder against itself.
* **Strict matching.** A replayed request must match the recorded one exactly,
  and in order. A permissive tape (first entry with a matching URL wins) turns
  every wire-format regression into a green test, which is precisely the failure
  a fixture exists to prevent. A mismatch names the field that differs.

Headers are never recorded, so a tape cannot leak an API key -- and a replay
needs no credentials at all, which is what makes it usable in CI.
"""

from __future__ import annotations

import base64
import json
import threading
from pathlib import Path
from typing import Any

from oneharness.errors import OneHarnessError

VERSION = 1


class TapeError(OneHarnessError):
    """A tape could not be recorded, loaded or matched."""


class Tape:
    """A list of request/response exchanges, either being written or read."""

    def __init__(self, *, path: Path | None = None, replaying: bool = False) -> None:
        self.path = Path(path) if path is not None else None
        self.replaying = replaying
        self.entries: list[dict[str, Any]] = []
        self.matched = 0
        self.recorded = 0
        self._lock = threading.Lock()

    @classmethod
    def record(cls, path: str | Path) -> Tape:
        return cls(path=Path(path), replaying=False)

    @classmethod
    def replay(cls, path: str | Path) -> Tape:
        source = Path(path)
        if not source.is_file():
            raise TapeError(f"no tape at {source}")
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise TapeError(f"{source} is not valid JSON: {exc}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
            raise TapeError(f"{source} has no `entries` list")
        version = data.get("version", VERSION)
        if version != VERSION:
            raise TapeError(f"{source} is tape version {version}, this build reads {VERSION}")
        tape = cls(path=source, replaying=True)
        tape.entries = data["entries"]
        return tape

    def __enter__(self) -> Tape:
        install(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        install(None)
        if not self.replaying:
            self.save()

    # -- recording -------------------------------------------------------

    def store(self, url: str, request: dict[str, Any], status: int, body: bytes) -> None:
        """Append one exchange. Callers record headers on their own conscience."""
        if self.replaying:
            return
        with self._lock:
            self.entries.append(
                {
                    "url": url,
                    "request": request,
                    "response": {
                        "status": status,
                        "body_b64": base64.b64encode(body).decode("ascii"),
                    },
                }
            )
            self.recorded += 1

    def save(self) -> None:
        if self.path is None:
            return
        document = {
            "version": VERSION,
            "hint": "recorded from a real endpoint; headers are deliberately absent",
            "entries": self.entries,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Indented on purpose: a tape is a fixture people read while debugging.
        self.path.write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    # -- replaying -------------------------------------------------------

    def exchange(self, url: str, request: dict[str, Any]) -> tuple[int, bytes]:
        """The recorded response for this request, or a loud failure."""
        with self._lock:
            if self.matched >= len(self.entries):
                raise TapeError(
                    f"tape {self.path} ran out after {len(self.entries)} exchange(s): this run "
                    f"asked for more than it recorded, starting with POST {url}"
                )
            entry = self.entries[self.matched]
            self._verify(entry, url, request)
            self.matched += 1
            response = entry.get("response") or {}

        return int(response.get("status", 200)), base64.b64decode(response["body_b64"])

    def _verify(self, entry: dict[str, Any], url: str, request: dict[str, Any]) -> None:
        index = self.matched + 1
        if entry.get("url") != url:
            raise TapeError(
                f"tape {self.path} exchange #{index}: recorded POST {entry.get('url')}, "
                f"this run used POST {url}"
            )
        recorded = entry.get("request") or {}
        if _canonical(recorded) == _canonical(request):
            return
        raise TapeError(
            f"tape {self.path} exchange #{index}: request differs at "
            f"{_first_difference(recorded, request, 'request')}. The wire format changed; "
            "re-record the tape if the change is intentional."
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path) if self.path else None,
            "mode": "replay" if self.replaying else "record",
            "entries": len(self.entries),
            "matched": self.matched,
            "recorded": self.recorded,
        }


def _canonical(payload: Any) -> str:
    """Key order is not part of a request. Everything else is."""
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)


def _first_difference(recorded: Any, actual: Any, path: str) -> str:
    """Name the first divergent field, so a broken tape says where to look."""
    if isinstance(recorded, dict) and isinstance(actual, dict):
        for key in sorted(set(recorded) | set(actual)):
            if key not in recorded:
                return f"{path}.{key} (the request added a field)"
            if key not in actual:
                return f"{path}.{key} (the request dropped a field)"
            if _canonical(recorded[key]) != _canonical(actual[key]):
                return _first_difference(recorded[key], actual[key], f"{path}.{key}")
        return path
    if isinstance(recorded, list) and isinstance(actual, list):
        if len(recorded) != len(actual):
            return f"{path} (length {len(recorded)} recorded, {len(actual)} sent)"
        for index, (left, right) in enumerate(zip(recorded, actual, strict=True)):
            if _canonical(left) != _canonical(right):
                return _first_difference(left, right, f"{path}[{index}]")
        return path
    return f"{path} (recorded {recorded!r}, sent {actual!r})"


_ACTIVE: Tape | None = None
_ACTIVE_LOCK = threading.Lock()


def install(tape: Tape | None) -> None:
    """Make `tape` the process-wide recorder for the transport.

    Deliberately process-wide rather than threaded through every provider
    constructor: it is a debugging switch, one run should record one tape, and
    sub agents run in worker threads that would otherwise each need the tape
    handed down to them.
    """
    global _ACTIVE
    with _ACTIVE_LOCK:
        _ACTIVE = tape


def active() -> Tape | None:
    return _ACTIVE
