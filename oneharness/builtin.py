"""Built-in tools: a sandboxed file surface and a shell runner."""

from __future__ import annotations

import codecs
import contextlib
import os
import signal
import subprocess
import threading
from pathlib import Path

from oneharness.errors import ToolError
from oneharness.tools import Tool, tool

MAX_READ_BYTES = 200_000
MAX_OUTPUT_CHARS = 20_000
# The cap on a command's output has to be enforced *while reading*, not after:
# `yes` writes 60 MB in 0.14s, and `capture_output` swallows every byte of it
# into memory before anyone can truncate it (measured: 182 MB peak RSS for a
# command whose answer is one line).
#
# Split in two halves rather than kept as a head, because the two ends answer
# different questions: the beginning says what the command started doing, the
# end says why it stopped. A failing test run is a wall of passes followed by
# the failures -- truncating to the head keeps the passes and loses the point.
HEAD_CHARS = MAX_OUTPUT_CHARS // 2
TAIL_CHARS = MAX_OUTPUT_CHARS - HEAD_CHARS
# After the child exits, how long to wait for the pipe to drain. A grandchild
# that inherited stdout can keep the pipe open after the shell is gone; waiting
# forever on it is how a tool call hangs on a command that already finished.
DRAIN_GRACE = 1.0


def _resolve(path: str, root: Path) -> Path:
    """Resolve `path` inside `root`, refusing anything that escapes the sandbox."""
    candidate = (root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ToolError(f"{path!r} is outside the workspace at {root}") from exc
    return candidate


def make_file_tools(root: Path | str = ".") -> list[Tool]:
    """read_file / write_file / list_dir, all confined to `root`."""
    workspace = Path(root).resolve()

    @tool
    def read_file(path: str) -> str:
        """Read a UTF-8 text file relative to the workspace root."""
        target = _resolve(path, workspace)
        if not target.is_file():
            raise ToolError(f"{path} is not a file")
        if target.stat().st_size > MAX_READ_BYTES:
            raise ToolError(f"{path} is larger than {MAX_READ_BYTES} bytes")
        return target.read_text(encoding="utf-8", errors="replace")

    @tool
    def write_file(path: str, content: str) -> str:
        """Write UTF-8 text to a file, creating parent directories as needed."""
        target = _resolve(path, workspace)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {target.relative_to(workspace)}"

    @tool
    def list_dir(path: str = ".") -> str:
        """List directory entries, one per line; directories end with a slash."""
        target = _resolve(path, workspace)
        if not target.is_dir():
            raise ToolError(f"{path} is not a directory")
        entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
        lines = [f"{entry.name}/" if entry.is_dir() else entry.name for entry in entries]
        return "\n".join(lines) if lines else "(empty)"

    return [read_file, write_file, list_dir]


def make_shell_tool(root: Path | str = ".", timeout: float = 60.0) -> Tool:
    """A shell tool that runs one command in the workspace with a hard timeout."""
    workspace = Path(root).resolve()

    @tool
    def run_shell(command: str) -> str:
        """Run a shell command in the workspace and return its combined output."""
        return _shell(command, workspace, timeout)

    return run_shell


class _Collector:
    """Drains a pipe on its own thread, keeping a bounded head and tail.

    A thread is not optional: a child that fills the 64 KB pipe buffer blocks in
    `write()` forever if nobody reads it, so `wait(timeout)` would report a
    timeout for a command that is merely talkative.

    Decoding uses an incremental decoder with `errors="replace"`, for the same
    reason the file tools do: `head -c 200 /dev/urandom` is a legitimate
    command, and answering it with `UnicodeDecodeError` tells the model nothing
    about what it ran. Incremental matters because a 64 KB read can end in the
    middle of a character, and a stateless decode would mangle both halves.
    """

    def __init__(self, head: int = HEAD_CHARS, tail: int = TAIL_CHARS) -> None:
        self.head_limit = head
        self.tail_limit = tail
        self.head: list[str] = []
        self.head_chars = 0
        self.tail: list[str] = []
        self.tail_chars = 0
        self.dropped = 0
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def drain(self, pipe) -> None:
        try:
            while True:
                chunk = pipe.read1(65536) if hasattr(pipe, "read1") else pipe.read(65536)
                if not chunk:
                    break
                self._feed(self._decoder.decode(chunk))
        except (OSError, ValueError):
            # The pipe was closed under us by the kill path. What arrived before
            # that is still the only copy of it, so it stays.
            pass
        finally:
            self._feed(self._decoder.decode(b"", True))
            with contextlib.suppress(OSError):
                pipe.close()

    def _feed(self, text: str) -> None:
        room = self.head_limit - self.head_chars
        if room > 0:
            piece = text[:room]
            self.head.append(piece)
            self.head_chars += len(piece)
            text = text[room:]
        if not text:
            return
        # Roll a chunk at a time, not a character at a time: a per-character
        # ring over a 60 MB stream is a Python loop over 60 million cells, and
        # the tool would spend more time trimming the answer than running it.
        self.tail.append(text)
        self.tail_chars += len(text)
        while self.tail_chars - len(self.tail[0]) >= self.tail_limit:
            gone = self.tail.pop(0)
            self.tail_chars -= len(gone)
            self.dropped += len(gone)
        excess = self.tail_chars - self.tail_limit
        if excess > 0:
            self.tail[0] = self.tail[0][excess:]
            self.tail_chars -= excess
            self.dropped += excess

    def text(self) -> str:
        head = "".join(self.head)
        tail = "".join(self.tail)
        if not self.dropped:
            return head + tail
        return f"{head}\n... ({self.dropped} chars dropped from the middle) ...\n{tail}"


def _terminate(process: subprocess.Popen) -> None:
    """Kill the whole process group, not just the shell.

    `shell=True` means the process we hold is `/bin/sh`, and `sh -c '... &'`
    leaves grandchildren that survive a kill aimed at the shell alone. Verified
    against the old implementation: a command that had "timed out and been
    killed" still wrote its file two seconds later. A timeout that only stops
    the shell is a timeout that lies about what it stopped.

    SIGTERM first, because a command may flush something worth keeping; SIGKILL
    second, because a handler that ignores SIGTERM must not own the tool call.
    """
    try:
        group = os.getpgid(process.pid)
    except (ProcessLookupError, PermissionError):
        group = None
    for sig, wait in ((signal.SIGTERM, 0.5), (signal.SIGKILL, 2.0)):
        if process.poll() is not None:
            return
        try:
            if group is not None:
                os.killpg(group, sig)
            else:
                process.kill()
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            continue


def _shell(command: str, workspace: Path, timeout: float) -> str:
    """Run one command and answer with what it printed, whether or not it finished.

    Three decisions worth stating, because each one is a way this can lie:

    * **A timeout is an error, but not an empty one.** The output produced before
      the stall is the most valuable thing in the answer -- a test suite that
      printed 40 failures before hanging has already told the model everything.
      Throwing it away to report only "timed out after 60s" is the same mistake
      v0.8 fixed for streams, in a different pipe.
    * **`stderr` goes into the same pipe as `stdout`.** Concatenating two
      captured buffers afterwards rearranges the transcript: a warning that came
      before the failure lands after it, and the model reads cause and effect
      backwards.
    * **The process gets its own session.** That is what makes the group kill
      possible, and `_terminate` on KeyboardInterrupt keeps a Ctrl-C from
      leaving the child running in the background.
    """
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=workspace,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as exc:  # cannot even fork: not a command failure
        raise ToolError(f"cannot start {command!r}: {exc}") from exc

    collector = _Collector()
    reader = threading.Thread(target=collector.drain, args=(process.stdout,), daemon=True)
    reader.start()

    timed_out = False
    try:
        code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate(process)
        code = process.returncode
    except BaseException:
        # Ctrl-C is a decision about this command too, not only about the CLI.
        _terminate(process)
        raise
    reader.join(DRAIN_GRACE)

    output = collector.text().strip()
    if timed_out:
        # An error result carries the partial output, so the model can see how
        # far it got instead of guessing what the timeout interrupted. No
        # `error:` prefix: `Registry.execute` adds one, and two of them read
        # like a bug in the tool rather than a sentence about the command.
        header = f"command timed out after {timeout}s: {command}"
        if code is not None:
            header += f" (killed, exit_code={code})"
        raise ToolError(f"{header}\n{output}" if output else header)

    header = f"exit_code={code}"
    return f"{header}\n{output}" if output else header


def default_tools(root: Path | str = ".", *, shell: bool = False) -> list[Tool]:
    """File tools always; the shell tool only when explicitly enabled."""
    items = make_file_tools(root)
    if shell:
        items.append(make_shell_tool(root))
    return items
