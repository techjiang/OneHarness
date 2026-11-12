"""End-to-end: a real socket that dies, seen from the CLI the way a user sees it.

`test_transport.py` checks the sentences. This file checks the interface: exit
codes, the half answer on stdout, no traceback. It exists because the v0.8
promise ("a cut stream keeps what already arrived") was only ever tested against
a provider that *raised `ProviderError`* -- and a real reset raises a
`ConnectionResetError`, which sails past every `except OneHarnessError` in the
codebase. The tests agreed with the docs, and both were wrong about the network.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import socket
import struct
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from oneharness import cli

SSE_HEAD = (
    b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n"
)


def delta(text: str) -> bytes:
    """One OpenAI SSE chunk carrying `text`."""
    body = {"choices": [{"index": 0, "delta": {"content": text}}]}
    payload = f"data: {json.dumps(body)}\n\n".encode()
    return b"%x\r\n%s\r\n" % (len(payload), payload)


class DyingStreamServer:
    """Sends a few SSE chunks, then kills the connection with a real RST.

    SO_LINGER(1, 0) makes `close()` send RST instead of FIN. Without it the
    client sees an orderly end of body and the parser is perfectly happy -- and
    the difference between "the answer is over" and "the answer was cut" is
    exactly what this file is about.
    """

    def __init__(self, pieces, *, terminate=False):
        self.pieces = list(pieces)
        self.terminate = terminate
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(2)
        self.base_url = f"http://127.0.0.1:{self.listener.getsockname()[1]}/v1"

    def __enter__(self):
        threading.Thread(target=self._serve, daemon=True).start()
        return self

    def __exit__(self, *exc):
        with contextlib.suppress(OSError):
            self.listener.close()

    def _serve(self):
        with contextlib.suppress(OSError):
            conn, _ = self.listener.accept()
            conn.recv(65536)
            conn.sendall(SSE_HEAD)
            for piece in self.pieces:
                conn.sendall(piece)
            if self.terminate:
                conn.sendall(b"0\r\n\r\n")  # a well-formed end, but no [DONE]
            else:
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            conn.close()


class ConnectionResetAtTheCliTests(unittest.TestCase):
    def _run(self, server, *extra):
        env = {"ONEHARNESS_PROVIDER": "ollama", "ONEHARNESS_BASE_URL": server.base_url}
        argv = ["run", "hi", "--stream", "--no-tools", *extra]
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            mock.patch.dict(os.environ, env, clear=True),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            code = cli.main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_a_reset_keeps_the_half_answer_and_prints_no_traceback(self):
        pieces = [delta("The timeout is "), delta("5 seconds")]
        whole = "The timeout is 5 seconds"
        with DyingStreamServer(pieces) as server:
            code, stdout, stderr = self._run(server)
        self.assertEqual(code, 1)
        self.assertNotIn("Traceback", stderr)
        self.assertIn("incomplete", stderr)

        # How much arrived before the RST is genuinely racy: a reset discards
        # whatever is still sitting in the kernel's receive buffer, so pinning
        # the exact text here would make this test fail for a reason that is not
        # the bug. What must hold is the promise itself -- nothing that arrived
        # is thrown away, and nothing that did not arrive is invented.
        kept = stdout.strip()
        self.assertTrue(whole.startswith(kept), f"{kept!r} is not a prefix of the answer")
        self.assertIn(f"kept {len(kept)} chars", stderr)
        if kept:
            self.assertEqual(stderr.count("kept"), 1)

    def test_nothing_arrived_reports_a_sentence_not_a_socket(self):
        with DyingStreamServer([]) as server:
            code, stdout, stderr = self._run(server)
        self.assertEqual(code, 1)
        self.assertEqual(stdout.strip(), "")
        self.assertNotIn("Traceback", stderr)
        self.assertIn("oneharness:", stderr)
        self.assertIn("lost the connection", stderr)

    def test_a_clean_end_without_a_terminal_marker_is_still_interrupted(self):
        # Distinct from the reset above: the body ended properly but the
        # provider never said it was finished. Calling that "done" is the lie
        # v0.8 fixed, and it must not come back through this door.
        with DyingStreamServer([delta("half a sentence")], terminate=True) as server:
            code, stdout, _ = self._run(server)
        self.assertEqual(code, 1)
        self.assertEqual(stdout.strip(), "half a sentence")


if __name__ == "__main__":
    unittest.main()
