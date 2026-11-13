"""When there is no answer, *which* no-answer? Four real failures, four verdicts.

    python3 examples/network_kinds_demo.py

v1.0 collapsed every "no answer" into one word, `network`, and told the reader
to check DNS, TLS and the proxy -- three guesses. This demo runs four failures
that are all "no answer" and shows that each one is fixed somewhere else:

1. **A certificate nobody trusts.** A real TLS handshake against a real
   self-signed certificate, refused by the real verifier. The network is fine;
   nothing is wrong with the host name either. Telling someone to check their
   connection here is a wrong instruction dressed as a diagnosis.
2. **A proxy that answers 407.** A real proxy, a real CONNECT tunnel, a real
   refusal. This used to fall through to `protocol` -- "our request body is
   malformed" -- which sends the reader into `providers/` looking for a field
   that was never wrong.
3. **A name that does not resolve.** One character in a base_url.
4. **A port nobody listens on.** The service is down, or the port is a typo.

The last block is about the other half of a diagnosis: whether retrying the
thing could possibly help. Each run is timed, because "it retried three times"
is visible on a terminal as three seconds of nothing.

Servers here are real sockets and real TLS; a mocked `URLError` would only prove
that this code can read its own assumptions back.
"""

from __future__ import annotations

import contextlib
import os
import socket
import ssl
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oneharness import doctor  # noqa: E402
from oneharness.errors import ProviderError  # noqa: E402
from oneharness.transport import post_json  # noqa: E402

CERT = ROOT / "tests" / "fixtures" / "tls" / "cert.pem"
KEY = ROOT / "tests" / "fixtures" / "tls" / "key.pem"


def banner(text: str) -> None:
    print(f"\n=== {text} ===")


def attempt(url: str, *, retries: int = 0) -> tuple[BaseException | None, float]:
    started = time.monotonic()
    try:
        post_json(url, {}, retries=retries, backoff=1.5)
    except BaseException as exc:
        return exc, time.monotonic() - started
    return None, time.monotonic() - started  # pragma: no cover - every case fails


def _wrap(text: str, width: int) -> str:
    """Cut the tail, not the middle: what identifies the failure is at the front."""
    return text if len(text) <= width else f"{text[: width - 1]}…"


def report(label: str, exc: BaseException | None, seconds: float) -> None:
    """One row: what happened, what kind it is, and what `doctor` would do next."""
    if exc is None:  # pragma: no cover - defensive
        print(f"  {label}: unexpectedly succeeded")
        return
    kind = getattr(exc, "kind", "-")
    status, detail = doctor.classify(exc) if isinstance(exc, ProviderError) else ("-", "")
    print(f"  {label}")
    print(f"    error type     : {type(exc).__name__} (kind={kind})")
    print(f"    doctor verdict : {status}")
    print(f"    what to do     : {doctor._next_step(status)}")
    print(f"    took           : {seconds:.2f}s")
    print(f"    the sentence   : {_wrap(str(exc).split(' | ', 1)[0], 74)}")


class UntrustedServer:
    """TLS with a certificate nobody trusts, surviving the rejected handshake.

    The handshake fails *inside* `accept`, so a plain `HTTPServer` would let that
    exception kill `serve_forever` -- and the second attempt would then meet a
    socket nobody is accepting on, which reads as a closed port rather than the
    certificate failure this is about.
    """

    def __init__(self) -> None:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(str(CERT), str(KEY))
        self.context = context
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(4)
        # By address: `localhost` resolves to ::1 on some hosts, and that would
        # report a closed port instead of the rejected certificate.
        self.url = f"https://127.0.0.1:{self.listener.getsockname()[1]}/v1/chat"

    def __enter__(self) -> UntrustedServer:
        threading.Thread(target=self._serve, daemon=True).start()
        return self

    def __exit__(self, *exc: object) -> None:
        with contextlib.suppress(OSError):
            self.listener.close()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self.listener.accept()
            except OSError:
                return
            with contextlib.suppress(OSError, ssl.SSLError):
                conn.settimeout(2.0)
                self.context.wrap_socket(conn, server_side=True).close()


class RefusingProxy:
    """A proxy that is reachable, answers, and says 407."""

    def __init__(self) -> None:
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def do_CONNECT(self) -> None:
                proxy.hits += 1
                self.send_response(407)
                self.send_header("proxy-authenticate", 'Basic realm="corp"')
                self.send_header("content-length", "0")
                self.end_headers()
                self.close_connection = True

            def do_POST(self) -> None:
                proxy.hits += 1
                self.send_response(407)
                self.send_header("content-length", "0")
                self.end_headers()

            do_GET = do_POST

            def log_message(self, *args: object) -> None:
                pass

        self.hits = 0
        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"

    def __enter__(self) -> RefusingProxy:
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@contextlib.contextmanager
def routed_through(url: str):
    """Make urllib use `url` as its proxy, then put every setting back."""
    keys = ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "no_proxy", "NO_PROXY")
    saved = {key: os.environ.get(key) for key in keys}
    for key in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"):
        os.environ[key] = url
    for key in ("no_proxy", "NO_PROXY"):
        os.environ.pop(key, None)  # a loopback exemption would bypass the proxy
    try:
        # Rebuilt while the environment says "use this proxy": urlopen caches one
        # opener for the life of the process, so the order here is the test.
        urllib.request.install_opener(urllib.request.build_opener())
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        # And rebuilt again afterwards, or the next section routes through a
        # proxy that no longer exists.
        urllib.request.install_opener(urllib.request.build_opener())


def untrusted_certificate() -> None:
    banner("1. a certificate nobody trusts")
    with UntrustedServer() as server:
        exc, seconds = attempt(server.url)
    report("https, self-signed certificate", exc, seconds)
    print("  the network is fine and the host name is right: this is a trust")
    print("  problem, and asking someone to check their connection is a wrong")
    print("  instruction wearing the clothes of a diagnosis.")


def proxy_wants_a_password() -> None:
    banner("2. a proxy that answers 407")
    with RefusingProxy() as proxy, routed_through(proxy.url):
        exc, seconds = attempt("http://model.example.invalid/v1/chat")
        print(f"  proxy hits: {proxy.hits}")
        report("plain http through a proxy", exc, seconds)
        exc, seconds = attempt("https://model.example.invalid/v1/chat")
        report("https tunnel through the same proxy", exc, seconds)
    print("  ^ the second one is the old `network` verdict at its worst: urllib")
    print("    reports a refused CONNECT as a bare OSError, the same face it")
    print("    wears for a typo in the host name.")


def name_does_not_resolve() -> None:
    banner("3. a name that does not resolve")
    exc, seconds = attempt("https://nothing-here.invalid/v1/chat")
    report("one character wrong in base_url", exc, seconds)


def port_is_closed() -> None:
    banner("4. a port nobody listens on")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    exc, seconds = attempt(f"http://127.0.0.1:{port}/v1/chat")
    report("the service is down, or the port is a typo", exc, seconds)


def retrying_what_cannot_change() -> None:
    banner("5. the second half of a diagnosis: could a retry possibly help?")
    with UntrustedServer() as server:
        exc, seconds = attempt(server.url, retries=2)
    print(f"  a bad certificate, retries=2: {seconds:.2f}s")
    assert exc is not None
    print(f"    retryable={getattr(exc, 'retryable', None)}  <- the backoff is not paid")
    print("  the backoff is a bet that the world changed in 1.5 seconds. A")
    print("  certificate nobody trusts will still be untrusted then, so the bet")
    print("  is provably lost -- and all it buys is three seconds of a terminal")
    print("  that looks frozen.")

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(2)
        url = f"http://127.0.0.1:{listener.getsockname()[1]}/v1"

        def hang_up() -> None:
            while True:
                try:
                    conn, _ = listener.accept()
                except OSError:
                    return
                conn.close()

        threading.Thread(target=hang_up, daemon=True).start()
        exc, seconds = attempt(url, retries=2)
    print(f"\n  a dropped connection, retries=2: {seconds:.2f}s")
    assert exc is not None
    print(f"    retryable={getattr(exc, 'retryable', None)}  <- worth another try")
    print("  same shape of failure, opposite decision. That is the whole point of")
    print("  naming the kind instead of printing one word for all six.")


def main() -> int:
    print("four ways to get no answer, and why they are not the same fact")
    untrusted_certificate()
    proxy_wants_a_password()
    name_does_not_resolve()
    port_is_closed()
    retrying_what_cannot_change()
    print("\nall four are 'no answer'. None of them is fixed in the same place.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
