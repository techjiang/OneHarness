"""No answer arrived -- but *which* no-answer?

Every one of these used to come out as the single word "network", or worse, as
`protocol`. The point of the file is that each cause is named, because each one
is fixed somewhere else: a certificate is a trust problem, a proxy 407 is a
credential for a different system, and neither is fixed by checking DNS.

The TLS and proxy cases run against real servers -- a real handshake, a real
CONNECT -- because the whole question is what urllib does with the bytes the
outside world sends back. A mocked `URLError` would only prove that this code
reads its own assumptions back.
"""

import contextlib
import os
import socket
import ssl
import threading
import time
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from oneharness import doctor, transport
from oneharness.errors import NetworkError, ProviderError

HERE = Path(__file__).resolve().parent
CERT = HERE / "fixtures" / "tls" / "cert.pem"
KEY = HERE / "fixtures" / "tls" / "key.pem"


class RefusingProxy:
    """A proxy that answers every CONNECT with 407 and every request with 407."""

    def __init__(self):
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def do_CONNECT(self):  # https:// through a proxy
                proxy.hits += 1
                self.send_response(407)
                self.send_header("proxy-authenticate", 'Basic realm="corp"')
                self.send_header("content-length", "0")
                self.end_headers()
                self.close_connection = True

            def do_POST(self):  # http:// through a proxy, same wall
                proxy.hits += 1
                self.send_response(407)
                self.send_header("content-length", "0")
                self.end_headers()

            do_GET = do_POST

            def log_message(self, *args):
                pass

        self.hits = 0
        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"

    def __enter__(self):
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


@contextlib.contextmanager
def proxy_env(url):
    """Point urllib at `url` for the duration, and put the old settings back.

    `no_proxy` is cleared explicitly: a machine that exempts the loopback would
    silently bypass the proxy and turn this test into a DNS test.
    """
    keys = ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "no_proxy", "NO_PROXY")
    saved = {key: os.environ.get(key) for key in keys}
    for key in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"):
        os.environ[key] = url
    for key in ("no_proxy", "NO_PROXY"):
        os.environ.pop(key, None)
    try:
        # `urlopen` builds its opener once and keeps it for the life of the
        # process, so the environment has to be in place *before* the rebuild --
        # otherwise this test quietly goes direct and passes for the wrong
        # reason.
        urllib.request.install_opener(urllib.request.build_opener())
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        # Rebuilt *after* the environment is back to normal: an opener built
        # while the proxy was still set keeps routing through a proxy that no
        # longer exists, and every later test fails with a confusing "refused".
        urllib.request.install_opener(urllib.request.build_opener())


class UntrustedServer:
    """Serves TLS with a certificate nobody trusts, and survives being rejected.

    The handshake fails *inside* `accept`, so a plain `HTTPServer` would let
    that exception kill `serve_forever` -- and the next test would then connect
    to a socket nobody is accepting on, which looks like a refused port rather
    than the certificate failure this is about. So the accept loop swallows the
    handshake error: refusing is the whole point of the fixture.
    """

    def __init__(self):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(str(CERT), str(KEY))
        self.context = context
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(4)
        # By address, not by name: `localhost` resolves to ::1 on some hosts, and
        # connecting there would report a refused port instead of the rejected
        # certificate this fixture exists to produce.
        self.url = f"https://127.0.0.1:{self.listener.getsockname()[1]}/v1/chat"

    def __enter__(self):
        threading.Thread(target=self._serve, daemon=True).start()
        return self

    def __exit__(self, *exc):
        with contextlib.suppress(OSError):
            self.listener.close()

    def _serve(self):
        while True:
            try:
                conn, _ = self.listener.accept()
            except OSError:
                return
            # The client is expected to abandon the handshake; that is a pass,
            # not a server bug.
            with contextlib.suppress(OSError, ssl.SSLError):
                conn.settimeout(2.0)
                self.context.wrap_socket(conn, server_side=True).close()


class TlsTests(unittest.TestCase):
    """A real self-signed certificate, a real handshake, a real rejection."""

    def setUp(self):
        if not (CERT.exists() and KEY.exists()):
            raise unittest.SkipTest("no test certificate; see tests/fixtures/tls/README.md")
        self.server = UntrustedServer().__enter__()
        self.addCleanup(self.server.__exit__)

    def fail_handshake(self):
        return self.assertRaises(NetworkError)

    @property
    def url(self):
        return self.server.url

    def test_an_untrusted_certificate_is_not_a_network_problem(self):
        self.server = self.server  # keeps the fixture visible to the reader
        with self.assertRaises(NetworkError) as ctx:
            transport.post_json(self.url, {}, retries=0)
        self.assertEqual(ctx.exception.kind, "tls")
        self.assertIn("TLS verification failed", str(ctx.exception))

    def test_doctor_says_tls_not_network(self):
        # The instruction a `network` verdict gives -- "check DNS, TLS, proxy" --
        # is three guesses. A certificate has exactly one fix.
        with self.assertRaises(NetworkError) as ctx:
            transport.post_json(self.url, {}, retries=0)
        self.assertEqual(doctor.classify(ctx.exception)[0], doctor.TLS)
        self.assertIn("certificate", doctor._next_step(doctor.TLS))

    def test_a_tls_failure_is_not_a_provider_rejection(self):
        # `status=None` matters: `is_overflow` and `classify` must not read this
        # as a body the provider chose to send.
        with self.assertRaises(NetworkError) as ctx:
            transport.post_json(self.url, {}, retries=0)
        self.assertIsNone(ctx.exception.status)

    def test_a_bad_certificate_is_not_retried(self):
        # Three attempts of a handshake that was refused for the same reason is
        # three seconds of a terminal that looks frozen, then the same sentence.
        # The backoff bets the world changed; a certificate is not a gamble.
        started = time.monotonic()
        with self.assertRaises(NetworkError) as ctx:
            transport.post_json(self.url, {}, retries=2, backoff=1.5)
        self.assertFalse(ctx.exception.retryable)
        self.assertLess(time.monotonic() - started, 1.0)  # no backoff was paid

    def test_it_does_not_leak_a_raw_ssl_error(self):
        # `SSLError` is an `OSError`, so without this conversion it sails past
        # every `except OneHarnessError` and reaches the terminal as a traceback.
        with self.assertRaises(ProviderError) as ctx:
            transport.post_json(self.url, {}, retries=0)
        self.assertNotIsInstance(ctx.exception, ssl.SSLError)


class DoctorVerdictTests(unittest.TestCase):
    """`doctor` is the thing a person runs when they have no idea what broke.

    A dummy key is set for the duration: without one every probe stops at
    `unconfigured` before it ever opens a socket, and this file would be
    asserting on the absence of a credential instead of on what came back.
    """

    def setUp(self):
        if not (CERT.exists() and KEY.exists()):
            raise unittest.SkipTest("no test certificate")
        self.server = UntrustedServer().__enter__()
        self.addCleanup(self.server.__exit__)
        self.addCleanup(os.environ.pop, "OPENAI_API_KEY", None)
        os.environ["OPENAI_API_KEY"] = "sk-not-a-real-key-for-a-probe"

    def test_an_untrusted_certificate_says_tls_and_names_one_fix(self):
        probe = doctor.check("openai", base_url=self.server.url, timeout=5.0)
        self.assertEqual(probe.status, doctor.TLS)
        self.assertTrue(probe.failed)
        self.assertIn("certificate", probe.hint)

    def test_a_proxy_in_the_way_is_not_reported_as_a_oneharness_bug(self):
        with (
            RefusingProxy() as proxy,
            proxy_env(proxy.url),
        ):
            probe = doctor.check("openai", base_url="http://model.example.invalid/v1", timeout=5.0)
        self.assertEqual(probe.status, doctor.PROXY)
        self.assertNotEqual(probe.status, doctor.PROTOCOL)


class ProxyTests(unittest.TestCase):
    """A proxy that is reachable, answers, and says no."""

    def test_a_407_on_a_plain_request_is_a_proxy_not_our_body(self):
        with (
            RefusingProxy() as proxy,
            proxy_env(proxy.url),
            self.assertRaises(NetworkError) as ctx,
        ):
            transport.post_json("http://model.example.invalid/v1/chat", {}, retries=0)
        self.assertEqual(ctx.exception.kind, "proxy")
        self.assertEqual(proxy.hits, 1)

    def test_a_407_on_a_plain_request_is_not_called_a_oneharness_bug(self):
        # This is the regression that mattered: 407 used to fall through to
        # `protocol`, i.e. "our request body is malformed", which sends the
        # reader into `providers/` looking for a field that was never wrong.
        with (
            RefusingProxy() as proxy,
            proxy_env(proxy.url),
            self.assertRaises(ProviderError) as ctx,
        ):
            transport.post_json("http://model.example.invalid/v1/chat", {}, retries=0)
        self.assertNotEqual(doctor.classify(ctx.exception)[0], doctor.PROTOCOL)

    def test_a_refused_connect_tunnel_is_a_proxy_not_dns(self):
        # `http.client` reports this as a bare OSError whose text is the only
        # evidence. urllib hands it over as `URLError`, the same face it wears
        # for a typo in the host name -- but the proxy did resolve, connect and
        # answer, so telling someone to check DNS is simply wrong.
        with (
            RefusingProxy() as proxy,
            proxy_env(proxy.url),
            self.assertRaises(NetworkError) as ctx,
        ):
            transport.post_json("https://model.example.invalid/v1/chat", {}, retries=0)
        self.assertEqual(ctx.exception.kind, "proxy")
        self.assertIn("407", str(ctx.exception))
        self.assertEqual(doctor.classify(ctx.exception)[0], doctor.PROXY)

    def test_a_dead_proxy_is_not_reported_as_a_dead_model_host(self):
        # `HTTPS_PROXY` pointing at a port nobody listens on produces
        # `ConnectionRefusedError`, and the URL in hand is the model host. But
        # the connection that was refused went to the proxy -- the model host
        # was never contacted, and may well be fine.
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            dead = sock.getsockname()[1]  # closed as soon as this block ends
        with proxy_env(f"http://127.0.0.1:{dead}"), self.assertRaises(NetworkError) as ctx:
            transport.post_json("https://model.example.invalid/v1/chat", {}, retries=0)
        self.assertEqual(ctx.exception.kind, "proxy")
        self.assertIn("proxy", str(ctx.exception))
        self.assertNotIn("nothing is listening at https://model", str(ctx.exception))

    def test_a_407_is_never_retried(self):
        # A proxy that wants a password will keep wanting it. Three attempts of
        # the same wall is three seconds of nothing on the user's screen.
        with (
            RefusingProxy() as proxy,
            proxy_env(proxy.url),
            self.assertRaises(NetworkError),
        ):
            transport.post_json("http://model.example.invalid/v1/chat", {}, retries=3, backoff=0.01)
        self.assertEqual(proxy.hits, 1)


class KindTests(unittest.TestCase):
    """The cheaper causes, asserted through the same vocabulary."""

    def test_dns_and_refused_are_different_kinds(self):
        with self.assertRaises(NetworkError) as ctx:
            transport.post_json("https://nothing.invalid/v1", {}, retries=0)
        self.assertEqual(ctx.exception.kind, "dns")

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        with self.assertRaises(NetworkError) as ctx:
            transport.post_json(f"http://127.0.0.1:{port}/v1", {}, retries=0)
        self.assertEqual(ctx.exception.kind, "refused")

    def test_a_lost_connection_is_retryable_and_so_says_the_error(self):
        # A reset before the first byte is transient; a rejected certificate is
        # not. Encoding that in the error means the retry loop stops guessing.
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            url = f"http://127.0.0.1:{listener.getsockname()[1]}/v1"

            def accept_and_close():
                conn, _ = listener.accept()
                conn.close()

            threading.Thread(target=accept_and_close, daemon=True).start()
            with self.assertRaises(NetworkError) as ctx:
                transport.post_json(url, {}, retries=0)
        self.assertEqual(ctx.exception.kind, "lost")
        self.assertTrue(ctx.exception.retryable)

    def test_a_network_error_is_still_a_provider_error(self):
        # The agent loop's "keep the half answer" branch catches ProviderError.
        # A sibling class would have slipped past it and thrown the half answer
        # away -- the very bug v1.0 set out to fix.
        self.assertTrue(issubclass(NetworkError, ProviderError))

    def test_every_kind_has_a_hint(self):
        for kind in ("dns", "refused", "tls", "proxy", "timeout", "lost"):
            self.assertTrue(NetworkError("x", kind=kind).hint, kind)


if __name__ == "__main__":
    unittest.main()
