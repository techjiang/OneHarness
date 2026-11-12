import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from oneharness import doctor
from oneharness.cli import main
from oneharness.errors import ProviderError
from oneharness.providers.openai import OpenAIProvider
from oneharness.tape import Tape
from oneharness.types import Message
from tests.test_tape import JSONServer
from tests.test_tape import reply as _reply

TAPES = Path(__file__).parent / "fixtures" / "tapes"


class ClassifyTests(unittest.TestCase):
    """The verdict is the product: `auth` and `protocol` must never be confused."""

    def check(self, status, body):
        return doctor.classify(ProviderError("failed", status=status, body=body))

    def test_401_is_auth(self):
        self.assertEqual(self.check(401, '{"error":{"message":"bad key"}}')[0], doctor.AUTH)

    def test_403_is_auth_not_endpoint(self):
        # Groq says nothing more than "Forbidden"; it is still a credential
        # answer, and calling it `endpoint` would send people to the wrong file.
        self.assertEqual(self.check(403, '{"error":{"message":"Forbidden"}}')[0], doctor.AUTH)

    def test_400_complaining_about_our_fields_is_a_bug(self):
        body = '{"error":{"message":"Unknown name \\"messages\\": Cannot find field."}}'
        self.assertEqual(self.check(400, body)[0], doctor.PROTOCOL)

    def test_400_about_the_key_is_auth(self):
        body = '{"error":{"message":"API key not valid. Please pass a valid API key."}}'
        self.assertEqual(self.check(400, body)[0], doctor.AUTH)

    def test_404_on_a_model_is_model(self):
        body = '{"error":{"message":"The model `gpt-9` does not exist"}}'
        self.assertEqual(self.check(404, body)[0], doctor.MODEL)

    def test_404_on_a_path_is_endpoint(self):
        self.assertEqual(self.check(404, '{"error":"not found"}')[0], doctor.ENDPOINT)

    def test_429_is_quota(self):
        self.assertEqual(self.check(429, "rate limited")[0], doctor.QUOTA)

    def test_5xx_is_the_network_not_the_key(self):
        self.assertEqual(self.check(503, "unavailable")[0], doctor.NETWORK)

    def test_no_status_is_network(self):
        self.assertEqual(doctor.classify(ProviderError("cannot reach host"))[0], doctor.NETWORK)

    def test_nested_message_is_extracted(self):
        body = json.dumps({"error": {"message": "  invalid x-api-key \n"}})
        status, detail = self.check(401, body)
        self.assertEqual(status, doctor.AUTH)
        self.assertEqual(detail, "invalid x-api-key")

    def test_non_json_body_is_kept_verbatim(self):
        self.assertEqual(self.check(401, "<html>denied</html>")[1], "<html>denied</html>")


class ProbeTests(unittest.TestCase):
    def test_missing_key_is_unconfigured_and_not_a_failure(self):
        import os

        saved = os.environ.pop("OPENAI_API_KEY", None)
        try:
            probe = doctor.check("openai")
        finally:
            if saved is not None:
                os.environ["OPENAI_API_KEY"] = saved
        self.assertEqual(probe.status, doctor.UNCONFIGURED)
        self.assertFalse(probe.failed, "absence of a key must not turn CI red")
        self.assertIn("OPENAI_API_KEY", probe.detail)

    def test_landing_on_a_closed_port_is_network(self):
        probe = doctor.check("ollama", base_url="http://127.0.0.1:1/v1", timeout=1)
        self.assertEqual(probe.status, doctor.NETWORK)

    def test_the_key_is_redacted_out_of_the_report(self):
        # A doctor report gets pasted into issues; the key must already be gone.
        with Tape.replay(TAPES / "openai.json"):
            probe = doctor.check("openai", api_key="sk-invalid-oneharness-abc")
        self.assertNotIn("sk-invalid-oneharness-abc", probe.detail)
        self.assertIn("auth", probe.status)

    def test_json_shape_is_stable(self):
        with Tape.replay(TAPES / "anthropic.json"):
            data = doctor.check("anthropic").as_dict()
        self.assertEqual(
            sorted(data),
            sorted(["preset", "status", "detail", "hint", "model", "base_url", "seconds", "usage"]),
        )


class CliTests(unittest.TestCase):
    def test_auth_failure_exits_non_zero(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["doctor", "openai", "--json", "--replay", str(TAPES / "openai.json")])
        self.assertEqual(code, 1, "an auth failure is a non-zero exit")
        self.assertEqual(json.loads(out.getvalue())[0]["status"], doctor.AUTH)

    def test_ok_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ok.json"
            with JSONServer(_reply("pong")) as server, Tape.record(path):
                OpenAIProvider("gpt-4o-mini", api_key="sk", base_url=server.url).complete(
                    [Message.user(doctor.PING)]
                )
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                args = [
                    "doctor",
                    "openai",
                    "--model",
                    "gpt-4o-mini",
                    "--base-url",
                    server.url,
                    "--json",
                ]
                code = main([*args, "--replay", str(path)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.getvalue())[0]["status"], doctor.OK)

    def test_unknown_preset_is_a_usage_error(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = main(["doctor", "nope"])
        self.assertEqual(code, 1)
        self.assertIn("unknown provider", err.getvalue())

    def test_record_and_replay_together_is_refused(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = main(["run", "hi", "--record", "a.json", "--replay", "b.json"])
        self.assertEqual(code, 1)
        self.assertIn("pick one", err.getvalue())


if __name__ == "__main__":
    unittest.main()
