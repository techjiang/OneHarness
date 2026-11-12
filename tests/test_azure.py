import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from oneharness.errors import ConfigError
from oneharness.providers import create_provider
from oneharness.providers.azure import AzureOpenAIProvider
from oneharness.types import Message


class Recorder:
    def __init__(self, reply):
        self.reply = reply
        self.requests = []
        recorder = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("content-length", 0)))
                recorder.requests.append(
                    {
                        "path": self.path,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "body": json.loads(raw or b"{}"),
                    }
                )
                payload = json.dumps(recorder.reply).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


REPLY = {
    "choices": [{"message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
}


class AzureTests(unittest.TestCase):
    def test_deployment_and_api_version_are_in_the_path(self):
        with Recorder(REPLY) as recorder:
            provider = AzureOpenAIProvider(
                "my-deployment", api_key="k", base_url=recorder.url, api_version="2024-06-01"
            )
            provider.complete([Message.user("ping")])
            self.assertEqual(
                recorder.requests[0]["path"],
                "/openai/deployments/my-deployment/chat/completions?api-version=2024-06-01",
            )

    def test_authentication_uses_the_api_key_header(self):
        with Recorder(REPLY) as recorder:
            provider = AzureOpenAIProvider("dep", api_key="secret", base_url=recorder.url)
            provider.complete([Message.user("ping")])
            headers = recorder.requests[0]["headers"]
            self.assertEqual(headers["api-key"], "secret")
            self.assertNotIn("authorization", headers)

    def test_openai_wire_format_is_reused(self):
        with Recorder(REPLY) as recorder:
            provider = AzureOpenAIProvider("dep", api_key="k", base_url=recorder.url)
            completion = provider.complete([Message.user("ping")])
        self.assertEqual(completion.text, "pong")
        self.assertEqual(completion.usage.total_tokens, 4)

    def test_preset_requires_a_model_and_an_endpoint(self):
        with self.assertRaises(ConfigError) as caught:
            create_provider("azure", api_key="k", base_url="https://x.openai.azure.com")
        # The message has to say *what* to pass, not just that something is missing.
        self.assertIn("deployment name", str(caught.exception))

        with self.assertRaises(ConfigError) as caught:
            create_provider("azure", model="dep", api_key="k")
        self.assertIn("AZURE_OPENAI_ENDPOINT", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
