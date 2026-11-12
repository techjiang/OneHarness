"""Azure OpenAI.

The wire format is OpenAI's. Everything that differs is about reaching it:

* the endpoint embeds the *deployment* name, not the model name;
* a deployment has no meaning without `api-version`, which is a query
  parameter rather than a header;
* authentication is `api-key`, not `Authorization: Bearer`.

Each of those is a one-line difference and a hard 404 or 401 if you get it
wrong, which is exactly the kind of thing that should live in one place.
"""

from __future__ import annotations

from typing import Any

from oneharness.providers.openai import OpenAIProvider


class AzureOpenAIProvider(OpenAIProvider):
    name = "azure"

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str,
        api_version: str = "2024-10-21",
        **options: Any,
    ) -> None:
        super().__init__(model, api_key=api_key, base_url=base_url, **options)
        self.api_version = api_version

    def _endpoint(self) -> str:
        # `model` is the deployment name here: Azure ignores the `model` field
        # in the payload and routes on the path, so one name has to do both.
        return (
            f"{self.base_url}/openai/deployments/{self.model}"
            f"/chat/completions?api-version={self.api_version}"
        )

    def _headers(self) -> dict[str, str]:
        headers = {"api-key": self.api_key}
        headers.update(self.extra_headers)
        return headers
