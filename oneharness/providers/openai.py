"""OpenAI Chat Completions wire format.

Also the de-facto standard for DeepSeek, Moonshot, Qwen, Ollama, vLLM and
llama.cpp servers, so one adapter covers all of them via `base_url`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from oneharness.errors import ProviderError
from oneharness.providers.base import Provider
from oneharness.transport import post_json
from oneharness.types import Completion, Message, ToolCall, Usage


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float = 120.0,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(model, temperature=temperature, max_tokens=max_tokens)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.extra_headers = dict(extra_headers or {})
        self.extra_body = dict(extra_body or {})

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [self._encode(m) for m in messages],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if tools:
            payload["tools"] = [{"type": "function", "function": dict(t)} for t in tools]

        payload.update(self.extra_body)
        headers = {"authorization": f"Bearer {self.api_key}"}
        headers.update(self.extra_headers)

        data = post_json(
            f"{self.base_url}/chat/completions", payload, headers, timeout=self.timeout
        )
        return self._decode(data)

    def _encode(self, message: Message) -> dict[str, Any]:
        if message.role == "tool":
            return {
                "role": "tool",
                "content": message.content,
                "tool_call_id": message.tool_call_id,
            }
        encoded: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_calls:
            encoded["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments_text()},
                }
                for call in message.tool_calls
            ]
            # OpenAI rejects an empty string alongside tool_calls.
            encoded["content"] = message.content or None
        return encoded

    def _decode(self, data: dict[str, Any]) -> Completion:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderError("response contained no choices", body=str(data)[:500])

        choice = choices[0] or {}
        raw_message = choice.get("message") or {}
        calls: list[ToolCall] = []
        for index, item in enumerate(raw_message.get("tool_calls") or ()):
            function = item.get("function") or {}
            calls.append(
                ToolCall.from_raw(
                    call_id=item.get("id") or f"call_{index}",
                    name=function.get("name") or "",
                    arguments=function.get("arguments"),
                )
            )

        message = Message.assistant(raw_message.get("content") or "", calls)
        usage_data = data.get("usage") or {}
        usage = Usage(
            input_tokens=int(usage_data.get("prompt_tokens") or 0),
            output_tokens=int(usage_data.get("completion_tokens") or 0),
        )
        return Completion(
            message=message,
            usage=usage,
            model=data.get("model") or self.model,
            stop_reason=choice.get("finish_reason"),
            raw=data,
        )
