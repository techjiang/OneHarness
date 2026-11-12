"""Anthropic Messages API wire format."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from oneharness.errors import ProviderError
from oneharness.providers.base import Provider
from oneharness.transport import post_json
from oneharness.types import Completion, Message, ToolCall, Usage


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str = "https://api.anthropic.com/v1",
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float = 120.0,
        anthropic_version: str = "2023-06-01",
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(model, temperature=temperature, max_tokens=max_tokens)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.anthropic_version = anthropic_version
        self.extra_headers = dict(extra_headers or {})
        self.extra_body = dict(extra_body or {})

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> Completion:
        system, turns = self._split(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": turns,
            "max_tokens": self.max_tokens or 4096,
            "temperature": self.temperature,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [
                {
                    "name": spec["name"],
                    "description": spec.get("description", ""),
                    "input_schema": spec.get("parameters") or {"type": "object", "properties": {}},
                }
                for spec in tools
            ]

        payload.update(self.extra_body)
        headers = {"x-api-key": self.api_key, "anthropic-version": self.anthropic_version}
        headers.update(self.extra_headers)

        data = post_json(f"{self.base_url}/messages", payload, headers, timeout=self.timeout)
        return self._decode(data)

    def _split(self, messages: Sequence[Message]) -> tuple[str, list[dict[str, Any]]]:
        """Anthropic wants system prompts hoisted out and tool results as user turns."""
        system_parts: list[str] = []
        turns: list[dict[str, Any]] = []

        for message in messages:
            if message.role == "system":
                if message.content:
                    system_parts.append(message.content)
                continue

            if message.role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": message.content,
                }
                if turns and turns[-1]["role"] == "user" and isinstance(turns[-1]["content"], list):
                    turns[-1]["content"].append(block)
                else:
                    turns.append({"role": "user", "content": [block]})
                continue

            if message.role == "assistant":
                blocks: list[dict[str, Any]] = []
                if message.content:
                    blocks.append({"type": "text", "text": message.content})
                for call in message.tool_calls:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.arguments,
                        }
                    )
                turns.append({"role": "assistant", "content": blocks or message.content})
                continue

            turns.append({"role": "user", "content": message.content})

        return "\n\n".join(system_parts), turns

    def _decode(self, data: dict[str, Any]) -> Completion:
        blocks = data.get("content")
        if not isinstance(blocks, list):
            raise ProviderError("response contained no content blocks", body=str(data)[:500])

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in blocks:
            kind = (block or {}).get("type")
            if kind == "text":
                text_parts.append(block.get("text") or "")
            elif kind == "tool_use":
                calls.append(
                    ToolCall.from_raw(
                        call_id=block.get("id") or f"call_{len(calls)}",
                        name=block.get("name") or "",
                        arguments=block.get("input") or {},
                    )
                )

        usage_data = data.get("usage") or {}
        usage = Usage(
            input_tokens=int(usage_data.get("input_tokens") or 0),
            output_tokens=int(usage_data.get("output_tokens") or 0),
        )
        return Completion(
            message=Message.assistant("".join(text_parts), calls),
            usage=usage,
            model=data.get("model") or self.model,
            stop_reason=data.get("stop_reason"),
            raw=data,
        )
