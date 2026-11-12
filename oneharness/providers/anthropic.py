"""Anthropic Messages API wire format."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from oneharness.errors import ProviderError
from oneharness.providers.base import Provider
from oneharness.transport import post_json, post_sse
from oneharness.types import (
    STREAM_ENDED_EARLY,
    Completion,
    Message,
    StreamEvent,
    ToolCall,
    Usage,
)


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
        data = post_json(
            self._endpoint(), self._payload(messages, tools), self._headers(), timeout=self.timeout
        )
        return self._decode(data)

    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> Iterator[StreamEvent]:
        payload = self._payload(messages, tools)
        payload["stream"] = True

        text: list[str] = []
        calls: list[ToolCall] = []
        partial_json: dict[int, list[str]] = {}
        blocks: dict[int, dict[str, Any]] = {}
        usage = Usage()
        stop_reason = None
        model = self.model

        for _, chunk in post_sse(self._endpoint(), payload, self._headers(), timeout=self.timeout):
            kind = chunk.get("type")
            if kind == "message_start":
                info = chunk.get("message") or {}
                model = info.get("model") or model
                raw_usage = info.get("usage") or {}
                usage = Usage(input_tokens=int(raw_usage.get("input_tokens") or 0))
            elif kind == "content_block_start":
                index = int(chunk.get("index", 0))
                block = chunk.get("content_block") or {}
                blocks[index] = block
                if block.get("type") == "tool_use":
                    partial_json[index] = []
                    # The id and the name arrive here, before any argument
                    # exists: announce it now, so a watcher knows a tool is
                    # coming while the model is still writing its input.
                    yield StreamEvent(
                        kind="tool",
                        index=index,
                        call=ToolCall(
                            id=block.get("id") or f"call_{index}",
                            name=block.get("name") or "",
                        ),
                    )
            elif kind == "content_block_delta":
                index = int(chunk.get("index", 0))
                delta = chunk.get("delta") or {}
                if delta.get("type") == "text_delta":
                    piece = delta.get("text") or ""
                    if piece:
                        text.append(piece)
                        yield StreamEvent(kind="text", text=piece)
                elif delta.get("type") == "input_json_delta":
                    fragment = delta.get("partial_json") or ""
                    partial_json.setdefault(index, []).append(fragment)
                    if fragment:
                        yield StreamEvent(
                            kind="tool",
                            index=index,
                            fragment=fragment,
                            call=ToolCall(
                                id=(blocks.get(index) or {}).get("id") or f"call_{index}",
                                name=(blocks.get(index) or {}).get("name") or "",
                            ),
                        )
            elif kind == "message_delta":
                stop_reason = (chunk.get("delta") or {}).get("stop_reason") or stop_reason
                raw_usage = chunk.get("usage") or {}
                if raw_usage.get("output_tokens") is not None:
                    usage.output_tokens = int(raw_usage["output_tokens"])

        # Tool inputs arrive as JSON fragments, one buffer per tool_use block.
        for index, block in sorted(blocks.items()):
            if block.get("type") != "tool_use":
                continue
            arguments: Any = block.get("input") or "{}"
            fragments = partial_json.get(index)
            if fragments:
                arguments = "".join(fragments) or "{}"
            calls.append(
                ToolCall.from_raw(
                    call_id=block.get("id") or f"call_{len(calls)}",
                    name=block.get("name") or "",
                    arguments=arguments,
                )
            )

        yield StreamEvent(
            kind="done",
            completion=Completion(
                message=Message.assistant("".join(text), calls),
                usage=usage,
                model=model,
                stop_reason=stop_reason,
                raw={"streamed": True},
                # `stop_reason` rides in `message_delta`; if it never came, the
                # body was cut after `content_block_delta` and the text is real
                # but unfinished.
                truncated=None if stop_reason else STREAM_ENDED_EARLY,
            ),
        )

    def _endpoint(self) -> str:
        return f"{self.base_url}/messages"

    def _headers(self) -> dict[str, str]:
        headers = {"x-api-key": self.api_key, "anthropic-version": self.anthropic_version}
        headers.update(self.extra_headers)
        return headers

    def _payload(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None,
    ) -> dict[str, Any]:
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
        return payload

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
                block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": message.content,
                }
                if message.is_error:
                    block["is_error"] = True
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
