"""OpenAI Chat Completions wire format.

Also the de-facto standard for DeepSeek, Moonshot, Qwen, Ollama, vLLM and
llama.cpp servers, so one adapter covers all of them via `base_url`.
"""

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
        cache: bool = False,
    ) -> None:
        super().__init__(model, temperature=temperature, max_tokens=max_tokens)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.extra_headers = dict(extra_headers or {})
        self.extra_body = dict(extra_body or {})
        # Accepted and ignored, and that is the correct behaviour rather than a
        # missing feature: OpenAI, DeepSeek, Groq and the rest of this wire
        # format cache repeated prefixes without being asked, and there is no
        # field to put a breakpoint in. Taking the flag keeps `--cache` portable
        # across presets instead of turning a shared config into a crash.
        self.cache = cache

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
        payload["stream_options"] = {"include_usage": True}

        text: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        model = self.model
        stop_reason = None
        usage = Usage()

        for _, chunk in post_sse(self._endpoint(), payload, self._headers(), timeout=self.timeout):
            if chunk.get("usage"):
                usage = _usage(chunk["usage"])
            model = chunk.get("model") or model
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0] or {}
            stop_reason = choice.get("finish_reason") or stop_reason
            delta = choice.get("delta") or {}

            piece = delta.get("content")
            if piece:
                text.append(piece)
                yield StreamEvent(kind="text", text=piece)

            for raw in delta.get("tool_calls") or ():
                index = int(raw.get("index", len(calls)))
                slot = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                if raw.get("id"):
                    slot["id"] = raw["id"]
                function = raw.get("function") or {}
                if function.get("name"):
                    slot["name"] = function["name"]
                fragment = function.get("arguments") or ""
                slot["arguments"] += fragment
                # Announced as it is assembled, not when the turn closes: the
                # name arrives in one chunk and the arguments in many, and
                # someone watching should not learn about a `write_file` only
                # after the whole body has been generated.
                yield StreamEvent(
                    kind="tool",
                    index=index,
                    fragment=fragment,
                    call=ToolCall(
                        id=slot["id"] or f"call_{index}", name=slot["name"], raw_arguments=""
                    ),
                )

        message = Message.assistant(
            "".join(text),
            [
                ToolCall.from_raw(
                    call_id=slot["id"] or f"call_{index}",
                    name=slot["name"],
                    arguments=slot["arguments"] or "{}",
                )
                for index, slot in sorted(calls.items())
            ],
        )
        yield StreamEvent(
            kind="done",
            completion=Completion(
                message=message,
                usage=usage,
                model=model,
                stop_reason=stop_reason,
                raw={"streamed": True},
                # No `finish_reason` means the body ended without the provider
                # saying it was done. Saying `done` here would be this adapter
                # inventing an ending the server never sent.
                truncated=None if stop_reason else STREAM_ENDED_EARLY,
            ),
        )

    def _endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"authorization": f"Bearer {self.api_key}"}
        headers.update(self.extra_headers)
        return headers

    def _payload(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None,
    ) -> dict[str, Any]:
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
        return payload

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
        return Completion(
            message=message,
            usage=_usage(data.get("usage") or {}),
            model=data.get("model") or self.model,
            stop_reason=choice.get("finish_reason"),
            raw=data,
        )


def _usage(raw: dict[str, Any]) -> Usage:
    """OpenAI-shaped usage, with the cached share pulled out of the prompt count.

    `prompt_tokens_details.cached_tokens` is a *subset* of `prompt_tokens`, so
    the total needs no arithmetic -- but DeepSeek (same wire format) reports it
    as a flat `prompt_cache_hit_tokens` instead, and several compatible servers
    report neither. Reading both spellings and defaulting to 0 keeps one adapter
    correct for all of them.
    """
    details = raw.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens") or raw.get("prompt_cache_hit_tokens") or 0
    return Usage(
        input_tokens=int(raw.get("prompt_tokens") or 0),
        output_tokens=int(raw.get("completion_tokens") or 0),
        cached_tokens=int(cached),
    )
