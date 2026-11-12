"""Google Gemini (generativelanguage) wire format.

Gemini differs from both OpenAI and Anthropic in ways that leak into the
agent loop: parts instead of content strings, `model` instead of `assistant`,
function arguments already decoded as objects, and no tool-call ids at all.
All of that is flattened here so callers still see plain `Message`s.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from oneharness.errors import ProviderError
from oneharness.providers.base import Provider
from oneharness.transport import post_json, post_sse
from oneharness.types import Completion, Message, StreamEvent, ToolCall, Usage


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
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
        data = post_json(
            self._endpoint("generateContent"),
            self._payload(messages, tools),
            self._headers(),
            timeout=self.timeout,
        )
        return self._decode(data)

    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> Iterator[StreamEvent]:
        text: list[str] = []
        parts: list[dict[str, Any]] = []
        usage = Usage()
        stop_reason: str | None = None
        model = self.model

        for _, chunk in post_sse(
            self._endpoint("streamGenerateContent", alt="sse"),
            self._payload(messages, tools),
            self._headers(),
            timeout=self.timeout,
        ):
            usage = self._usage(chunk.get("usageMetadata")) or usage
            model = chunk.get("modelVersion") or model
            candidates = chunk.get("candidates") or []
            if not candidates:
                continue
            candidate = candidates[0] or {}
            stop_reason = candidate.get("finishReason") or stop_reason
            for part in (candidate.get("content") or {}).get("parts") or ():
                if "text" in part:
                    piece = part.get("text") or ""
                    if piece:
                        text.append(piece)
                        yield StreamEvent(kind="text", text=piece)
                elif "functionCall" in part:
                    parts.append(part)

        calls = [
            ToolCall.from_raw(
                call_id=f"call_{index}",
                name=(part["functionCall"] or {}).get("name") or "",
                arguments=(part["functionCall"] or {}).get("args") or {},
            )
            for index, part in enumerate(parts)
        ]
        yield StreamEvent(
            kind="done",
            completion=Completion(
                message=Message.assistant("".join(text), calls),
                usage=usage,
                model=model,
                stop_reason=stop_reason,
                raw={"streamed": True},
            ),
        )

    def _endpoint(self, method: str, **params: str) -> str:
        query = "&".join(f"{key}={value}" for key, value in params.items())
        suffix = f"?{query}" if query else ""
        return f"{self.base_url}/models/{self.model}:{method}{suffix}"

    def _headers(self) -> dict[str, str]:
        headers = {"x-goog-api-key": self.api_key}
        headers.update(self.extra_headers)
        return headers

    def _payload(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        system = "\n\n".join(m.content for m in messages if m.role == "system" and m.content)
        payload: dict[str, Any] = {
            "contents": self._encode_turns(messages),
            "generationConfig": {"temperature": self.temperature},
        }
        if self.max_tokens is not None:
            payload["generationConfig"]["maxOutputTokens"] = self.max_tokens
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": spec["name"],
                            "description": spec.get("description", ""),
                            "parameters": spec.get("parameters") or {"type": "object"},
                        }
                        for spec in tools
                    ]
                }
            ]

        # `streamGenerateContent` needs the SSE alternative to be JSON, not chunks.
        payload.update(self.extra_body)
        return payload

    def _encode_turns(self, messages: Sequence[Message]) -> list[dict[str, Any]]:
        turns: list[dict[str, Any]] = []

        def push(role: str, parts: list[dict[str, Any]]) -> None:
            if turns and turns[-1]["role"] == role:
                turns[-1]["parts"].extend(parts)
            else:
                turns.append({"role": role, "parts": parts})

        for message in messages:
            if message.role == "system":
                continue
            if message.role == "tool":
                push(
                    "user",
                    [
                        {
                            "functionResponse": {
                                "name": message.name or "tool",
                                "response": {"result": message.content},
                            }
                        }
                    ],
                )
                continue
            if message.role == "assistant":
                parts: list[dict[str, Any]] = []
                if message.content:
                    parts.append({"text": message.content})
                parts.extend(
                    {"functionCall": {"name": call.name, "args": call.arguments}}
                    for call in message.tool_calls
                )
                if parts:
                    push("model", parts)
                continue
            push("user", [{"text": message.content}])

        return turns

    def _decode(self, data: dict[str, Any]) -> Completion:
        candidates = data.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ProviderError("response contained no candidates", body=str(data)[:500])

        candidate = candidates[0] or {}
        parts = (candidate.get("content") or {}).get("parts") or []
        texts: list[str] = []
        calls: list[ToolCall] = []
        for part in parts:
            if "text" in part:
                texts.append(part.get("text") or "")
            if "functionCall" in part:
                call = part["functionCall"] or {}
                calls.append(
                    ToolCall.from_raw(
                        call_id=f"call_{len(calls)}",  # Gemini has no ids of its own
                        name=call.get("name") or "",
                        arguments=call.get("args") or {},
                    )
                )

        return Completion(
            message=Message.assistant("".join(texts), calls),
            usage=self._usage(data.get("usageMetadata")) or Usage(),
            model=data.get("modelVersion") or self.model,
            stop_reason=candidate.get("finishReason"),
            raw=data,
        )

    @staticmethod
    def _usage(meta: Any) -> Usage | None:
        if not isinstance(meta, dict):
            return None
        return Usage(
            input_tokens=int(meta.get("promptTokenCount") or 0),
            output_tokens=int(meta.get("candidatesTokenCount") or 0),
        )
