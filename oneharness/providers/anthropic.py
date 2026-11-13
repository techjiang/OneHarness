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
    # Nothing is cached unless a breakpoint says where, so this is the one place
    # `--cache` changes what goes on the wire rather than only what is reported.
    cache_mode = "explicit"

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
        cache: bool = False,
    ) -> None:
        super().__init__(model, temperature=temperature, max_tokens=max_tokens)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.anthropic_version = anthropic_version
        self.extra_headers = dict(extra_headers or {})
        self.extra_body = dict(extra_body or {})
        # Anthropic is the one provider that caches *only* where you ask it to:
        # a request with no `cache_control` breakpoint is billed in full, every
        # time. That makes a tool loop -- which resends the whole history on
        # every step by construction -- the worst possible shape for it unless
        # the breakpoints are placed.
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
                usage = _usage(info.get("usage") or {})
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
        if self.cache:
            _place_breakpoints(payload)
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

        return Completion(
            message=Message.assistant("".join(text_parts), calls),
            usage=_usage(data.get("usage") or {}),
            model=data.get("model") or self.model,
            stop_reason=data.get("stop_reason"),
            raw=data,
        )


def _usage(raw: dict[str, Any]) -> Usage:
    """Anthropic usage, where `input_tokens` means "the part you paid for".

    This is the one provider whose prompt count is *exclusive*: a request that
    read 2000 tokens out of the cache reports `input_tokens: 12` and
    `cache_read_input_tokens: 2000`. Reported as-is, a run that is 99% cached
    looks like a run that got cheaper -- and it did, but it also looks like a
    prompt of 12 tokens, which no budget arithmetic should ever believe.
    So the total is summed here, once, and the cached half kept beside it.
    """
    read = int(raw.get("cache_read_input_tokens") or 0)
    write = int(raw.get("cache_creation_input_tokens") or 0)
    return Usage(
        input_tokens=int(raw.get("input_tokens") or 0) + read + write,
        output_tokens=int(raw.get("output_tokens") or 0),
        cached_tokens=read,
        cache_write_tokens=write,
    )


# Anthropic accepts at most four `cache_control` breakpoints per request, and
# they are read as a prefix: everything before the last one is a candidate for
# the cache. Three are placed, which is the number that matches how this loop
# actually is shaped.
MAX_BREAKPOINTS = 4


def _place_breakpoints(payload: dict[str, Any]) -> None:
    """Mark the parts of the request that repeat, so they are paid for once.

    Order matters on the wire: the cacheable prefix runs tools -> system ->
    messages, so a breakpoint in the messages can only ever be a hit if the
    tools and the system prompt are also inside the cached region. Marking only
    the last message on a request whose tool schemas changed every step would
    therefore cache nothing at all.

    Three breakpoints:

    * **the last tool** -- twenty tool schemas are a fixed 1600 tokens sitting
      in front of every request of the run;
    * **the system prompt** -- fixed for the whole run, by definition;
    * **the end of the last message** -- the sliding part. Step N+1 sends
      everything step N sent plus one turn, so this breakpoint turns the
      previous step's whole history into a cache read, and only the new turn is
      billed at full price.
    """
    tools = payload.get("tools") or []
    if tools:
        tools[-1]["cache_control"] = {"type": "ephemeral"}

    system = payload.get("system")
    if isinstance(system, str) and system:
        payload["system"] = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]

    turns = payload.get("messages") or []
    if turns:
        content = turns[-1].get("content")
        if isinstance(content, str):
            if content:
                turns[-1]["content"] = [
                    {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
                ]
        elif isinstance(content, list) and content:
            # A tool result turn or a block list: mark the last block, which is
            # the end of the prefix a later request will re-send.
            content[-1]["cache_control"] = {"type": "ephemeral"}
