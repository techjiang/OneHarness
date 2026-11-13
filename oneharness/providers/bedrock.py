"""AWS Bedrock Converse API.

Bedrock is the one provider where the interesting problem is not the shape of
the JSON. Two things are genuinely different from writing another chat-completions
adapter:

* every request is signed, and the signature covers the exact body bytes;
* the streaming response is a binary event-stream, not SSE lines.

Both of those live in their own modules (`sigv4.py`, `eventstream.py`) so they
can be tested against the specs instead of against Bedrock. This file is only
the Converse translation: messages with content blocks, `toolUse` / `toolResult`
pairs, and the `messages` role alternation that Converse insists on.
"""

from __future__ import annotations

import urllib.parse
from collections.abc import Iterator, Sequence
from typing import Any

from oneharness.errors import ProviderError
from oneharness.eventstream import decode_frames
from oneharness.providers.base import Provider
from oneharness.sigv4 import Credentials, sign_request
from oneharness.transport import Signer, post_chunks, post_json
from oneharness.types import (
    STREAM_ENDED_EARLY,
    Completion,
    Message,
    StreamEvent,
    ToolCall,
    Usage,
)

SERVICE = "bedrock"
DEFAULT_REGION = "us-east-1"


class BedrockProvider(Provider):
    name = "bedrock"
    # Nothing is cached unless a breakpoint says where, so this is the one place
    # `--cache` changes what goes on the wire rather than only what is reported.
    cache_mode = "explicit"

    def __init__(
        self,
        model: str,
        *,
        credentials: Credentials,
        region: str = DEFAULT_REGION,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float = 120.0,
        cache: bool = False,
    ) -> None:
        super().__init__(model, temperature=temperature, max_tokens=max_tokens)
        self.credentials = credentials
        self.region = region
        self.base_url = (base_url or f"https://bedrock-runtime.{region}.amazonaws.com").rstrip("/")
        self.timeout = timeout
        # Same models as Anthropic, same rule: a breakpoint (`cachePoint`) has to
        # be placed or the prefix is billed again on every step.
        self.cache = cache

    # -- requests --------------------------------------------------------

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> Completion:
        data = post_json(
            self._endpoint("converse"),
            self._payload(messages, tools),
            timeout=self.timeout,
            sign=self._signer(),
        )
        return self._decode(data)

    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> Iterator[StreamEvent]:
        text: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        stop_reason: str | None = None
        usage = Usage()
        model = self.model

        chunks = post_chunks(
            self._endpoint("converse-stream"),
            self._payload(messages, tools),
            accept="application/vnd.amazon.eventstream",
            timeout=self.timeout,
            sign=self._signer(),
        )
        for frame in decode_frames(chunks):
            if frame.message_type == "exception":
                raise ProviderError(
                    f"bedrock {frame.event}", body=frame.payload.decode("utf-8", "replace")
                )
            event = frame.event
            body = frame.json()
            if event == "contentBlockStart":
                # The tool name arrives once, before its arguments: the id has
                # to be remembered here because the deltas that follow carry
                # only fragments of JSON.
                start = (body.get("start") or {}).get("toolUse") or {}
                index = int(body.get("contentBlockIndex", 0))
                calls[index] = {
                    "id": start.get("toolUseId") or f"tool_{index}",
                    "name": start.get("name") or "",
                    "arguments": "",
                }
                yield StreamEvent(
                    kind="tool",
                    index=index,
                    call=ToolCall(id=calls[index]["id"], name=calls[index]["name"]),
                )
            elif event == "contentBlockDelta":
                delta = body.get("delta") or {}
                if "text" in delta:
                    piece = delta.get("text") or ""
                    if piece:
                        text.append(piece)
                        yield StreamEvent(kind="text", text=piece)
                elif "toolUse" in delta:
                    index = int(body.get("contentBlockIndex", 0))
                    slot = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                    fragment = (delta.get("toolUse") or {}).get("input") or ""
                    slot["arguments"] += fragment
                    if fragment:
                        yield StreamEvent(
                            kind="tool",
                            index=index,
                            fragment=fragment,
                            call=ToolCall(id=slot["id"] or f"tool_{index}", name=slot["name"]),
                        )
            elif event == "messageStop":
                stop_reason = body.get("stopReason") or stop_reason
            elif event == "metadata":
                usage = self._usage(body.get("usage")) or usage
                model = body.get("model") or model

        assembled = [
            ToolCall.from_raw(
                call_id=slot["id"] or f"tool_{index}",
                name=slot["name"],
                arguments=slot["arguments"] or "{}",
            )
            for index, slot in sorted(calls.items())
        ]
        yield StreamEvent(
            kind="done",
            completion=self._finish(
                Completion(
                    message=Message.assistant("".join(text), assembled),
                    usage=usage,
                    model=model,
                    raw={"streamed": True},
                    # `messageStop` is the last frame of a complete turn;
                    # without it the event stream ended mid-sentence.
                    # `stopReason: max_tokens` is a frame that did arrive -- the
                    # turn is complete and the answer is not.
                    truncated=None if stop_reason else STREAM_ENDED_EARLY,
                ),
                stop_reason,
            ),
        )

    def _endpoint(self, operation: str) -> str:
        # The model id goes in the path and must be percent-encoded: every
        # inference profile is `arn:aws:bedrock:...` once you use cross-region,
        # and a raw colon in a signed path is a signature mismatch.
        model = urllib.parse.quote(self.model, safe="")
        return f"{self.base_url}/model/{model}/{operation}"

    def _signer(self) -> Signer:
        def sign(url: str, body: bytes, headers: dict[str, str]) -> dict[str, str]:
            prepared = {key.lower(): value for key, value in headers.items()}
            prepared["host"] = urllib.parse.urlsplit(url).netloc
            return sign_request(
                method="POST",
                url=url,
                body=body,
                headers=prepared,
                credentials=self.credentials,
                service=SERVICE,
                region=self.region,
            )

        return sign

    def _payload(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        system, turns = self._split(messages)
        payload: dict[str, Any] = {
            "messages": turns,
            "inferenceConfig": {"temperature": self.temperature},
        }
        if self.max_tokens is not None:
            payload["inferenceConfig"]["maxTokens"] = self.max_tokens
        if system:
            payload["system"] = [{"text": system}]
        if tools:
            payload["toolConfig"] = {
                "tools": [
                    {
                        "toolSpec": {
                            "name": spec["name"],
                            "description": spec.get("description", ""),
                            # Converse wants the schema nested under `json`,
                            # unlike everyone else who takes it inline.
                            "inputSchema": {"json": spec.get("parameters") or {"type": "object"}},
                        }
                    }
                    for spec in tools
                ]
            }
        if self.cache:
            _place_breakpoints(payload)
        return payload

    def _split(self, messages: Sequence[Message]) -> tuple[str, list[dict[str, Any]]]:
        """Fold messages into Converse turns.

        Converse rejects two consecutive messages with the same role, so tool
        results have to be collected into a single user turn, and a system
        prompt is a list of text blocks rather than a message.
        """
        system_parts: list[str] = []
        turns: list[dict[str, Any]] = []

        def push(role: str, blocks: list[dict[str, Any]]) -> None:
            if not blocks:
                return
            if turns and turns[-1]["role"] == role:
                turns[-1]["content"].extend(blocks)
            else:
                turns.append({"role": role, "content": blocks})

        for message in messages:
            if message.role == "system":
                if message.content:
                    system_parts.append(message.content)
            elif message.role == "tool":
                block: dict[str, Any] = {
                    "toolResult": {
                        "toolUseId": message.tool_call_id,
                        "content": [{"text": message.content}],
                    }
                }
                if message.is_error:
                    block["toolResult"]["status"] = "error"
                push("user", [block])
            elif message.role == "assistant":
                blocks: list[dict[str, Any]] = []
                if message.content:
                    blocks.append({"text": message.content})
                blocks.extend(
                    {
                        "toolUse": {
                            "toolUseId": call.id,
                            "name": call.name,
                            "input": call.arguments,
                        }
                    }
                    for call in message.tool_calls
                )
                # Converse refuses an empty assistant turn, and a model that
                # answered with nothing but a tool call is normal here.
                push("assistant", blocks or [{"text": " "}])
            else:
                push("user", [{"text": message.content}])

        return "\n\n".join(system_parts), turns

    def _decode(self, data: dict[str, Any]) -> Completion:
        output = (data.get("output") or {}).get("message")
        if not isinstance(output, dict):
            raise ProviderError("response contained no output message", body=str(data)[:500])

        texts: list[str] = []
        calls: list[ToolCall] = []
        for block in output.get("content") or ():
            if "text" in block:
                texts.append(block.get("text") or "")
            elif "toolUse" in block:
                use = block["toolUse"] or {}
                calls.append(
                    ToolCall.from_raw(
                        call_id=use.get("toolUseId") or f"tool_{len(calls)}",
                        name=use.get("name") or "",
                        arguments=use.get("input") or {},
                    )
                )

        return self._finish(
            Completion(
                message=Message.assistant("".join(texts), calls),
                usage=self._usage(data.get("usage")) or Usage(),
                model=data.get("model") or self.model,
                raw=data,
            ),
            data.get("stopReason"),
        )

    @staticmethod
    def _usage(raw: Any) -> Usage | None:
        """Converse usage, with the cache fields the Converse API added later.

        Like Anthropic (same models, same cache), `inputTokens` excludes what
        was read from the cache; `cacheReadInputTokens` and
        `cacheWriteInputTokens` sit beside it. Old Converse responses simply do
        not carry them, which is why every field is read with `or 0` rather than
        indexed.
        """
        if not isinstance(raw, dict):
            return None
        read = int(raw.get("cacheReadInputTokens") or 0)
        write = int(raw.get("cacheWriteInputTokens") or 0)
        return Usage(
            input_tokens=int(raw.get("inputTokens") or 0) + read + write,
            output_tokens=int(raw.get("outputTokens") or 0),
            cached_tokens=read,
            cache_write_tokens=write,
        )


# Converse spells breakpoints differently from the Messages API -- a
# `{"cachePoint": {"type": "default"}}` block inside the same list the content
# lives in -- but the placement logic is identical, and so is the reason for it.
def _place_breakpoints(payload: dict[str, Any]) -> None:
    """Mark the repeating prefix: tools, then system, then the newest turn.

    Converse walks the request in that order, so a breakpoint that is not on the
    last tool cannot cover the tools at all. Two blocks are also *not* enough:
    the system list keeps its own breakpoint, and the messages list gets one at
    the very end so the next step's history is a read.
    """
    point = {"cachePoint": {"type": "default"}}
    tool_config = payload.get("toolConfig")
    if isinstance(tool_config, dict) and tool_config.get("tools"):
        tool_config["tools"].append(point)

    system = payload.get("system")
    if isinstance(system, list) and system:
        system.append(point)

    turns = payload.get("messages") or []
    if turns and turns[-1].get("content"):
        turns[-1]["content"].append(point)
