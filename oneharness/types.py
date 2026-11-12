"""Data types exchanged between providers, tools and the agent loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]

# Why a completion is incomplete. Both live here rather than in `agent` because
# the providers have to set them: only the adapter knows whether the stream it
# was reading ever reached its terminal marker.
INTERRUPTED_BY_USER = "interrupted by user (Ctrl-C)"
STREAM_ENDED_EARLY = "stream ended without a final completion"


@dataclass(slots=True)
class ToolCall:
    """A model request to run one tool.

    `raw_arguments` keeps the provider's original JSON text so a malformed
    payload can be echoed back to the model instead of being silently lost.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str | None = None
    arguments_error: str | None = None

    @classmethod
    def from_raw(cls, *, call_id: str, name: str, arguments: Any) -> ToolCall:
        """Build a call from a provider payload, tolerating bad or partial JSON."""
        if isinstance(arguments, dict):
            return cls(id=call_id, name=name, arguments=arguments, raw_arguments=dumps(arguments))
        text = "" if arguments is None else str(arguments)
        if not text.strip():
            return cls(id=call_id, name=name, raw_arguments=text)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return cls(
                id=call_id,
                name=name,
                raw_arguments=text,
                arguments_error=f"arguments were not valid JSON: {exc}",
            )
        if not isinstance(parsed, dict):
            return cls(
                id=call_id,
                name=name,
                raw_arguments=text,
                arguments_error=f"arguments must be a JSON object, got {type(parsed).__name__}",
            )
        return cls(id=call_id, name=name, arguments=parsed, raw_arguments=text)

    def arguments_text(self) -> str:
        """The JSON text to send back to a provider."""
        if self.raw_arguments is not None:
            return self.raw_arguments
        return dumps(self.arguments)

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"id": self.id, "name": self.name, "arguments": self.arguments}
        if self.arguments_error:
            data["arguments_error"] = self.arguments_error
        return data


@dataclass(slots=True)
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    # A failed tool call is still a message to the model, but Anthropic and
    # Bedrock both distinguish "here is output" from "this went wrong" on the
    # wire. OpenAI does not, so it simply ignores this flag.
    is_error: bool = False

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str = "", tool_calls: list[ToolCall] | None = None) -> Message:
        return cls(role="assistant", content=content, tool_calls=list(tool_calls or ()))

    @classmethod
    def tool(
        cls,
        tool_call_id: str,
        content: str,
        name: str | None = None,
        is_error: bool = False,
    ) -> Message:
        return cls(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            name=name,
            is_error=is_error,
        )

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [call.as_dict() for call in self.tool_calls]
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.name:
            data["name"] = self.name
        return data


@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )

    def __sub__(self, other: Usage) -> Usage:
        """Used to isolate what a sub agent spent during one run."""
        return Usage(
            input_tokens=self.input_tokens - other.input_tokens,
            output_tokens=self.output_tokens - other.output_tokens,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(slots=True)
class Completion:
    """One provider response: the assistant message plus bookkeeping."""

    message: Message
    usage: Usage = field(default_factory=Usage)
    model: str | None = None
    stop_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    # Set when the stream ended before the provider said it was done. The text
    # is whatever had already been produced, which is worth more than nothing.
    truncated: str | None = None

    @property
    def text(self) -> str:
        return self.message.content

    @property
    def tool_calls(self) -> list[ToolCall]:
        return self.message.tool_calls


@dataclass(slots=True)
class StreamEvent:
    """One step of a streamed response.

    Providers yield `text` events as tokens arrive and exactly one `done`
    event carrying the assembled Completion, so the agent loop has a single
    place to pick the finished turn up from.

    `tool` events are the third kind: a tool call as it is being *assembled*.
    The id and name arrive once, then each `fragment` is the next slice of the
    argument JSON. Nobody has to wait for the closing brace to know a tool is
    about to run -- which is what a person watching a long `write_file` wants
    to see, and what the `done` event alone cannot tell them.
    """

    kind: Literal["text", "tool", "done"]
    text: str = ""
    completion: Completion | None = None
    call: ToolCall | None = None
    fragment: str = ""
    index: int = 0


@dataclass(slots=True)
class ToolResult:
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False
    # How long this one call took, measured around the tool function itself.
    # Calls in one turn can overlap, so the sum of these is not the wall clock
    # of the turn -- keeping both numbers is what makes `--parallel` visible.
    seconds: float = 0.0

    def as_message(self) -> Message:
        return Message.tool(self.tool_call_id, self.content, self.name, self.is_error)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": self.content,
            "is_error": self.is_error,
            "seconds": round(self.seconds, 3),
        }


def dumps(value: Any) -> str:
    """Compact JSON used when handing tool results back to a model."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
