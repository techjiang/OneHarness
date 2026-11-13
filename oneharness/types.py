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
    """What one request cost, normalised across five vocabularies.

    `input_tokens` is the whole prompt side, always. `cached_tokens` is how much
    of it the provider served from its prompt cache -- a subset, never an
    addition. That normalisation is not cosmetic, because the providers do not
    agree on it:

    * OpenAI and the OpenAI-shaped services put the cache inside the prompt
      count (`prompt_tokens_details.cached_tokens` is a slice of
      `prompt_tokens`; DeepSeek reports `prompt_cache_hit_tokens` the same way);
    * Anthropic's `input_tokens` counts only the *uncached* input, with
      `cache_read_input_tokens` and `cache_creation_input_tokens` alongside it;
    * Gemini's `cachedContentTokenCount` is a slice of `promptTokenCount`;
    * Bedrock reports `cacheReadInputTokens` / `cacheWriteInputTokens` next to
      `inputTokens`, like Anthropic.

    Adding the two together naively would double-count on four providers and
    under-count on the fifth. So every adapter reports the prompt total here and
    the cached share separately, and nothing above the adapter has to know which
    wire format it came from.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    # Tokens written *into* the cache. Only the explicit-breakpoint providers
    # (Anthropic, Bedrock) have this concept; the implicit-cache ones charge
    # nothing for the write, so 0 there means "not applicable", not "measured 0".
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def uncached_tokens(self) -> int:
        """The part of the prompt that was billed at full price."""
        return max(0, self.input_tokens - self.cached_tokens)

    @property
    def cache_hit_rate(self) -> float:
        return self.cached_tokens / self.input_tokens if self.input_tokens else 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )

    def __sub__(self, other: Usage) -> Usage:
        """Used to isolate what a sub agent spent during one run."""
        return Usage(
            input_tokens=self.input_tokens - other.input_tokens,
            output_tokens=self.output_tokens - other.output_tokens,
            cached_tokens=self.cached_tokens - other.cached_tokens,
            cache_write_tokens=self.cache_write_tokens - other.cache_write_tokens,
        )

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }
        # Only when there is something to say: a `cache` block of zeros on a
        # provider that has no cache reads like "caching is broken", and the
        # absence of the key reads like "nothing was cached", which is true.
        if self.cached_tokens or self.cache_write_tokens:
            data["cache"] = {
                "read_tokens": self.cached_tokens,
                "write_tokens": self.cache_write_tokens,
                "hit_rate": round(self.cache_hit_rate, 3),
            }
        return data


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
    # Set when the provider says it stopped because it ran out of output
    # budget rather than because it was finished. This is a complete response,
    # not a broken one -- which is exactly why it used to be the quietest
    # failure in the loop: `finish_reason: length` sets no `truncated`, the
    # answer parses, and the run reports success with half a sentence.
    cut_off: str | None = None

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


# The five providers do not agree on the word for "I ran out of output", and
# the word is the whole signal: OpenAI-shaped services say `length` (DeepSeek
# keeps it), Anthropic says `max_tokens`, Gemini says `MAX_TOKENS`, and Bedrock
# -- being the odd one out -- says `max_tokens` too. Matched case-insensitively
# so one set covers both spellings.
CUT_OFF_REASONS = frozenset({"length", "max_tokens"})

# The tool call that was being written when the output ran out. Arguments are
# JSON, so the half that arrived does not parse. Marking it here rather than
# dropping it keeps the declaration (the model asked for this) and states the
# defect (the arguments never finished), which is a true pair of facts.
INCOMPLETE_ARGUMENTS = "tool arguments were cut off by the output limit"


def stops_for_length(stop_reason: str | None) -> bool:
    """True when a finish reason means "out of budget", not "finished"."""
    if not stop_reason:
        return False
    return stop_reason.strip().lower().replace("-", "_") in CUT_OFF_REASONS


def cut_off_reason(provider: str, stop_reason: str | None) -> str | None:
    """The sentence for a turn the provider ended at its output limit."""
    if not stops_for_length(stop_reason):
        return None
    return (
        f"{provider} stopped at the output token limit (stop_reason={stop_reason}): "
        "this response is complete as a response and unfinished as an answer"
    )


def mark_incomplete_arguments(calls: list[ToolCall]) -> list[ToolCall]:
    """Flag the last call when its argument JSON does not parse.

    Only the call being written can be half-done: the ones before it were
    closed before this one started. The test is whether the JSON *parses*, not
    whether it looks short: a cut can land right after the closing quote of a
    value, and `{"path": "app.conf"` is missing only one brace.
    """
    if not calls:
        return calls
    last = calls[-1]
    text = (last.raw_arguments or "").strip()
    if not last.arguments_error and text and _parses(text):
        return calls
    last.arguments_error = INCOMPLETE_ARGUMENTS
    return calls


def _parses(text: str) -> bool:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(value, dict)


def dumps(value: Any) -> str:
    """Compact JSON used when handing tool results back to a model."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
