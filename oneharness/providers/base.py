"""Provider interface: everything the agent loop needs from a backend."""

from __future__ import annotations

import abc
from collections.abc import Iterator, Sequence
from typing import Any

from oneharness.types import Completion, Message, StreamEvent


class Provider(abc.ABC):
    """Turns a message list plus a tool schema into one Completion."""

    name: str = "provider"
    # Whether prompt caching needs to be asked for on the wire.
    #
    #   "automatic"  the provider caches repeated prefixes on its own (OpenAI,
    #                Gemini, DeepSeek and the rest of the OpenAI-shaped family);
    #                there is nothing for a harness to do but count what came back
    #   "explicit"   nothing is cached unless a breakpoint is placed (Anthropic,
    #                Bedrock); a tool loop that resends its whole history every
    #                step pays for that history every step otherwise
    #
    # This distinction is why `--cache` is not a switch that every provider
    # honours: asking an implicit-cache provider to place breakpoints would be
    # inventing a wire field it does not have.
    cache_mode: str = "automatic"

    def __init__(
        self, model: str, *, temperature: float = 0.0, max_tokens: int | None = None
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abc.abstractmethod
    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> Completion:
        """Send one request and normalise the reply into a Completion."""

    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> Iterator[StreamEvent]:
        """Stream a reply as text deltas plus one final Completion.

        The default implementation cheats: it does the blocking request and
        yields the whole answer in one delta, so `--stream` still works on a
        provider that has not implemented real streaming.
        """
        completion = self.complete(messages, tools)
        if completion.text:
            yield StreamEvent(kind="text", text=completion.text)
        yield StreamEvent(kind="done", completion=completion)
