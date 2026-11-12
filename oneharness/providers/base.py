"""Provider interface: everything the agent loop needs from a backend."""

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any

from oneharness.types import Completion, Message


class Provider(abc.ABC):
    """Turns a message list plus a tool schema into one Completion."""

    name: str = "provider"

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
