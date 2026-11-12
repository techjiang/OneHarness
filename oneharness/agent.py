"""The agent loop: prompt in, tool calls executed, final answer out."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from oneharness.errors import ProviderError
from oneharness.providers.base import Provider
from oneharness.tools import Registry
from oneharness.types import Completion, Message, ToolResult, Usage

Observer = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class Step:
    """One assistant turn plus the tool results it triggered."""

    completion: Completion
    tool_results: list[ToolResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.completion.text,
            "tool_calls": [call.as_dict() for call in self.completion.tool_calls],
            "tool_results": [result.as_dict() for result in self.tool_results],
            "usage": self.completion.usage.as_dict(),
        }


@dataclass(slots=True)
class AgentResult:
    """The outcome of a full run."""

    text: str
    messages: list[Message]
    steps: list[Step]
    usage: Usage
    stopped_reason: str  # "done" | "max_steps"
    model: str | None = None

    @property
    def ok(self) -> bool:
        return self.stopped_reason == "done"

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "stopped_reason": self.stopped_reason,
            "model": self.model,
            "usage": self.usage.as_dict(),
            "steps": [step.as_dict() for step in self.steps],
        }


class Agent:
    """Runs a provider/tool loop until the model answers or the budget runs out.

    Every failure that the model could recover from (bad arguments, a tool that
    raised, an unknown tool name) is fed back as a tool result instead of being
    raised, so a single mistake never ends the run.
    """

    def __init__(
        self,
        provider: Provider,
        tools: Registry | Sequence[Any] | None = None,
        *,
        system: str | None = None,
        max_steps: int = 8,
        on_event: Observer | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.provider = provider
        self.tools = tools if isinstance(tools, Registry) else Registry(tools or ())
        self.system = system
        self.max_steps = max_steps
        self.on_event = on_event

    def run(
        self,
        prompt: str,
        *,
        history: Sequence[Message] | None = None,
        stream: bool = False,
    ) -> AgentResult:
        """Drive the loop to completion; `stream=True` also emits text deltas.

        Streaming does not change the result shape: token events go out through
        `on_event` as they arrive, and the loop still works on a finished
        Completion, so a mid-run tool call behaves exactly as before.
        """
        messages = self._seed(prompt, history)
        specs = self.tools.specs()
        steps: list[Step] = []
        total = Usage()
        stopped = "max_steps"
        final_text = ""

        for index in range(self.max_steps):
            self._emit({"type": "step_start", "step": index + 1})
            completion = self._complete(messages, specs, stream=stream, step=index + 1)

            total = total + completion.usage
            messages.append(completion.message)
            step = Step(completion=completion)
            steps.append(step)
            self._emit(
                {
                    "type": "assistant",
                    "step": index + 1,
                    "text": completion.text,
                    "tool_calls": [call.as_dict() for call in completion.tool_calls],
                }
            )

            if completion.text:
                final_text = completion.text

            if not completion.tool_calls:
                stopped = "done"
                break

            for call in completion.tool_calls:
                self._emit({"type": "tool_start", "name": call.name, "arguments": call.arguments})
                result = self.tools.execute(call)
                step.tool_results.append(result)
                messages.append(result.as_message())
                self._emit(
                    {
                        "type": "tool_end",
                        "name": call.name,
                        "is_error": result.is_error,
                        "content": result.content,
                    }
                )
        else:
            # Budget exhausted while the model still wanted tools.
            stopped = "max_steps"
            self._emit({"type": "max_steps", "steps": self.max_steps})

        return AgentResult(
            text=final_text,
            messages=messages,
            steps=steps,
            usage=total,
            stopped_reason=stopped,
            model=getattr(self.provider, "model", None),
        )

    def _seed(self, prompt: str, history: Sequence[Message] | None) -> list[Message]:
        messages: list[Message] = []
        if self.system:
            messages.append(Message.system(self.system))
        messages.extend(history or ())
        if prompt:
            messages.append(Message.user(prompt))
        return messages

    def _complete(
        self,
        messages: list[Message],
        specs: Sequence[dict[str, Any]],
        *,
        stream: bool,
        step: int,
    ) -> Completion:
        if not stream:
            return self.provider.complete(messages, specs)

        text: list[str] = []
        finished: Completion | None = None
        for event in self.provider.stream(messages, specs):
            if event.kind == "text":
                text.append(event.text)
                self._emit({"type": "text_delta", "step": step, "text": event.text})
            elif event.completion is not None:
                finished = event.completion
        if finished is None:
            raise ProviderError("stream ended without a final completion")
        if not finished.message.content and text:
            finished.message = Message.assistant("".join(text), finished.message.tool_calls)
        return finished

    def _emit(self, event: dict[str, Any]) -> None:
        if self.on_event is not None:
            self.on_event(event)
