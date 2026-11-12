"""The agent loop: prompt in, tool calls executed, final answer out."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from oneharness.errors import ProviderError
from oneharness.providers.base import Provider
from oneharness.tools import Registry
from oneharness.types import Completion, Message, ToolCall, ToolResult, Usage

Observer = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class Step:
    """One assistant turn plus the tool results it triggered."""

    completion: Completion
    tool_results: list[ToolResult] = field(default_factory=list)
    seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.completion.text,
            "tool_calls": [call.as_dict() for call in self.completion.tool_calls],
            "tool_results": [result.as_dict() for result in self.tool_results],
            "usage": self.completion.usage.as_dict(),
            "seconds": round(self.seconds, 3),
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
        max_parallel: int = 1,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if max_parallel < 1:
            raise ValueError("max_parallel must be at least 1")
        self.provider = provider
        self.tools = tools if isinstance(tools, Registry) else Registry(tools or ())
        self.system = system
        self.max_steps = max_steps
        self.on_event = on_event
        self.max_parallel = max_parallel

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
            started = time.monotonic()
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

            step.tool_results.extend(self._run_tools(completion.tool_calls, step_index=index + 1))
            for result in step.tool_results:
                messages.append(result.as_message())
            step.seconds = time.monotonic() - started
            self._emit(
                {
                    "type": "step_end",
                    "step": index + 1,
                    "seconds": round(step.seconds, 3),
                    "tool_calls": len(completion.tool_calls),
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

    def _run_tools(self, calls: Sequence[ToolCall], *, step_index: int) -> list[ToolResult]:
        """Execute one turn's tool calls, in order or side by side.

        A model that asks for six independent reads should not wait for six
        round trips. Parallel execution is opt-in (`max_parallel`) because
        tools are not assumed to be thread safe: a shell tool and a file tool
        racing on the same path is a real hazard, and the caller knows better.
        """
        for call in calls:
            self._emit({"type": "tool_start", "name": call.name, "arguments": call.arguments})

        if self.max_parallel == 1 or len(calls) == 1:
            results = []
            for call in calls:
                result = self.tools.execute(call)
                results.append(result)
                self._emit_tool_end(result)
            return results

        workers = min(self.max_parallel, len(calls))
        finished: dict[int, ToolResult] = {}
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="oneharness-tool") as pool:
            futures = {
                pool.submit(self.tools.execute, call): index for index, call in enumerate(calls)
            }
            # Events fire as things actually finish (that is what a watcher wants
            # to see), while the returned results keep the model's own order, so
            # the message list stays predictable.
            for future in as_completed(futures):
                result = future.result()
                finished[futures[future]] = result
                self._emit_tool_end(result)
        return [finished[index] for index in range(len(calls))]

    def _emit_tool_end(self, result: ToolResult) -> None:
        self._emit(
            {
                "type": "tool_end",
                "name": result.name,
                "is_error": result.is_error,
                "content": result.content,
            }
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
