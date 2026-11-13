"""The agent loop: prompt in, tool calls executed, final answer out."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from oneharness import context as context_module
from oneharness.errors import ProviderError
from oneharness.providers.base import Provider
from oneharness.tools import Registry
from oneharness.types import (
    INTERRUPTED_BY_USER,
    STREAM_ENDED_EARLY,
    Completion,
    Message,
    StreamEvent,
    ToolCall,
    ToolResult,
    Usage,
)

# The one interruption that is a human decision rather than a failure; the CLI
# maps it back to 130 so a shell still sees "the user stopped this".
Observer = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class Step:
    """One assistant turn plus the tool results it triggered.

    A step is two waits in a row, and "2.4s" alone does not say which one hurt.
    Waiting for the model and waiting for a tool have opposite fixes -- a
    smaller context or a faster model on one side, a narrower tool or more
    parallelism on the other -- so both are measured separately, and `seconds`
    stays the honest wall clock that neither of them explains on its own.
    """

    completion: Completion
    tool_results: list[ToolResult] = field(default_factory=list)
    seconds: float = 0.0
    model_seconds: float = 0.0
    tool_seconds: float = 0.0
    # How many requests this step actually sent. More than one means at least
    # one was refused for size and retried -- time that a "how long did the
    # model take" number would otherwise hide.
    model_requests: int = 1
    # How long the request ran before its first byte of anything came back.
    # Only streaming can observe it; a blocking request has no "first".
    first_token_seconds: float | None = None

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "text": self.completion.text,
            "tool_calls": [call.as_dict() for call in self.completion.tool_calls],
            "tool_results": [result.as_dict() for result in self.tool_results],
            "usage": self.completion.usage.as_dict(),
            "seconds": round(self.seconds, 3),
            "model_seconds": round(self.model_seconds, 3),
            "tool_seconds": round(self.tool_seconds, 3),
            "model_requests": self.model_requests,
        }
        if self.first_token_seconds is not None:
            data["first_token_seconds"] = round(self.first_token_seconds, 3)
        return data


@dataclass(slots=True)
class ModelCall:
    """One logical request to the provider, with what it cost in wall clock.

    A single step can issue more than one request (an overflow is refused and
    retried under a smaller budget), and all of them cost real time, so the
    timing lives here rather than being sampled around the step.
    """

    completion: Completion
    trim: context_module.Trim | None = None
    seconds: float = 0.0
    requests: int = 0
    first_token_seconds: float | None = None


@dataclass(slots=True)
class AgentResult:
    """The outcome of a full run."""

    text: str
    messages: list[Message]
    steps: list[Step]
    usage: Usage
    stopped_reason: str  # "done" | "max_steps" | "interrupted"
    model: str | None = None
    # What the budget cut on the way to the last request. Present only when
    # something actually had to go, so a run that fitted looks as it always did.
    trim: dict[str, int] | None = None
    # Why the stream stopped early, when it did. The partial answer in `text`
    # is what the model had already said out loud; throwing it away because the
    # connection died would discard the only thing that arrived.
    interrupted: str | None = None
    # Wall clock of the whole run, measured by the loop itself.
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.stopped_reason == "done"

    @property
    def timing(self) -> dict[str, Any]:
        """Where the run's wall clock went, and which step owned the most of it.

        `model_seconds + tool_seconds` does not have to equal `total_seconds`:
        both halves are measured around the loop's own waits, so the difference
        (`unaccounted_seconds`) is everything it did not wrap in a timer --
        building messages, reading history, writing to a terminal. Publishing
        the remainder separately is the point: a breakdown whose parts always
        add up exactly is a breakdown that invents the numbers it cannot see.
        """
        model = sum(step.model_seconds for step in self.steps)
        tools = sum(step.tool_seconds for step in self.steps)
        # By position, not by identity: two steps of a scripted run can compare
        # equal, and `list.index` would then report the first of them.
        slowest = 0
        for number, step in enumerate(self.steps, start=1):
            if step.seconds > self.steps[slowest].seconds:
                slowest = number - 1
        firsts = [
            step.first_token_seconds for step in self.steps if step.first_token_seconds is not None
        ]
        return {
            "total_seconds": round(self.seconds, 3),
            "model_seconds": round(model, 3),
            "tool_seconds": round(tools, 3),
            "tool_call_seconds": round(
                sum(result.seconds for step in self.steps for result in step.tool_results), 3
            ),
            "unaccounted_seconds": round(max(0.0, self.seconds - model - tools), 3),
            "first_token_seconds": round(sum(firsts) / len(firsts), 3) if firsts else None,
            "steps": len(self.steps),
            "model_requests": sum(step.model_requests for step in self.steps),
            "tool_calls": sum(len(step.tool_results) for step in self.steps),
            "slowest_step": (
                None
                if not self.steps
                else {
                    "step": slowest + 1,
                    "seconds": round(self.steps[slowest].seconds, 3),
                }
            ),
        }

    @property
    def interrupted_by_user(self) -> bool:
        """True when a human stopped the run, which is not a failure."""
        return self.interrupted == INTERRUPTED_BY_USER

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "text": self.text,
            "stopped_reason": self.stopped_reason,
            "model": self.model,
            "usage": self.usage.as_dict(),
            "steps": [step.as_dict() for step in self.steps],
        }
        # Absent, not null, when nothing was cut: an empty object is easier to
        # mistake for a broken run than a missing key.
        if self.trim is not None:
            data["trim"] = self.trim
        if self.interrupted:
            data["interrupted"] = self.interrupted
        data["timing"] = self.timing
        return data


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
        context_budget: int = 0,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if max_parallel < 1:
            raise ValueError("max_parallel must be at least 1")
        if context_budget < 0:
            raise ValueError("context_budget must be 0 (off) or at least 1")
        self.provider = provider
        self.tools = tools if isinstance(tools, Registry) else Registry(tools or ())
        self.system = system
        self.max_steps = max_steps
        self.on_event = on_event
        self.max_parallel = max_parallel
        # 0 means unlimited: a run that fitted before keeps behaving the same.
        self.context_budget = context_budget

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
        trim: context_module.Trim | None = None
        interrupted: str | None = None
        run_started = time.monotonic()

        for index in range(self.max_steps):
            started = time.monotonic()
            self._emit({"type": "step_start", "step": index + 1})
            turn = self._complete_budgeted(messages, specs, stream=stream, step=index + 1)
            completion = turn.completion
            if turn.trim is not None:
                trim = turn.trim

            total = total + completion.usage
            messages.append(completion.message)
            step = Step(
                completion=completion,
                model_seconds=turn.seconds,
                model_requests=turn.requests,
                first_token_seconds=turn.first_token_seconds,
            )
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

            if completion.truncated:
                # A half-answer is not a turn: it has no tool calls to run and
                # no promise of being finished, so the loop stops here and says
                # why rather than pretending the model chose to end.
                interrupted = completion.truncated
                stopped = "interrupted"
                step.seconds = time.monotonic() - started
                # No `step_end` here: the `interrupted` event is what closes a
                # step that never finished, and emitting both would tell a
                # watcher the step ended normally.
                self._emit(
                    {
                        "type": "interrupted",
                        "step": index + 1,
                        "reason": interrupted,
                        "chars": len(completion.text),
                    }
                )
                break

            if not completion.tool_calls:
                # The step that answers is the one people want the duration of,
                # and it used to be the only one still reporting 0.0 -- the
                # number was set after the tool loop, which a final step never
                # reaches.
                step.seconds = time.monotonic() - started
                self._emit_step_end(index + 1, step, tool_calls=0)
                stopped = "done"
                break

            tools_started = time.monotonic()
            step.tool_results.extend(self._run_tools(completion.tool_calls, step_index=index + 1))
            # The wall clock of the tool phase, not the sum of the calls in it:
            # with `--parallel` those two numbers differ, and the difference is
            # exactly the parallelism that was bought.
            step.tool_seconds = time.monotonic() - tools_started
            for result in step.tool_results:
                messages.append(result.as_message())
            step.seconds = time.monotonic() - started
            self._emit_step_end(index + 1, step, tool_calls=len(completion.tool_calls))
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
            trim=trim.as_dict() if trim is not None and trim.changed else None,
            interrupted=interrupted,
            seconds=time.monotonic() - run_started,
        )

    def _emit_step_end(self, number: int, step: Step, *, tool_calls: int) -> None:
        """Close a step with both halves of its wait, not just the total.

        The split is the whole point: a step that spent 3s in the model and one
        that spent 3s in a shell look identical in a bare duration and have
        completely different fixes.
        """
        self._emit(
            {
                "type": "step_end",
                "step": number,
                "seconds": round(step.seconds, 3),
                "model_seconds": round(step.model_seconds, 3),
                "tool_seconds": round(step.tool_seconds, 3),
                "tool_calls": tool_calls,
            }
        )

    def _run_tools(self, calls: Sequence[ToolCall], *, step_index: int) -> list[ToolResult]:
        """Execute one turn's tool calls, in order or side by side.

        A model that asks for six independent reads should not wait for six
        round trips. Parallel execution is opt-in (`max_parallel`) because
        tools are not assumed to be thread safe: a shell tool and a file tool
        racing on the same path is a real hazard, and the caller knows better.
        """
        if self.max_parallel == 1 or len(calls) == 1:
            return [self._run_one(call) for call in calls]

        workers = min(self.max_parallel, len(calls))
        finished: dict[int, ToolResult] = {}
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="oneharness-tool") as pool:
            futures = {pool.submit(self._run_one, call): index for index, call in enumerate(calls)}
            # Events fire as things actually finish (that is what a watcher wants
            # to see), while the returned results keep the model's own order, so
            # the message list stays predictable.
            for future in as_completed(futures):
                result = future.result()
                finished[futures[future]] = result
        return [finished[index] for index in range(len(calls))]

    def _run_one(self, call: ToolCall) -> ToolResult:
        """Run one call, announcing it at the moment it actually starts.

        The announcement used to go out for the whole batch before the first
        tool ran, which made the event stream claim a call was running while it
        was still queued behind another one. Now that every call carries its
        own duration, that lie would be measurable: the gap between
        `tool_start` and `tool_end` has to be the time the tool really spent.
        """
        self._emit(
            {
                "type": "tool_start",
                "name": call.name,
                "id": call.id,
                "arguments": call.arguments,
            }
        )
        started = time.monotonic()
        result = self.tools.execute(call)
        result.seconds = time.monotonic() - started
        self._emit_tool_end(result)
        return result

    def _emit_tool_end(self, result: ToolResult) -> None:
        self._emit(
            {
                "type": "tool_end",
                "name": result.name,
                "is_error": result.is_error,
                "content": result.content,
                "seconds": round(result.seconds, 3),
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

    def _complete_budgeted(
        self,
        messages: list[Message],
        specs: Sequence[dict[str, Any]],
        *,
        stream: bool,
        step: int,
    ) -> ModelCall:
        """Send the request, shrinking the budget if the provider says it is too big.

        The retry is not a transport retry: the payload itself was the problem,
        so the same bytes would fail again. Each pass honestly reduces the
        estimate and remembers what it cut; on success the caller can report the
        final shape in `--json`.

        A refused attempt still spent seconds on the wire, so its duration is
        added to the step rather than discarded: the point of measuring a step
        is to explain what the run waited for, and a wasted round trip is part
        of the answer.
        """
        budget = self.context_budget
        used: context_module.Trim | None = None
        spent = 0.0
        requests = 0
        for attempt in range(context_module.MAX_SHRINKS + 1):
            outgoing = messages
            if budget > 0:
                outgoing, trim = context_module.fit(messages, budget, specs=list(specs))
                if trim.changed:
                    used = trim
                    self._emit({"type": "context_trim", "step": step, **trim.as_dict()})
            started = time.monotonic()
            try:
                completion, first_token = self._complete(outgoing, specs, stream=stream, step=step)
            except ProviderError as exc:
                spent += time.monotonic() - started
                requests += 1
                if attempt == context_module.MAX_SHRINKS or not context_module.is_overflow(exc):
                    raise
                estimated = context_module.estimate(messages, list(specs))
                budget = context_module.shrink(budget, estimated)
                # Retrying with a payload the provider will refuse again only
                # hides the real problem. If the smaller budget cannot remove a
                # single character (a lone oversized prompt has no history to
                # elide), the window is not what has to give -- the caller's
                # prompt is, and that is not this loop's decision to make.
                preview, _ = context_module.fit(messages, budget, specs=list(specs))
                if context_module.estimate(preview, list(specs)) >= estimated:
                    raise ProviderError(
                        "context overflow: the provider refused this request for size and "
                        "there is nothing left to elide (the prompt itself is over the "
                        "window); shorten it instead of retrying",
                        status=exc.status,
                        body=exc.body,
                    ) from None
                self._emit(
                    {
                        "type": "context_overflow",
                        "step": step,
                        "estimated": estimated,
                        "retry_budget": budget,
                    }
                )
                continue
            spent += time.monotonic() - started
            requests += 1
            return ModelCall(
                completion=completion,
                trim=used if used is not None and used.changed else None,
                seconds=spent,
                requests=requests,
                first_token_seconds=first_token,
            )
        raise AssertionError("unreachable: the loop returns or raises")

    def _complete(
        self,
        messages: list[Message],
        specs: Sequence[dict[str, Any]],
        *,
        stream: bool,
        step: int,
    ) -> tuple[Completion, float | None]:
        """One request; the second value is the wait until the first event.

        Only streaming can report that: a blocking request returns once and has
        no "first". Time-to-first-token is the number that separates a slow
        model from a slow pipe -- a provider that thinks for eight seconds and
        then writes fast looks nothing like one that starts immediately and
        dribbles, and both feel the same when all you print is the total.
        """
        if not stream:
            return self.provider.complete(messages, specs), None

        text: list[str] = []
        finished: Completion | None = None
        started = time.monotonic()
        first_token: float | None = None
        events = self.provider.stream(messages, specs)
        try:
            for event in events:
                if first_token is None and (event.kind != "done" or event.completion is not None):
                    first_token = time.monotonic() - started
                if event.kind == "text":
                    text.append(event.text)
                    self._emit({"type": "text_delta", "step": step, "text": event.text})
                elif event.kind == "tool":
                    self._emit_tool_call(event, step)
                elif event.completion is not None:
                    finished = event.completion
        except KeyboardInterrupt:
            # Ctrl-C is a decision, not a crash. Whatever the model already
            # said out loud goes into the result instead of being thrown away
            # together with the connection.
            finished = self._truncated("".join(text), INTERRUPTED_BY_USER)
        except ProviderError as exc:
            if not text:
                raise
            # A connection that dies halfway is the same situation: half an
            # answer that arrived beats an exception that hides it.
            finished = self._truncated("".join(text), f"stream failed: {exc}")
        finally:
            close = getattr(events, "close", None)
            if close is not None:
                close()  # stop reading the body; nothing should keep draining it
        if finished is None:
            if not text:
                raise ProviderError("stream ended without a final completion")
            # The body simply stopped: no `done`, no finish reason. That is the
            # same situation as a reset connection, and the text that arrived is
            # just as worth keeping -- this is exactly the case that used to
            # throw away a half-written answer because the JSON was incomplete.
            finished = self._truncated("".join(text), STREAM_ENDED_EARLY)
        if not finished.message.content and text:
            finished.message = Message.assistant("".join(text), finished.message.tool_calls)
        return finished, first_token

    @staticmethod
    def _truncated(text: str, reason: str) -> Completion:
        """A completion that ended early, keeping the text that did arrive."""
        return Completion(message=Message.assistant(text), truncated=reason)

    def _emit_tool_call(self, event: StreamEvent, step: int) -> None:
        """Announce a tool call while it is still being assembled.

        `fragment` empty means "the call exists, its arguments do not yet"; each
        later fragment is the next slice of the argument JSON. A watcher learns
        that a `write_file` is coming before the whole body has been generated,
        which is the only moment when knowing it is still worth anything.
        """
        call = event.call
        self._emit(
            {
                "type": "tool_call_delta",
                "step": step,
                "index": event.index,
                "id": call.id if call is not None else "",
                "name": call.name if call is not None else "",
                "fragment": event.fragment,
            }
        )

    def _emit(self, event: dict[str, Any]) -> None:
        if self.on_event is not None:
            self.on_event(event)
