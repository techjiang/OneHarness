"""Sub agents: hand a task to another agent run and keep only its conclusion.

The point is context isolation, not parallelism. A sub agent gets its own
system prompt, its own tool registry and its own message list, so the twenty
intermediate reads it did never enter the parent's context -- the parent only
sees the answer. That is the difference between "a delegation" and "calling a
tool": the stuff in between is thrown away.

Several delegations in one turn run side by side (the tool executor's
`max_parallel` covers them), so every run gets a label and its own line in the
ledger: three sub agents talking at once produce three interleaved transcripts,
and "who spent what" must still be answerable afterwards.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from oneharness.agent import Agent, Observer
from oneharness.errors import OneHarnessError
from oneharness.providers.base import Provider
from oneharness.tools import Registry, Tool, tool
from oneharness.types import Usage

DEFAULT_SYSTEM = (
    "You are a sub agent working on one delegated task. You have the same tools "
    "as the agent that asked you. Finish the task, then answer with the result "
    "only: no preamble, no restating the task, no offer to help further. If you "
    "cannot finish, say what blocked you in one or two sentences."
)


# A sub agent must never get these back, or it could recurse forever. The names
# live in one place so adding a delegation tool cannot forget the filter.
RECURSIVE_TOOLS = frozenset({"delegate", "delegate_many"})


class DelegationError(OneHarnessError):
    """The sub agent finished without producing an answer."""


class Delegator:
    """Spawns sub agents and tracks what they cost.

    Sub-agent spend is accounted separately (`usage`, `by_label`, `runs`)
    instead of being folded into the parent's numbers, because "$0.02 for the
    parent, $0.30 for the sub agent" is a fact worth being able to see.
    """

    def __init__(
        self,
        provider: Provider,
        tools: Sequence[Tool] | Registry = (),
        *,
        system: str | None = None,
        max_steps: int = 6,
        max_parallel: int = 1,
        on_event: Observer | None = None,
        label_prefix: str = "sub",
        context_budget: int = 0,
    ) -> None:
        registry = tools if isinstance(tools, Registry) else Registry(tools)
        # A sub agent must not be able to delegate again: recursion stays shallow
        # by construction rather than by a depth counter nobody enforces.
        self._inner = [item for item in registry if item.name not in RECURSIVE_TOOLS]
        self.provider = provider
        self.system = DEFAULT_SYSTEM if system is None else system
        self.max_steps = max_steps
        self.max_parallel = max_parallel
        self.on_event = on_event
        self.label_prefix = label_prefix
        # A sub agent explores harder than its parent and then throws the
        # exploring away, so it is the one that hits the window first.
        self.context_budget = context_budget
        # Parallel runs land here at the same time; `total = total + x` is a
        # read-modify-write, so the ledger needs a lock or it silently drops
        # tokens (verified: 40 concurrent runs lost 1000 input tokens).
        self._lock = threading.Lock()
        self._spawned = 0
        self.usage = Usage()
        self.by_label: dict[str, Usage] = {}
        self.runs = 0

    def run(self, task: str, *, label: str | None = None) -> str:
        """Run one sub agent to completion and return its final answer."""
        label = label or self._next_label()
        self._emit({"type": "subagent_start", "task": task, "label": label})
        agent = Agent(
            self.provider,
            Registry(self._inner),
            system=self.system,
            max_steps=self.max_steps,
            max_parallel=self.max_parallel,
            on_event=self._sub_observer(label),
            context_budget=self.context_budget,
        )
        result = agent.run(task)
        self._account(label, result.usage)
        self._emit(
            {
                "type": "subagent_end",
                "task": task,
                "label": label,
                "steps": len(result.steps),
                "usage": result.usage.as_dict(),
            }
        )
        if not result.text:
            raise DelegationError(
                f"sub agent {label} stopped after {len(result.steps)} steps without an answer "
                f"({result.stopped_reason})"
            )
        return result.text

    def _reserve(self, count: int) -> list[str]:
        """Hand out `count` labels in one go, so a batch is numbered without gaps."""
        with self._lock:
            start = self._spawned
            self._spawned += count
        return [f"{self.label_prefix}-{index}" for index in range(start + 1, start + count + 1)]

    def _next_label(self) -> str:
        return self._reserve(1)[0]

    def many(self, tasks: Sequence[str], *, max_parallel: int | None = None) -> str:
        """Run several sub agents at once and return their answers, one per line.

        The parent asked for N independent things; waiting for them one by one
        is the whole cost of the fan-out. A sub agent that fails does not sink
        the batch: its line says so and the other answers still come back, in
        the order the model asked for them.
        """
        ordered = [task for task in tasks if task.strip()]
        if not ordered:
            raise DelegationError("no tasks given; nothing to delegate")
        pairs = list(zip(self._reserve(len(ordered)), ordered, strict=True))
        workers = min(max_parallel or self.max_parallel, len(pairs))
        if workers <= 1:
            return "\n".join(self._one(label, task) for label, task in pairs)

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="oneharness-sub") as pool:
            futures = [pool.submit(self._one, label, task) for label, task in pairs]
            # Results are read back in submission order, so a task that finishes
            # early cannot jump the queue ahead of the one before it.
            return "\n".join(future.result() for future in futures)

    def _one(self, label: str, task: str) -> str:
        """One sub agent, with its failure folded into the line it returns."""
        try:
            return f"[{label}] {self.run(task, label=label)}"
        except DelegationError as exc:
            return f"[{label}] failed: {exc}"
        except OneHarnessError as exc:
            return f"[{label}] failed: {type(exc).__name__}: {exc}"

    def _account(self, label: str, usage: Usage) -> None:
        with self._lock:
            self.runs += 1
            self.usage = self.usage + usage
            self.by_label[label] = self.by_label.get(label, Usage()) + usage

    def _sub_observer(self, label: str) -> Observer | None:
        """Tag sub-agent events so a printer can indent and attribute them."""
        if self.on_event is None:
            return None

        parent = self.on_event

        def forward(event: dict[str, Any]) -> None:
            parent({**event, "depth": 1, "label": label})

        return forward

    def as_dict(self) -> dict[str, Any]:
        return {
            "runs": self.runs,
            "usage": self.usage.as_dict(),
            "by_label": {label: usage.as_dict() for label, usage in self.by_label.items()},
        }

    def _emit(self, event: dict[str, Any]) -> None:
        if self.on_event is not None:
            self.on_event(event)

    def tool(
        self,
        *,
        name: str = "delegate",
        description: str | None = None,
    ) -> Tool:
        """Expose `run` to the parent model as an ordinary tool."""

        def delegate(task: str) -> str:
            return self.run(task)

        # `@tool` reads the signature, so the schema says what the model needs
        # to know: one required string.
        return tool(
            delegate,
            name=name,
            description=description
            or (
                "Delegate one self-contained task to a sub agent that has the same "
                "tools but a fresh context, and get back only its final answer. Use "
                "this for work that takes many exploratory steps; you will not see "
                "the steps, only the result. Independent tasks may be delegated "
                "several at a time in one turn and will run side by side."
            ),
        )

    def batch_tool(
        self,
        *,
        name: str = "delegate_many",
        description: str | None = None,
    ) -> Tool:
        """Expose the fan-out to the parent model as one tool call."""

        def delegate_many(tasks: list[str]) -> str:
            return self.many(tasks)

        return tool(
            delegate_many,
            name=name,
            description=description
            or (
                "Delegate several independent tasks at once, each to its own sub "
                "agent with a fresh context. Returns one `[label] answer` line per "
                "task, in the order given. Use this instead of calling `delegate` "
                "repeatedly when the tasks do not depend on each other."
            ),
        )


def make_delegate_tool(
    provider: Provider,
    tools: Sequence[Tool] | Registry = (),
    *,
    system: str | None = None,
    max_steps: int = 6,
    max_parallel: int = 1,
    context_budget: int = 0,
) -> Tool:
    """Convenience wrapper: one Delegator, one tool, no handle to keep."""
    return Delegator(
        provider,
        tools,
        system=system,
        max_steps=max_steps,
        max_parallel=max_parallel,
        context_budget=context_budget,
    ).tool()
