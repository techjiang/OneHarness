"""Sub agents: hand a task to another agent run and keep only its conclusion.

The point is context isolation, not parallelism. A sub agent gets its own
system prompt, its own tool registry and its own message list, so the twenty
intermediate reads it did never enter the parent's context -- the parent only
sees the answer. That is the difference between "a delegation" and "calling a
tool": the stuff in between is thrown away.
"""

from __future__ import annotations

from collections.abc import Sequence
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


class DelegationError(OneHarnessError):
    """The sub agent finished without producing an answer."""


class Delegator:
    """Spawns sub agents and tracks what they cost.

    Sub-agent spend is accounted separately (`usage`, `runs`) instead of being
    folded into the parent's numbers, because "$0.02 for the parent, $0.30 for
    the sub agent" is a fact worth being able to see.
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
    ) -> None:
        registry = tools if isinstance(tools, Registry) else Registry(tools)
        # A sub agent must not be able to delegate again: recursion stays shallow
        # by construction rather than by a depth counter nobody enforces.
        self._inner = [item for item in registry if item.name != "delegate"]
        self.provider = provider
        self.system = DEFAULT_SYSTEM if system is None else system
        self.max_steps = max_steps
        self.max_parallel = max_parallel
        self.on_event = on_event
        self.usage = Usage()
        self.runs = 0

    def run(self, task: str) -> str:
        """Run one sub agent to completion and return its final answer."""
        self.runs += 1
        self._emit({"type": "subagent_start", "task": task})
        agent = Agent(
            self.provider,
            Registry(self._inner),
            system=self.system,
            max_steps=self.max_steps,
            max_parallel=self.max_parallel,
            on_event=self._sub_observer(),
        )
        result = agent.run(task)
        self.usage = self.usage + result.usage
        self._emit(
            {
                "type": "subagent_end",
                "task": task,
                "steps": len(result.steps),
                "usage": result.usage.as_dict(),
            }
        )
        if not result.text:
            raise DelegationError(
                f"sub agent stopped after {len(result.steps)} steps without an answer "
                f"({result.stopped_reason})"
            )
        return result.text

    def _sub_observer(self) -> Observer | None:
        """Tag sub-agent events so a printer can indent them instead of guessing."""
        if self.on_event is None:
            return None

        parent = self.on_event

        def forward(event: dict[str, Any]) -> None:
            parent({**event, "depth": 1})

        return forward

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
                "the steps, only the result."
            ),
        )


def make_delegate_tool(
    provider: Provider,
    tools: Sequence[Tool] | Registry = (),
    *,
    system: str | None = None,
    max_steps: int = 6,
    max_parallel: int = 1,
) -> Tool:
    """Convenience wrapper: one Delegator, one tool, no handle to keep."""
    return Delegator(
        provider, tools, system=system, max_steps=max_steps, max_parallel=max_parallel
    ).tool()
