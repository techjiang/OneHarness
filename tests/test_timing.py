"""Timing has to be evidence, not decoration.

A duration is only worth printing if it can be checked: a step spent in a tool
must show up in `tool_seconds` and not in `model_seconds`, parallel calls must
sum to more than the wall clock they took, and `tool_start` must fire when the
tool starts rather than when the batch is announced.
"""

import time
import unittest

from oneharness.agent import Agent
from oneharness.providers.base import Provider
from oneharness.tools import tool
from oneharness.types import Completion, Message, StreamEvent, ToolCall, Usage


def text_reply(text, usage=(10, 5)):
    return Completion(message=Message.assistant(text), usage=Usage(*usage))


def tool_reply(name, arguments, call_id="c1"):
    call = ToolCall.from_raw(call_id=call_id, name=name, arguments=arguments)
    return Completion(message=Message.assistant("", [call]), usage=Usage(10, 5))


class ScriptedProvider(Provider):
    name = "scripted"

    def __init__(self, *completions):
        super().__init__(model="scripted-1")
        self.script = list(completions)

    def complete(self, messages, tools=None):
        return self.script.pop(0)


class SlowProvider(Provider):
    """Spends real time in the "model", so the split can be checked."""

    name = "slow"

    def __init__(self, delay, *completions):
        super().__init__(model="slow-1")
        self.delay = delay
        self.script = list(completions)

    def complete(self, messages, tools=None):
        time.sleep(self.delay)
        return self.script.pop(0)


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def sleepy(delay: float) -> str:
    """Sleep, so a tool's own duration is bigger than the loop's overhead."""
    time.sleep(delay)
    return f"slept {delay}"


class StepTimingTests(unittest.TestCase):
    def test_model_time_is_not_tool_time(self):
        provider = SlowProvider(0.12, text_reply("done"))
        result = Agent(provider).run("go")

        step = result.steps[0]
        self.assertGreaterEqual(step.model_seconds, 0.1)
        self.assertLess(step.tool_seconds, 0.05)
        self.assertGreaterEqual(step.seconds, step.model_seconds)

    def test_tool_time_lands_in_the_tool_half(self):
        provider = ScriptedProvider(tool_reply("sleepy", {"delay": 0.12}), text_reply("done"))
        result = Agent(provider, [sleepy]).run("go")

        first = result.steps[0]
        self.assertGreaterEqual(first.tool_seconds, 0.1)
        self.assertLess(first.model_seconds, 0.05)
        self.assertGreaterEqual(result.steps[0].tool_results[0].seconds, 0.1)

    def test_a_step_reports_both_halves_on_its_end_event(self):
        events = []
        provider = ScriptedProvider(tool_reply("sleepy", {"delay": 0.08}), text_reply("done"))
        Agent(provider, [sleepy], on_event=events.append).run("go")

        ends = [event for event in events if event["type"] == "step_end"]
        self.assertEqual(len(ends), 2)
        for event in ends:
            self.assertIn("model_seconds", event)
            self.assertIn("tool_seconds", event)
        self.assertGreater(ends[0]["tool_seconds"], ends[0]["model_seconds"])
        self.assertLess(ends[1]["tool_seconds"], ends[0]["tool_seconds"])

    def test_tool_end_carries_the_duration_of_that_call(self):
        events = []
        provider = ScriptedProvider(tool_reply("sleepy", {"delay": 0.1}), text_reply("done"))
        Agent(provider, [sleepy], on_event=events.append).run("go")

        end = next(event for event in events if event["type"] == "tool_end")
        self.assertGreaterEqual(end["seconds"], 0.09)


class ParallelTimingTests(unittest.TestCase):
    def test_calls_sum_to_more_than_the_wall_clock_when_they_overlap(self):
        calls = [
            ToolCall.from_raw(call_id=f"c{index}", name="sleepy", arguments={"delay": 0.15})
            for index in range(3)
        ]
        provider = ScriptedProvider(
            Completion(message=Message.assistant("", calls)), text_reply("done")
        )
        result = Agent(provider, [sleepy], max_parallel=3).run("go")

        timing = result.timing
        self.assertGreaterEqual(timing["tool_call_seconds"], 0.4)
        self.assertLess(timing["tool_seconds"], 0.3)
        # the gap between the two is the parallelism that was actually bought
        self.assertGreater(timing["tool_call_seconds"], timing["tool_seconds"])
        self.assertEqual(timing["tool_calls"], 3)

    def test_sequential_calls_sum_to_the_wall_clock(self):
        calls = [
            ToolCall.from_raw(call_id=f"c{index}", name="sleepy", arguments={"delay": 0.06})
            for index in range(2)
        ]
        provider = ScriptedProvider(
            Completion(message=Message.assistant("", calls)), text_reply("done")
        )
        result = Agent(provider, [sleepy]).run("go")

        timing = result.timing
        self.assertAlmostEqual(timing["tool_call_seconds"], timing["tool_seconds"], delta=0.03)


class ToolStartIsHonestTests(unittest.TestCase):
    def test_tool_start_fires_when_the_tool_starts(self):
        """The second call is announced after the first one finished.

        The old behaviour announced the whole batch up front, so a queued call
        looked like it was already running -- and once every call reports its
        own duration, that gap is a number someone will trust.
        """

        @tool
        def wait_for(name: str, delay: float) -> str:
            """Sleep, then report which call this was."""
            time.sleep(delay)
            return name

        calls = [
            ToolCall.from_raw(
                call_id="c1", name="wait_for", arguments={"name": "first", "delay": 0.1}
            ),
            ToolCall.from_raw(
                call_id="c2", name="wait_for", arguments={"name": "second", "delay": 0.0}
            ),
        ]
        provider = ScriptedProvider(
            Completion(message=Message.assistant("", calls)), text_reply("done")
        )
        marks: list[tuple[str, float]] = []
        started = time.monotonic()

        def observe(event):
            if event["type"] in ("tool_start", "tool_end"):
                marks.append((event["type"], time.monotonic() - started))

        Agent(provider, [wait_for], on_event=observe).run("go")  # sequential

        starts = [moment for kind, moment in marks if kind == "tool_start"]
        ends = [moment for kind, moment in marks if kind == "tool_end"]
        self.assertEqual(len(starts), 2)
        # sequential: the second tool cannot have started before the first ended
        self.assertLess(ends[0], starts[1])

    def test_parallel_starts_all_land_before_the_first_end(self):
        @tool
        def wait_for(name: str, delay: float) -> str:
            """Sleep, then report which call this was."""
            time.sleep(delay)
            return name

        calls = [
            ToolCall.from_raw(
                call_id=f"c{index}", name="wait_for", arguments={"name": str(index), "delay": 0.1}
            )
            for index in range(3)
        ]
        provider = ScriptedProvider(
            Completion(message=Message.assistant("", calls)), text_reply("done")
        )
        marks: list[str] = []

        def observe(event):
            if event["type"] in ("tool_start", "tool_end"):
                marks.append(event["type"])

        Agent(provider, [wait_for], max_parallel=3, on_event=observe).run("go")

        # They overlap, so every call is announced before any of them returns.
        self.assertEqual(marks[:3], ["tool_start"] * 3)
        self.assertEqual(marks[3:], ["tool_end"] * 3)


class DelegatedTimeTests(unittest.TestCase):
    def test_sub_agent_time_is_reported_separately_from_tools(self):
        """A `delegate` call is a tool call, but waiting on it is not tool time.

        Folding the sub agent's clock into `tool_seconds` would make every
        fan-out look like a slow shell -- a wrong diagnosis with a confident
        number attached to it.
        """
        from oneharness.delegate import Delegator

        class SlowSub(ScriptedProvider):
            def complete(self, messages, tools=None):
                time.sleep(0.1)
                return text_reply("sub answer")

        delegator = Delegator(SlowSub(), [add])
        delegator.run("do something")

        self.assertGreaterEqual(delegator.seconds, 0.09)
        self.assertGreaterEqual(delegator.as_dict()["seconds"], 0.09)

    def test_concurrent_sub_agent_time_is_not_double_counted_per_label(self):
        from oneharness.delegate import Delegator

        class SlowSub(ScriptedProvider):
            def complete(self, messages, tools=None):
                time.sleep(0.05)
                return text_reply("ok")

        delegator = Delegator(SlowSub(), [add], max_parallel=3)
        delegator.many(["a", "b", "c"])

        self.assertEqual(delegator.runs, 3)
        # three runs of ~50ms each: the ledger counts the time it waited for
        # them, one run at a time, not the wall clock of the batch
        self.assertGreaterEqual(delegator.seconds, 0.12)


class ResultTimingTests(unittest.TestCase):
    def test_timing_is_in_the_json(self):
        provider = ScriptedProvider(tool_reply("add", '{"a": 1, "b": 2}'), text_reply("3"))
        payload = Agent(provider, [add]).run("go").as_dict()

        timing = payload["timing"]
        self.assertEqual(timing["steps"], 2)
        self.assertEqual(timing["tool_calls"], 1)
        self.assertEqual(timing["slowest_step"]["step"], 1)
        self.assertIn("unaccounted_seconds", timing)
        self.assertIn("seconds", payload["steps"][0]["tool_results"][0])

    def test_a_retried_request_is_counted(self):
        """Two requests for one step is a fact about the step, not a detail."""

        from oneharness.errors import ProviderError

        class Refusing(Provider):
            name = "refusing"

            def __init__(self):
                super().__init__(model="m")
                self.rounds = 0

            def complete(self, messages, tools=None):
                self.rounds += 1
                if self.rounds == 1:
                    raise ProviderError("too big", status=400, body="prompt is too long")
                return text_reply("done")

        result = Agent(Refusing()).run(
            "go", history=[Message.user("earlier"), Message.tool("c0", "x" * 4000, "read_file")]
        )
        self.assertEqual(result.steps[0].model_requests, 2)
        self.assertEqual(result.timing["model_requests"], 2)

    def test_first_token_is_absent_without_streaming(self):
        provider = ScriptedProvider(text_reply("hello"))
        payload = Agent(provider).run("go").as_dict()
        self.assertIsNone(payload["timing"]["first_token_seconds"])
        self.assertNotIn("first_token_seconds", payload["steps"][0])

    def test_first_token_is_measured_on_a_stream(self):
        class SlowThinking(Provider):
            name = "thinking"

            def __init__(self):
                super().__init__(model="m")

            def complete(self, messages, tools=None):
                raise AssertionError("stream mode must not call complete()")

            def stream(self, messages, tools=None):
                time.sleep(0.1)  # the pause a person actually stares at
                yield StreamEvent(kind="text", text="an")
                yield StreamEvent(kind="text", text="swer")
                yield StreamEvent(
                    kind="done", completion=Completion(message=Message.assistant("answer"))
                )

        result = Agent(SlowThinking()).run("go", stream=True)
        self.assertGreaterEqual(result.steps[0].first_token_seconds, 0.09)
        self.assertGreaterEqual(result.timing["first_token_seconds"], 0.09)

    def test_unaccounted_time_is_never_negative(self):
        """The parts are measured around the waits; the whole is wall clock.

        Rounding must not turn "we did not measure this" into a negative number.
        """
        provider = ScriptedProvider(text_reply("instant"))
        timing = Agent(provider).run("go").timing
        self.assertGreaterEqual(timing["unaccounted_seconds"], 0.0)

    def test_a_refused_attempt_still_counts_against_the_step(self):
        """A wasted round trip is part of the wait, so it is part of the number."""

        from oneharness.errors import ProviderError

        class Refusing(Provider):
            name = "refusing"

            def __init__(self):
                super().__init__(model="m")
                self.rounds = 0

            def complete(self, messages, tools=None):
                self.rounds += 1
                time.sleep(0.05)
                if self.rounds == 1:
                    raise ProviderError(
                        "too big", status=400, body="prompt is too long: 9000 tokens"
                    )
                return text_reply("done")

        provider = Refusing()
        agent = Agent(provider, context_budget=0)
        # The retry needs something to elide, otherwise the loop refuses to try
        # again; a history with a fat tool result gives it that.
        result = agent.run(
            "go",
            history=[Message.user("earlier"), Message.tool("c0", "x" * 4000, "read_file")],
        )
        self.assertEqual(result.text, "done")
        self.assertGreaterEqual(result.steps[0].model_seconds, 0.09)


if __name__ == "__main__":
    unittest.main()
