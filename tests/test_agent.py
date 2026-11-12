import time
import unittest

from oneharness.agent import Agent
from oneharness.errors import ProviderError
from oneharness.providers.base import Provider
from oneharness.tools import tool
from oneharness.types import Completion, Message, StreamEvent, ToolCall, Usage


class ScriptedProvider(Provider):
    """Returns canned completions in order and records what it was asked."""

    name = "scripted"

    def __init__(self, *completions):
        super().__init__(model="scripted-1")
        self.script = list(completions)
        self.calls = []

    def complete(self, messages, tools=None):
        self.calls.append({"messages": list(messages), "tools": list(tools or ())})
        if not self.script:
            raise AssertionError("provider called more times than scripted")
        return self.script.pop(0)


def text_reply(text):
    return Completion(message=Message.assistant(text), usage=Usage(10, 5))


def tool_reply(name, arguments, call_id="c1"):
    call = ToolCall.from_raw(call_id=call_id, name=name, arguments=arguments)
    return Completion(message=Message.assistant("", [call]), usage=Usage(10, 5))


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


class AgentLoopTests(unittest.TestCase):
    def test_plain_answer_without_tools(self):
        provider = ScriptedProvider(text_reply("hello"))
        result = Agent(provider).run("hi")
        self.assertEqual(result.text, "hello")
        self.assertEqual(result.stopped_reason, "done")
        self.assertTrue(result.ok)
        self.assertEqual(result.usage.total_tokens, 15)

    def test_single_tool_round_trip(self):
        provider = ScriptedProvider(tool_reply("add", '{"a": 2, "b": 3}'), text_reply("5"))
        result = Agent(provider, [add]).run("2+3?")

        self.assertEqual(result.text, "5")
        self.assertEqual(len(result.steps), 2)
        self.assertEqual(result.steps[0].tool_results[0].content, "5")
        self.assertFalse(result.steps[0].tool_results[0].is_error)

        # the provider must see the tool result as a tool message
        second_call = provider.calls[1]["messages"]
        self.assertEqual(second_call[-1].role, "tool")
        self.assertEqual(second_call[-1].tool_call_id, "c1")
        self.assertEqual(second_call[-1].content, "5")

    def test_tool_schemas_are_passed_through(self):
        provider = ScriptedProvider(text_reply("ok"))
        Agent(provider, [add]).run("hi")
        specs = provider.calls[0]["tools"]
        self.assertEqual(specs[0]["name"], "add")
        self.assertEqual(specs[0]["parameters"]["required"], ["a", "b"])

    def test_system_prompt_prepended_once(self):
        provider = ScriptedProvider(text_reply("ok"))
        Agent(provider, system="be terse").run("hi")
        messages = provider.calls[0]["messages"]
        self.assertEqual(messages[0].role, "system")
        self.assertEqual(messages[0].content, "be terse")
        self.assertEqual(messages[1].role, "user")

    def test_history_is_honoured(self):
        provider = ScriptedProvider(text_reply("again"))
        history = [Message.user("first"), Message.assistant("ok"), Message.user("second")]
        result = Agent(provider, system="sys").run("third", history=history)
        roles = [m.role for m in result.messages]
        self.assertEqual(roles[:4], ["system", "user", "assistant", "user"])

    def test_bad_tool_arguments_are_fed_back(self):
        provider = ScriptedProvider(tool_reply("add", '{"a": "x"}'), text_reply("sorry, retrying"))
        result = Agent(provider, [add]).run("2+3?")
        first = result.steps[0].tool_results[0]
        self.assertTrue(first.is_error)
        self.assertIn("bad arguments", first.content)
        self.assertTrue(result.ok)

    def test_unknown_tool_is_fed_back(self):
        provider = ScriptedProvider(tool_reply("ghost", "{}"), text_reply("no such tool"))
        result = Agent(provider, [add]).run("go")
        self.assertTrue(result.steps[0].tool_results[0].is_error)
        self.assertIn("unknown tool", result.steps[0].tool_results[0].content)

    def test_multiple_tool_calls_in_one_turn(self):
        call = ToolCall.from_raw(call_id="c1", name="add", arguments={"a": 1, "b": 1})
        call2 = ToolCall.from_raw(call_id="c2", name="add", arguments={"a": 2, "b": 2})
        provider = ScriptedProvider(
            Completion(message=Message.assistant("", [call, call2])), text_reply("2 and 4")
        )
        result = Agent(provider, [add]).run("sums")
        results = result.steps[0].tool_results
        self.assertEqual([r.content for r in results], ["2", "4"])
        tool_messages = [m for m in result.messages if m.role == "tool"]
        self.assertEqual([m.tool_call_id for m in tool_messages], ["c1", "c2"])

    def test_max_steps_stops_a_looping_model(self):
        script = [tool_reply("add", '{"a": 1, "b": 1}', call_id=f"c{i}") for i in range(3)]
        provider = ScriptedProvider(*script)
        result = Agent(provider, [add], max_steps=3).run("loop")
        self.assertEqual(result.stopped_reason, "max_steps")
        self.assertFalse(result.ok)
        self.assertEqual(len(result.steps), 3)
        # usage accumulates across every step
        self.assertEqual(result.usage.input_tokens, 30)

    def test_max_steps_must_be_positive(self):
        with self.assertRaises(ValueError):
            Agent(ScriptedProvider(), max_steps=0)

    def test_provider_error_propagates(self):
        class Boom(Provider):
            def complete(self, messages, tools=None):
                raise ProviderError("upstream 500", status=500)

        with self.assertRaises(ProviderError):
            Agent(Boom("m")).run("hi")

    def test_events_are_emitted_in_order(self):
        events = []
        provider = ScriptedProvider(tool_reply("add", '{"a": 1, "b": 2}'), text_reply("3"))
        Agent(provider, [add], on_event=events.append).run("go")
        kinds = [e["type"] for e in events]
        self.assertEqual(
            kinds,
            [
                "step_start",
                "assistant",
                "tool_start",
                "tool_end",
                "step_end",
                "step_start",
                "assistant",
                # The answering step is timed too: it used to be the only one
                # reporting 0.0, which is a duration nobody should trust.
                "step_end",
            ],
        )
        self.assertEqual(events[2]["name"], "add")
        self.assertEqual(events[4]["tool_calls"], 1)
        self.assertEqual(events[7]["tool_calls"], 0)

    def test_stream_mode_emits_deltas_and_keeps_result_shape(self):
        class StreamingProvider(Provider):
            name = "streamy"

            def __init__(self):
                super().__init__(model="m")
                self.rounds = 0

            def complete(self, messages, tools=None):
                raise AssertionError("stream mode must not call complete()")

            def stream(self, messages, tools=None):
                self.rounds += 1
                if self.rounds == 1:
                    call = ToolCall.from_raw(call_id="c1", name="add", arguments={"a": 2, "b": 3})
                    for piece in ("let me ", "add"):
                        yield StreamEvent(kind="text", text=piece)
                    yield StreamEvent(
                        kind="done",
                        completion=Completion(message=Message.assistant("", [call])),
                    )
                    return
                yield StreamEvent(kind="text", text="it is ")
                yield StreamEvent(kind="text", text="5")
                yield StreamEvent(kind="done", completion=Completion(message=Message.assistant("")))

        events = []
        result = Agent(StreamingProvider(), [add], on_event=events.append).run("2+3?", stream=True)

        self.assertEqual(result.text, "it is 5")
        self.assertEqual(result.stopped_reason, "done")
        self.assertEqual(result.steps[0].tool_results[0].content, "5")
        deltas = [e["text"] for e in events if e["type"] == "text_delta"]
        self.assertEqual("".join(deltas), "let me addit is 5")

    def test_stream_that_just_stops_keeps_its_text(self):
        # No `done`, no finish reason: the body simply ended. The text that did
        # arrive is kept and the run is marked truncated -- raising here would
        # discard the only thing that came back.
        class Mute(Provider):
            name = "mute"

            def complete(self, messages, tools=None):
                raise AssertionError("unused")

            def stream(self, messages, tools=None):
                yield StreamEvent(kind="text", text="half")

        result = Agent(Mute("m")).run("hi", stream=True)
        self.assertEqual(result.text, "half")
        self.assertEqual(result.stopped_reason, "interrupted")
        self.assertIn("without a final completion", result.interrupted)

    def test_stream_that_produces_nothing_still_raises(self):
        class Silent(Provider):
            name = "silent"

            def complete(self, messages, tools=None):
                raise AssertionError("unused")

            def stream(self, messages, tools=None):
                return
                yield  # pragma: no cover - makes this a generator

        with self.assertRaises(ProviderError):
            Agent(Silent("m")).run("hi", stream=True)

    def test_default_stream_falls_back_to_complete(self):
        """A provider without real streaming still works under --stream."""
        provider = ScriptedProvider(text_reply("pong"))
        events = []
        result = Agent(provider, on_event=events.append).run("ping", stream=True)
        self.assertEqual(result.text, "pong")
        self.assertEqual([e["text"] for e in events if e["type"] == "text_delta"], ["pong"])

    def test_parallel_tool_calls_overlap(self):
        """Two slow tools in one turn should finish in about one tool's time."""

        @tool
        def slow(label: str) -> str:
            """Sleep briefly, then answer."""
            time.sleep(0.2)
            return label

        first = ToolCall.from_raw(call_id="c1", name="slow", arguments={"label": "a"})
        second = ToolCall.from_raw(call_id="c2", name="slow", arguments={"label": "b"})
        provider = ScriptedProvider(
            Completion(message=Message.assistant("", [first, second])), text_reply("both done")
        )
        started = time.monotonic()
        result = Agent(provider, [slow], max_parallel=2).run("go")
        elapsed = time.monotonic() - started

        self.assertEqual([r.content for r in result.steps[0].tool_results], ["a", "b"])
        self.assertLess(elapsed, 0.38)

    def test_parallel_results_keep_the_model_order(self):
        @tool
        def sleepy(name: str, delay: float) -> str:
            """Sleep, then return the name so ordering is visible."""
            time.sleep(delay)
            return name

        calls = [
            ToolCall.from_raw(
                call_id="c1", name="sleepy", arguments={"name": "slow", "delay": 0.2}
            ),
            ToolCall.from_raw(
                call_id="c2", name="sleepy", arguments={"name": "fast", "delay": 0.01}
            ),
        ]
        provider = ScriptedProvider(
            Completion(message=Message.assistant("", calls)), text_reply("done")
        )
        events = []
        result = Agent(provider, [sleepy], max_parallel=2, on_event=events.append).run("go")

        # results follow the request order, not the completion order
        self.assertEqual([r.content for r in result.steps[0].tool_results], ["slow", "fast"])
        ended = [e["content"] for e in events if e["type"] == "tool_end"]
        self.assertEqual(ended, ["fast", "slow"])

    def test_parallel_failure_is_isolated(self):
        @tool
        def boom() -> str:
            """Always fails."""
            raise ValueError("kaboom")

        calls = [
            ToolCall.from_raw(call_id="c1", name="boom", arguments={}),
            ToolCall.from_raw(call_id="c2", name="add", arguments={"a": 1, "b": 1}),
        ]
        provider = ScriptedProvider(
            Completion(message=Message.assistant("", calls)), text_reply("recovered")
        )
        result = Agent(provider, [boom, add], max_parallel=2).run("go")
        results = result.steps[0].tool_results
        self.assertTrue(results[0].is_error)
        self.assertEqual(results[1].content, "2")
        self.assertTrue(result.ok)

    def test_max_parallel_must_be_positive(self):
        with self.assertRaises(ValueError):
            Agent(ScriptedProvider(), max_parallel=0)

    def test_step_seconds_are_recorded(self):
        provider = ScriptedProvider(tool_reply("add", '{"a": 1, "b": 2}'), text_reply("3"))
        result = Agent(provider, [add]).run("go")
        self.assertGreaterEqual(result.steps[0].seconds, 0.0)
        self.assertGreaterEqual(result.steps[1].seconds, 0.0)
        self.assertIn("seconds", result.as_dict()["steps"][0])

    def test_tool_call_deltas_reach_the_observer_before_the_tool_runs(self):
        class StreamingTools(Provider):
            name = "streamy"

            def __init__(self):
                super().__init__(model="m")
                self.rounds = 0

            def complete(self, messages, tools=None):
                raise AssertionError("stream mode must not call complete()")

            def stream(self, messages, tools=None):
                self.rounds += 1
                if self.rounds == 1:
                    yield StreamEvent(kind="tool", index=0, call=ToolCall(id="c1", name="add"))
                    yield StreamEvent(
                        kind="tool",
                        index=0,
                        call=ToolCall(id="c1", name="add"),
                        fragment='{"a": 1, "b": 2}',
                    )
                    yield StreamEvent(kind="done", completion=tool_reply("add", {"a": 1, "b": 2}))
                else:
                    yield StreamEvent(kind="text", text="2")
                    yield StreamEvent(kind="done", completion=text_reply("2"))

        events = []
        Agent(StreamingTools(), [add], on_event=events.append).run("go", stream=True)
        kinds = [e["type"] for e in events]
        # the announcement lands before the tool is started, which is the whole
        # point: a watcher learns what is about to run while it is being written
        self.assertLess(kinds.index("tool_call_delta"), kinds.index("tool_start"))
        announced = [e for e in events if e["type"] == "tool_call_delta"]
        self.assertEqual(announced[0]["name"], "add")
        self.assertEqual(announced[0]["fragment"], "")
        self.assertEqual(announced[1]["fragment"], '{"a": 1, "b": 2}')

    def test_mid_stream_failure_keeps_the_text_that_arrived(self):
        from oneharness.errors import ProviderError

        class DyingStream(Provider):
            name = "dying"

            def __init__(self):
                super().__init__(model="m")

            def complete(self, messages, tools=None):
                raise AssertionError("must not be called")

            def stream(self, messages, tools=None):
                yield StreamEvent(kind="text", text="The timeout is ")
                yield StreamEvent(kind="text", text="five seconds")
                raise ProviderError("connection reset by peer")

        result = Agent(DyingStream()).run("go", stream=True)
        # half an answer that arrived is worth more than an exception that hides it
        self.assertEqual(result.text, "The timeout is five seconds")
        self.assertEqual(result.stopped_reason, "interrupted")
        self.assertIn("connection reset", result.interrupted)
        self.assertFalse(result.interrupted_by_user)
        self.assertFalse(result.ok)
        self.assertEqual(result.as_dict()["interrupted"], result.interrupted)

    def test_failure_before_any_text_still_raises(self):
        from oneharness.errors import ProviderError

        class DeadStream(Provider):
            name = "dead"

            def __init__(self):
                super().__init__(model="m")

            def complete(self, messages, tools=None):
                raise AssertionError("must not be called")

            def stream(self, messages, tools=None):
                raise ProviderError("connection reset by peer")
                yield  # pragma: no cover - makes this a generator

        with self.assertRaises(ProviderError):
            Agent(DeadStream()).run("go", stream=True)

    def test_interrupt_keeps_partial_text_and_exits_130(self):
        class Interrupted(Provider):
            name = "interrupted"

            def __init__(self):
                super().__init__(model="m")

            def complete(self, messages, tools=None):
                raise AssertionError("must not be called")

            def stream(self, messages, tools=None):
                yield StreamEvent(kind="text", text="half a thou")
                raise KeyboardInterrupt

        events = []
        result = Agent(Interrupted(), on_event=events.append).run("go", stream=True)
        self.assertEqual(result.text, "half a thou")
        self.assertTrue(result.interrupted_by_user)
        self.assertEqual(result.stopped_reason, "interrupted")
        self.assertIn("interrupted", [e["type"] for e in events])

    def test_result_serialises(self):
        provider = ScriptedProvider(tool_reply("add", '{"a": 1, "b": 2}'), text_reply("3"))
        payload = Agent(provider, [add]).run("go").as_dict()
        self.assertEqual(payload["text"], "3")
        self.assertEqual(payload["stopped_reason"], "done")
        self.assertEqual(payload["steps"][0]["tool_results"][0]["content"], "3")
        self.assertEqual(payload["usage"]["total_tokens"], 30)


if __name__ == "__main__":
    unittest.main()
