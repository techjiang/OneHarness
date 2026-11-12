import unittest

from oneharness.agent import Agent
from oneharness.errors import ProviderError
from oneharness.providers.base import Provider
from oneharness.tools import tool
from oneharness.types import Completion, Message, ToolCall, Usage


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
        provider = ScriptedProvider(
            tool_reply("add", '{"a": "x"}'), text_reply("sorry, retrying")
        )
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
            kinds, ["step_start", "assistant", "tool_start", "tool_end", "step_start", "assistant"]
        )
        self.assertEqual(events[2]["name"], "add")

    def test_result_serialises(self):
        provider = ScriptedProvider(tool_reply("add", '{"a": 1, "b": 2}'), text_reply("3"))
        payload = Agent(provider, [add]).run("go").as_dict()
        self.assertEqual(payload["text"], "3")
        self.assertEqual(payload["stopped_reason"], "done")
        self.assertEqual(payload["steps"][0]["tool_results"][0]["content"], "3")
        self.assertEqual(payload["usage"]["total_tokens"], 30)


if __name__ == "__main__":
    unittest.main()
