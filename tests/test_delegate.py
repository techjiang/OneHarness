import unittest

from oneharness.agent import Agent
from oneharness.delegate import DelegationError, Delegator, make_delegate_tool
from oneharness.errors import ToolError
from oneharness.providers.base import Provider
from oneharness.tools import Registry, tool
from oneharness.types import Completion, Message, ToolCall, Usage


class ScriptedProvider(Provider):
    """Canned completions, keyed by who asked: parent or sub agent."""

    name = "scripted"

    def __init__(self, *completions):
        super().__init__(model="scripted-1")
        self.script = list(completions)
        self.calls = []

    def complete(self, messages, tools=None):
        self.calls.append({"messages": list(messages), "tools": [t["name"] for t in tools or ()]})
        if not self.script:
            raise AssertionError("provider called more times than scripted")
        return self.script.pop(0)


def text_reply(text, usage=None):
    return Completion(message=Message.assistant(text), usage=usage or Usage(10, 5))


def tool_reply(name, arguments, call_id="c1"):
    call = ToolCall.from_raw(call_id=call_id, name=name, arguments=arguments)
    return Completion(message=Message.assistant("", [call]), usage=Usage(10, 5))


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def boom() -> str:
    """Always fails."""
    raise ToolError("nope")


class DelegatorTests(unittest.TestCase):
    def test_sub_agent_answer_reaches_the_parent_as_a_tool_result(self):
        provider = ScriptedProvider(
            tool_reply("delegate", {"task": "add 2 and 3"}),
            text_reply("5"),  # sub agent
            text_reply("the sub agent says 5"),
        )
        delegator = Delegator(provider, [add])
        result = Agent(provider, Registry([add, delegator.tool()])).run("2+3?")

        self.assertEqual(result.text, "the sub agent says 5")
        self.assertEqual(result.steps[0].tool_results[0].content, "5")
        self.assertFalse(result.steps[0].tool_results[0].is_error)

    def test_sub_agent_gets_a_fresh_context(self):
        provider = ScriptedProvider(
            tool_reply("delegate", {"task": "look it up"}),
            text_reply("done"),
            text_reply("ok"),
        )
        delegator = Delegator(provider, [add], system="be a sub agent")
        Agent(provider, Registry([delegator.tool()])).run("please delegate")

        later_parent_turn = provider.calls[2]  # parent turn, sub turn, parent turn
        roles = [m.role for m in provider.calls[1]["messages"]]
        self.assertEqual(provider.calls[1]["messages"][0].content, "be a sub agent")
        self.assertEqual(roles, ["system", "user"])
        # the parent's own prompt never leaked into the sub agent
        self.assertNotIn("please delegate", [m.content for m in provider.calls[1]["messages"]])
        # the parent sees one tool message, not the sub agent's transcript
        self.assertEqual(len(later_parent_turn["messages"]), 3)
        self.assertEqual(later_parent_turn["messages"][-1].role, "tool")

    def test_sub_agent_cannot_delegate_again(self):
        provider = ScriptedProvider(
            tool_reply("delegate", {"task": "outer"}),
            text_reply("inner"),
            text_reply("ok"),
        )
        delegator = Delegator(provider, [add])
        Agent(provider, Registry([add, delegator.tool()])).run("go")
        # `delegate` is offered to the parent but never to the sub agent
        self.assertIn("delegate", provider.calls[0]["tools"])
        self.assertNotIn("delegate", provider.calls[1]["tools"])

    def test_usage_is_tracked_separately_from_the_parent(self):
        provider = ScriptedProvider(
            tool_reply("delegate", {"task": "work"}),
            text_reply("sub answer", Usage(100, 40)),
            text_reply("parent answer"),
        )
        delegator = Delegator(provider, [add])
        parent = Agent(provider, Registry([delegator.tool()]))
        result = parent.run("go")

        self.assertEqual(delegator.runs, 1)
        self.assertEqual(delegator.usage.input_tokens, 100)
        self.assertEqual(delegator.usage.output_tokens, 40)
        # the parent's own usage does not include the sub agent's 140
        self.assertEqual(result.usage.total_tokens, 30)

    def test_sub_agent_tool_failure_does_not_kill_the_parent(self):
        provider = ScriptedProvider(
            tool_reply("delegate", {"task": "try it"}),
            tool_reply("boom", {}, call_id="c2"),
            text_reply("I could not do it"),  # sub agent reports the failure
            text_reply("understood"),
        )
        delegator = Delegator(provider, [boom])
        result = Agent(provider, Registry([boom, delegator.tool()])).run("go")
        self.assertEqual(result.steps[0].tool_results[0].content, "I could not do it")
        self.assertTrue(result.ok)

    def test_empty_sub_answer_is_an_error_result_not_a_crash(self):
        # sub agent burns its whole budget on tool calls and never answers
        provider = ScriptedProvider(
            tool_reply("delegate", {"task": "loop"}),
            tool_reply("add", {"a": 1, "b": 1}, call_id="c2"),
            tool_reply("add", {"a": 1, "b": 1}, call_id="c3"),
            text_reply("parent gave up too"),
        )
        delegator = Delegator(provider, [add], max_steps=2)
        result = Agent(provider, Registry([add, delegator.tool()])).run("go")
        first = result.steps[0].tool_results[0]
        self.assertTrue(first.is_error)
        self.assertIn("without an answer", first.content)

    def test_run_raises_when_the_sub_agent_produces_nothing(self):
        provider = ScriptedProvider(tool_reply("add", {"a": 1, "b": 1}))
        delegator = Delegator(provider, [add], max_steps=1)
        with self.assertRaises(DelegationError):
            delegator.run("nothing to say")

    def test_tool_schema_describes_one_required_task(self):
        spec = Delegator(ScriptedProvider(), [add]).tool().spec()
        self.assertEqual(spec["name"], "delegate")
        self.assertEqual(spec["parameters"]["required"], ["task"])
        self.assertIn("sub agent", spec["description"])

    def test_sub_agent_events_are_tagged_with_depth(self):
        events = []
        provider = ScriptedProvider(
            tool_reply("delegate", {"task": "work"}),
            text_reply("sub answer"),
            text_reply("parent answer"),
        )
        delegator = Delegator(provider, [add], on_event=events.append)
        Agent(provider, Registry([delegator.tool()]), on_event=events.append).run("go")

        kinds = [e["type"] for e in events]
        self.assertIn("subagent_start", kinds)
        self.assertIn("subagent_end", kinds)
        nested = [e for e in events if e.get("depth") == 1]
        self.assertTrue(nested)
        self.assertTrue(all("depth" not in e for e in events if e["type"] == "subagent_start"))

    def test_make_delegate_tool_is_the_shortcut(self):
        provider = ScriptedProvider(text_reply("answer"))
        item = make_delegate_tool(provider, [add])
        self.assertEqual(item.name, "delegate")
        self.assertEqual(item(task="anything"), "answer")


if __name__ == "__main__":
    unittest.main()
