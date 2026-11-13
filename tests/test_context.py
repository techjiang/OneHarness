import unittest

from oneharness import context, tokens
from oneharness.agent import Agent
from oneharness.errors import ConfigError, ProviderError
from oneharness.providers.base import Provider
from oneharness.tools import tool
from oneharness.types import Completion, Message, ToolCall, Usage


def text_reply(text, usage=(10, 5)):
    return Completion(message=Message.assistant(text), usage=Usage(*usage))


def tool_reply(name, arguments, call_id="c1"):
    call = ToolCall.from_raw(call_id=call_id, name=name, arguments=arguments)
    return Completion(message=Message.assistant("", [call]), usage=Usage(10, 5))


class ScriptedProvider(Provider):
    """Records every request so a test can see what went on the wire."""

    name = "scripted"

    def __init__(self, *completions):
        super().__init__(model="scripted-1")
        self.script = list(completions)
        self.sent = []

    def complete(self, messages, tools=None):
        self.sent.append(list(messages))
        if not self.script:
            raise AssertionError("provider called more times than scripted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class OverflowThenOk(ScriptedProvider):
    """Rejects anything over a real token count, the way an endpoint does."""

    def __init__(self, limit, *completions):
        super().__init__(*completions)
        self.limit = limit

    def complete(self, messages, tools=None):
        self.sent.append(list(messages))
        if context.estimate(messages) > self.limit:
            raise ProviderError(
                "request to provider failed",
                status=400,
                body=(
                    '{"error":{"message":"prompt is too long: '
                    f'{context.estimate(messages)} tokens > {self.limit} maximum"}}'
                ),
            )
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@tool
def read_file(path: str) -> str:
    """Read a file."""
    return "x" * 4000


class EstimateTests(unittest.TestCase):
    def test_estimate_counts_content_arguments_and_schemas(self):
        message = Message.tool("c1", "a" * 40, "read_file")
        content_only = tokens.count_message(message)
        self.assertEqual(context.estimate([message]), content_only)
        call = Message.assistant(
            "", [ToolCall.from_raw(call_id="c1", name="read_file", arguments={"path": "a"})]
        )
        self.assertEqual(context.estimate([call]), tokens.count_message(call))
        # The tool schemas ride along on every request and used to be counted
        # as zero, so a budget never saw them.
        specs = [{"name": "read_file", "description": "Read a file.", "parameters": {}}]
        self.assertEqual(
            context.estimate([message], specs), content_only + tokens.count_specs(specs)
        )

    def test_an_empty_message_list_is_zero_but_a_string_is_never_zero(self):
        self.assertEqual(context.estimate([]), 0)
        self.assertEqual(tokens.count("a"), 1)
        self.assertEqual(tokens.count(""), 0)


class FitTests(unittest.TestCase):
    def test_no_budget_means_no_change(self):
        messages = [Message.user("hi"), Message.tool("c1", "x" * 10_000, "read_file")]
        fitted, trim = context.fit(messages, 0)
        self.assertEqual(fitted, messages)
        self.assertFalse(trim.changed)

    def test_a_list_that_fits_is_untouched(self):
        messages = [Message.user("hi")]
        fitted, trim = context.fit(messages, 1000)
        self.assertIs(fitted[0], messages[0])
        self.assertFalse(trim.changed)

    def test_oldest_tool_results_are_elided_first(self):
        messages = [
            Message.user("look"),
            Message.assistant(
                "", [ToolCall.from_raw(call_id="c1", name="read_file", arguments={})]
            ),
            Message.tool("c1", "old" * 1000, "read_file"),
            Message.assistant(
                "", [ToolCall.from_raw(call_id="c2", name="read_file", arguments={})]
            ),
            Message.tool("c2", "new" * 1000, "read_file"),
        ]
        # The budget is in the new estimate's units: eliding the first result is
        # enough, so the second one must survive untouched.
        fitted, trim = context.fit(messages, 1400)
        self.assertEqual(trim.elided, 1)
        self.assertIn("elided", fitted[2].content)
        self.assertEqual(fitted[4].content, messages[4].content)  # newest kept intact
        self.assertLessEqual(trim.after, trim.budget)

    def test_assistant_reasoning_is_not_dropped_before_results(self):
        messages = [
            Message.user("go"),
            Message.assistant("important reasoning" * 10),
            Message.tool("c1", "x" * 8000, "read_file"),
        ]
        fitted, trim = context.fit(messages, 500)
        self.assertEqual(fitted[1].content, messages[1].content)
        self.assertGreater(trim.elided, 0)

    def test_multiple_calls_in_one_turn_are_not_orphaned(self):
        """The shape that produced an HTTP 400: three calls, one assistant turn.

        `test_a_call_is_never_separated_from_its_results` above covers one call
        per turn, which happened to delete cleanly. With three calls in one
        turn, the trimmer stepped over the assistant message and left three
        results answering nothing, and OpenAI rejects that outright:

            HTTP 400: messages with role 'tool' must be a response to a
            preceding message with 'tool_calls'

        The budget matters: at 124 the three results are already elided, so the
        first round to go is the one that carries the calls, and every budget
        from there until the next round fits lands on this shape. A budget that
        is wrong only inside a narrow window is still a request that never
        leaves the machine.
        """
        messages = [Message.system("be brief"), Message.user("read all three")]
        messages.append(
            Message.assistant(
                "",
                [
                    ToolCall.from_raw(call_id=f"c{index}", name="read_file", arguments={})
                    for index in range(3)
                ],
            )
        )
        messages.append(Message.assistant("plan: compare them, starting with the parser"))
        for index in range(3):
            messages.append(Message.tool(f"c{index}", "z" * 3000, "read_file"))
        messages.append(Message.user("ok next"))

        fitted, trim = context.fit(messages, 124)

        self.assertGreater(trim.dropped, 0)
        # The whole group goes: the prose, the turn that declares the calls and
        # the three results behind it. Anything less leaves a request that the
        # endpoint rejects.
        self.assertEqual(fitted[1].role, "user")
        self.assertEqual([m for m in fitted if m.role == "tool"], [])
        self.assertEqual([m for m in fitted if m.tool_calls], [])

    def test_a_round_that_cannot_go_whole_does_not_go_in_pieces(self):
        """Prose with no calls is its own round; the results must not be split off."""
        messages = [Message.user("go"), Message.assistant("x" * 400)]
        messages.append(
            Message.assistant(
                "",
                [
                    ToolCall.from_raw(call_id=f"c{index}", name="read_file", arguments={})
                    for index in range(2)
                ],
            )
        )
        for index in range(2):
            messages.append(Message.tool(f"c{index}", "y" * 4000, "read_file"))

        fitted, trim = context.fit(messages, 40)

        self.assertGreater(trim.dropped, 0)
        self.assertEqual([m.role for m in fitted if m.role == "tool"], [])
        calls = {call.id for message in fitted for call in message.tool_calls}
        self.assertEqual(calls, set())

    def test_the_newest_user_turn_is_never_the_thing_that_gets_dropped(self):
        messages = [Message.user("old question"), Message.tool("c0", "z" * 8000, "read_file")]
        messages.append(Message.user("the real task"))
        fitted, trim = context.fit(messages, 30)
        self.assertEqual(fitted[-1].content, "the real task")
        self.assertGreater(trim.dropped, 0)

    def test_the_transcript_itself_is_never_mutated(self):
        messages = [Message.user("go"), Message.tool("c1", "x" * 8000, "read_file")]
        original = list(messages)
        context.fit(messages, 100)
        self.assertEqual(messages, original)

    def test_dropping_keeps_system_and_newest_user_turn(self):
        messages = [Message.system("be brief"), Message.user("old question")]
        for index in range(6):
            messages.append(Message.assistant(f"thought {index} " * 40))
            messages.append(Message.tool(f"c{index}", "y" * 800, "read_file"))
        messages.append(Message.user("the real task"))

        fitted, trim = context.fit(messages, 400)
        self.assertEqual(fitted[0].content, "be brief")
        self.assertEqual(fitted[-1].content, "the real task")
        self.assertGreater(trim.dropped, 0)

    def test_a_call_is_never_separated_from_its_results(self):
        messages = [Message.user("go")]
        for index in range(5):
            messages.append(
                Message.assistant(
                    "",
                    [
                        ToolCall.from_raw(
                            call_id=f"c{index}", name="read_file", arguments={"path": "a"}
                        )
                    ],
                )
            )
            messages.append(Message.tool(f"c{index}", "z" * 2000, "read_file"))
        messages.append(Message.user("carry on"))

        fitted, trim = context.fit(messages, 100)
        self.assertGreater(trim.dropped, 0)
        ids = {call.id for message in fitted for call in message.tool_calls}
        orphaned = [m for m in fitted if m.role == "tool" and m.tool_call_id not in ids]
        self.assertEqual(orphaned, [])  # an orphan tool message is an invalid request

    def test_the_first_non_system_turn_stays_a_user_turn(self):
        messages = [Message.system("be brief"), Message.user("the original task")]
        for index in range(6):
            messages.append(Message.assistant(f"thought {index} " * 40))
            messages.append(Message.tool(f"c{index}", "y" * 800, "read_file"))
        messages.append(Message.user("and now this"))

        fitted, _ = context.fit(messages, 150)
        self.assertEqual(fitted[0].role, "system")
        self.assertEqual(fitted[1].role, "user")  # Anthropic rejects a leading assistant turn

    def test_trim_as_dict_is_json_ready(self):
        message = Message.user("x" * 8000)
        _, trim = context.fit([message], 100)
        payload = trim.as_dict()
        self.assertEqual(payload["tokens_before"], context.estimate([message]))
        self.assertEqual(payload["budget"], 100)
        self.assertIn("dropped", payload)
        self.assertEqual(payload["tools"], 0)


@tool
def many_tools(path: str) -> str:
    """A tool whose schema is a real part of every request."""
    return path


class SchemaTests(unittest.TestCase):
    """The tool schemas are payload, and they used to be free."""

    def test_a_budget_that_only_fits_without_the_tools_is_not_fitting(self):
        specs = [many_tools.spec()] * 20
        message = Message.user("hello")
        self.assertLess(context.estimate([message]), 100)
        self.assertGreater(context.estimate([message], specs), 1000)

    def test_the_agent_counts_the_schemas_it_sends(self):
        specs = [read_file.spec()] * 20
        before = context.estimate([Message.user("hi")], specs)
        after = context.estimate([Message.user("hi")])
        self.assertGreater(before, after)

    def test_trim_reports_the_schema_cost(self):
        _, trim = context.fit([Message.user("x")], 1_000_000, specs=[read_file.spec()] * 10)
        self.assertGreater(trim.tools, 0)
        self.assertEqual(trim.as_dict()["tools"], trim.tools)


class UnmetTests(unittest.TestCase):
    """Trimming has a floor, and pretending otherwise is the bug."""

    def test_tiny_results_are_not_replaced_by_a_longer_stub(self):
        # Eliding "ok" into a 130-character explanation makes the request bigger.
        messages = [Message.user("go")]
        for index in range(8):
            messages.append(
                Message.assistant(
                    "", [ToolCall.from_raw(call_id=f"c{index}", name="read_file", arguments={})]
                )
            )
            messages.append(Message.tool(f"c{index}", "ok", "read_file"))
        outgoing, trim = context.fit(messages, 60)
        self.assertEqual(trim.elided, 0)
        # The stub swap is what is being tested, so the check is on the results
        # that survived: a small result that got "elided" into a longer notice
        # would be the bug, whether or not whole rounds were dropped as well.
        for message in outgoing:
            if message.role == "tool":
                self.assertEqual(message.content, "ok")
        self.assertGreater(trim.after, 0)

    def test_what_the_trimmer_cannot_reach_is_reported(self):
        messages = [Message.system("be brief"), Message.user("x" * 4000)]
        _, trim = context.fit(messages, 100)
        self.assertEqual(trim.dropped, 0)  # the prompt is the only thing left
        self.assertGreater(trim.unmet, 0)
        self.assertEqual(trim.as_dict()["unmet"], trim.after - 100)

    def test_a_trim_that_reaches_the_budget_reports_nothing_left(self):
        messages = [Message.user("go"), Message.tool("c1", "y" * 8000, "read_file")]
        _, trim = context.fit(messages, 500)
        self.assertEqual(trim.unmet, 0)

    def test_a_budget_smaller_than_the_schemas_says_so(self):
        specs = [read_file.spec()] * 20
        _, trim = context.fit([Message.user("hi")], 100, specs=specs)
        self.assertGreater(trim.tools, trim.budget)
        self.assertGreater(trim.unmet, 0)


class LegacyComparisonTests(unittest.TestCase):
    """The old rule is kept only so the change can be measured."""

    def test_chinese_was_underestimated_by_a_factor_of_two(self):
        text = "请帮我把这个仓库里的网络层重构一下，先看 transport 的错误分类。"
        old = context.legacy_estimate([Message.user(text)])
        new = context.estimate([Message.user(text)])
        self.assertGreater(new, old * 2)

    def test_indentation_was_overestimated_by_a_factor_of_twenty(self):
        text = "    " * 2000
        old = context.legacy_estimate([Message.user(text)])
        new = context.estimate([Message.user(text)])
        self.assertGreater(old, new * 10)

    def test_english_is_where_the_old_rule_worked(self):
        text = "the quick brown fox jumps over the lazy dog " * 20
        old = context.legacy_estimate([Message.user(text)])
        new = context.estimate([Message.user(text)])
        self.assertLess(abs(new - old) / old, 0.25)


class OverflowTests(unittest.TestCase):
    def test_known_overflow_bodies_are_recognised(self):
        bodies = [
            "prompt is too long: 208310 tokens > 200000 maximum",
            "This model's maximum context length is 128000 tokens",
            '{"error":{"code":"context_length_exceeded"}}',
            "Input is too long for requested model",
        ]
        for body in bodies:
            with self.subTest(body=body):
                self.assertTrue(context.is_overflow(ProviderError("x", status=400, body=body)))

    def test_a_wire_format_400_is_not_an_overflow(self):
        exc = ProviderError("x", status=400, body="Unknown name 'messages' at 'request'")
        self.assertFalse(context.is_overflow(exc))

    def test_auth_and_server_errors_are_not_overflows(self):
        too_long = "prompt is too long"
        self.assertFalse(context.is_overflow(ProviderError("x", status=401, body=too_long)))
        self.assertFalse(context.is_overflow(ProviderError("x", status=500, body=too_long)))

    def test_shrink_is_proportional_and_floored(self):
        self.assertEqual(context.shrink(10_000, 10_000), 6000)
        self.assertEqual(context.shrink(0, 10_000), 6000)
        self.assertEqual(context.shrink(300, 300), context.MIN_BUDGET)


class AgentBudgetTests(unittest.TestCase):
    def test_budget_off_sends_the_untrimmed_list(self):
        provider = ScriptedProvider(tool_reply("read_file", {"path": "a"}), text_reply("done"))
        result = Agent(provider, [read_file], max_steps=3).run("go")
        self.assertIsNone(result.trim)
        self.assertEqual(len(provider.sent[1]), 3)  # user + assistant call + result

    def test_a_configured_budget_trims_and_reports(self):
        provider = ScriptedProvider(
            tool_reply("read_file", {"path": "a"}),
            tool_reply("read_file", {"path": "b"}, call_id="c2"),
            text_reply("done"),
        )
        agent = Agent(provider, [read_file], max_steps=4, context_budget=200)
        result = agent.run("go")

        self.assertEqual(result.text, "done")
        self.assertIsNotNone(result.trim)
        self.assertGreater(result.trim["elided"], 0)
        self.assertLessEqual(result.trim["tokens_after"], 200)
        self.assertIn("trim", result.as_dict())
        # the outgoing request was trimmed, the transcript was not
        self.assertIn("elided", provider.sent[-1][2].content)
        self.assertEqual(len(result.messages), 6)  # transcript keeps every turn

    def test_an_overflow_triggers_one_honest_shrink(self):
        # The first request is small; the second carries a 500-token file and is
        # refused against a 300-token window. That is the shape of a real
        # overrun: it happens mid-loop, not on the first request.
        provider = OverflowThenOk(300, tool_reply("read_file", {"path": "a"}), text_reply("done"))
        result = Agent(provider, [read_file], max_steps=3).run("look at a")

        self.assertEqual(result.text, "done")
        self.assertEqual(len(provider.sent), 3)  # refused once, then succeeded
        self.assertIsNotNone(result.trim)
        self.assertGreater(result.trim["elided"], 0)

    def test_the_retry_budgets_even_when_none_was_configured(self):
        events = []
        provider = OverflowThenOk(300, tool_reply("read_file", {"path": "a"}), text_reply("done"))
        agent = Agent(provider, [read_file], on_event=events.append)
        result = agent.run("look at a")

        self.assertEqual(result.text, "done")
        self.assertIn("context_overflow", [event["type"] for event in events])
        self.assertIsNotNone(result.trim)  # a retry always reports what it cut

    def test_a_prompt_that_is_too_long_on_its_own_is_not_retried_forever(self):
        # Nothing can be elided from a single oversized user message: dropping
        # the task to fit the window would be answering a different question.
        provider = OverflowThenOk(100, text_reply("never"))
        with self.assertRaises(ProviderError):
            Agent(provider).run("go" * 1000)
        self.assertEqual(len(provider.sent), 1)  # no retry with an identical payload

    def test_it_gives_up_instead_of_looping(self):
        # The window is below what even the elided result costs, so no amount of
        # shrinking can help. The loop must stop on its own, not lean on --max-steps.
        #
        # One request, not five: the check is "did the smaller budget actually
        # remove anything", and here it cannot, so retrying is provably useless.
        # Spending four more requests to discover that would be the old behaviour.
        provider = OverflowThenOk(5, tool_reply("read_file", {"path": "a"}), text_reply("never"))
        with self.assertRaises(ProviderError) as ctx:
            Agent(provider, [read_file], max_steps=4).run("look at a")

        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(len(provider.sent), 1)

    def test_a_wire_format_error_is_not_retried(self):
        provider = ScriptedProvider(
            ProviderError("bad", status=400, body="Unknown name 'messages'")
        )
        with self.assertRaises(ProviderError):
            Agent(provider, max_steps=2).run("go")
        self.assertEqual(len(provider.sent), 1)

    def test_bad_configuration_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            Agent(ScriptedProvider(), context_budget=-1)


class BudgetConfigTests(unittest.TestCase):
    def test_negative_budget_in_env_is_an_error(self):
        import os
        from unittest import mock

        from oneharness.config import load_config

        with (
            mock.patch.dict(os.environ, {"ONEHARNESS_CONTEXT_BUDGET": "-5"}, clear=True),
            self.assertRaises(ConfigError),
        ):
            load_config()

    def test_budget_from_env(self):
        import os
        from unittest import mock

        from oneharness.config import load_config

        with mock.patch.dict(os.environ, {"ONEHARNESS_CONTEXT_BUDGET": "8000"}, clear=True):
            self.assertEqual(load_config().context_budget, 8000)


if __name__ == "__main__":
    unittest.main()
