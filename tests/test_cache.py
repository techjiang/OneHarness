"""Prompt caching: what the providers report, and what the harness does about it.

Two different problems, and only one of them is a harness decision:

* every provider already tells you how much of the prompt came back from its
  cache, and every provider spells it differently. Before v1.3 all five
  adapters dropped those fields on the floor, so `usage.input_tokens` on
  Anthropic counted only the uncached part -- a run that read 2000 cached tokens
  and 12 fresh ones looked like a 12-token prompt.
* Anthropic and Bedrock cache only where a breakpoint says so. OpenAI, Gemini
  and DeepSeek cache repeated prefixes without being asked, so the only correct
  behaviour there is to report what came back.
"""

import unittest

from oneharness import context
from oneharness.agent import Agent
from oneharness.providers.anthropic import AnthropicProvider, _place_breakpoints
from oneharness.providers.bedrock import BedrockProvider
from oneharness.providers.bedrock import _place_breakpoints as _bedrock_points
from oneharness.providers.gemini import GeminiProvider
from oneharness.providers.openai import OpenAIProvider
from oneharness.tools import tool
from oneharness.types import Completion, Message, ToolCall, Usage


def openai_usage(**overrides):
    raw = {"prompt_tokens": 1000, "completion_tokens": 50}
    raw.update(overrides)
    return raw


class NormalisationTests(unittest.TestCase):
    """One `Usage`, five wire formats, and no double counting anywhere."""

    def test_openai_cached_is_a_slice_of_the_prompt(self):
        provider = OpenAIProvider("gpt", api_key="sk", base_url="http://127.0.0.1:1")
        completion = provider._decode(
            {
                "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                "usage": openai_usage(prompt_tokens_details={"cached_tokens": 768}),
            }
        )
        usage = completion.usage
        self.assertEqual(usage.input_tokens, 1000)
        self.assertEqual(usage.cached_tokens, 768)
        self.assertEqual(usage.uncached_tokens, 232)
        self.assertAlmostEqual(usage.cache_hit_rate, 0.768)

    def test_deepseek_spells_it_differently_and_still_lands(self):
        provider = OpenAIProvider("deepseek-chat", api_key="sk", base_url="http://127.0.0.1:1")
        completion = provider._decode(
            {
                "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 500,
                    "completion_tokens": 10,
                    "prompt_cache_hit_tokens": 384,
                },
            }
        )
        self.assertEqual(completion.usage.cached_tokens, 384)
        self.assertEqual(completion.usage.input_tokens, 500)

    def test_anthropic_input_tokens_excludes_the_cache(self):
        """The one adapter that has to add, and therefore the one worth testing."""
        provider = AnthropicProvider("claude", api_key="sk", base_url="http://127.0.0.1:1")
        completion = provider._decode(
            {
                "content": [{"type": "text", "text": "hi"}],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 12,
                    "cache_read_input_tokens": 2000,
                    "cache_creation_input_tokens": 40,
                    "output_tokens": 5,
                },
            }
        )
        usage = completion.usage
        # 2052, not 12: the prompt really was 2052 tokens, and the fact that
        # 2000 of them were cheap is a separate number.
        self.assertEqual(usage.input_tokens, 2052)
        self.assertEqual(usage.cached_tokens, 2000)
        self.assertEqual(usage.cache_write_tokens, 40)
        self.assertEqual(usage.total_tokens, 2057)

    def test_gemini_cached_is_also_a_slice(self):
        provider = GeminiProvider("gem", api_key="k", base_url="http://127.0.0.1:1")
        completion = provider._decode(
            {
                "candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}],
                "usageMetadata": {
                    "promptTokenCount": 900,
                    "candidatesTokenCount": 20,
                    "cachedContentTokenCount": 640,
                },
            }
        )
        self.assertEqual(completion.usage.input_tokens, 900)
        self.assertEqual(completion.usage.cached_tokens, 640)

    def test_bedrock_adds_like_anthropic(self):
        usage = BedrockProvider._usage(
            {
                "inputTokens": 30,
                "outputTokens": 4,
                "cacheReadInputTokens": 1000,
                "cacheWriteInputTokens": 8,
            }
        )
        self.assertEqual(usage.input_tokens, 1038)
        self.assertEqual(usage.cached_tokens, 1000)
        self.assertEqual(usage.cache_write_tokens, 8)

    def test_bedrock_without_cache_fields_changes_nothing(self):
        """Old Converse responses have none of these keys; nothing should move."""
        usage = BedrockProvider._usage({"inputTokens": 30, "outputTokens": 4})
        self.assertEqual(usage.input_tokens, 30)
        self.assertEqual(usage.cached_tokens, 0)
        self.assertEqual(usage.input_tokens, 30 + usage.cached_tokens)

    def test_zero_cache_is_absent_from_json_not_zero(self):
        """`cache: {read: 0}` reads as a broken cache; a missing key reads as none."""
        self.assertNotIn("cache", Usage(10, 2).as_dict())
        self.assertIn("cache", Usage(10, 2, cached_tokens=4).as_dict())

    def test_adding_usage_adds_the_cache_half(self):
        total = Usage(100, 10, cached_tokens=80) + Usage(100, 10, cached_tokens=60)
        self.assertEqual(total.input_tokens, 200)
        self.assertEqual(total.cached_tokens, 140)
        self.assertEqual(total.uncached_tokens, 60)


class BreakpointTests(unittest.TestCase):
    """Where the breakpoints go, and why there and not somewhere else."""

    def payload(self, **overrides):
        base = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "system": "be terse",
            "tools": [{"name": "a"}, {"name": "b"}],
        }
        base.update(overrides)
        return base

    def test_three_breakpoints_tools_system_and_the_last_turn(self):
        payload = self.payload()
        _place_breakpoints(payload)
        self.assertIn("cache_control", payload["tools"][-1])
        self.assertNotIn("cache_control", payload["tools"][0])
        self.assertEqual(payload["system"][0]["cache_control"], {"type": "ephemeral"})
        last = payload["messages"][-1]["content"][-1]
        self.assertEqual(last["cache_control"]["type"], "ephemeral")

    def test_system_becomes_a_block_list_only_when_caching(self):
        """A plain string stays a plain string: `cache_control` needs a block."""
        provider = AnthropicProvider("m", api_key="k", base_url="http://x.test")
        plain = provider._payload([Message.system("be terse"), Message.user("hi")], None)
        self.assertEqual(plain["system"], "be terse")
        cached = AnthropicProvider("m", api_key="k", base_url="http://x.test", cache=True)
        marked = cached._payload([Message.system("be terse"), Message.user("hi")], None)
        self.assertIsInstance(marked["system"], list)

    def test_a_tool_result_turn_is_marked_on_its_last_block(self):
        messages = [
            Message.tool("c1", "42", "add"),
            Message.tool("c2", "43", "add"),
        ]
        caching = AnthropicProvider("m", api_key="k", base_url="http://x.test", cache=True)
        payload = caching._payload(messages, None)
        blocks = payload["messages"][-1]["content"]
        self.assertIn("cache_control", blocks[-1])
        self.assertNotIn("cache_control", blocks[0])

    def test_no_breakpoint_without_the_flag(self):
        provider = AnthropicProvider("m", api_key="k", base_url="http://x.test")
        payload = provider._payload([Message.user("hi")], [{"name": "a", "parameters": {}}])
        self.assertNotIn("cache_control", payload["tools"][-1])
        self.assertEqual(payload["messages"][-1]["content"], "hi")

    def test_bedrock_uses_cache_point_blocks(self):
        payload = {
            "messages": [{"role": "user", "content": [{"text": "hi"}]}],
            "system": [{"text": "be terse"}],
            "toolConfig": {"tools": [{"toolSpec": {"name": "a"}}]},
        }
        _bedrock_points(payload)
        self.assertEqual(payload["toolConfig"]["tools"][-1], {"cachePoint": {"type": "default"}})
        self.assertEqual(payload["system"][-1], {"cachePoint": {"type": "default"}})
        self.assertEqual(payload["messages"][-1]["content"][-1]["cachePoint"]["type"], "default")

    def test_bedrock_without_tools_still_marks_system_and_turn(self):
        payload = {
            "messages": [{"role": "user", "content": [{"text": "hi"}]}],
            "system": [{"text": "s"}],
        }
        _bedrock_points(payload)
        self.assertEqual(len(payload["system"]), 2)
        self.assertNotIn("toolConfig", payload)


class CacheFlagPortabilityTests(unittest.TestCase):
    """`--cache` has to be safe to leave on in a shared config file.

    It was not: the flag went straight into the constructor, and every preset
    whose cache is implicit rejected the keyword. A `cache = true` in
    `oneharness.toml` then crashed the run on OpenAI -- which is worse than not
    caching, because the user asked for something harmless.
    """

    def test_every_preset_accepts_the_flag(self):
        from oneharness.providers import PRESETS, create_provider

        for preset in sorted(PRESETS):
            options = {"cache": True}
            if PRESETS[preset].get("auth") == "aws":
                from oneharness.sigv4 import Credentials

                options["credentials"] = Credentials("AKIA", "secret")
            elif PRESETS[preset].get("requires_model"):
                options.update(api_key="sk", base_url="https://x.openai.azure.com", model="dep")
            elif preset != "ollama":
                options["api_key"] = "sk"
            with self.subTest(preset=preset):
                provider = create_provider(preset, **options)
                self.assertTrue(provider.cache)

    def test_an_implicit_provider_ignores_it_on_the_wire(self):
        """Accepted, but nothing is added: there is no breakpoint field to add."""
        provider = OpenAIProvider("gpt", api_key="sk", base_url="http://x.test", cache=True)
        payload = provider._payload([Message.user("hi")], [{"name": "a", "parameters": {}}])
        self.assertEqual(payload["messages"], [{"role": "user", "content": "hi"}])
        self.assertNotIn("cache_control", payload["tools"][0])


class CacheModeTests(unittest.TestCase):
    def test_explicit_providers_are_the_two_that_need_breakpoints(self):
        self.assertEqual(AnthropicProvider.cache_mode, "explicit")
        self.assertEqual(BedrockProvider.cache_mode, "explicit")
        self.assertEqual(OpenAIProvider.cache_mode, "automatic")
        self.assertEqual(GeminiProvider.cache_mode, "automatic")


@tool
def ping() -> str:
    """Answer."""
    return "pong"


class EstimateAgainstProviderTests(unittest.TestCase):
    """The estimator, checked against a counter that is not its own."""

    def test_the_step_records_what_it_sent(self):
        from oneharness.providers.base import Provider

        class Scripted(Provider):
            name = "scripted"

            def __init__(self):
                super().__init__(model="m")
                self.rounds = 0

            def complete(self, messages, tools=None):
                self.rounds += 1
                if self.rounds == 1:
                    call = ToolCall.from_raw(call_id="c1", name="ping", arguments={})
                    return Completion(message=Message.assistant("", [call]), usage=Usage(100, 5))
                return Completion(message=Message.assistant("done"), usage=Usage(120, 3))

        result = Agent(Scripted(), [ping]).run("go")
        self.assertEqual(len(result.steps), 2)
        for step in result.steps:
            self.assertGreater(step.prompt_tokens_estimated, 0)
        # the second request carries the tool result, so it must estimate larger
        self.assertGreater(
            result.steps[1].prompt_tokens_estimated, result.steps[0].prompt_tokens_estimated
        )

    def test_a_trimmed_step_estimates_the_trimmed_list(self):
        """The provider only ever saw the trimmed list, so that is what counts."""
        from oneharness.providers.base import Provider

        class Chatty(Provider):
            name = "chatty"

            def __init__(self):
                super().__init__(model="m")

            def complete(self, messages, tools=None):
                return Completion(message=Message.assistant("done"), usage=Usage(50, 2))

        history = [Message.tool(f"c{i}", "x " * 2000, "read_file") for i in range(6)]
        agent = Agent(Chatty(), context_budget=400)
        result = agent.run("go", history=history)
        sent = result.steps[0].prompt_tokens_estimated
        untrimmed = context.estimate(result.messages)
        self.assertLess(sent, untrimmed)
        self.assertLessEqual(sent, 400 + 200)  # the budget, plus the unavoidable parts


if __name__ == "__main__":
    unittest.main()
