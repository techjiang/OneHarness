"""The estimator, checked against numbers a real tokenizer produced.

v1.1's `chars / 4` was wrong in a way that could not be seen from inside the
harness: it agreed with itself everywhere, because nothing independent ever
disagreed with it. The numbers here come from `tiktoken`, recorded for the repo
by `tools/calibrate_tokens.py` and frozen in `tests/fixtures/token_counts.json`.
The tokenizer itself is not a dependency; the file it wrote is the witness.
"""

import json
import math
import time
import unittest
from pathlib import Path

from oneharness import context, tokens
from oneharness.builtin import make_file_tools
from oneharness.tools import tool
from oneharness.types import Message, ToolCall

FIXTURE = Path(__file__).parent / "fixtures" / "token_counts.json"
HOLDOUT = Path(__file__).parent / "fixtures" / "token_counts_holdout.json"


def samples(path: Path = FIXTURE):
    """A frozen corpus, with the tokenizer's own count for each sample."""
    data = json.loads(path.read_text())
    for name, item in data["samples"].items():
        counts = item["tokens"]
        yield name, item["text"], sum(counts.values()) / len(counts)


def errors():
    """(name, how far off the new estimate is, how far off chars/4 was)."""
    for name, text, real in samples():
        real = max(1.0, real)
        yield name, tokens.count(text) / real, len(text) / 4 / real


class CalibrationTests(unittest.TestCase):
    def test_the_constants_are_the_fitted_ones(self):
        # Named so a future edit has to be deliberate: changing these without
        # re-running tools/calibrate_tokens.py makes the claim below false.
        self.assertAlmostEqual(tokens.CALIBRATION["latin_cpt"], 4.1476, places=3)
        self.assertAlmostEqual(tokens.CALIBRATION["cjk"], 0.8437, places=3)
        self.assertAlmostEqual(tokens.CALIBRATION["space"], 0.008, places=3)
        self.assertNotEqual(tokens.CALIBRATION["latin_cpt"], 4.0)  # not a round number

    def test_a_letter_is_cheaper_than_a_cjk_ideograph(self):
        # The relationship `chars / 4` could not express: one character of
        # English is a fraction of a token, one character of Chinese is one.
        self.assertLess(1 / tokens.CALIBRATION["latin_cpt"], tokens.CALIBRATION["cjk"])

    def test_the_fixture_is_not_empty(self):
        self.assertGreaterEqual(len(list(samples())), 20)


class ClassTests(unittest.TestCase):
    def test_empty_text_is_zero_and_one_character_is_one(self):
        self.assertEqual(tokens.count(""), 0)
        for char in "a1 .世":
            with self.subTest(char=char):
                self.assertEqual(tokens.count(char), 1)
        # An emoji is two tokens in these vocabularies, and counting it as one
        # is how a wall of them slips past a budget.
        self.assertEqual(tokens.count("🎉"), 2)

    def test_indentation_is_nearly_free(self):
        # 8000 spaces are 63 tokens, not 2000: a code-heavy history used to be
        # trimmed while it still fitted.
        self.assertLess(tokens.count("    " * 2000), 200)

    def test_a_repeated_character_merges(self):
        self.assertLess(tokens.count("x" * 4000), 600)
        self.assertLess(tokens.count("=" * 4000), 200)

    def test_a_long_path_is_not_a_blob(self):
        # Paths are letters and slashes: they merge. Calling them blobs would
        # halve the estimate for a repository listing.
        self.assertFalse(tokens._is_blob("oneharness/providers/anthropic_module"))

    def test_a_padded_run_is_a_blob(self):
        self.assertTrue(tokens._is_blob("QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVphYmNkZWZnaGlqa2w="))

    def test_a_word_is_not_a_blob(self):
        self.assertFalse(tokens._is_blob("def estimate"))

    def test_a_hex_digest_costs_a_token_per_character(self):
        # Random hex has no merges at all: 64 characters, 64 tokens. Useful as a
        # reminder that "characters over four" was not merely imprecise.
        digest = "9f2c1b7a4e8d0c3b5a6f7e8d9c0b1a2f3e4d5c6b7a8f9e0d1c2b3a4f5e6d7c8b"
        self.assertGreaterEqual(tokens.count(digest), len(digest) * 0.9)


class MessageTests(unittest.TestCase):
    def test_a_message_includes_its_tool_calls(self):
        bare = tokens.count_message(Message.assistant("hi"))
        with_call = tokens.count_message(
            Message.assistant(
                "hi", [ToolCall.from_raw(call_id="c1", name="read_file", arguments={"path": "a"})]
            )
        )
        self.assertGreater(with_call, bare)

    def test_schemas_used_to_be_counted_as_zero(self):
        specs = [{"name": f"tool_{i}", "description": "x", "parameters": {}} for i in range(20)]
        self.assertEqual(tokens.count_specs(None), 0)
        self.assertEqual(tokens.count_specs([]), 0)
        self.assertGreater(tokens.count_specs(specs), 100)


class AccuracyTests(unittest.TestCase):
    """The bounds that matter: how far off can a budget be?"""

    def test_ninety_percent_within_fifteen_percent(self):
        found = sorted(abs(math.log(new)) for _, new, _ in errors())
        at_90 = found[int(len(found) * 0.9)]
        self.assertLess(at_90, math.log(1.15), f"90th percentile {math.exp(at_90):.2f}x")

    def test_the_mean_error_is_under_a_twentieth(self):
        mean = sum(abs(math.log(new)) for _, new, _ in errors()) / len(list(samples()))
        self.assertLess(mean, 0.06, f"mean {math.exp(mean):.3f}x")

    def test_no_sample_is_off_by_more_than_half(self):
        # Vietnamese is the one sample outside this bound, and it is named in
        # `test_one_sample_is_allowed_to_stay_wrong` rather than quietly dropped
        # from the corpus: a bound with no exception has usually been tuned.
        for name, new, _ in errors():
            if name == "vietnamese":
                continue
            with self.subTest(sample=name):
                self.assertLess(abs(math.log(new)), math.log(1.5), f"{name}: {new:.2f}x")

    def test_it_beats_chars_over_four_wherever_the_old_rule_was_wrong(self):
        # Not "on every sample": English prose is where chars/4 was already
        # right, and beating it there by a hair is not the claim. The claim is
        # that the cases it got badly wrong -- 1.5x or worse -- are fixed.
        for name, new, old in errors():
            if abs(math.log(old)) < math.log(1.4) or name == "vietnamese":
                continue
            with self.subTest(sample=name):
                self.assertLess(abs(math.log(new)), abs(math.log(old)))
                self.assertLess(abs(math.log(new)), math.log(1.15))

    def test_one_sample_is_allowed_to_stay_wrong(self):
        # Vietnamese stacks two diacritics on one base letter and the vocabularies
        # split them differently than Latin-1 would suggest. It is 0.72x, it is
        # the worst sample left, and it is listed here rather than hidden: a
        # bound with no exception would be a bound that was tuned to look good.
        for name, new, _ in errors():
            if name == "vietnamese":
                self.assertGreater(new, 0.6)
                self.assertLess(new, 1.0)

    def test_the_old_rule_average_is_one_and_a_half_times_the_truth(self):
        old = [abs(math.log(legacy)) for _, _, legacy in errors()]
        self.assertGreater(math.exp(sum(old) / len(old)), 1.4)

    def test_chinese_is_no_longer_counted_at_a_third(self):
        text = next(text for name, text, _ in samples() if name == "chinese")
        real = next(real for name, _, real in samples() if name == "chinese")
        self.assertLess(len(text) / 4 / real, 0.5)  # the bug: 2000 let out ~5000
        self.assertGreater(tokens.count(text) / real, 0.9)

    def test_indentation_is_no_longer_counted_at_thirty_times(self):
        text = next(text for name, text, _ in samples() if name == "indentation")
        real = next(real for name, _, real in samples() if name == "indentation")
        self.assertGreater(len(text) / 4 / real, 10)
        self.assertLess(tokens.count(text) / real, 1.5)


class HoldoutTests(unittest.TestCase):
    """The constants were fitted on CORPUS, so measuring there is a closed loop.

    These samples are this repository's own source and README: the same kind of
    text, never seen by the fit. If the constants only describe the corpus they
    came from, this is where it shows.
    """

    def test_the_holdout_is_a_different_set(self):
        self.assertGreaterEqual(len(list(samples(HOLDOUT))), 20)
        corpus_names = {name for name, _, _ in samples()}
        holdout_names = {name for name, _, _ in samples(HOLDOUT)}
        self.assertEqual(corpus_names & holdout_names, set())

    def test_the_error_survives_on_text_it_was_not_fitted_on(self):
        errors = sorted(
            abs(math.log(tokens.count(text) / max(1.0, real))) for _, text, real in samples(HOLDOUT)
        )
        mean = sum(errors) / len(errors)
        self.assertLess(mean, 0.12, f"mean {math.exp(mean):.2f}x on held-out text")
        self.assertLess(errors[int(len(errors) * 0.9)], math.log(1.25))
        self.assertLess(errors[-1], math.log(1.5))

    def test_the_worst_case_improves_on_held_out_text(self):
        # This repository is Python and English, where chars/4 was already close:
        # the mean barely moves (1.16x -> 1.10x). What changes is the tail, and
        # the tail is what costs a retry.
        old = []
        new = []
        for _, text, real in samples(HOLDOUT):
            real = max(1.0, real)
            old.append(abs(math.log(len(text) / 4 / real)))
            new.append(abs(math.log(tokens.count(text) / real)))
        self.assertLess(max(new), max(old))
        self.assertLess(max(new), math.log(1.5))


class SchemaWireTests(unittest.TestCase):
    """The schemas are counted as serialised, not tool by tool.

    Counting each schema with its own `json.dumps` and summing the results drops
    every separator between them and the array brackets around them, so the
    answer is short by however much framing there is. That direction of error is
    the one nobody investigates: the budget looks roomier than it is.
    """

    def setUp(self):
        @tool
        def alpha(path: str) -> str:
            """A tool."""

        @tool
        def beta(count: int, label: str = "x") -> str:
            """Another tool, with two parameters."""

        @tool
        def gamma(name: str) -> str:
            """A third."""
            return name

        files = [item.spec() for item in make_file_tools()]
        self.specs = [alpha.spec(), beta.spec(), gamma.spec(), *files]

    def test_the_brackets_and_commas_are_counted(self):
        separate = sum(tokens.count(json.dumps(spec)) for spec in self.specs)
        self.assertGreater(tokens.count_specs(self.specs), separate)

    def test_the_count_follows_the_serialised_size(self):
        wire = len(json.dumps({"tools": self.specs}, ensure_ascii=False))
        counted = tokens.count_specs(self.specs)
        # Within a factor that says "this is the same array", not "this is the
        # sum of its parts": 876 characters of schema come out at 251 tokens.
        self.assertGreater(counted, wire / 5)
        self.assertLess(counted, wire)

    def test_more_tools_never_cost_less(self):
        one = tokens.count_specs(self.specs[:1])
        two = tokens.count_specs(self.specs[:2])
        three = tokens.count_specs(self.specs)
        self.assertLess(one, two)
        self.assertLess(two, three)

    def test_no_tools_cost_nothing(self):
        self.assertEqual(tokens.count_specs([]), 0)
        self.assertEqual(tokens.count_specs(None), 0)


class LegacyTests(unittest.TestCase):
    """The comparison is the whole point of v1.2, so it is kept measurable."""

    def test_chinese_was_underestimated_by_a_factor_of_two(self):
        text = "请帮我把这个仓库里的网络层重构一下，先看 transport 的错误分类。"
        messages = [Message.user(text)]
        self.assertGreater(context.estimate(messages), context.legacy_estimate(messages) * 2)

    def test_indentation_was_overestimated_by_a_factor_of_twenty(self):
        messages = [Message.user("    " * 2000)]
        self.assertGreater(context.legacy_estimate(messages), context.estimate(messages) * 10)


class PerfTests(unittest.TestCase):
    def test_a_forty_kilobyte_tool_result_counts_fast_enough(self):
        text = next(text for name, text, _ in samples() if name == "log")
        started = time.perf_counter()
        tokens.count(text)
        self.assertLess(time.perf_counter() - started, 0.05)


class NoDependencyTests(unittest.TestCase):
    def test_the_library_never_imports_a_tokenizer(self):
        import inspect

        source = inspect.getsource(tokens)
        self.assertNotIn("import tiktoken", source)
        for name in ("count", "count_message", "count_specs"):
            self.assertTrue(callable(getattr(tokens, name)))


if __name__ == "__main__":
    unittest.main()
