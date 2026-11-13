"""Context budgeting: keep a long tool loop inside the model's window.

A tool loop only grows. Every step appends an assistant message carrying the
call and a tool message carrying the result, and nothing ever leaves -- so a run
that fitted at step two is past the window by step nine, and the provider
answers with a 400 that reads like a malformed request rather than a budget
overrun. `doctor` classified that body as `protocol`, i.e. "this is a OneHarness
bug", which sends the reader into the wrong file.

Two rules here:

* **Pruning is for the request, not the transcript.** Only the list that goes on
  the wire is trimmed; `AgentResult.messages` keeps everything, because a
  history that is both truncated and presented as the record is worse than no
  record at all.
* **The cheapest thing to lose goes first.** A tool result can be produced
  again by calling the tool; the reasoning that decided to call it cannot. So
  results are elided before anything is dropped, and the stub says what
  happened, so the model can ask again instead of assuming the file was empty.

Counting is a calibrated per-character-class estimate (`oneharness.tokens`),
not a tokenizer. It is provider-independent and stable across package upgrades,
but it is no longer a flat `chars / 4`: that guess was 0.38x on Chinese and 31x
on indentation, so it wasted round trips on one and threw away context on the
other. The tool schemas are counted too -- twenty tools are about 1600 tokens
that no budget ever saw, on every request of a run.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from oneharness import tokens
from oneharness.errors import ProviderError
from oneharness.types import Message

MIN_BUDGET = 256
MAX_SHRINKS = 2

ELIDED = (
    "[elided {chars} chars to fit the context budget; call the tool again if you still need this]"
)

# Words real endpoints use for "the payload, not the shape, is the problem".
# These come from published error strings (Anthropic: "prompt is too long:
# 208310 tokens > 200000 maximum"; OpenAI: "This model's maximum context length
# is 128000 tokens"), so they are not backed by a tape the way the `auth`
# verdict is -- there is no delta between them and a local server either.
_OVERFLOW_WORDS = (
    "context length",
    "context window",
    "context_length_exceeded",
    "maximum context",
    "prompt is too long",
    "too many tokens",
    "input is too long",
    "reduce the length",
    "exceeds the maximum",
    "request too large",
    "string too long",
)


@dataclass(slots=True)
class Trim:
    """What one request had to give up to fit, and the estimate either side."""

    budget: int = 0
    before: int = 0
    after: int = 0
    elided: int = 0
    dropped: int = 0
    # Tokens spent on the tool schemas, which are sent with every request and
    # were previously not counted at all. Kept apart from `before`/`after`
    # because a budget cannot elide them: either the tools fit or the tools fit.
    tools: int = 0
    # Tokens still over the budget after everything the trimmer is allowed to
    # do. Non-zero means the request went out too big on purpose: the only
    # things left to cut are the system prompt, the first user turn or the tool
    # schemas, and each of those changes the question rather than shortening it.
    unmet: int = 0

    @property
    def changed(self) -> bool:
        return self.elided > 0 or self.dropped > 0

    def as_dict(self) -> dict[str, int]:
        return {
            "budget": self.budget,
            "tokens_before": self.before,
            "tokens_after": self.after,
            "elided": self.elided,
            "dropped": self.dropped,
            "tools": self.tools,
            "unmet": self.unmet,
        }


def estimate(
    messages: list[Message] | tuple[Message, ...],
    specs: list[dict[str, Any]] | None = None,
) -> int:
    """Estimated tokens for a request: the messages plus the tool schemas.

    `specs` is part of the request, not decoration. Leaving it out is how a run
    with twenty tools overshoots a budget by 1600 tokens and gets a 400 that
    reads like a wire-format bug.
    """
    counted = sum(tokens.count_message(message) for message in messages)
    return counted + tokens.count_specs(specs)


def legacy_estimate(messages: list[Message] | tuple[Message, ...]) -> int:
    """The v1.1 rule: every character counts as a quarter of a token.

    Kept because the difference between this and `estimate` is the whole point
    of v1.2, and a claim about an improvement needs the old number next to the
    new one. Nothing in the harness calls it.
    """
    total = 0
    for message in messages:
        total += len(message.content or "")
        total += len(message.name or "")
        for call in message.tool_calls:
            total += len(call.name) + len(call.arguments_text())
    return total // 4


def fit(
    messages: list[Message],
    budget: int,
    *,
    specs: list[dict[str, Any]] | None = None,
) -> tuple[list[Message], Trim]:
    """Return the messages to send plus a record of what was cut.

    `budget <= 0` means no budgeting: the list comes back untouched, which is
    the default, so an agent that fitted before still behaves identically.
    """
    working = list(messages)
    before = estimate(working, specs)
    trim = Trim(budget=budget, before=before, after=before, tools=tokens.count_specs(specs))
    if budget <= 0 or before <= budget:
        return working, trim

    trim.elided = _elide_results(working, budget, specs)
    if estimate(working, specs) > budget:
        trim.dropped = _drop_middle(working, budget, specs)
    trim.after = estimate(working, specs)
    trim.unmet = max(0, trim.after - budget)
    return working, trim


def _elide_results(messages: list[Message], budget: int, specs: list[dict[str, Any]] | None) -> int:
    """Blank the oldest tool results first: they are the reproducible part.

    A result shorter than the stub that replaces it is skipped. Eliding `ok`
    into a 130-character explanation makes the request *bigger*: the trimmer
    would then keep going, blanking every message it can reach and still
    finishing over budget, which reads as "the budget is impossible" when the
    truth is "this trade was not worth making".
    """
    elided = 0
    for index, message in enumerate(messages):
        if estimate(messages, specs) <= budget:
            break
        if message.role != "tool" or not message.content:
            continue
        stub = ELIDED.format(chars=len(message.content))
        if len(stub) >= len(message.content):
            continue
        messages[index] = replace(message, content=stub)
        elided += 1
    return elided


def _drop_middle(messages: list[Message], budget: int, specs: list[dict[str, Any]] | None) -> int:
    """Delete whole rounds from the start, never splitting a call from its results.

    A round is a run of messages that only makes sense together: any prose and
    the assistant turn that declares the calls, plus every `tool` message
    answering them. Rounds go or stay as a unit. Two things depend on that, and
    only one of them is about size:

    * Deeper in the tool loop the older rounds are the least likely to still
      matter, so they are the cheapest thing to lose after elided results.
    * A request whose history starts with a `tool` message is not a smaller
      request, it is an invalid one. OpenAI says so directly:

            HTTP 400: messages with role 'tool' must be a response to a
            preceding message with 'tool_calls'

      Every provider that carries tool results has an equivalent rule, because
      a result with no call is not something a model can act on.

    Three messages are never dropped: the system prompt (who to be), the first
    user turn (what the task is) and the newest user turn. The newest one is
    also what keeps the request *valid*: Anthropic and Bedrock require the first
    non-system message to be a user turn, so trimming may not promote whatever
    came before the last question into that slot.

    The rounds are measured by walking the list that is actually going out, not
    by counting ahead: an earlier version stepped `head` over an assistant turn
    and left its three results behind. The endpoint's answer to that is the 400
    above, and `doctor` files a 400 about `tool_calls` as a OneHarness bug --
    the right verdict for the wrong reason, since the request never should have
    been shaped that way in the first place.
    """
    head = 1 if messages and messages[0].role == "system" else 0
    if head < len(messages) and messages[head].role == "user":
        head += 1
    dropped = 0
    while head < len(messages) and estimate(messages, specs) > budget:
        start, end = _round_end(messages, head)
        if end <= start:  # only the anchors are left
            break
        dropped += end - start
        del messages[start:end]
        head = start
    return dropped


def _round_end(messages: list[Message], head: int) -> tuple[int, int]:
    """The slice of `messages` that one round occupies, given where to start.

    The caller proposes `head` -- the message after the anchors -- and gets back
    what would actually go. The two can differ, for two reasons:

    **A result does not outlive its call.** Every provider that carries tool
    results requires one to follow the turn that asked for it; OpenAI says it
    plainly:

        HTTP 400: messages with role 'tool' must be a response to a preceding
        message with 'tool_calls'

    So a turn that declares calls and the `tool` messages behind it leave
    together. A previous version stepped over the assistant turn and deleted
    only what followed it, which is why the bug needed two calls in a turn
    before it showed: with a single call the two halves happened to be removed
    in order, and the request stayed valid by accident.

    **The group, not the message.** A stretch of assistant messages followed by
    `tool` messages is one thing: text that announces a call, the call, and the
    results. Starting anywhere inside it takes the whole thing.

    The newest user turn is the boundary, not part of any round: dropping it
    would answer question N as if it were question N+1.
    """
    newest_user = _last_user_index(messages, head)
    limit = newest_user if newest_user > head else len(messages)
    start = head
    while start > 0 and messages[start - 1].role in ("assistant", "tool"):
        start -= 1
    if head < limit and messages[head].role in ("assistant", "tool"):
        end = head + 1
        while end < limit and messages[end].role in ("assistant", "tool"):
            end += 1
        return start, end
    return start, start + 1


def _last_user_index(messages: list[Message], head: int) -> int:
    for index in range(len(messages) - 1, head - 1, -1):
        if messages[index].role == "user":
            return index
    return head


def is_overflow(exc: ProviderError) -> bool:
    """True when a provider refused the request for being too big.

    A 400 that names the context is the one 400 that is not about our field
    names: same status as a wire-format bug, opposite meaning. Reading it as a
    protocol error is how a budget problem gets debugged in the wrong file.
    """
    if exc.status == 413:
        return True  # "payload too large" has no other reading
    if exc.status not in (400, 422):
        return False
    body = (exc.body or "").lower()
    return any(word in body for word in _OVERFLOW_WORDS)


def shrink(budget: int, estimated: int) -> int:
    """A smaller budget after an overflow, never below a floor that could loop.

    When nothing was configured the estimate of what was just sent is the only
    measurement available, so the retry targets a fraction of it.
    """
    base = budget if budget > 0 else estimated
    return max(MIN_BUDGET, int(base * 0.6))


__all__ = [
    "ELIDED",
    "MAX_SHRINKS",
    "MIN_BUDGET",
    "Trim",
    "estimate",
    "legacy_estimate",
    "fit",
    "is_overflow",
    "shrink",
]
