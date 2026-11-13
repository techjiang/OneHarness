"""Counting tokens without a tokenizer, calibrated against one.

v1.1 counted `len(text) // 4`. That is not an estimate, it is a guess that
happens to work for English prose. Against tiktoken over the corpus in
`tools/calibrate_tokens.py` it is off by a factor of 2 on average and 25x in the
worst case -- and the error has a direction, so it is not noise:

* **Chinese counted at 0.38x.** 435 tokens read as 166, so a 2000-token budget
  let a 5000-token request out to be refused. Every CJK prompt paid a wasted
  round trip and a re-trim.
* **Indentation counted at 32x.** 8 KB of spaces is 63 tokens, not 2000, so a
  code-heavy history was trimmed while it still fitted.

Both directions cost real work: a request refused, and context discarded for
nothing. So the flat ratio is gone. This module counts per class of character
and merges within word runs, with the constants fitted offline against a real
tokenizer.

**The tokenizer is a calibration tool, not a dependency.** `tiktoken` produced
the numbers and the error bounds in `tests/test_tokens.py`; nothing imports it
at runtime. Shipping it would add tens of megabytes and tie the estimate to one
vendor's vocabulary, which is exactly what a harness with five providers must
not do. What is left is arithmetic over `ord()`: no download, no drift, and the
same answer for every provider.

Measured over 195 samples of Python, JSON, logs, English/Chinese/Japanese/
Korean/European prose, base64, hashes and emoji: mean error 4.6%, 90th
percentile 10%, worst 1.5x. `chars / 4` is mean 62%, worst 32x.
"""

from __future__ import annotations

import json
import re
from typing import Any

# What a token costs, in tokens per character or characters per token. The
# `*_cpt` names are characters per token, because that is how people talk about
# a tokenizer ("about four characters per token"); the bare names are tokens per
# character, because a run that merges is not a per-character quantity.
CALIBRATION: dict[str, float] = {
    # Inside a word run BPE merges several characters into one token, so these
    # are ratios rather than weights. 3.8 for letters is the familiar
    # "four characters per token", and it is the one number `chars / 4` got
    # right -- for English only.
    "latin_cpt": 4.1476,
    "digit_cpt": 0.9234,
    "cyr_cpt": 2.625,
    "grk_cpt": 1.4,
    # A run of letters is at least one token: "a" is not a third of one.
    "word_min": 0.9898,
    # Whitespace merges into long runs -- an indented file is nearly free.
    "space": 0.008,
    "punct": 0.4679,
    # Scripts without spaces between words: one character is roughly one token.
    "cjk": 0.8437,
    "kana": 0.6825,
    "hangul": 0.91,
    "wide": 2.26,  # emoji, flags, anything astral: two tokens per character
    "other": 2.0434,  # accented latin and everything not named above
    # Long runs with no short merges left in the vocabulary.
    "hex_cpt": 1.0,
    "b64_cpt": 1.55,
    # Runs of one character merge almost perfectly, and how perfectly depends on
    # the character: 8000 spaces are 63 tokens (127 chars/token), 4000 "=" are
    # 63, but 4000 "x" are 500 (8 per token) -- a repeated letter is a rarer
    # pattern in the vocabulary.
    "repeat_alnum_cpt": 8.0,
    "repeat_punct_cpt": 230.3092,
}

_PER_MESSAGE = 4  # role, separators: the fixed cost of one message on the wire

_LATIN = re.compile(r"[A-Za-z0-9\u00c0-\u024f\u1e00-\u1eff]+")
# A long run of one character -- a separator line, a progress bar, base64
# padding -- has merges for the whole repetition. Counting it at the letter or
# punctuation weight overestimates it by an order of magnitude.
_REPEAT = re.compile(r"(.)\1{7,}")
_CYRILLIC = re.compile(r"[\u0400-\u04ff]+")
_GREEK = re.compile(r"[\u0370-\u03ff]+")
_HEX = re.compile(r"(?<![0-9A-Za-z])[0-9a-fA-F]{24,}(?![0-9A-Za-z])")
_SHAPED = re.compile(r"[A-Za-z0-9+/_-]{24,}={0,2}")


def _is_blob(run: str) -> bool:
    """True for base64, hex digests and jwt-ish runs, where merges run out.

    Requiring a digit is what keeps long paths and identifiers out: those are
    letters and slashes, and they do merge -- treating them as blobs would
    underestimate a repository listing by half.
    """
    if not any(char.isdigit() for char in run):
        return False
    upper = any(char.isupper() for char in run)
    lower = any(char.islower() for char in run)
    return (upper and lower) or run.endswith("=") or len(run) >= 48


def _classify(char: str) -> str:
    """The name of the class one non-word character belongs to."""
    code = ord(char)
    if code < 128:
        return "space" if char in " \t\n\r\f\v" else "punct"
    if 0x3040 <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF or 0xFF66 <= code <= 0xFF9F:
        return "kana"
    if 0xAC00 <= code <= 0xD7AF or 0x1100 <= code <= 0x11FF:
        return "hangul"
    if 0x2E80 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF or 0x20000 <= code <= 0x2FA1F:
        return "cjk"
    if code >= 0x1F000 or 0x2600 <= code <= 0x27BF or 0xFE00 <= code <= 0xFE0F or code > 0xFFFF:
        return "wide"
    return "other"


def count(text: str, calibration: dict[str, float] | None = None) -> int:
    """Estimated tokens in one string.

    Never returns 0 for non-empty text: a message that costs "nothing" is the
    kind of zero that makes a budget loop forever.

    Three passes, in order of how much they change the answer:

    1. **Runs of one character.** `"x" * 4000` is 500 tokens and `"=" * 4000` is
       63; measured at the letter weight they would be 1050 and 1780, and a log
       like that gets trimmed for nothing.
    2. **Long machine-generated runs** (hex digests, base64): the vocabulary has
       no short merges left in them.
    3. **Everything else**, per character, with word runs counted as words.
    """
    if not text:
        return 0
    weights = calibration or CALIBRATION
    total = 0.0

    # 1. Repeated characters. Whitespace is left alone: its own weight already
    # describes a run, and an indented file must stay nearly free.
    rest: list[str] = []
    position = 0
    for match in _REPEAT.finditer(text):
        char = match.group(1)
        if char not in " \t\n\r\f\v":
            rest.append(text[position : match.start()])
            length = len(match.group(0))
            name = "repeat_alnum_cpt" if char.isalnum() else "repeat_punct_cpt"
            total += length / weights[name]
            position = match.end()
    text = "".join(rest) + text[position:]

    # 2. Long machine-generated runs.
    hexes = 0
    rest = []
    position = 0
    for match in _HEX.finditer(text):
        rest.append(text[position : match.start()])
        hexes += len(match.group(0))
        position = match.end()
    text = "".join(rest) + text[position:]

    blobs = 0
    rest = []
    position = 0
    for match in _SHAPED.finditer(text):
        if _is_blob(match.group(0)):
            rest.append(text[position : match.start()])
            blobs += len(match.group(0))
            position = match.end()
    text = "".join(rest) + text[position:]
    total += hexes / weights["hex_cpt"] + blobs / weights["b64_cpt"]

    # 3. Word runs, per script. Counting a run as a unit is what makes "the" one
    # token and "they" still one token, instead of 3 and 4 characters' worth.
    taken = bytearray(len(text))
    for pattern, key, with_digits in (
        (_LATIN, "latin_cpt", True),
        (_CYRILLIC, "cyr_cpt", False),
        (_GREEK, "grk_cpt", False),
    ):
        for match in pattern.finditer(text):
            run = match.group(0)
            if with_digits:
                digits = sum(1 for char in run if char.isdigit())
                cost = (len(run) - digits) / weights[key] + digits / weights["digit_cpt"]
            else:
                cost = len(run) / weights[key]
            total += max(weights["word_min"], cost)
            for index in range(match.start(), match.end()):
                taken[index] = 1

    for index, char in enumerate(text):
        if not taken[index]:
            total += weights[_classify(char)]

    return max(1, round(total))


def count_message(message: Any) -> int:
    """Tokens for one message: its content, its name, and any tool calls."""
    total = _PER_MESSAGE
    total += count(message.content or "")
    if message.name:
        total += count(message.name)
    for call in message.tool_calls:
        total += count(call.name) + count(call.arguments_text())
    return total


def count_specs(specs: list[dict[str, Any]] | None) -> int:
    """Tokens for the tool schemas, which ride along on every request.

    This used to count as zero, and that omission cannot be discovered from the
    message list: twenty tools is roughly 1600 tokens a budget never saw, over
    and over, for the whole run.
    """
    if not specs:
        return 0
    return count(json.dumps(specs, ensure_ascii=False))


__all__ = ["CALIBRATION", "count", "count_message", "count_specs"]
