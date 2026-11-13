"""Fit `oneharness/tokens.py` against a real tokenizer, and record the truth.

    python3 tools/calibrate_tokens.py            # report the error, rewrite the fixture
    python3 tools/calibrate_tokens.py --check    # fail if the constants drifted

This script is the only place `tiktoken` is used, and it is not part of the
package: the harness ships arithmetic, not a vocabulary. It does two jobs.

1. **Fitting.** The constants in `tokens.CALIBRATION` came from matching this
   corpus. They are coordinate-descent products, not round numbers, because
   round numbers were the bug: a single divisor for every kind of text is what
   `chars / 4` was.

2. **Recording.** It writes `tests/fixtures/token_counts.json`: the corpus plus
   what a real tokenizer counted. The tests and the demo compare against that
   file instead of importing tiktoken, so the accuracy claim is checked on every
   CI run without a tokenizer anywhere in the dependency tree -- and the numbers
   are still someone else's, not the estimator agreeing with itself.

Two vocabularies are averaged (`cl100k_base`, `o200k_base`) because no single
one is the truth for five providers. Anthropic's tokenizer is not public; the
point of the constant is to be within a sensible band of all of them.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oneharness import tokens  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "token_counts.json"
# Fitted on `corpus()`, so measuring on it is a closed loop. This one is the way
# out: real files from this repository, never seen by `fit()`, recorded by the
# same tokenizer. If the constants only describe the corpus they came from, this
# is where it shows.
HOLDOUT = ROOT / "tests" / "fixtures" / "token_counts_holdout.json"

ZH = (
    "请帮我把这个仓库里的网络层重构一下，先看 transport 的错误分类，"
    "再把重试策略改成按 Retry-After 走。这个 harness 有一个问题："
    "它估算 token 的方式是字符数除以四，对中文来说差了将近三倍。"
)


def repo_pieces() -> list[tuple[str, str]]:
    """The repository's own source, one 1.5 KB piece per file, in a stable order."""
    paths = sorted((ROOT / "oneharness").rglob("*.py")) + sorted((ROOT / "examples").glob("*.py"))
    pieces = []
    for path in paths:
        piece = path.read_text()[:1500]
        if len(piece) > 500:
            # Relative, not the bare name: there are several `__init__.py`, and
            # two samples with one key silently collapse into one.
            pieces.append((str(path.relative_to(ROOT)), piece))
    return pieces


def corpus() -> dict[str, str]:
    """Text that a tool-calling agent actually carries.

    Not a benchmark: the classes here are the ones that broke the old rule, plus
    the ones that quietly worked, so a regression in either direction shows up.

    Each sample is cut to about 4 KB: the fixture is committed, and a bigger one
    buys nothing -- the error is a ratio, and ratios do not need length.
    """

    def clip(text: str, limit: int = 4000) -> str:
        return text[:limit]

    return {
        "english": (
            "The quick brown fox jumps over the lazy dog. "
            "It was the best of times, it was the worst of times. "
        )
        * 8,
        "python": (
            "from __future__ import annotations\n\n"
            "def fit(messages, budget):\n"
            "    # trim the request, not the transcript\n"
            "    working = list(messages)\n"
            "    return working\n"
        )
        * 10,
        "json": clip(
            json.dumps(
                {
                    "path": "app.conf",
                    "lines": [{"n": i, "text": f"key{i} = value{i}"} for i in range(200)],
                },
                ensure_ascii=False,
            )
        ),
        "log": clip(
            "\n".join(
                f"2026-11-13T04:25:54Z INFO step={i} tool=read_file "
                f"path=oneharness/agent.py elapsed=0.9"
                for i in range(400)
            )
        ),
        "chinese": ZH * 6,
        "chinese_code": "这个函数 `estimate()` 用的是 chars/4，对中文差 3 倍。" * 8,
        "japanese": (
            "このリポジトリのネットワーク層をリファクタリングしてください。テストも追加してください。"
        )
        * 6,
        "korean": "이 저장소의 네트워크 계층을 리팩터링하고 테스트를 추가하세요." * 6,
        "french": "L'agent doit estimer les jetons sans dépendre d'un paquet tiers. " * 8,
        "german": "Der Agent muss die Token schätzen, ohne von einem Paket abzuhängen. " * 8,
        "russian": "Агент должен оценивать токены без сторонних зависимостей. " * 8,
        "emoji": "🎉👍🏽 ok ✅ done 🚀🇨🇳 the end " * 50,
        "base64": (
            "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVphYmNkZWZnaGlqa2xtbm9wcXJzdHV2d3h5ejAxMjM0NTY3ODk="
        )
        * 12,
        "hash": "9f2c1b7a4e8d0c3b5a6f7e8d9c0b1a2f3e4d5c6b7a8f9e0d1c2b3a4f5e6d7c8b " * 10,
        "hexdump": ("00000000: 7f45 4c46 0201 0100 0000 0000 0000 0000  .ELF............\n") * 40,
        "vietnamese": "Trợ lý phải ước lượng token mà không cần thư viện bên ngoài. " * 8,
        "greek": "Το μοντέλο πρέπει να μετράει τα διακριτικά χωρίς εξαρτήσεις. " * 8,
        "indentation": "    " * 2000,
        "markdown": (
            "| kind | who says it | where to fix |\n|---|---|---|\n| `dns` | resolver | typo |\n"
        )
        * 20,
        "traceback": (
            'Traceback (most recent call last):\n  File "oneharness/agent.py", line 417, in run\n'
            '    raise ProviderError("x")\nProviderError: nope\n'
        )
        * 20,
        "tool_schemas": json.dumps(
            [
                {
                    "name": f"tool_{i}",
                    "description": "Read a file and return its contents line by line.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                }
                for i in range(20)
            ]
        ),
        "tool_result": "x" * 4000,
        # Half the repository's own source. Python and English prose is the most
        # common thing this counters, so the constants have to fit it -- and the
        # other half is kept back for `holdout()`.
        **{
            f"repo:{name}": text
            for index, (name, text) in enumerate(repo_pieces())
            if index % 2 == 0
        },
    }


def holdout() -> dict[str, str]:
    """Text the fit never saw: the other half of the source, plus README cuts.

    The split is by file, not by character, so a held-out piece is a different
    document rather than the next paragraph of one the fit already read. This is
    the number to trust: the fitting corpus can be matched by constants that
    describe nothing else, and the first run of this script proved it did -- the
    fit chose a Latin ratio that was 0.5x worse on real code than on the corpus.
    """
    out: dict[str, str] = {
        f"repo:{name}": text for index, (name, text) in enumerate(repo_pieces()) if index % 2 == 1
    }
    readme = (ROOT / "README.md").read_text()
    for start in range(0, min(len(readme), 15_000), 3_000):
        out[f"README.md@{start}"] = readme[start : start + 3_000]
    return out


def measure(text: str) -> dict[str, int]:
    import tiktoken

    return {
        name: len(tiktoken.get_encoding(name).encode(text))
        for name in ("cl100k_base", "o200k_base")
    }


def truth(counts: dict[str, int]) -> float:
    return sum(counts.values()) / len(counts)


def score(calibration: dict[str, float], data: list[tuple[str, str, float]]) -> tuple:
    errors = sorted(abs(math.log(tokens.count(text, calibration) / real)) for _, text, real in data)
    return (
        sum(errors) / len(errors),
        errors[int(len(errors) * 0.9)],
        errors[-1],
    )


def fit(data: list[tuple[str, str, float]], rounds: int = 40) -> dict[str, float]:
    """Coordinate descent on multiplicative factors, minimising mean log error."""
    calibration = dict(tokens.CALIBRATION)
    best = score(calibration, data)[0]
    for _ in range(rounds):
        moved = False
        for name in calibration:
            for factor in (0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25):
                trial = dict(calibration)
                trial[name] = calibration[name] * factor
                candidate = score(trial, data)[0]
                if candidate < best - 1e-7:
                    best, calibration[name], moved = candidate, trial[name], True
        if not moved:
            break
    return calibration


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="do not rewrite anything")
    args = parser.parse_args()
    try:
        import tiktoken  # noqa: F401
    except ImportError:
        print("tiktoken is not installed: pip install tiktoken", file=sys.stderr)
        return 2

    samples = {name: {"text": text, "tokens": measure(text)} for name, text in corpus().items()}
    data = [(name, item["text"], truth(item["tokens"])) for name, item in samples.items()]

    fitted = fit(data)
    print("fitted constants:")
    for name, value in fitted.items():
        was = tokens.CALIBRATION[name]
        print(f"  {name:<12} {value:8.4f}  (was {was:8.4f})")

    for label, calibration in (("chars/4", None), ("fitted", fitted)):
        rows = []
        for name, text, real in data:
            estimate = len(text) / 4 if calibration is None else tokens.count(text, calibration)
            rows.append((name, len(text), real, estimate))
        errors = sorted(abs(math.log(row[3] / max(1.0, row[2]))) for row in rows)
        print(
            f"\n{label:>8}: mean {math.exp(sum(errors) / len(errors)):.2f}x  "
            f"p90 {math.exp(errors[int(len(errors) * 0.9)]):.2f}x  "
            f"worst {math.exp(errors[-1]):.2f}x"
        )
        for name, chars, real, estimate in rows:
            if name in ("chinese", "indentation", "log", "base64", "tool_schemas", "english"):
                print(
                    f"  {name:<13} {chars:>6} chars  real {real:>7.0f}  "
                    f"estimate {estimate:>7.0f}  ({estimate / max(1.0, real):.2f}x)"
                )

    held = {name: {"text": text, "tokens": measure(text)} for name, text in holdout().items()}
    held_data = [(name, item["text"], truth(item["tokens"])) for name, item in held.items()]
    held_errors = sorted(
        abs(math.log(tokens.count(text) / max(1.0, real))) for _, text, real in held_data
    )
    print(
        f"\nheld out ({len(held_data)} pieces of this repo, never fitted on): "
        f"mean {math.exp(sum(held_errors) / len(held_errors)):.2f}x  "
        f"p90 {math.exp(held_errors[int(len(held_errors) * 0.9)]):.2f}x  "
        f"worst {math.exp(held_errors[-1]):.2f}x"
    )

    if not args.check:
        note = (
            "Recorded by tools/calibrate_tokens.py with tiktoken "
            "(cl100k_base and o200k_base, averaged). Frozen here so the tests and "
            "the demo can check the estimate without a tokenizer anywhere in the tree."
        )
        for path, payload in (
            (FIXTURE, {"note": note + " This is the fitting corpus.", "samples": samples}),
            (
                HOLDOUT,
                {
                    "note": note + " Held out: never used for fitting.",
                    "samples": held,
                },
            ),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
            print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
