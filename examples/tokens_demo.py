"""Counting tokens without a tokenizer: how wrong the old rule was, and where.

    python3 examples/tokens_demo.py

The numbers in this demo are not produced by the estimator grading itself. They
were recorded by `tools/calibrate_tokens.py` with `tiktoken` (`cl100k_base` and
`o200k_base`, averaged) and frozen in `tests/fixtures/token_counts.json`. This
file only reads that fixture, so a tokenizer is not needed to check the claim --
and the reference is still somebody else's number.

Four parts:

1. Per-sample error. `chars / 4` is not a rough approximation, it is a rule with
   a direction: it is right for English prose and wrong by 3x to 30x on the
   things a tool result is made of.
2. The same data through the real budget code: the same budget, the same
   history, two answers -- one run is refused for size, the other fits.
3. The tool schemas, which were not counted at all. Twenty tools is about 1000
   tokens that no budget ever saw, on every request of a run.
4. Fitting is still possible when the trimmer has nothing left to cut, and now
   says so instead of going out over budget in silence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oneharness import context, tokens  # noqa: E402
from oneharness.types import Message, ToolCall  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "token_counts.json"
HOLDOUT = ROOT / "tests" / "fixtures" / "token_counts_holdout.json"


def recorded(path: Path = FIXTURE) -> dict[str, tuple[str, float]]:
    """name -> (text, the tokenizer's own count) from a frozen fixture."""
    data = json.loads(path.read_text())
    out = {}
    for name, item in data["samples"].items():
        counts = item["tokens"]
        out[name] = (item["text"], sum(counts.values()) / len(counts))
    return out


def part_one() -> None:
    print("=== 1. 同一段文本，两套数法 ===")
    print(f"{'sample':<14}{'chars':>7}{'real':>8}{'chars/4':>10}{'new':>8}   verdict")
    for name, (text, real) in recorded().items():
        if name.startswith("repo:"):
            continue  # the source files are part 5's business, not a table row
        old = len(text) / 4
        new = tokens.count(text)
        verdict = "ok" if abs(new / real - 1) < 0.2 else "off"
        if abs(old / real - 1) > 0.5:
            verdict += f"  (chars/4 {old / real:.2f}x)"
        print(
            f"{name:<14}{len(text):>7}{real:>8.0f}{old:>10.0f}{new:>8}  {new / real:.2f}x {verdict}"
        )


def history_in(text: str, turns: int = 6) -> list[Message]:
    """A tool loop whose results are all `text`: six reads, like a real run."""
    messages = [Message.user("读一遍这些文件，然后告诉我哪里出了问题")]
    for index in range(turns):
        messages.append(
            Message.assistant(
                "",
                [ToolCall.from_raw(call_id=f"c{index}", name="read_file", arguments={"path": "a"})],
            )
        )
        messages.append(Message.tool(f"c{index}", text, "read_file"))
    return messages


def part_two() -> None:
    """One window, two histories: the old rule and the new one decide differently."""
    print("\n=== 2. 同一个窗口，两段历史，一个放行一个先裁 ===")
    for label, text in (
        (
            "中文工具结果",
            "这个函数估算 token 的方式不对，请看一下 transport 的错误分类和重试策略。" * 60,
        ),
        ("缩进很深的代码", ("def f():\n" + "    " * 6 + "return 1\n") * 200),
    ):
        messages = history_in(text)
        old = context.legacy_estimate(messages)
        new = context.estimate(messages)
        # The window is placed between the two answers on purpose: that is the
        # only place a disagreement is visible. Part 1 shows which number is
        # closer to the truth; this part shows that a budget is a *decision*,
        # and two decisions cannot both be right.
        budget = (old + new) // 2
        _, trimmed = context.fit(messages, budget)
        print(f"\n  {label}，窗口取两数之间：{budget}")
        old_verdict = "先裁（删掉本来塞得下的内容）" if old > budget else "放行（以为塞得下）"
        new_verdict = f"先裁 {trimmed.elided} 条工具结果" if new > budget else "放行"
        print(f"    chars/4  {old:>6} tokens -> {old_verdict}")
        print(f"    新估算  {new:>6} tokens -> {new_verdict}")
    print("\n  方向性错误的代价是双向的：一个白白被 provider 拒一次，一个白白删掉塞得下的内容")


def part_three() -> None:
    """The schemas ride along on every request and used to be free."""
    print("\n=== 3. 工具 schema 一直是 0 ===")
    specs = [
        {
            "name": f"tool_{i}",
            "description": "Read a file and return its contents line by line.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "start": {"type": "integer"}},
                "required": ["path"],
            },
        }
        for i in range(20)
    ]
    prompt = [Message.user("hi")]
    schema_tokens = tokens.count_specs(specs)
    print(f"  20 个工具：schema 本身 {schema_tokens} tokens，消息只有 {context.estimate(prompt)}")
    print(f"  以前这个数算 0，现在算 {context.estimate(prompt, specs)}")
    print("  它每一次请求都在，而预算从来看不见它")


def part_four() -> None:
    """What the trimmer cannot reach, it now reports instead of hiding."""
    print("\n=== 4. 裁不动的时候，说出来 ===")
    messages = [Message.system("be brief"), Message.user("这个提示词本身就超过窗口 " * 300)]
    fitted, trim = context.fit(messages, 256)
    print(f"  budget {trim.budget} | {trim.before} -> {trim.after} tokens")
    print(f"  elided {trim.elided} | dropped {trim.dropped} | 仍然超出 {trim.unmet}")
    verdict = "没裁到预算" if trim.unmet else "裁到预算了"
    print(f"  {verdict}：剩下的部分是 system 和用户的问题，删掉就是回答别的问题")

    tiny = [Message.user("go")]
    for index in range(8):
        tiny.append(
            Message.assistant(
                "", [ToolCall.from_raw(call_id=f"c{index}", name="read_file", arguments={})]
            )
        )
        tiny.append(Message.tool(f"c{index}", "ok", "read_file"))
    _, small = context.fit(tiny, 60)
    print(f"  另一头：8 个 'ok' 结果，budget 60 -> elided {small.elided}")
    print(
        f"  {'把 ok 换成 130 字的占位符只会让请求更大，所以跳过' if not small.elided else 'elided'}"
    )


def part_five() -> None:
    """Where the constants come from, and whether they survive new text."""
    print("\n=== 5. 常数是拟合的，但不在拟合集上自证 ===")
    for label, path in (("拟合集", FIXTURE), ("留出集：本仓库另一半源码，没见过", HOLDOUT)):
        rows = [
            (abs(tokens.count(text) / max(1.0, real) - 1), abs(len(text) / 4 / max(1.0, real) - 1))
            for text, real in recorded(path).values()
        ]
        rows.sort()
        new_mean = sum(row[0] for row in rows) / len(rows)
        old_mean = sum(row[1] for row in rows) / len(rows)
        print(
            f"  {label}，{len(rows)} 个样本："
            f"平均 {new_mean * 100:.0f}%（chars/4 {old_mean * 100:.0f}%），"
            f"最差 {rows[-1][0] * 100:.0f}%（chars/4 {rows[-1][1] * 100:.0f}%）"
        )
    print("  留出集和拟合集一样好，才说明常数描述的是英文和代码，不是那几十个样本")
    print("  常数里没有一个是整数：整数的那个就是 chars/4")
    print("  重新拟合：python3 tools/calibrate_tokens.py（需要 pip install tiktoken，仅此一次）")


def main() -> None:
    part_one()
    part_two()
    part_three()
    part_four()
    part_five()


if __name__ == "__main__":
    main()
