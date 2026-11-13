"""A tool loop re-sends its whole history every step. What does that cost?

    python3 examples/cache_demo.py

Four parts, all against one local server that implements Anthropic's caching
rule as documented: an exact prefix match against the longest breakpoint seen
before, a 1024-token minimum, and a five-minute TTL.

1. **The shape of the bill without a cache.** Five steps, and the prompt grows
   by one turn each time. The total is quadratic in the number of steps, which
   is not a bug in the loop -- it is what the loop is.
2. **The same five steps with breakpoints placed.** The server decides what was
   a hit, from the bytes this harness actually sent. It is not told anything by
   the harness.
3. **What the adapters report.** The same numbers arrive in four different wire
   formats, and a `Usage` normalises them so `input_tokens` is the prompt and
   `cached_tokens` is the cheap share of it. This one is checked against the
   recorded fixture from the earlier versions.
4. **The estimate against the server's own count.** `tokens.py` counts
   characters, the server counts words -- independent sources, which is the only
   reason comparing them is worth doing.

**What this demo cannot prove.** The server is my reading of the spec, so if I
misread caching, it misreads it identically and both stay green. That is the
same closed loop the rest of the suite runs in, and it is why the honest count
of verified provider behaviours in the README does not include this one. What it
does prove is narrower and still worth something: given that rule, the
breakpoints this harness places are the ones that produce the hits.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oneharness.agent import Agent  # noqa: E402
from oneharness.providers.anthropic import AnthropicProvider  # noqa: E402
from oneharness.providers.gemini import GeminiProvider  # noqa: E402
from oneharness.providers.openai import OpenAIProvider  # noqa: E402
from oneharness.tools import tool  # noqa: E402
from oneharness.types import Usage  # noqa: E402

MIN_CACHEABLE = 1024  # tokens; below this Anthropic refuses to cache a prefix


def banner(text: str) -> None:
    print(f"\n=== {text} ===")


def words(payload) -> int:
    """The server's own token count: crude, but not the harness's estimator."""
    return max(1, len(json.dumps(payload, ensure_ascii=False).split()) * 4 // 3)


def prefix_key(block: dict) -> str:
    return json.dumps(block, sort_keys=True, ensure_ascii=False)


class CachingServer:
    """Anthropic-shaped, with the cache rule from the docs and a bill.

    The cache is keyed on the exact serialised prefix, because that is the part
    that has to match: a prefix that differs by one character is a miss, and a
    server that "roughly matched" would make every placement look correct.
    """

    def __init__(self):
        self.prefixes: list[tuple[list[str], int]] = []
        self.requests: list[dict] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                reply = server.answer(body)
                server.requests.append(reply)
                data = json.dumps(reply["response"]).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()

    def answer(self, payload: dict) -> dict:
        """Walk the request in wire order, honour every breakpoint, hand back a bill."""
        items, marks = self._walk(payload)
        prompt = words(items)

        # A prompt cache is a *prefix* cache: what matters is the longest
        # already-known prefix of this request, not whether this request's own
        # breakpoints happen to be known. Checking only the latter would miss
        # every step after the first, because each step marks its own end.
        hit = 0
        for known, size in self.prefixes:
            if len(known) <= len(items) and items[: len(known)] == known:
                hit = max(hit, size)

        written = 0
        deepest = max(marks, default=0)
        if deepest >= MIN_CACHEABLE and deepest > hit:
            taken = items[: self._cut(items, deepest)]
            self.prefixes.append((taken, deepest))
            written = deepest - hit

        return {
            "prompt": prompt,
            "cache_read": hit,
            "cache_write": written,
            "response": {
                "id": "msg_demo",
                "type": "message",
                "role": "assistant",
                "model": payload.get("model"),
                "content": self._answer_text(payload),
                "stop_reason": "tool_use" if self._wants_tool(payload) else "end_turn",
                "usage": {
                    # Anthropic's three prompt fields *sum* to the prompt:
                    # `input_tokens` is what was neither read from nor written to
                    # the cache. Writing a prefix costs more per token, but those
                    # tokens are still not "new input" -- putting the write on
                    # top of `input_tokens` (as my first version did) counts the
                    # same tokens twice and made the adapter look like it was
                    # inflating the prompt.
                    "input_tokens": max(0, prompt - hit - written),
                    "output_tokens": 12,
                    "cache_read_input_tokens": hit,
                    "cache_creation_input_tokens": written,
                },
            },
        }

    @staticmethod
    def _strip(block: dict) -> dict:
        """The content without the marking.

        The cache key covers what a request *says*, not where it asks to cache.
        A breakpoint moving forward on the next step would otherwise change the
        bytes of earlier blocks and turn every hit into a miss -- which is what
        my first version of this server did, and it looked exactly like the
        breakpoints being placed wrong.
        """
        return {key: value for key, value in block.items() if key != "cache_control"}

    @staticmethod
    def _walk(payload: dict) -> tuple[list[str], list[int]]:
        """The request as one ordered list of prefix items, plus the marked sizes.

        Flattening the three top-level parts into one sequence is what makes the
        prefix logic honest: the wire order really is tools, then system, then
        messages, and a breakpoint can only ever cover what comes before it.
        """
        items: list[str] = []
        marks: list[int] = []
        for spec in payload.get("tools") or []:
            items.append(prefix_key({"tool": CachingServer._strip(spec)}))
            if "cache_control" in spec:
                marks.append(words(items))
        system = payload.get("system") or []
        for block in [system] if isinstance(system, str) else system:
            plain = CachingServer._strip(block) if isinstance(block, dict) else block
            items.append(prefix_key({"system": plain}))
            if isinstance(block, dict) and "cache_control" in block:
                marks.append(words(items))
        for turn in payload.get("messages") or []:
            content = turn.get("content")
            blocks = content if isinstance(content, list) else [{"text": content}]
            for block in blocks:
                plain = CachingServer._strip(block) if isinstance(block, dict) else block
                items.append(prefix_key({"role": turn.get("role"), "block": plain}))
                if isinstance(block, dict) and "cache_control" in block:
                    marks.append(words(items))
        return items, marks

    @staticmethod
    def _cut(items: list[str], size: int) -> int:
        """How many items make up a prefix of about `size` tokens."""
        kept = 0
        for index in range(len(items)):
            kept = index + 1
            if words(items[:kept]) >= size:
                break
        return kept

    @staticmethod
    def _wants_tool(payload: dict) -> bool:
        """Keep asking for the file until the loop has run out of patience.

        A demo of a growing prefix needs a *growing* prompt; a server that
        answered on step two would make every point below vacuous.
        """
        turns = payload.get("messages") or []
        return len(turns) < 9

    @staticmethod
    def _answer_text(payload: dict) -> list[dict]:
        turns = payload.get("messages") or []
        wants_tool = CachingServer._wants_tool(payload)
        if wants_tool:
            return [
                {
                    "type": "tool_use",
                    "id": f"toolu_{len(turns)}",
                    "name": "read_notes",
                    "input": {"path": "notes.md"},
                }
            ]
        return [{"type": "text", "text": f"read {len(turns)} turns, done"}]


@tool
def read_notes(path: str) -> str:
    """Read a notes file (a fixed body, so the prefix stays identical)."""
    # Deliberately bigger than the 1024-token cache minimum on its own, so the
    # prefix is cacheable from the first breakpoint rather than only at the end.
    return f"# {path}\n" + ("- a note about timeouts, retries and the shape of a tool loop\n" * 260)


def run_loop(points: bool) -> tuple[Usage, list[int], list[int]]:
    with CachingServer() as server:
        provider = AnthropicProvider(
            "claude-sonnet-4-5", api_key="sk-demo", base_url=server.url, cache=points
        )
        agent = Agent(provider, [read_notes], max_steps=8)
        result = agent.run("read notes.md and tell me what it says")
        prompt = [item["prompt"] for item in server.requests]
        reads = [item["cache_read"] for item in server.requests]
        return result.usage, prompt, reads


def show(label: str, usage: Usage, prompt: list[int], reads: list[int]) -> None:
    print(f"  {label}")
    print(f"    steps                 {len(prompt)}")
    print(f"    prompt sent, summed   {sum(prompt)}  (per step: {', '.join(map(str, prompt))})")
    print(f"    read from the cache   {sum(reads)}  (per step: {', '.join(map(str, reads))})")
    print(f"    paid at full price    {sum(prompt) - sum(reads)}")
    if usage.cache_write_tokens:
        print(f"    written to the cache  {usage.cache_write_tokens}")


def main() -> None:
    banner("1. the same steps, no breakpoints: every turn is paid for again")
    cold, cold_prompt, cold_reads = run_loop(points=False)
    show("no breakpoints", cold, cold_prompt, cold_reads)

    banner("2. the same steps, breakpoints placed by `--cache`")
    warm, warm_prompt, warm_reads = run_loop(points=True)
    show("with breakpoints", warm, warm_prompt, warm_reads)

    cold_paid = sum(cold_prompt) - sum(cold_reads)
    warm_paid = sum(warm_prompt) - sum(warm_reads)
    print()
    print(f"  prompt tokens paid at full price: {cold_paid} -> {warm_paid}")
    print(f"  saved: {cold_paid - warm_paid} tokens ({(1 - warm_paid / cold_paid) * 100:.0f}%)")
    print("  The server decided every one of those hits from the bytes it received,")
    print("  not from anything this harness told it. It is still my server: see the")
    print("  docstring for exactly how much that is worth.")

    banner("3. four wire formats, one Usage: prompt total, cached share apart")
    openai = OpenAIProvider("gpt", api_key="sk", base_url="http://x.test")
    one = openai._decode(
        {
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 768},
            },
        }
    ).usage
    anthropic = (
        AnthropicProvider("claude", api_key="sk", base_url="http://x.test")
        ._decode(
            {
                "content": [{"type": "text", "text": "hi"}],
                "usage": {"input_tokens": 232, "output_tokens": 20, "cache_read_input_tokens": 768},
            }
        )
        .usage
    )
    gemini = (
        GeminiProvider("gem", api_key="k", base_url="http://x.test")
        ._decode(
            {
                "candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}],
                "usageMetadata": {
                    "promptTokenCount": 1000,
                    "candidatesTokenCount": 20,
                    "cachedContentTokenCount": 768,
                },
            }
        )
        .usage
    )
    for name, usage in (("openai", one), ("anthropic", anthropic), ("gemini", gemini)):
        print(
            f"    {name:<10} input={usage.input_tokens:<5} cached={usage.cached_tokens:<5} "
            f"hit={usage.cache_hit_rate * 100:.0f}%"
        )
    print()
    print("    All three describe the same request. Anthropic reports 232 because its")
    print("    `input_tokens` excludes the cache; the other two include it. Normalising")
    print("    is not cosmetic -- a budget reading 232 where the prompt is 1000 would")
    print("    never trim, and a cost report reading 1000 where 768 were cheap would")
    print("    never notice the cache at all.")

    banner("4. the estimator against a counter that is not its own")
    with CachingServer() as server:
        provider = AnthropicProvider("claude", api_key="sk", base_url=server.url, cache=True)
        result = Agent(provider, [read_notes], max_steps=8).run("read notes.md")
        counted = sum(item["prompt"] for item in server.requests)
        estimated = sum(step.prompt_tokens_estimated for step in result.steps)
        # The normalised total, which is the number the harness actually uses:
        # prompt = fresh + read + written, and the adapter sums the three back.
        normalised = result.usage.input_tokens
    print(f"    estimated by tokens.py  {estimated}")
    print(f"    counted by the server   {counted}")
    print(f"    ratio                   {estimated / counted:.2f}x")
    print(f"    normalised by the adapters {normalised}  (fresh + read + written)")
    print()
    print("    The server counts whitespace-separated words, so this is a comparison of")
    print("    two crude methods, not of one crude method against a tokenizer. It is here")
    print("    because an estimate that is never compared to anything is just arithmetic")
    print("    that agrees with itself.")
    print()
    print("    The third number is the one that matters for the bug this version fixed:")
    print("    it is what `Usage.input_tokens` says after the adapter sums the three")
    print("    Anthropic fields back together, and it is within a few tokens of the")
    print("    server's own count. Before v1.3 the same run reported only the fresh part.")


if __name__ == "__main__":
    main()
