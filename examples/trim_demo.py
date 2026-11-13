"""What the context trimmer sends, and whether the endpoint accepts it.

    python3 examples/trim_demo.py

v1.7 fixed two things that were true in this repo for six versions, and both of
them were about the *shape* of a request rather than its size:

1. **The tool schemas were measured out of context.** `count_specs` summed one
   `json.dumps` per schema, so every separator and both brackets went
   unmeasured: the array on the wire is 876 characters where the sum of its
   parts describes 808. Small in relative terms, and in the flattering
   direction -- the budget looked roomier than it was -- which is why nobody
   chased it. It is still the wrong number feeding a hard limit.

2. **Trimming could produce a request OpenAI rejects.** A turn that asks for
   three files is one `assistant` message followed by three `tool` messages.
   Removing the turn and keeping the results leaves results that answer nothing,
   and the endpoint refuses the whole request:

       HTTP 400: messages with role 'tool' must be a response to a preceding
       message with 'tool_calls'

   A turn with a single call happened to delete cleanly, so the existing test
   passed and the bug survived.

Part 1 measures the schema count. Part 2 builds the transcript that used to
break, trims it with the current code, and hands the result to a local
OpenAI-shaped server that enforces the same rule the real one does. Part 3 does
the same for the old behaviour, so the difference is a pair of responses rather
than a claim.

No network, no API key.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oneharness import context as context_module  # noqa: E402
from oneharness.builtin import default_tools  # noqa: E402
from oneharness.errors import ProviderError  # noqa: E402
from oneharness.providers.openai import OpenAIProvider  # noqa: E402
from oneharness.tokens import count  # noqa: E402
from oneharness.types import Message, ToolCall  # noqa: E402


def banner(text: str) -> None:
    print(f"\n=== {text} ===")


def part_one() -> None:
    banner("1. the tool schemas, counted the old way and the real way")
    specs = [item.spec() for item in default_tools(".", shell=True)]
    separate = sum(count(json.dumps(spec, ensure_ascii=False)) for spec in specs)
    whole = count(json.dumps({"tools": specs}, ensure_ascii=False))
    wire = len(json.dumps({"tools": specs}, ensure_ascii=False))
    print(f"  {len(specs)} built-in tools")
    print(f"  per-schema sum, separators dropped   {separate}")
    print(f"  the array as it goes on the wire     {whole}   ({wire} chars)")
    print(f"  share of the cost the old count saw  {separate / whole:.0%}")
    print()
    print("  The difference is the framing: `},{` between the tools and the two")
    print("  brackets around the array. Forty separator characters on four tools")
    print("  is a small share; the point is that the number feeds a hard limit,")
    print("  so an estimate taken out of context is wrong in the direction that")
    print("  lets an oversized request through.")


def transcript() -> list[Message]:
    """The shape a tool loop produces when one turn asks for three files."""
    calls = [
        ToolCall.from_raw(call_id=f"c{index}", name="read_file", arguments={"path": path})
        for index, path in enumerate(("parser.py", "lexer.py", "notes.md"))
    ]
    messages = [Message.system("be brief"), Message.user("read all three and compare them")]
    messages.append(Message.assistant("", calls))
    messages.append(Message.assistant("plan: start with the parser, it owns the token boundary"))
    for index in range(3):
        messages.append(Message.tool(f"c{index}", f"line {index}\n" * 400, "read_file"))
    messages.append(Message.user("now tell me which one is wrong"))
    return messages


def old_trim(messages: list[Message], budget: int) -> list[Message]:
    """v1.6's pipeline, verbatim: elide results, then drop whole exchanges.

    The `_drop_middle` here is the one that shipped: it stepped over an
    assistant turn and then kept walking, which is what left the three results
    behind. `fit` in this module's parent repo now does the round instead.
    """
    working = list(messages)
    if context_module.estimate(working) > budget:
        context_module._elide_results(working, budget, None)
    head = 1 if working and working[0].role == "system" else 0
    if head < len(working) and working[head].role == "user":
        head += 1
    tail = head
    for index in range(len(working) - 1, head - 1, -1):
        if working[index].role == "user":
            tail = index
            break
    index = head
    while index < tail and context_module.estimate(working) > budget:
        end = index + 1
        if working[index].role == "assistant":
            while end < tail and working[end].role == "tool":
                end += 1
        del working[index:end]
        tail -= end - index
    return working


def orphans(messages: list[Message]) -> list[str]:
    declared = {call.id for message in messages for call in message.tool_calls}
    return [m.tool_call_id for m in messages if m.role == "tool" and m.tool_call_id not in declared]


def part_two() -> None:
    banner("2. a trimmed request, handed to an endpoint that checks the rule")
    source = transcript()
    # Below the elided size, above the size of everything except the oldest
    # round: the budget that has to drop a round to fit, which is where the two
    # versions disagree.
    budget = 150
    trimmed, record = context_module.fit(list(source), budget)
    print(f"  budget {budget}, {record.before} tokens before, {record.after} after")
    print(f"  dropped {record.dropped}, elided {record.elided}")
    print(f"  outgoing roles: {[m.role for m in trimmed]}")
    print(f"  orphaned results: {orphans(trimmed) or 'none'}")
    print()
    status, body = post_to_endpoint(trimmed)
    print(f"  endpoint says: HTTP {status}")
    print(f"    {body}")


def part_three() -> None:
    banner("3. the same transcript, trimmed the way v1.6 trimmed it")
    source = transcript()
    broken = old_trim(source, 150)
    print(f"  outgoing roles: {[m.role for m in broken]}")
    print(f"  orphaned results: {orphans(broken) or 'none'}")
    print()
    status, body = post_to_endpoint(broken)
    print(f"  endpoint says: HTTP {status}")
    print(f"    {body}")
    print()
    print('  That 400 is classified by `doctor` as `protocol`, i.e. "this is a')
    print('  OneHarness bug" -- and it is, but not in the file the message points')
    print("  at: the request never should have been shaped this way.")


def part_four() -> None:
    banner("4. why it survived: one call per turn deletes cleanly")
    one = [Message.system("s"), Message.user("read it")]
    call = ToolCall.from_raw(call_id="c0", name="read_file", arguments={})
    one.append(Message.assistant("", [call]))
    one.append(Message.tool("c0", "x" * 4000, "read_file"))
    one.append(Message.user("thanks, now fix it"))
    for label, build in (("v1.6", old_trim), ("now", lambda m, b: context_module.fit(m, b)[0])):
        out = build(list(one), 120)
        print(f"  {label}: orphaned results {orphans(out) or 'none'}")
    print()
    print("  The regression test that existed used exactly this shape, so it was")
    print("  green the whole time the three-call shape was producing 400s.")


def post_to_endpoint(messages: list[Message]) -> tuple[int, str]:
    """Send `messages` to a local server that enforces OpenAI's tool rule."""
    seen: dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            declared: set[str] = set()
            problem = None
            for message in body["messages"]:
                if message["role"] == "assistant":
                    declared |= {call["id"] for call in message.get("tool_calls") or ()}
                elif message["role"] == "tool" and message["tool_call_id"] not in declared:
                    problem = message["tool_call_id"]
                    break
            if problem:
                payload = {
                    "error": {
                        "message": (
                            "Invalid parameter: messages with role 'tool' must be a "
                            "response to a previous message with 'tool_calls' "
                            f"(missing {problem})"
                        ),
                        "type": "invalid_request_error",
                    }
                }
                raw = json.dumps(payload).encode()
                status = 400
            else:
                raw = json.dumps(
                    {
                        "id": "x",
                        "object": "chat.completion",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "ok"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    }
                ).encode()
                status = 200
            seen["message"] = json.loads(raw)["error"]["message"] if status == 400 else "accepted"
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    provider = OpenAIProvider(
        "gpt-demo", api_key="sk-demo", base_url=f"http://127.0.0.1:{server.server_port}/v1"
    )
    try:
        provider.complete(messages)
        status = 200
    except ProviderError as exc:
        status = exc.status or 0
    finally:
        server.shutdown()
        server.server_close()
    return status, seen.get("message", "accepted")


def main() -> None:
    part_one()
    part_two()
    part_three()
    part_four()
    print("\nParts 2 and 3 sent real requests to a local server that applies")
    print("OpenAI's rule. The 400 in part 3 is what the user saw; the 200 in")
    print("part 2 is the same transcript, trimmed whole instead of in halves.")


if __name__ == "__main__":
    main()
