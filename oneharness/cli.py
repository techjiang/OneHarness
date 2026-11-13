"""Command line interface: `oneharness run`, `chat`, `tools`, `config`."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import threading
from collections.abc import Sequence
from typing import Any

from oneharness import __version__, transport
from oneharness import doctor as doctor_module
from oneharness.agent import Agent, AgentResult
from oneharness.builtin import default_tools
from oneharness.config import Config, McpServerConfig, load_config
from oneharness.delegate import Delegator
from oneharness.errors import ConfigError, OneHarnessError
from oneharness.mcp import McpClient, McpError
from oneharness.providers import create_provider, list_presets, preset_defaults
from oneharness.tape import Tape
from oneharness.tools import Registry, Tool
from oneharness.types import Message

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oneharness",
        description="Run a tool-calling LLM agent against any provider with one interface.",
    )
    parser.add_argument("--version", action="version", version=f"oneharness {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("-c", "--config", help="path to oneharness.toml")
        sub.add_argument("-p", "--provider", choices=list_presets(), help="provider preset")
        sub.add_argument("-m", "--model", help="model name")
        sub.add_argument("-s", "--system", help="system prompt")
        sub.add_argument("--max-steps", type=int, help="max tool round-trips (default 8)")
        sub.add_argument("--workspace", default=".", help="workspace root for file tools")
        sub.add_argument("--shell", action="store_true", help="enable the run_shell tool")
        sub.add_argument("--no-tools", action="store_true", help="run without any tools")
        sub.add_argument(
            "--parallel",
            type=int,
            metavar="N",
            help="run up to N tool calls of one turn at once (default 1, sequential)",
        )
        sub.add_argument(
            "--context-budget",
            type=int,
            metavar="TOKENS",
            help="token ceiling for one request; old tool results are elided to fit",
        )
        sub.add_argument(
            "--cache",
            action="store_true",
            help="place prompt-cache breakpoints (Anthropic/Bedrock); others cache on their own",
        )
        sub.add_argument(
            "--delegate",
            action="store_true",
            help="expose `delegate` / `delegate_many` tools that hand work to sub agents",
        )
        sub.add_argument(
            "--record",
            metavar="PATH",
            help="write every provider exchange of this run to a tape file",
        )
        sub.add_argument(
            "--replay",
            metavar="PATH",
            help="answer provider requests from a tape instead of the network (no key needed)",
        )
        sub.add_argument("--json", action="store_true", help="emit machine-readable JSON")
        sub.add_argument("--quiet", action="store_true", help="hide progress events")
        sub.add_argument("--stream", action="store_true", help="print tokens as they arrive")
        sub.add_argument(
            "--timing",
            action="store_true",
            help="print where the run's wall clock went (model vs tools, per step)",
        )
        sub.add_argument(
            "--mcp",
            action="append",
            default=[],
            metavar="CMD",
            help="spawn an MCP server and expose its tools (repeatable): --mcp 'npx -y server'",
        )

    run = subparsers.add_parser("run", help="run one prompt and print the answer")
    run.add_argument("prompt", nargs="+", help="the prompt to send")
    add_common(run)

    chat = subparsers.add_parser("chat", help="interactive multi-turn session")
    add_common(chat)

    subparsers.add_parser("tools", help="list the built-in tool names")
    subparsers.add_parser("presets", help="list provider presets and their default endpoints")

    doctor = subparsers.add_parser(
        "doctor",
        help="send one real minimal request per provider and classify the answer",
    )
    doctor.add_argument(
        "providers",
        nargs="*",
        metavar="PRESET",
        help="presets to check; default is every preset with credentials present",
    )
    doctor.add_argument(
        "--all", action="store_true", help="check every preset, not only the configured ones"
    )
    doctor.add_argument("--timeout", type=float, default=20.0, help="seconds per request")
    doctor.add_argument(
        "-m", "--model", help="override the model/deployment name for every preset checked"
    )
    doctor.add_argument("--base-url", help="override the endpoint for every preset checked")
    doctor.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    doctor.add_argument("--record", metavar="PATH", help="save the exchanges to a tape file")
    doctor.add_argument(
        "--replay",
        metavar="PATH",
        help="classify a recorded exchange instead of asking the network",
    )

    config_cmd = subparsers.add_parser("config", help="print the resolved configuration")
    config_cmd.add_argument("-c", "--config", help="path to oneharness.toml")

    return parser


def _resolve_config(args: argparse.Namespace) -> Config:
    config = load_config(getattr(args, "config", None))
    if getattr(args, "provider", None):
        config.provider.preset = args.provider
    if getattr(args, "model", None):
        config.provider.model = args.model
    if getattr(args, "system", None):
        config.system = args.system
    if getattr(args, "max_steps", None) is not None:
        if args.max_steps < 1:
            raise ConfigError("--max-steps must be at least 1")
        config.max_steps = args.max_steps
    if getattr(args, "parallel", None) is not None:
        if args.parallel < 1:
            raise ConfigError("--parallel must be at least 1")
        config.max_parallel = args.parallel
    if getattr(args, "context_budget", None) is not None:
        if args.context_budget < 0:
            raise ConfigError("--context-budget must be 0 (unlimited) or a positive token count")
        config.context_budget = args.context_budget
    if getattr(args, "cache", False):
        config.provider.cache = True
    if getattr(args, "delegate", False):
        config.delegate.enabled = True
    return config


def _build_agent(
    args: argparse.Namespace, config: Config, observer=None, extra=()
) -> tuple[Agent, Delegator | None]:
    provider = create_provider(config.provider.preset, **config.provider.options())
    tools: list[Tool] = []
    if not getattr(args, "no_tools", False):
        tools = default_tools(args.workspace, shell=getattr(args, "shell", False))
    tools.extend(extra)

    delegator: Delegator | None = None
    if config.delegate.enabled:
        # The sub agent gets the same tools minus `delegate` itself, so it can
        # never recurse, and the same provider, so the parent can hand off a
        # task without a second round of configuration.
        delegator = Delegator(
            provider,
            tools,
            system=config.delegate.system,
            max_steps=config.delegate.max_steps,
            max_parallel=config.delegate.max_parallel or config.max_parallel,
            on_event=observer,
            # Same model, same window: a sub agent inherits the ceiling rather
            # than getting a second knob that can drift out of sync with it.
            context_budget=config.context_budget,
        )
        tools.append(delegator.tool())
        tools.append(delegator.batch_tool())

    agent = Agent(
        provider,
        Registry(tools),
        system=config.system,
        max_steps=config.max_steps,
        max_parallel=config.max_parallel,
        on_event=observer,
        context_budget=config.context_budget,
    )
    return agent, delegator


def _mcp_servers(args: argparse.Namespace, config: Config) -> list[McpServerConfig]:
    """Config-file servers plus any `--mcp 'cmd args'` given on the command line."""
    servers = list(config.mcp)
    for raw in getattr(args, "mcp", ()) or ():
        parts = shlex.split(raw)
        if not parts:
            raise ConfigError("--mcp needs a command")
        servers.append(McpServerConfig(command=parts))
    return servers


def _connect_mcp(
    servers: list[McpServerConfig], observer=None
) -> tuple[list[Tool], list[McpClient]]:
    """Spawn every configured server and collect the tools it offers."""
    tools: list[Tool] = []
    clients: list[McpClient] = []
    for server in servers:
        client = McpClient(server.command, cwd=server.cwd, timeout=server.timeout)
        try:
            client.start()
            exposed = client.tools(prefix=server.prefix)
        except McpError as exc:
            client.close()
            for started in clients:
                started.close()
            # The error already names its server; wrapping it in the command
            # again produced "MCP server 'python3': MCP server 'python3' ...".
            raise McpError(str(exc)) from exc

        taken = {item.name for item in tools}
        clashes = sorted({item.name for item in exposed} & taken)
        if clashes:
            client.close()
            for started in clients:
                started.close()
            raise ConfigError(
                f"MCP server {client.name!r} exposes tools that already exist: "
                f"{', '.join(clashes)}; set `prefix` for that server to avoid the clash"
            )

        clients.append(client)
        for item in exposed:
            if observer is not None:
                observer({"type": "mcp_tool", "server": client.name, "name": item.name})
        tools.extend(exposed)
    return tools, clients


class _Printer:
    """The observer, plus a count of how much answer text actually reached stdout.

    The count matters: `_report` must not assume a streamed run printed
    anything. A provider that answers in one `done` event (or one that does not
    implement `stream()` at all) produces no deltas, and "the tokens already
    went to stdout" would then print a blank line instead of the answer.
    """

    def __init__(self, stream: bool, quiet: bool = False) -> None:
        self.stream = stream
        self.quiet = quiet
        self.open_line = False
        # Which assembled call currently owns the growing stderr line, if any.
        # Two calls in one turn interleave their fragments, so the line has to
        # say whose arguments these are -- otherwise the reader cannot tell
        # whether a `,` belongs to the first call or the second.
        self.drawing: str | None = None
        self.announced: dict[str, str] = {}  # call id -> the label it was drawn under
        self.chars = 0
        self._lock = threading.Lock()

    def __call__(self, event: dict[str, Any]) -> None:
        if self.quiet:
            return
        with self._lock:
            _render(event, self, self.stream)

    @property
    def printed_text(self) -> bool:
        return self.chars > 0


def _printer(quiet: bool, stream: bool = False):
    """Progress on stderr, streamed answer text on stdout.

    Streaming puts tokens on stdout without newlines, so anything printed to
    stderr in between would land on the same visual line; close the line first.

    Sub-agent events arrive with `depth: 1` and a `label`, and get indented and
    tagged: three sub agents interleaving their steps in one unmarked column is
    unreadable, and the tag is what makes the parallelism legible. Those events
    come from several threads at once, so the whole thing sits behind one lock.
    """
    return _Printer(stream, quiet)


def _render(event: dict[str, Any], state: Any, stream: bool) -> None:
    kind = event.get("type")
    depth = int(event.get("depth", 0))
    label = event.get("label")
    pad = "    " * depth
    tag = f"{pad}[{label}] " if label else pad

    if kind == "tool_call_delta":
        # Drawn as a growing line: the arguments exist one fragment at a time,
        # and showing that growth is the whole point of surfacing them. When the
        # call is finally complete this line is simply closed, instead of being
        # printed a second time by `tool_start`.
        _render_tool_delta(event, state, tag)
        return

    if kind == "text_delta" and stream:
        # A sub agent's tokens are not the answer the user asked for; they
        # would splice straight into the parent's sentence on stdout.
        if not depth:
            piece = event.get("text", "")
            print(piece, end="", flush=True)
            state.open_line = True
            state.chars += len(piece)
        return

    if state.open_line:
        print(flush=True)  # keep streamed text and tool chatter on separate lines
        state.open_line = False

    if kind == "tool_start" and str(event.get("id")) in state.announced:
        # The growing line already showed this tool and its arguments; all that
        # is left is to close the parenthesis it left open, if it is still the
        # line on screen.
        if state.drawing == state.announced.pop(str(event.get("id"))):
            print(")", file=sys.stderr)
            state.drawing = None
        return

    _close_assembly(state)

    if kind == "tool_start":
        args = json.dumps(event.get("arguments", {}), ensure_ascii=False)
        if len(args) > 140:
            args = args[:137] + "..."
        print(f"{tag}→ {event['name']}({args})", file=sys.stderr)
    elif kind == "tool_end":
        mark = "!" if event.get("is_error") else "✓"
        preview = str(event.get("content", "")).replace("\n", " ")[:120]
        print(f"{tag}{mark} {event['name']}: {preview}", file=sys.stderr)
    elif kind == "context_trim":
        print(
            f"{tag}~ context {event['tokens_before']}→{event['tokens_after']} tokens "
            f"(budget {event['budget']}, elided {event['elided']}, dropped {event['dropped']})",
            file=sys.stderr,
        )
        # A trim that could not reach the budget is not a success, and it is not
        # a silent overshoot either: say which part refused to shrink.
        if event.get("unmet"):
            left = f"{event['unmet']} tokens over"
            if event.get("tools", 0) > event["budget"]:
                left += f" -- the tool schemas alone are {event['tools']}"
            print(f"{tag}! still {left}: the rest is the prompt itself", file=sys.stderr)
    elif kind == "context_overflow":
        print(
            f"{tag}! provider refused {event['estimated']} tokens; retrying under "
            f"{event['retry_budget']}",
            file=sys.stderr,
        )
    elif kind == "provider_retry":
        # The wait is announced *before* it is taken: three silent seconds read
        # as a hang, and a person who thinks it hung starts Ctrl-C-ing the very
        # run the wait was protecting.
        status = event.get("status")
        why = f"HTTP {status}" if status else "connection lost"
        print(
            f"{tag}~ {why}, waiting {_seconds(event['delay'])} "
            f"({event['source']}) then retrying "
            f"[{event['attempt']}/{event['retries']}]",
            file=sys.stderr,
        )
    elif kind == "mcp_tool":
        print(f"{tag}+ mcp {event['server']}: {event['name']}", file=sys.stderr)
    elif kind == "interrupted":
        print(
            f"{tag}! {event['reason']} -- kept {event['chars']} chars it had already said",
            file=sys.stderr,
        )
    elif kind == "subagent_start":
        task = str(event.get("task", "")).replace("\n", " ")
        print(f"{tag}⇢ {task[:100]}", file=sys.stderr)
    elif kind == "subagent_end":
        usage = event.get("usage") or {}
        print(
            f"{tag}⇠ done: {event.get('steps')} step(s), {usage.get('total_tokens', 0)} tokens",
            file=sys.stderr,
        )


def _close_assembly(state: _Printer) -> None:
    """End a growing tool line, so the next message starts on its own row."""
    if state.drawing is None:
        return
    # `…` not `)`: these arguments may still be arriving, and a closing bracket
    # would claim the call is complete when it is not.
    print("…", file=sys.stderr)
    state.drawing = None


def _render_tool_delta(event: dict[str, Any], state: _Printer, tag: str) -> None:
    """One growing stderr line per tool call being assembled.

    The label carries the provider's own index (`write_file#0`), because two
    calls in one turn interleave their fragments and a reader has to be able to
    tell whose argument this is. Reading the label along with the fragment
    reconstructs both calls, which no amount of pretty-printing can replace.
    """
    if state.open_line:
        print(flush=True)
        state.open_line = False

    call_id = str(event.get("id"))
    label = f"{event.get('name', '')}#{event.get('index', 0)}"
    state.announced[call_id] = label

    if state.drawing != label:
        if state.drawing is not None:
            print("…", file=sys.stderr)
        print(f"{tag}… {label}(", end="", file=sys.stderr, flush=True)
        state.drawing = label
    _write_fragment(event.get("fragment") or "")


def _write_fragment(fragment: str) -> None:
    """Newlines and tabs are shown escaped: this is one line that keeps growing.

    A fragment of `"content": "timeout=30\n"` printed literally would break the
    line into pieces and push the rest of the call off the screen, which defeats
    the point of drawing it at all.
    """
    if not fragment:
        return
    escaped = fragment.replace("\n", "\\n").replace("\t", "\\t")
    print(escaped, end="", file=sys.stderr, flush=True)


def _report(
    result: AgentResult,
    as_json: bool,
    streamed: bool = False,
    delegation: dict[str, Any] | None = None,
) -> None:
    """`streamed` means the answer text already reached stdout, not "was a stream"."""
    if as_json:
        payload = result.as_dict()
        if delegation is not None:
            payload["delegation"] = delegation
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if streamed and result.text:
        print()  # tokens already went to stdout; just close the line
    else:
        print(result.text or "(no text returned)")
    if result.stopped_reason == "max_steps":
        print(
            f"[stopped after {len(result.steps)} steps without a final answer]",
            file=sys.stderr,
        )
    elif result.interrupted:
        # The reason itself was already printed by the `interrupted` event while
        # the run was going; repeating it here would just be the same sentence
        # twice. What matters at the end is that the answer is not whole.
        print("[the answer above is incomplete]", file=sys.stderr)


def _report_timing(result: AgentResult, delegation: dict[str, Any] | None = None) -> None:
    """A per-step split on stderr: the answer goes to stdout, this is diagnosis.

    Nothing here is rounded into agreement. `other` is whatever the loop did not
    measure around its two waits, and it is printed even when it is the largest
    number -- a breakdown whose parts always add up exactly is a breakdown that
    is inventing the missing seconds.

    Delegated time gets its own line because it would otherwise be filed under
    "tools": a `delegate` call *is* a tool call, but what it waits for is
    another agent thinking. Folding it in would make every fan-out look like a
    slow shell.
    """
    timing = result.timing
    total = timing["total_seconds"]
    print(
        f"[timing] {timing['steps']} step(s), {timing['tool_calls']} tool call(s), "
        f"{_seconds(total)} total",
        file=sys.stderr,
    )
    width = len(str(len(result.steps)))
    for number, step in enumerate(result.steps, start=1):
        line = (
            f"{number:>{width}}  {_seconds(step.seconds):>8}  "
            f"model {_seconds(step.model_seconds):>8}  tools {_seconds(step.tool_seconds):>8}"
        )
        if step.first_token_seconds is not None:
            line += f"  first token {_seconds(step.first_token_seconds)}"
        if step.tool_results:
            slowest = max(step.tool_results, key=lambda item: item.seconds)
            line += f"  slowest call {slowest.name} {_seconds(slowest.seconds)}"
        print("     " + line, file=sys.stderr)

    model = timing["model_seconds"]
    tools = timing["tool_seconds"]
    other = timing["unaccounted_seconds"]
    print(
        f"     model {_seconds(model)} ({_share(model, total)})  "
        f"tools {_seconds(tools)} ({_share(tools, total)})  "
        f"other {_seconds(other)} ({_share(other, total)})",
        file=sys.stderr,
    )
    # Sum of every call against the wall clock of the phase they ran in. Two
    # calls in 60ms each either take 120ms of wall clock (serial) or roughly 60
    # (overlapping), so the ratio -- not the difference -- is the signal. A fixed
    # tolerance like "bigger by 50ms" calls 60+60 vs 80 serial, which is how a
    # slow machine quietly turned this line off.
    calls = timing["tool_call_seconds"]
    if calls > tools * 1.25:
        print(
            f"     calls summed {_seconds(calls)} vs wall clock {_seconds(tools)}: "
            f"{calls / tools:.1f}x, so they overlapped and the gap is what "
            f"--parallel bought",
            file=sys.stderr,
        )
    requests = timing["model_requests"]
    if requests > timing["steps"]:
        print(
            f"     {requests} request(s) for {timing['steps']} step(s): "
            f"{requests - timing['steps']} of them were refused for size and retried",
            file=sys.stderr,
        )
    delegated = (delegation or {}).get("seconds") or 0.0
    if delegated:
        print(
            f"     {_seconds(delegated)} of the tool time was sub agents thinking "
            f"({(delegation or {}).get('runs', 0)} run(s)), not a slow tool",
            file=sys.stderr,
        )
    if timing["first_token_seconds"] is not None:
        print(
            f"     first token {_seconds(timing['first_token_seconds'])} on average "
            "(streamed steps only)",
            file=sys.stderr,
        )

    slowest = timing["slowest_step"]
    if slowest is not None:
        print(
            f"     slowest step {slowest['step']} at {_seconds(slowest['seconds'])}",
            file=sys.stderr,
        )
    # Only worth printing when the model time is big enough to divide by: on a
    # local fake provider this line reads 12000 tok/s, which is a unit error
    # dressed up as a measurement.
    firsts = timing["first_token_seconds"]
    output = result.usage.output_tokens
    if output and model >= 0.2:
        if firsts is not None and model > firsts:
            # What the model actually wrote per second, with the thinking
            # stripped out: that is the number a shorter prompt can improve.
            print(
                f"     {output} output token(s) written in {_seconds(model - firsts)} "
                f"after the first one = {output / (model - firsts):.0f} tok/s",
                file=sys.stderr,
            )
        else:
            print(
                f"     {output} output token(s) over {_seconds(model)} incl. the wait "
                f"= {output / model:.0f} tok/s",
                file=sys.stderr,
            )


def _report_cache(result: AgentResult, *, quiet: bool = False) -> None:
    """How much of the prompt came back from the cache, when any did.

    Silence when nothing was cached, on purpose: a `cache: 0 of 0` line would
    show up on every run against a provider that has no cache at all, and a
    number that is always zero stops being read.
    """
    if quiet:
        return
    usage = result.usage
    if not usage.cached_tokens and not usage.cache_write_tokens:
        return
    line = (
        f"[cache] {usage.cached_tokens} of {usage.input_tokens} prompt tokens read "
        f"({usage.cache_hit_rate * 100:.0f}%)"
    )
    if usage.cache_write_tokens:
        line += f", {usage.cache_write_tokens} written"
    print(line, file=sys.stderr)


def _report_estimate(result: AgentResult, *, quiet: bool = False) -> None:
    """The estimator against the provider's own prompt count, summed over the run.

    This is the only place the two can meet. `tokens.py` counts characters; the
    provider counts tokens. Independent sources are what makes the comparison
    worth printing at all -- and v1.2 claimed a 5% average error, so here is
    that claim being checked against a real counter rather than a frozen
    fixture.

    Summed over steps, and each step contributes the estimate of the list it
    actually sent (after trimming). Comparing a trimmed request against the
    untrimmed transcript would produce a gap that is not the estimator's fault.

    Skipped when nothing was estimated or nothing was reported: a provider that
    does not count tokens cannot disagree with anything.
    """
    if quiet:
        return
    estimated = sum(step.prompt_tokens_estimated for step in result.steps)
    reported = result.usage.input_tokens
    if estimated <= 0 or reported <= 0:
        return
    ratio = estimated / reported
    verdict = f"{abs(ratio - 1) * 100:.0f}% off" if abs(ratio - 1) > 0.02 else "within 2%"
    print(
        f"[estimate] {estimated} token(s) estimated vs {reported} counted by the provider "
        f"over {len(result.steps)} step(s): {verdict}",
        file=sys.stderr,
    )


def _seconds(value: float) -> str:
    """Milliseconds below a second, seconds above: 0.004s reads as noise."""
    return f"{value * 1000:.0f}ms" if value < 1 else f"{value:.2f}s"


def _share(value: float, total: float) -> str:
    return f"{value / total * 100:.0f}%" if total > 0 else "n/a"


def _report_delegation(delegator: Delegator | None, quiet: bool = False) -> None:
    """Sub-agent tokens are not the parent's, so they get their own line.

    It goes to stderr even under `--json`: stdout stays a pure document, and a
    human watching the run still learns that someone else spent tokens.
    """
    if delegator is None or not delegator.runs or quiet:
        return
    usage = delegator.usage
    print(
        f"[delegated {delegator.runs} task(s): "
        f"{usage.input_tokens} in / {usage.output_tokens} out tokens]",
        file=sys.stderr,
    )
    # Per-sub-agent numbers only when there is more than one: a single run would
    # just repeat the total on a second line.
    if delegator.runs > 1:
        for label, spent in delegator.by_label.items():
            print(
                f"  {label}: {spent.input_tokens} in / {spent.output_tokens} out",
                file=sys.stderr,
            )


def _with_mcp(
    args: argparse.Namespace, config: Config, observer
) -> tuple[list[Tool], list[McpClient]]:
    servers = _mcp_servers(args, config)
    if not servers:
        return [], []
    return _connect_mcp(servers, observer)


def cmd_run(args: argparse.Namespace) -> int:
    config = _resolve_config(args)
    stream = args.stream and not args.json
    observer = _printer(args.quiet, stream)
    with _tape(args) as tape, _retries(observer):
        tools, clients = _with_mcp(args, config, observer)
        try:
            agent, delegator = _build_agent(args, config, observer, tools)
            result = agent.run(" ".join(args.prompt), stream=stream)
        finally:
            for client in clients:
                client.close()
    delegation = delegator.as_dict() if delegator is not None and delegator.runs else None
    # Not `stream`: a streamed run whose tokens never reached stdout (a provider
    # that answers in one `done` event) still needs the answer printed here.
    _report(result, args.json, stream and observer.printed_text, delegation)
    if args.timing and not args.json:
        _report_timing(result, delegation)
    _report_cache(result, quiet=args.quiet)
    if args.timing and not args.json:
        # Diagnosis, not a headline: `--timing` is where numbers about numbers
        # live. The cache line above stays unconditional because it is about
        # money, and `--json` carries both in `usage`.
        _report_estimate(result, quiet=args.quiet)
    _report_delegation(delegator, args.quiet)
    if tape is not None and not args.quiet:
        _report_tape(tape)
    if result.interrupted_by_user:
        return EXIT_INTERRUPTED  # what a shell expects from Ctrl-C
    return EXIT_OK if result.ok else EXIT_ERROR


def cmd_chat(args: argparse.Namespace) -> int:
    config = _resolve_config(args)
    stream = args.stream
    observer = _printer(args.quiet, stream)
    with _tape(args), _retries(observer):
        return _chat(args, config, observer, stream)


def _chat(args: argparse.Namespace, config: Config, observer, stream: bool) -> int:
    tools, clients = _with_mcp(args, config, observer)
    history: list[Message] = []

    try:
        agent, _ = _build_agent(args, config, observer, tools)
        if not args.quiet:
            print(f"oneharness {__version__} · {config.provider.preset}/{agent.provider.model}")
            print("Type your message. Ctrl-D or 'exit' to quit.\n")

        while True:
            try:
                line = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return EXIT_OK
            if not line:
                continue
            if line in ("exit", "quit", ":q"):
                return EXIT_OK

            try:
                result = agent.run(line, history=history, stream=stream)
            except OneHarnessError as exc:
                print(f"error: {exc}", file=sys.stderr)
                continue

            history = [m for m in result.messages if m.role != "system"]
            if stream and result.text:
                print()
            else:
                print(f"\nassistant> {result.text}\n")
            if args.timing:
                _report_timing(result)
    finally:
        for client in clients:
            client.close()


def cmd_tools(_: argparse.Namespace) -> int:
    for item in default_tools(".", shell=True):
        params = ", ".join(item.parameters.get("properties", {}))
        print(f"{item.name}({params})\n    {item.description}")
    return EXIT_OK


def cmd_presets(_: argparse.Namespace) -> int:
    print(f"{'preset':<12} {'default model':<44} base_url")
    for name in list_presets():
        defaults = preset_defaults(name)
        model = defaults["default_model"] or "(deployment name required)"
        base_url = defaults["base_url"] or "(set AZURE_OPENAI_ENDPOINT)"
        print(f"{name:<12} {model:<44} {base_url}")
    return EXIT_OK


def cmd_config(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    # ensure_ascii=False: this is a human debugging command, not a data feed.
    print(json.dumps(config.as_dict(), indent=2, ensure_ascii=False))
    return EXIT_OK


def _tape(args: argparse.Namespace) -> Tape:
    """`--record` / `--replay`, or a no-op that installs nothing.

    Replay is the reason this exists as a context manager: a recorded run can be
    replayed on a machine with no keys and no network, and the provider code
    path is byte-for-byte the one that talked to the real endpoint.
    """
    record = getattr(args, "record", None)
    replay = getattr(args, "replay", None)
    if record and replay:
        raise ConfigError("--record and --replay are different runs; pick one")
    if replay:
        return Tape.replay(replay)
    if record:
        return Tape.record(record)
    return _NullTape()


class _retries:
    """Route transport retry notices to the printer for the duration of a command.

    Process-wide while it lasts (a retry happens deep inside a provider call, on
    whatever thread), and cleared afterwards: a leftover sink would let one
    command's terminal output leak into the next caller's library code.
    """

    def __init__(self, observer: Any) -> None:
        self.observer = observer

    def __enter__(self) -> None:
        transport.watch_retries(self.observer)

    def __exit__(self, *exc: Any) -> None:
        transport.watch_retries(None)


class _NullTape:
    """Stands in for `Tape` when the flag was not given, without installing."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> None:
        return None


def _report_tape(tape: Tape) -> None:
    """One line on stderr: a tape is a side effect, the answer is the product."""
    if tape.replaying:
        print(f"[replayed {tape.matched} exchange(s) from {tape.path}]", file=sys.stderr)
    else:
        print(f"[recorded {tape.recorded} exchange(s) to {tape.path}]", file=sys.stderr)


def cmd_doctor(args: argparse.Namespace) -> int:
    names = list(args.providers)
    if not names:
        names = list_presets() if args.all else doctor_module.configured_presets() or list_presets()
    unknown = [name for name in names if name not in list_presets()]
    if unknown:
        raise ConfigError(
            f"unknown provider(s): {', '.join(unknown)}; known: {', '.join(list_presets())}"
        )

    overrides = {
        key: value for key, value in (("model", args.model), ("base_url", args.base_url)) if value
    }
    with _tape(args):
        probes = doctor_module.check_many(names, timeout=args.timeout, **overrides)

    if args.json:
        print(json.dumps([probe.as_dict() for probe in probes], indent=2, ensure_ascii=False))
    else:
        _print_probes(probes)
    return EXIT_ERROR if any(probe.failed for probe in probes) else EXIT_OK


def _print_probes(probes: Sequence[doctor_module.Probe]) -> None:
    """A table on stdout: this is the whole product, so it is not stderr chatter."""
    width = max(len(probe.preset) for probe in probes)
    marks = {
        doctor_module.OK: "✓",
        doctor_module.AUTH: "✗",
        doctor_module.MODEL: "✗",
        doctor_module.ENDPOINT: "✗",
        doctor_module.QUOTA: "!",
        doctor_module.PROTOCOL: "✗",
        doctor_module.CONTEXT: "✗",
        doctor_module.TLS: "✗",
        doctor_module.PROXY: "✗",
        doctor_module.NETWORK: "✗",
        doctor_module.UNCONFIGURED: "-",
    }
    for probe in probes:
        mark = marks.get(probe.status, "?")
        print(f"{mark} {probe.preset:<{width}}  {probe.status:<12} {probe.detail}")
        if probe.status != doctor_module.OK and probe.hint:
            print(f"  {'':<{width}}  {probe.hint}")


COMMANDS = {
    "run": cmd_run,
    "chat": cmd_chat,
    "tools": cmd_tools,
    "presets": cmd_presets,
    "config": cmd_config,
    "doctor": cmd_doctor,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except OneHarnessError as exc:
        print(f"oneharness: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
