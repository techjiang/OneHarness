"""Command line interface: `oneharness run`, `chat`, `tools`, `config`."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import threading
from collections.abc import Sequence
from typing import Any

from oneharness import __version__
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
            raise McpError(f"MCP server {server.command[0]!r}: {exc}") from exc

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


def _printer(quiet: bool, stream: bool = False):
    """Progress on stderr, streamed answer text on stdout.

    Streaming puts tokens on stdout without newlines, so anything printed to
    stderr in between would land on the same visual line; close the line first.

    Sub-agent events arrive with `depth: 1` and a `label`, and get indented and
    tagged: three sub agents interleaving their steps in one unmarked column is
    unreadable, and the tag is what makes the parallelism legible. Those events
    come from several threads at once, so the whole thing sits behind one lock.
    """
    state = {"open_line": False}
    lock = threading.Lock()

    def observe(event: dict[str, Any]) -> None:
        if quiet:
            return
        with lock:
            _render(event, state, stream)

    return observe


def _render(event: dict[str, Any], state: dict[str, bool], stream: bool) -> None:
    kind = event.get("type")
    depth = int(event.get("depth", 0))
    label = event.get("label")
    pad = "    " * depth
    tag = f"{pad}[{label}] " if label else pad

    if kind == "text_delta" and stream:
        # A sub agent's tokens are not the answer the user asked for; they
        # would splice straight into the parent's sentence on stdout.
        if not depth:
            print(event.get("text", ""), end="", flush=True)
            state["open_line"] = True
        return

    if state["open_line"]:
        print(flush=True)  # keep streamed text and tool chatter on separate lines
        state["open_line"] = False

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
    elif kind == "context_overflow":
        print(
            f"{tag}! provider refused {event['estimated']} tokens; retrying under "
            f"{event['retry_budget']}",
            file=sys.stderr,
        )
    elif kind == "mcp_tool":
        print(f"{tag}+ mcp {event['server']}: {event['name']}", file=sys.stderr)
    elif kind == "subagent_start":
        task = str(event.get("task", "")).replace("\n", " ")
        print(f"{tag}⇢ {task[:100]}", file=sys.stderr)
    elif kind == "subagent_end":
        usage = event.get("usage") or {}
        print(
            f"{tag}⇠ done: {event.get('steps')} step(s), {usage.get('total_tokens', 0)} tokens",
            file=sys.stderr,
        )


def _report(
    result: AgentResult,
    as_json: bool,
    streamed: bool = False,
    delegation: dict[str, Any] | None = None,
) -> None:
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
    with _tape(args) as tape:
        tools, clients = _with_mcp(args, config, observer)
        try:
            agent, delegator = _build_agent(args, config, observer, tools)
            result = agent.run(" ".join(args.prompt), stream=stream)
        finally:
            for client in clients:
                client.close()
    delegation = delegator.as_dict() if delegator is not None and delegator.runs else None
    _report(result, args.json, stream, delegation)
    _report_delegation(delegator, args.quiet)
    if tape is not None and not args.quiet:
        _report_tape(tape)
    return EXIT_OK if result.ok else EXIT_ERROR


def cmd_chat(args: argparse.Namespace) -> int:
    config = _resolve_config(args)
    stream = args.stream
    observer = _printer(args.quiet, stream)
    with _tape(args):
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
