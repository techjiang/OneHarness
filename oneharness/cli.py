"""Command line interface: `oneharness run`, `chat`, `tools`, `config`."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from oneharness import __version__
from oneharness.agent import Agent, AgentResult
from oneharness.builtin import default_tools
from oneharness.config import Config, load_config
from oneharness.errors import OneHarnessError
from oneharness.providers import create_provider, list_presets
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
        sub.add_argument("--json", action="store_true", help="emit machine-readable JSON")
        sub.add_argument("--quiet", action="store_true", help="hide progress events")

    run = subparsers.add_parser("run", help="run one prompt and print the answer")
    run.add_argument("prompt", nargs="+", help="the prompt to send")
    add_common(run)

    chat = subparsers.add_parser("chat", help="interactive multi-turn session")
    add_common(chat)

    subparsers.add_parser("tools", help="list the built-in tool names")

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
    if getattr(args, "max_steps", None):
        config.max_steps = args.max_steps
    return config


def _build_agent(args: argparse.Namespace, config: Config, observer=None) -> Agent:
    provider = create_provider(config.provider.preset, **config.provider.options())
    tools: list[Tool] = []
    if not getattr(args, "no_tools", False):
        tools = default_tools(args.workspace, shell=getattr(args, "shell", False))
    return Agent(
        provider,
        Registry(tools),
        system=config.system,
        max_steps=config.max_steps,
        on_event=observer,
    )


def _printer(quiet: bool):
    def observe(event: dict[str, Any]) -> None:
        if quiet:
            return
        kind = event.get("type")
        if kind == "tool_start":
            args = json.dumps(event.get("arguments", {}), ensure_ascii=False)
            print(f"  → {event['name']}({args})", file=sys.stderr)
        elif kind == "tool_end":
            mark = "!" if event.get("is_error") else "✓"
            preview = str(event.get("content", "")).replace("\n", " ")[:120]
            print(f"  {mark} {event['name']}: {preview}", file=sys.stderr)

    return observe


def _report(result: AgentResult, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
        return
    print(result.text or "(no text returned)")
    if result.stopped_reason == "max_steps":
        print(
            f"[stopped after {len(result.steps)} steps without a final answer]",
            file=sys.stderr,
        )


def cmd_run(args: argparse.Namespace) -> int:
    config = _resolve_config(args)
    agent = _build_agent(args, config, _printer(args.quiet))
    result = agent.run(" ".join(args.prompt))
    _report(result, args.json)
    return EXIT_OK if result.ok else EXIT_ERROR


def cmd_chat(args: argparse.Namespace) -> int:
    config = _resolve_config(args)
    agent = _build_agent(args, config, _printer(args.quiet))
    history: list[Message] = []

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
            result = agent.run(line, history=history)
        except OneHarnessError as exc:
            print(f"error: {exc}", file=sys.stderr)
            continue

        history = [m for m in result.messages if m.role != "system"]
        print(f"\nassistant> {result.text}\n")


def cmd_tools(_: argparse.Namespace) -> int:
    for item in default_tools(".", shell=True):
        params = ", ".join(item.parameters.get("properties", {}))
        print(f"{item.name}({params})\n    {item.description}")
    return EXIT_OK


def cmd_config(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    print(json.dumps(config.as_dict(), indent=2))
    return EXIT_OK


COMMANDS = {
    "run": cmd_run,
    "chat": cmd_chat,
    "tools": cmd_tools,
    "config": cmd_config,
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
