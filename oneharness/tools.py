"""Tool definition, schema generation and execution."""

from __future__ import annotations

import inspect
import re
import types
import typing
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal, get_args, get_origin

from oneharness.errors import ToolError
from oneharness.types import ToolCall, ToolResult, dumps

_UNION_TYPE = getattr(types, "UnionType", None)  # `int | None` on 3.10+
_NAME = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_-]{0,63}$")
_PRIMITIVES: dict[Any, str] = {str: "string", int: "integer", float: "number", bool: "boolean"}


@dataclass(frozen=True)
class Tool:
    """A callable exposed to the model, with a JSON schema for its arguments."""

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]

    def spec(self) -> dict[str, Any]:
        """The schema shape shared by every provider (OpenAI style)."""
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Any:
    """Turn a function into a Tool, deriving the argument schema from its signature."""

    def wrap(func: Callable[..., Any]) -> Tool:
        tool_name = name or func.__name__
        if not _NAME.match(tool_name):
            raise ToolError(f"invalid tool name {tool_name!r}")
        doc = description if description is not None else (inspect.getdoc(func) or "").strip()
        return Tool(
            name=tool_name,
            description=doc.split("\n\n")[0].strip() or tool_name,
            parameters=_schema_for(func),
            fn=func,
        )

    return wrap(fn) if fn is not None else wrap


def _schema_for(func: Callable[..., Any]) -> dict[str, Any]:
    try:
        hints = typing.get_type_hints(func)
    except Exception:  # unresolved forward refs: fall back to untyped params
        hints = {}

    signature = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param in signature.parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        properties[param.name] = _describe(hints.get(param.name, Any))
        if param.default is inspect.Parameter.empty:
            required.append(param.name)
        else:
            default = param.default
            if default is not None and not isinstance(default, (list, dict)):
                properties[param.name]["default"] = default

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _describe(annotation: Any) -> dict[str, Any]:
    """Map a Python annotation onto a JSON schema fragment."""
    if annotation in _PRIMITIVES:
        return {"type": _PRIMITIVES[annotation]}
    if annotation is Any or annotation is inspect.Parameter.empty:
        return {"type": "string"}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if annotation is dict:
        return {"type": "object"}
    if annotation in (list, set, tuple, frozenset):
        return {"type": "array", "items": {"type": "string"}}

    if origin is Literal:
        values = list(args)
        schema: dict[str, Any] = {"enum": values}
        kinds = {_PRIMITIVES.get(type(v)) for v in values}
        if len(kinds) == 1 and None not in kinds:
            schema["type"] = kinds.pop()
        return schema
    if origin in (list, set, tuple, frozenset):
        item = _describe(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item}
    if origin is dict:
        return {"type": "object"}
    if origin is typing.Union or origin is _UNION_TYPE:
        members = [a for a in args if a is not type(None)]
        if len(members) == 1:
            return _describe(members[0])
        return {"anyOf": [_describe(m) for m in members]}
    if annotation is type(None):
        return {"type": "null"}
    return {"type": "string"}


class Registry:
    """An ordered, name-indexed collection of tools."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for item in tools:
            self.add(item)

    def add(self, item: Tool) -> Tool:
        if not isinstance(item, Tool):
            raise ToolError(f"expected a Tool, got {type(item).__name__}")
        if item.name in self._tools:
            raise ToolError(f"tool {item.name!r} is already registered")
        self._tools[item.name] = item
        return item

    def remove(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        """Schemas for providers that take a `tools` list; empty when unused."""
        return [item.spec() for item in self._tools.values()]

    def execute(self, call: ToolCall) -> ToolResult:
        """Run one tool call, converting any failure into a model-readable result."""
        item = self._tools.get(call.name)
        if item is None:
            available = ", ".join(self.names()) or "none"
            return ToolResult(
                call.id,
                call.name,
                f"error: unknown tool {call.name!r}; available: {available}",
                is_error=True,
            )
        if call.arguments_error:
            return ToolResult(call.id, call.name, f"error: {call.arguments_error}", is_error=True)

        try:
            value = item(**call.arguments)
        except ToolError as exc:
            return ToolResult(call.id, call.name, f"error: {exc}", is_error=True)
        except TypeError as exc:
            return ToolResult(call.id, call.name, f"error: bad arguments: {exc}", is_error=True)
        except Exception as exc:  # a broken tool must not kill the agent loop
            return ToolResult(
                call.id, call.name, f"error: {type(exc).__name__}: {exc}", is_error=True
            )
        return ToolResult(call.id, call.name, value if isinstance(value, str) else dumps(value))
