"""Configuration loading: defaults, TOML file, environment overrides."""

from __future__ import annotations

import os
import shlex
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from oneharness.errors import ConfigError

CONFIG_FILENAME = "oneharness.toml"
ENV_PREFIX = "ONEHARNESS_"


@dataclass(slots=True)
class ProviderConfig:
    preset: str = "openai"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    timeout: float = 120.0
    # Bedrock routes on region, not on the URL; Azure routes on a deployment
    # name in the path. Both are per-provider settings that belong next to the
    # model rather than in environment variables three layers away.
    region: str | None = None
    api_version: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)

    def options(self) -> dict[str, Any]:
        """Kwargs for `create_provider`, dropping unset values."""
        raw = {
            "model": self.model,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "region": self.region,
            "api_version": self.api_version,
            "extra_headers": self.extra_headers or None,
            "extra_body": self.extra_body or None,
        }
        return {key: value for key, value in raw.items() if value is not None}


@dataclass(slots=True)
class McpServerConfig:
    """One MCP server to spawn and borrow tools from."""

    command: list[str] = field(default_factory=list)
    prefix: str = ""
    cwd: str | None = None
    timeout: float = 30.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "command": list(self.command),
            "prefix": self.prefix,
            "cwd": self.cwd,
            "timeout": self.timeout,
        }


@dataclass(slots=True)
class DelegateConfig:
    """Settings for the `delegate` tool that hands work to a sub agent."""

    enabled: bool = False
    max_steps: int = 6
    # 0 means "inherit the top-level max_parallel": a fan-out of sub agents
    # rarely wants exactly the same width as the parent's tool batch.
    max_parallel: int = 0
    system: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_steps": self.max_steps,
            "max_parallel": self.max_parallel,
            "system": self.system,
        }


@dataclass(slots=True)
class Config:
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    system: str | None = None
    max_steps: int = 8
    max_parallel: int = 1
    # 0 = no ceiling. A token budget is a provider property, but it is the
    # caller's decision whether to enforce one at all.
    context_budget: int = 0
    delegate: DelegateConfig = field(default_factory=DelegateConfig)
    mcp: list[McpServerConfig] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": {
                "preset": self.provider.preset,
                "model": self.provider.model,
                "base_url": self.provider.base_url,
                "temperature": self.provider.temperature,
                "max_tokens": self.provider.max_tokens,
                "region": self.provider.region,
                "api_version": self.provider.api_version,
            },
            "system": self.system,
            "max_steps": self.max_steps,
            "max_parallel": self.max_parallel,
            "context_budget": self.context_budget,
            "delegate": self.delegate.as_dict(),
            "mcp": [server.as_dict() for server in self.mcp],
        }


def find_config_file(start: Path | None = None) -> Path | None:
    """Walk up from `start` looking for oneharness.toml, then check the home dir."""
    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        candidate = directory / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
    home_candidate = Path.home() / ".config" / CONFIG_FILENAME
    return home_candidate if home_candidate.is_file() else None


def load_config(path: Path | str | None = None) -> Config:
    """Load config from an explicit path, discovered file, and environment."""
    config = Config()

    resolved = Path(path) if path else find_config_file()
    if path and not resolved.is_file():
        raise ConfigError(f"config file not found: {path}")
    if resolved:
        config = _merge(config, _read_toml(resolved))

    return _apply_env(config)


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path} is not valid TOML: {exc}") from exc
    known = {
        "provider",
        "system",
        "max_steps",
        "max_parallel",
        "context_budget",
        "agent",
        "delegate",
        "mcp",
    }
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"{path} has unknown sections: {', '.join(sorted(unknown))}")
    return data


def _merge(config: Config, data: dict[str, Any]) -> Config:
    provider_data = data.get("provider", {})
    if not isinstance(provider_data, dict):
        raise ConfigError("[provider] must be a table")

    allowed = set(ProviderConfig.__slots__)
    unknown = set(provider_data) - allowed
    if unknown:
        raise ConfigError(f"unknown [provider] keys: {', '.join(sorted(unknown))}")
    for key, value in provider_data.items():
        setattr(config.provider, key, value)

    agent_data = data.get("agent", {})
    if not isinstance(agent_data, dict):
        raise ConfigError("[agent] must be a table")
    # Top level wins over [agent], which is the older spelling of the same keys.
    top = ("system", "max_steps", "max_parallel", "context_budget")
    settings = {**agent_data, **{key: data[key] for key in top if key in data}}
    if "system" in settings:
        config.system = settings["system"]
    if "max_steps" in settings:
        config.max_steps = _positive(settings["max_steps"], "max_steps")
    if "max_parallel" in settings:
        config.max_parallel = max(1, int(settings["max_parallel"]))
    if "context_budget" in settings:
        config.context_budget = _budget(settings["context_budget"], "context_budget")

    # A present `[delegate]` table turns the tool on; leaving it out keeps it off.
    if "delegate" in data:
        config.delegate = _read_delegate(data["delegate"])
    config.mcp = _read_mcp(data.get("mcp") or [])
    return config


def _positive(value: Any, name: str) -> int:
    """A count that has no meaning below 1: clamping it silently is how a file
    that says `max_steps = 0` runs with 8 and nobody notices."""
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be an integer") from exc
    if number < 1:
        raise ConfigError(f"{name} must be at least 1")
    return number


def _budget(value: Any, name: str) -> int:
    """A token budget: 0 turns the ceiling off, negative is a typo, not a policy."""
    try:
        budget = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be an integer token count") from exc
    if budget < 0:
        raise ConfigError(f"{name} must be 0 (unlimited) or a positive token count")
    return budget


def _read_delegate(entry: Any) -> DelegateConfig:
    """`[delegate]` table: turn the sub-agent tool on and size its budget."""
    if not isinstance(entry, dict):
        raise ConfigError("delegate must be a table ([delegate])")
    unknown = set(entry) - set(DelegateConfig.__slots__)
    if unknown:
        raise ConfigError(f"unknown [delegate] keys: {', '.join(sorted(unknown))}")
    return DelegateConfig(
        enabled=bool(entry.get("enabled", True)),
        max_steps=_positive(entry.get("max_steps", 6), "[delegate] max_steps"),
        max_parallel=max(0, int(entry.get("max_parallel") or 0)),
        system=entry.get("system"),
    )


def _read_mcp(entries: Any) -> list[McpServerConfig]:
    """`[[mcp]]` tables: each one spawns a server via `command`."""
    if not isinstance(entries, list):
        raise ConfigError("mcp must be an array of tables ([[mcp]])")

    servers: list[McpServerConfig] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ConfigError(f"mcp[{index}] must be a table")
        unknown = set(entry) - set(McpServerConfig.__slots__)
        if unknown:
            raise ConfigError(f"unknown mcp[{index}] keys: {', '.join(sorted(unknown))}")

        command = entry.get("command")
        if isinstance(command, str):
            command = shlex.split(command)
        if not isinstance(command, list) or not command:
            raise ConfigError(f"mcp[{index}] needs a non-empty command list")
        servers.append(
            McpServerConfig(
                command=[str(part) for part in command],
                prefix=str(entry.get("prefix") or ""),
                cwd=entry.get("cwd"),
                timeout=float(entry.get("timeout") or 30.0),
            )
        )
    return servers


def _apply_env(config: Config) -> Config:
    """`ONEHARNESS_PROVIDER`, `ONEHARNESS_MODEL`, etc. win over the file."""
    mapping = {
        "PROVIDER": ("provider", "preset"),
        "MODEL": ("provider", "model"),
        "API_KEY": ("provider", "api_key"),
        "BASE_URL": ("provider", "base_url"),
        "SYSTEM": ("system", None),
        "MAX_STEPS": ("max_steps", None),
        "MAX_PARALLEL": ("max_parallel", None),
        "CONTEXT_BUDGET": ("context_budget", None),
    }
    for suffix, (section, attr) in mapping.items():
        raw = os.environ.get(f"{ENV_PREFIX}{suffix}")
        if raw is None or raw == "":
            continue
        if suffix == "CONTEXT_BUDGET":
            config.context_budget = _budget(raw, f"{ENV_PREFIX}{suffix}")
        elif suffix in ("MAX_STEPS", "MAX_PARALLEL"):
            try:
                value = int(raw)
            except ValueError as exc:
                raise ConfigError(f"{ENV_PREFIX}{suffix} must be an integer") from exc
            if suffix == "MAX_STEPS":
                if value < 1:
                    raise ConfigError(f"{ENV_PREFIX}{suffix} must be at least 1")
                config.max_steps = value
            else:
                if value < 1:
                    raise ConfigError(f"{ENV_PREFIX}{suffix} must be at least 1")
                config.max_parallel = value
        elif section == "provider":
            setattr(config.provider, attr, raw)
        else:
            setattr(config, attr, raw)
    return config
