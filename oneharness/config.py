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
class Config:
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    system: str | None = None
    max_steps: int = 8
    mcp: list[McpServerConfig] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": {
                "preset": self.provider.preset,
                "model": self.provider.model,
                "base_url": self.provider.base_url,
                "temperature": self.provider.temperature,
                "max_tokens": self.provider.max_tokens,
            },
            "system": self.system,
            "max_steps": self.max_steps,
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
    unknown = set(data) - {"provider", "system", "max_steps", "agent", "mcp"}
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
    if "system" in data:
        config.system = data["system"]
    if "max_steps" in data:
        config.max_steps = int(data["max_steps"])
    if isinstance(agent_data, dict):
        if "system" in agent_data:
            config.system = agent_data["system"]
        if "max_steps" in agent_data:
            config.max_steps = int(agent_data["max_steps"])

    config.mcp = _read_mcp(data.get("mcp") or [])
    return config


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
    }
    for suffix, (section, attr) in mapping.items():
        raw = os.environ.get(f"{ENV_PREFIX}{suffix}")
        if raw is None or raw == "":
            continue
        if suffix == "MAX_STEPS":
            try:
                config.max_steps = int(raw)
            except ValueError as exc:
                raise ConfigError(f"{ENV_PREFIX}MAX_STEPS must be an integer") from exc
        elif section == "provider":
            setattr(config.provider, attr, raw)
        else:
            setattr(config, attr, raw)
    return config
