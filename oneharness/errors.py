"""Exception hierarchy. Everything raised by OneHarness derives from OneHarnessError."""

from __future__ import annotations


class OneHarnessError(Exception):
    """Base class for all OneHarness failures."""


class ConfigError(OneHarnessError):
    """Invalid or incomplete configuration."""


class ProviderError(OneHarnessError):
    """A provider rejected the request or returned something unusable."""

    def __init__(self, message: str, *, status: int | None = None, body: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.body = body

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.status is not None:
            parts.append(f"status={self.status}")
        if self.body:
            parts.append(f"body={self.body[:500]}")
        return " | ".join(parts)


class ToolError(OneHarnessError):
    """A tool failed in a way the model is allowed to see and react to."""
