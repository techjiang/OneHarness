"""OneHarness — a small, dependency-free harness for tool-calling LLM agents."""

from oneharness.agent import Agent, AgentResult
from oneharness.errors import ConfigError, OneHarnessError, ProviderError, ToolError
from oneharness.providers import create_provider, list_presets
from oneharness.tools import Registry, Tool, tool
from oneharness.types import Completion, Message, ToolCall, ToolResult, Usage

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentResult",
    "Completion",
    "ConfigError",
    "Message",
    "OneHarnessError",
    "ProviderError",
    "Registry",
    "Tool",
    "ToolCall",
    "ToolError",
    "ToolResult",
    "Usage",
    "__version__",
    "create_provider",
    "list_presets",
    "tool",
]
