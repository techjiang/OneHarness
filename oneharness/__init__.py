"""OneHarness — a small, dependency-free harness for tool-calling LLM agents."""

from oneharness.agent import Agent, AgentResult
from oneharness.errors import ConfigError, OneHarnessError, ProviderError, ToolError
from oneharness.mcp import McpClient, connect_mcp
from oneharness.providers import create_provider, list_presets
from oneharness.tools import Registry, Tool, tool
from oneharness.types import Completion, Message, StreamEvent, ToolCall, ToolResult, Usage

__version__ = "0.2.0"

__all__ = [
    "Agent",
    "AgentResult",
    "Completion",
    "ConfigError",
    "McpClient",
    "Message",
    "OneHarnessError",
    "ProviderError",
    "Registry",
    "StreamEvent",
    "Tool",
    "ToolCall",
    "ToolError",
    "ToolResult",
    "Usage",
    "__version__",
    "connect_mcp",
    "create_provider",
    "list_presets",
    "tool",
]
