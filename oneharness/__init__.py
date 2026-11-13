"""OneHarness — a small, dependency-free harness for tool-calling LLM agents."""

from oneharness.agent import Agent, AgentResult
from oneharness.builtin import default_tools
from oneharness.delegate import DelegationError, Delegator, make_delegate_tool
from oneharness.doctor import Probe, check
from oneharness.errors import (
    ConfigError,
    NetworkError,
    OneHarnessError,
    ProviderError,
    ToolError,
)
from oneharness.mcp import McpClient, McpError, McpServerGone, connect_mcp
from oneharness.providers import create_provider, list_presets
from oneharness.tape import Tape, TapeError
from oneharness.tools import Registry, Tool, tool
from oneharness.types import (
    INTERRUPTED_BY_USER,
    STREAM_ENDED_EARLY,
    Completion,
    Message,
    StreamEvent,
    ToolCall,
    ToolResult,
    Usage,
)

__version__ = "1.6.0"

__all__ = [
    "Agent",
    "AgentResult",
    "Completion",
    "ConfigError",
    "INTERRUPTED_BY_USER",
    "DelegationError",
    "Delegator",
    "McpClient",
    "McpError",
    "McpServerGone",
    "Message",
    "NetworkError",
    "OneHarnessError",
    "Probe",
    "ProviderError",
    "Registry",
    "STREAM_ENDED_EARLY",
    "StreamEvent",
    "Tape",
    "TapeError",
    "Tool",
    "ToolCall",
    "ToolError",
    "ToolResult",
    "Usage",
    "__version__",
    "check",
    "connect_mcp",
    "create_provider",
    "default_tools",
    "list_presets",
    "make_delegate_tool",
    "tool",
]
