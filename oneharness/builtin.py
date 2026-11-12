"""Built-in tools: a sandboxed file surface and a shell runner."""

from __future__ import annotations

import subprocess
from pathlib import Path

from oneharness.errors import ToolError
from oneharness.tools import Tool, tool

MAX_READ_BYTES = 200_000
MAX_OUTPUT_CHARS = 20_000


def _resolve(path: str, root: Path) -> Path:
    """Resolve `path` inside `root`, refusing anything that escapes the sandbox."""
    candidate = (root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ToolError(f"{path!r} is outside the workspace at {root}") from exc
    return candidate


def make_file_tools(root: Path | str = ".") -> list[Tool]:
    """read_file / write_file / list_dir, all confined to `root`."""
    workspace = Path(root).resolve()

    @tool
    def read_file(path: str) -> str:
        """Read a UTF-8 text file relative to the workspace root."""
        target = _resolve(path, workspace)
        if not target.is_file():
            raise ToolError(f"{path} is not a file")
        if target.stat().st_size > MAX_READ_BYTES:
            raise ToolError(f"{path} is larger than {MAX_READ_BYTES} bytes")
        return target.read_text(encoding="utf-8", errors="replace")

    @tool
    def write_file(path: str, content: str) -> str:
        """Write UTF-8 text to a file, creating parent directories as needed."""
        target = _resolve(path, workspace)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {target.relative_to(workspace)}"

    @tool
    def list_dir(path: str = ".") -> str:
        """List directory entries, one per line; directories end with a slash."""
        target = _resolve(path, workspace)
        if not target.is_dir():
            raise ToolError(f"{path} is not a directory")
        entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
        lines = [f"{entry.name}/" if entry.is_dir() else entry.name for entry in entries]
        return "\n".join(lines) if lines else "(empty)"

    return [read_file, write_file, list_dir]


def make_shell_tool(root: Path | str = ".", timeout: float = 60.0) -> Tool:
    """A shell tool that runs one command in the workspace with a hard timeout."""
    workspace = Path(root).resolve()

    @tool
    def run_shell(command: str) -> str:
        """Run a shell command in the workspace and return its combined output."""
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolError(f"command timed out after {timeout}s: {command}") from exc

        output = (completed.stdout or "") + (completed.stderr or "")
        output = output.strip()
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + f"\n... (truncated, {len(output)} chars total)"
        header = f"exit_code={completed.returncode}"
        return f"{header}\n{output}" if output else header

    return run_shell


def default_tools(root: Path | str = ".", *, shell: bool = False) -> list[Tool]:
    """File tools always; the shell tool only when explicitly enabled."""
    items = make_file_tools(root)
    if shell:
        items.append(make_shell_tool(root))
    return items
