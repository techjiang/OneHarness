"""A minimal MCP server used by the tests: two tools, stdio JSON-RPC."""

from __future__ import annotations

import json
import sys

TOOLS = [
    {
        "name": "echo",
        "description": "Echo the given text back.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "boom",
        "description": "Always fails, to prove errors reach the model.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def handle(request):
    method = request.get("method")
    if method == "initialize":
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake", "version": "1.0"},
        }
    if method == "tools/list":
        return {"tools": TOOLS}
    if method == "tools/call":
        params = request.get("params") or {}
        if params.get("name") == "echo":
            text = (params.get("arguments") or {}).get("text", "")
            return {"content": [{"type": "text", "text": f"echo:{text}"}]}
        return {"isError": True, "content": [{"type": "text", "text": "boom happened"}]}
    raise KeyError(method)


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        if "id" not in request:  # notification
            continue
        try:
            reply = {"jsonrpc": "2.0", "id": request["id"], "result": handle(request)}
        except Exception as exc:  # noqa: BLE001 - report anything to the client
            reply = {
                "jsonrpc": "2.0",
                "id": request["id"],
                "error": {"code": -32601, "message": str(exc)},
            }
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
