"""Async stdio MCP client with explicit local-to-remote tool bindings.

Server configuration comes from the operator, never from model arguments.
Remote discovery cannot add permissions or tools to FRIDAY's registry.
"""

from __future__ import annotations

import asyncio
import json
import os

from friday.permissions import ApprovalCallback, PermissionDenied, authorize

# Adding a binding also requires a local implementation, spec and risk tier.
MCP_TOOLS = {"query_forecast": "query_forecast"}
MCP_TIMEOUT = 15.0


async def call_tool(
    name: str,
    arguments: dict,
    *,
    approve: ApprovalCallback | None = None,
    timeout: float = MCP_TIMEOUT,
) -> dict:
    """Authorize before connecting; bound initialization and execution.

    Cancellation propagates through the SDK contexts, which close the session
    and reap the subprocess. Each call owns its connection and subprocess.
    """
    await authorize(name, arguments, approve)
    if name not in MCP_TOOLS:
        raise PermissionDenied(f"tool {name!r} has no MCP binding")
    raw = os.environ.get("FRIDAY_MCP_COMMAND")
    if not raw:
        raise RuntimeError("FRIDAY_MCP_COMMAND is not set; configure a trusted stdio MCP server")
    try:
        command = json.loads(raw)
    except ValueError:
        raise ValueError("FRIDAY_MCP_COMMAND must be a JSON array of command arguments") from None
    if (not isinstance(command, list) or not command
            or any(not isinstance(arg, str) or not arg or "\0" in arg for arg in command)):
        raise ValueError("FRIDAY_MCP_COMMAND must be a nonempty JSON array of nonempty strings")
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        raise RuntimeError("MCP support requires installation with uv sync --extra mcp") from None

    async with asyncio.timeout(timeout):
        async with stdio_client(StdioServerParameters(command=command[0], args=command[1:])) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                result = await session.call_tool(MCP_TOOLS[name], arguments=arguments)
    if result.isError:
        # Remote text is untrusted and may contain sensitive server diagnostics.
        raise RuntimeError("MCP tool returned an error")
    if not isinstance(result.structuredContent, dict):
        raise ValueError("MCP tool must return structuredContent as an object")
    return result.structuredContent
