"""Offline MCP server for client lifecycle and result-contract tests."""

import asyncio
import os
import sys
from pathlib import Path

with Path(sys.argv[1]).open("a") as pid_log:
    pid_log.write(f"{os.getpid()}\n")

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent

server = FastMCP("forecast-fixture")


@server.tool()
async def query_forecast(series_id: str, horizon_steps: int) -> CallToolResult:
    if series_id == "slow":
        await asyncio.sleep(60)
    if series_id == "remote-error":
        return CallToolResult(isError=True, content=[TextContent(type="text", text="private diagnostic")])
    if series_id == "unstructured":
        return CallToolResult(content=[TextContent(type="text", text='{"values": [1]}')])
    return CallToolResult(content=[], structuredContent={
        "series_id": series_id, "unit": "kW", "values": list(range(1, horizon_steps + 1)),
    })


if __name__ == "__main__":
    server.run(transport="stdio")
