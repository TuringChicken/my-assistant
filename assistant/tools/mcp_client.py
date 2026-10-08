"""MCP client layer (active from stage 3).

Reads config/mcp_servers.json, launches each configured MCP server over stdio,
and registers its tools into the shared ToolRegistry. If no servers are
configured, this is a no-op, so stages 1-2 run without any MCP dependency.

Config format (same shape as Claude Desktop), e.g.:

    {
      "mcpServers": {
        "web_search": {
          "command": "python",
          "args": ["mcp_servers/web_search/server.py"],
          "env": {}
        }
      }
    }
"""
from __future__ import annotations

import json
import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from .registry import Tool, ToolRegistry

from ..config import ROOT

DEFAULT_CONFIG = ROOT / "config" / "mcp_servers.json"


class MCPManager:
    """Owns the lifetime of all MCP sessions via a single AsyncExitStack.

    Use as an async context manager so sessions stay open for as long as the
    app runs:

        async with MCPManager() as mcp:
            await mcp.connect_all(registry)
            ...  # run the app here
    """

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG) -> None:
        self.config_path = Path(config_path)
        self._stack = AsyncExitStack()
        self.sessions: dict[str, Any] = {}

    async def __aenter__(self) -> "MCPManager":
        await self._stack.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> Any:
        return await self._stack.__aexit__(*exc)

    def _load_config(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return {}
        with open(self.config_path, encoding="utf-8") as f:
            return json.load(f).get("mcpServers", {})

    async def connect_all(self, registry: ToolRegistry) -> None:
        servers = self._load_config()
        if not servers:
            return

        # Import lazily so stages 1-2 don't require the `mcp` package.
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        for name, cfg in servers.items():
            params = StdioServerParameters(
                command=cfg["command"],
                args=cfg.get("args", []),
                env={**os.environ, **cfg.get("env", {})},
            )
            read, write = await self._stack.enter_async_context(stdio_client(params))
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.sessions[name] = session

            listed = await session.list_tools()
            for tool in listed.tools:
                registry.register(self._wrap(session, tool))

    @staticmethod
    def _wrap(session: Any, tool: Any) -> Tool:
        async def handler(args: dict[str, Any]) -> str:
            result = await session.call_tool(tool.name, args)
            parts = [
                c.text for c in result.content if getattr(c, "type", None) == "text"
            ]
            return "\n".join(parts) if parts else "(tool returned no text output)"

        return Tool(
            name=tool.name,
            description=tool.description or "",
            parameters=tool.inputSchema or {"type": "object", "properties": {}},
            handler=handler,
        )
