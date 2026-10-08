"""Unified tool registry.

Both built-in tools and MCP tools are normalized into the same `Tool`
abstraction so the agent loop never needs to know where a tool came from.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

# A tool handler takes a dict of arguments and returns a string result.
ToolHandler = Callable[[dict[str, Any]], Awaitable[str]]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for the arguments
    handler: ToolHandler

    def to_ollama(self) -> dict[str, Any]:
        """Convert to the format Ollama expects in its `tools` parameter."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def specs(self) -> list[dict[str, Any]]:
        """All tool specs in Ollama's expected format."""
        return [t.to_ollama() for t in self._tools.values()]

    async def call(self, name: str, args: dict[str, Any] | None) -> str:
        """Execute a tool by name. Errors are returned as text so the model
        can see what went wrong and recover."""
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'."
        try:
            return await tool.handler(args or {})
        except Exception as e:  # noqa: BLE001 - surface any failure to the model
            return f"Error while running tool '{name}': {e}"

    def __len__(self) -> int:
        return len(self._tools)
