"""Entry point: wire everything together and start the CLI."""
from __future__ import annotations

import asyncio

from .agent.loop import Agent
from .config import load_settings
from .llm.client import LLMClient
from .tools.builtin import register_builtin
from .tools.mcp_client import MCPManager
from .tools.registry import ToolRegistry
from .ui.cli import run_cli


def _log_event(text: str) -> None:
    print(f"  · {text}")


async def amain() -> None:
    settings = load_settings()

    llm = LLMClient(**settings["ollama"])

    registry = ToolRegistry()
    register_builtin(registry)

    # MCP sessions must stay open for the whole app lifetime, so keep the CLI
    # running inside the manager's context. With no servers configured this
    # does nothing.
    async with MCPManager() as mcp:
        await mcp.connect_all(registry)

        agent = Agent(
            llm=llm,
            registry=registry,
            max_iterations=settings["agent"]["max_iterations"],
            on_event=_log_event,
        )
        await run_cli(agent, settings["agent"]["system_prompt"])


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
