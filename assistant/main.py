"""Entry point: wire everything together and start the CLI or web UI."""
from __future__ import annotations

import argparse
import asyncio

from .agent.loop import Agent
from .config import load_settings
from .llm.client import LLMClient
from .tools.builtin import register_builtin
from .tools.mcp_client import MCPManager
from .tools.registry import ToolRegistry
from .ui.cli import run_cli


async def amain(web: bool) -> None:
    settings = load_settings()

    llm = LLMClient(**settings["ollama"])

    registry = ToolRegistry()
    register_builtin(registry)

    # MCP sessions must stay open for the whole app lifetime, so keep the UI
    # running inside the manager's context. With no servers configured this
    # does nothing.
    async with MCPManager() as mcp:
        await mcp.connect_all(registry)

        agent = Agent(
            llm=llm,
            registry=registry,
            max_iterations=settings["agent"]["max_iterations"],
        )
        system_prompt = settings["agent"]["system_prompt"]
        if web:
            # Imported lazily so the CLI doesn't require Gradio.
            from .ui.web import run_web

            await run_web(agent, system_prompt, **settings["web"])
        else:
            await run_cli(agent, system_prompt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local LLM assistant")
    parser.add_argument("--web", action="store_true", help="start the Gradio web UI")
    args = parser.parse_args()
    try:
        asyncio.run(amain(web=args.web))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
