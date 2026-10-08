"""Minimal command-line chat interface (stage 1)."""
from __future__ import annotations

import asyncio

from ..agent.history import History
from ..agent.loop import Agent

_EXIT_WORDS = {"exit", "quit", ":q"}


async def run_cli(agent: Agent, system_prompt: str | None) -> None:
    history = History(system_prompt)
    print("Local assistant ready. Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            # input() is blocking; run it off the event loop thread.
            user = (await asyncio.to_thread(input, "you > ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user.lower() in _EXIT_WORDS:
            break
        if not user:
            continue

        history.add_user(user)
        answer = await agent.run(history)
        print(f"\nassistant > {answer}\n")
