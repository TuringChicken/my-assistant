"""The core tool-calling loop.

    user question
        -> send [messages + tool specs] to the model
        -> model replies: answer directly? -> return it, done
                          : wants a tool?   -> run it, append result, repeat
        (bounded by max_iterations)
"""
from __future__ import annotations

from typing import Any, Callable

from ..llm.client import LLMClient
from ..tools.registry import ToolRegistry
from .history import History

# Optional callback to surface progress to the UI (e.g. "calling tool ...").
EventHook = Callable[[str], None]


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        max_iterations: int = 5,
        on_event: EventHook | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.max_iterations = max_iterations
        self._on_event = on_event

    def _emit(self, text: str) -> None:
        if self._on_event:
            self._on_event(text)

    async def run(self, history: History) -> str:
        """Drive the conversation until the model answers or we hit the cap."""
        tools = self.registry.specs() if len(self.registry) else None

        for _ in range(self.max_iterations):
            response = await self.llm.chat(history.messages, tools=tools)
            message = response.message
            history.add_assistant(message.model_dump())

            if not message.tool_calls:
                return message.content or ""

            # Execute every requested tool and feed results back.
            for call in message.tool_calls:
                name = call.function.name
                args = call.function.arguments  # already a dict in the Ollama SDK
                self._emit(f"calling tool: {name}({args})")
                result = await self.registry.call(name, args)
                history.add_tool_result(name, result)

        # Hit the iteration cap: ask once more without tools for a final answer.
        self._emit("max tool iterations reached; requesting a final answer")
        response = await self.llm.chat(history.messages, tools=None)
        history.add_assistant(response.message.model_dump())
        return response.message.content or "(No final answer produced.)"
