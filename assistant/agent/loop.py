"""The core tool-calling loop.

    user question
        -> send [messages + tool specs] to the model
        -> model replies: answer directly? -> return it, done
                          : wants a tool?   -> run it, append result, repeat
        (bounded by max_iterations)
"""
from __future__ import annotations

from typing import Any, Callable

from ollama import Message

from ..llm.client import LLMClient
from ..tools.registry import ToolRegistry
from .history import History

# Optional callback to surface progress to the UI (e.g. "calling tool ...").
EventHook = Callable[[str], None]
# Optional callback receiving answer text as it streams in.
TokenHook = Callable[[str], None]


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        max_iterations: int = 5,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.max_iterations = max_iterations

    async def _complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        on_token: TokenHook | None,
    ) -> Message:
        """Stream one model turn, forwarding tokens, and return the full message."""
        content: list[str] = []
        thinking: list[str] = []
        tool_calls: list[Message.ToolCall] = []

        async for chunk in await self.llm.chat(messages, tools=tools):
            msg = chunk.message
            if msg.content:
                content.append(msg.content)
                if on_token:
                    on_token(msg.content)
            if msg.thinking:
                thinking.append(msg.thinking)
            if msg.tool_calls:
                tool_calls.extend(msg.tool_calls)

        return Message(
            role="assistant",
            content="".join(content),
            thinking="".join(thinking) or None,
            tool_calls=tool_calls or None,
        )

    async def run(
        self,
        history: History,
        on_token: TokenHook | None = None,
        on_event: EventHook | None = None,
    ) -> str:
        """Drive the conversation until the model answers or we hit the cap.

        Callbacks are per call (not per agent), so one Agent can serve several
        UI sessions concurrently.
        """
        emit = on_event or (lambda _text: None)
        tools = self.registry.specs() if len(self.registry) else None

        for _ in range(self.max_iterations):
            message = await self._complete(history.messages, tools, on_token)
            history.add_assistant(message.model_dump(exclude_none=True))

            if not message.tool_calls:
                return message.content or ""

            # Execute every requested tool and feed results back.
            for call in message.tool_calls:
                name = call.function.name
                args = call.function.arguments  # already a dict in the Ollama SDK
                emit(f"calling tool: {name}({args})")
                result = await self.registry.call(name, args)
                history.add_tool_result(name, result)

        # Hit the iteration cap: ask once more without tools for a final answer.
        emit("max tool iterations reached; requesting a final answer")
        message = await self._complete(history.messages, None, on_token)
        history.add_assistant(message.model_dump(exclude_none=True))
        return message.content or "(No final answer produced.)"
