"""Conversation history.

Stores the full message list in the structure Ollama expects, including the
assistant messages that carry `tool_calls` and the `tool` result messages.
Keeping this structure intact is what lets the model remember it just called
a tool.
"""
from __future__ import annotations

from typing import Any


class History:
    def __init__(self, system_prompt: str | None = None) -> None:
        self._messages: list[dict[str, Any]] = []
        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})

    def add_user(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_assistant(self, message: dict[str, Any]) -> None:
        """Append a raw assistant message (may contain `tool_calls`)."""
        self._messages.append(message)

    def add_tool_result(self, name: str, content: str) -> None:
        self._messages.append(
            {"role": "tool", "tool_name": name, "content": content}
        )

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self._messages
