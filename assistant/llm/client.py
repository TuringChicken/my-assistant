"""Thin async wrapper around the Ollama chat API.

This is the only place that talks to the model. Swapping the backend later
(e.g. a different local server) means changing only this file.
"""
from __future__ import annotations

from typing import Any, AsyncIterator

from ollama import AsyncClient, ChatResponse


class LLMClient:
    def __init__(self, host: str, model: str, temperature: float = 0.7) -> None:
        self._client = AsyncClient(host=host)
        self.model = model
        self.temperature = temperature

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[ChatResponse]:
        """Send one chat request and return an async stream of response chunks.

        Content arrives token by token; each tool call arrives whole inside a
        single chunk. The caller is responsible for accumulating the chunks.
        """
        return await self._client.chat(
            model=self.model,
            messages=messages,
            tools=tools or None,
            options={"temperature": self.temperature},
            stream=True,
        )
