"""Thin async wrapper around the Ollama chat API.

This is the only place that talks to the model. Swapping the backend later
(e.g. a different local server) means changing only this file.
"""
from __future__ import annotations

from typing import Any

from ollama import AsyncClient


class LLMClient:
    def __init__(self, host: str, model: str, temperature: float = 0.7) -> None:
        self._client = AsyncClient(host=host)
        self.model = model
        self.temperature = temperature

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Send one chat request and return the raw Ollama response.

        Note: tool calling is incompatible with streaming in Ollama, so we
        always request a non-streamed response here. Streaming of the *final*
        answer can be added later in the UI layer.
        """
        return await self._client.chat(
            model=self.model,
            messages=messages,
            tools=tools or None,
            options={"temperature": self.temperature},
            stream=False,
        )
