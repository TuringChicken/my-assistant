"""Gradio web chat interface.

Threading model: Gradio serves requests on its own thread and event loop, but
MCP sessions are bound to the main event loop that opened them. Each turn is
therefore submitted to the main loop with `run_coroutine_threadsafe`, and
streamed tokens / tool events are passed back to the Gradio side through a
queue. Calling `agent.run` directly from a Gradio handler would break MCP tools.
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

import gradio as gr

from ..agent.history import History
from ..agent.loop import Agent

ChatLog = list[dict[str, Any]]


def build_app(
    agent: Agent, system_prompt: str | None, main_loop: asyncio.AbstractEventLoop
) -> gr.Blocks:
    async def respond(
        user_text: str, chat: ChatLog, history: History | None
    ) -> AsyncIterator[tuple[Any, ChatLog, History]]:
        user_text = user_text.strip()
        if not user_text:
            yield gr.skip(), chat, history
            return

        # One History per browser session, created on the first message.
        if history is None:
            history = History(system_prompt)
        history.add_user(user_text)
        chat = chat + [{"role": "user", "content": user_text}]
        yield "", chat, history

        ui_loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()

        def put(kind: str, payload: str = "") -> None:
            # Called from the main loop's thread; hand items over thread-safely.
            ui_loop.call_soon_threadsafe(queue.put_nowait, (kind, payload))

        future = asyncio.run_coroutine_threadsafe(
            agent.run(
                history,
                on_token=lambda t: put("token", t),
                on_event=lambda e: put("event", e),
            ),
            main_loop,
        )
        future.add_done_callback(lambda _f: put("done"))

        answer: dict[str, Any] | None = None  # assistant bubble being streamed
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "done":
                    break
                if kind == "event":
                    # Tool activity shows as a collapsed "thought" bubble.
                    chat.append(
                        {
                            "role": "assistant",
                            "content": "",
                            "metadata": {"title": payload, "status": "done"},
                        }
                    )
                    answer = None  # text after a tool call starts a new bubble
                else:
                    if answer is None:
                        answer = {"role": "assistant", "content": ""}
                        chat.append(answer)
                    answer["content"] += payload
                yield gr.skip(), chat, history

            try:
                future.result()
            except Exception as exc:  # e.g. Ollama not running
                raise gr.Error(f"{type(exc).__name__}: {exc}") from exc
        finally:
            # Browser closed or request cancelled: stop the agent turn too.
            if not future.done():
                future.cancel()

    with gr.Blocks(title="Local Assistant") as app:
        gr.Markdown("## Local Assistant")
        chatbot = gr.Chatbot(height=560, buttons=["copy"])
        history_state = gr.State(None)
        with gr.Row():
            textbox = gr.Textbox(
                placeholder="Ask something… (Enter to send)",
                show_label=False,
                scale=8,
                autofocus=True,
            )
            send = gr.Button("Send", variant="primary", scale=1)
        clear = gr.Button("New chat", size="sm")

        gr.on(
            triggers=[textbox.submit, send.click],
            fn=respond,
            inputs=[textbox, chatbot, history_state],
            outputs=[textbox, chatbot, history_state],
        )
        clear.click(lambda: ([], None), None, [chatbot, history_state], queue=False)

    return app


async def run_web(
    agent: Agent, system_prompt: str | None, host: str, port: int
) -> None:
    """Serve the UI until interrupted, keeping the main loop (and MCP) alive."""
    app = build_app(agent, system_prompt, asyncio.get_running_loop())
    app.queue()
    app.launch(server_name=host, server_port=port, prevent_thread_lock=True)
    try:
        await asyncio.Event().wait()  # run until Ctrl+C cancels this task
    finally:
        app.close()
