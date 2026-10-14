# Local LLM Assistant

A personal assistant powered by a local LLM (Ollama + Qwen3), using
**Function Calling + MCP**. Tools are decoupled from the assistant: each tool
is (or will be) a standalone MCP server, and the assistant is the MCP client.

See `local-llm-assistant-plan.md` for the full design and roadmap.

## Status

- [x] Stage 1 — Chat: multi-turn CLI chat against Ollama
- [x] Stage 2 — First tools: the tool-calling loop, proven with built-in
      `get_current_time` and `calculate` tools
- [ ] Stage 3 — First MCP server (web search via SearXNG)
- [ ] Stage 4 — File search / RAG
- [ ] Stage 5 — Web UI + persistent memory
- [ ] Stage 6 — Polish

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) running locally with the model pulled:

  ```powershell
  ollama pull qwen3:8b
  ```

## Setup

```powershell
# From the project root (D:\my-assistant)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -e .
# or, if you only want to run stages 1-2 without MCP yet:
# pip install ollama pyyaml python-dotenv
```

Optional configuration:

```powershell
Copy-Item .env.example .env   # then edit if needed
```

## Run

```powershell
python -m assistant.main          # command-line chat
python -m assistant.main --web    # Gradio web UI at http://127.0.0.1:7860
```

Try:

- `what time is it now?`  → model calls `get_current_time`
- `what is (3 + 4) * 2?`  → model calls `calculate`
- or just chat normally

Type `exit` or `quit` to leave.

## Configuration

- `config/settings.yaml` — model name, temperature, system prompt,
  `max_iterations`.
- `config/mcp_servers.json` — MCP servers to connect (empty for now; from
  stage 3 onward it uses the same format as Claude Desktop).
- `.env` — `OLLAMA_HOST` overrides the host in `settings.yaml`.

## Project layout

```
assistant/
  config.py          # load settings.yaml + .env
  main.py            # entry point
  llm/client.py      # Ollama communication (async)
  agent/
    loop.py          # core tool-calling loop
    history.py       # conversation history
  tools/
    registry.py      # unified tool abstraction (built-in + MCP)
    builtin.py       # get_current_time, calculate
    mcp_client.py    # connects MCP servers over stdio
  ui/cli.py          # command-line interface
  ui/web.py          # Gradio web UI (--web)
config/              # settings.yaml, mcp_servers.json
mcp_servers/         # one standalone MCP server per tool (from stage 3)
data/                # docs the assistant may search + vector DB
```
