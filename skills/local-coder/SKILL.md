---
name: local-coder
description: "Leverage a locally-hosted LLM (llama.cpp/GGUF) for coding tasks — Q&A, file editing, and multi-turn chat — without sending code to a remote API. Use when an agent needs to delegate coding questions, code edits, or code review to the local LLM, or when the user asks to 'use local-coder', 'run the local model', or 'ask the local LLM'. Also use when setting up or configuring local-coder's MCP server in Claude Code, Cursor, or other MCP-compatible agents."
---

# local-coder Skill

local-coder runs a GGUF model locally via llama.cpp and exposes it as MCP tools.

## Quick start — running the MCP server

The server must be started from the local-coder project root using its virtualenv:

```bash
/path/to/local-coder/llm/bin/python /path/to/local-coder/local-coder/scripts/server.py
```

Once running, it exposes five tools over stdio:

| Tool | Purpose |
|------|---------|
| `ask` | Single-turn Q&A with optional file context |
| `chat` | Stateful multi-turn conversation (use `session_id` to continue) |
| `edit` | Code editing — model reads files, applies changes, returns summary |
| `get_model` | Inspect current model path, context size, GPU layers |
| `set_model` | Switch to a different GGUF model file |

## Using the tools

### ask — one-off question
```
ask(prompt="What does the agent loop in @agent.py do?")
ask(prompt="Explain the caching logic", files=["cache.py", "store.py"])
ask(prompt="Write a pytest fixture for the DB", max_tokens=1024)
```

### chat — multi-turn session
```python
# First turn (omit session_id to start a new session)
result = chat(message="What does main.py do?")
sid = result["session_id"]

# Subsequent turns
result = chat(message="How does the agentic loop work?", session_id=sid)
result = chat(message="Can you simplify it?", session_id=sid)
```

### edit — file editing
```
edit(prompt="Refactor @helpers.py to use pathlib throughout")
edit(prompt="Add type annotations to all functions in @agent.py", max_tokens=3000)
```

### get_model / set_model
```
get_model()
set_model(model_path="/models/codellama-13b.Q4_K_M.gguf")
```

## File references

In `ask`, `chat`, and `edit` prompts you can use `@filepath` to inject file contents:
```
ask(prompt="What is wrong with @src/auth.py?")
```
Or pass paths explicitly via the `files` parameter — both approaches inject the file
contents into the LLM context as `<file path='...'>...</file>` blocks.

## Setup guide for agents

See `references/setup.md` for how to configure the MCP server in Claude Code, Cursor,
and other MCP-compatible agents.
