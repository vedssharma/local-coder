# local-coder MCP Server — Agent Setup Guide

## Prerequisites

- local-coder project cloned with its virtualenv set up (`pip install -r requirements.txt`)
- A GGUF model file configured (check `~/.local-coder/config.json` or run `python main.py models`)
- Node.js installed (required for the MCP filesystem sub-server used by `edit`)

## Configuring in Claude Code

Add to `~/.claude/settings.json` (or the project-level `.claude/settings.json`):

```json
{
  "mcpServers": {
    "local-coder": {
      "command": "/absolute/path/to/local-coder/llm/bin/python",
      "args": ["/absolute/path/to/local-coder/local-coder/scripts/server.py"],
      "cwd": "/absolute/path/to/local-coder"
    }
  }
}
```

Then restart Claude Code. The tools `ask`, `chat`, `edit`, `get_model`, and `set_model`
will appear in the available tools list.

## Configuring in Cursor

Add to `.cursor/mcp.json` in your workspace (or `~/.cursor/mcp.json` globally):

```json
{
  "mcpServers": {
    "local-coder": {
      "command": "/absolute/path/to/local-coder/llm/bin/python",
      "args": ["/absolute/path/to/local-coder/local-coder/scripts/server.py"],
      "cwd": "/absolute/path/to/local-coder"
    }
  }
}
```

## Configuring in any MCP-compatible agent

The server uses stdio transport. Point the agent to:

```
command:  /path/to/local-coder/llm/bin/python
args:     ["/path/to/local-coder/local-coder/scripts/server.py"]
cwd:      /path/to/local-coder   (required — modules are resolved relative to cwd)
```

## Finding your local-coder path

```bash
# If cloned to ~/projects/local-coder:
realpath ~/projects/local-coder
```

## Testing the server manually

```bash
cd /path/to/local-coder
./llm/bin/python local-coder/scripts/server.py
# Server waits for MCP JSON-RPC on stdin; Ctrl-C to stop
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: llama_cpp` | Run server with the local-coder virtualenv python (`llm/bin/python`) |
| `Model file not found` | Run `python main.py models` and confirm the path in `~/.local-coder/config.json` |
| `MCP filesystem unavailable` | Install Node.js; the edit tool uses `@modelcontextprotocol/server-filesystem` |
| Slow first response | Normal — the GGUF model loads on first use and stays resident |
