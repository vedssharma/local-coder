# MCP setup

Install Local Coder's Python requirements and ripgrep. Configure a GGUF model or an OpenAI-compatible local server using `python main.py models`. Node is not required for the native coding tools.

Use your host's supported stdio MCP configuration format. The essential fields are:

```json
{
  "mcpServers": {
    "local-coder": {
      "command": "/absolute/path/to/local-coder/llm/bin/python",
      "args": ["/absolute/path/to/local-coder/skills/local-coder/scripts/server.py"],
      "cwd": "/absolute/path/to/the/project-to-work-on",
      "env": {
        "LOCAL_CODER_PERMISSION_MODE": "read-only"
      }
    }
  }
}
```

Launch from the project to work on; the server resolves its own Python imports separately. `LOCAL_CODER_CONFIG_DIR` can select a configuration/session directory. Permission mode defaults to `read-only`; use `workspace-edit` for patches or `execute` for commands. Command execution uses host privileges, not an OS sandbox. Use environment variables for API authentication and never embed key values in source or shared settings.

`ask`, `chat`, and `edit` return structured results including `status`, `text`, and `session_id`. Continue with `chat(message="...", session_id="...")`. Sessions persist across server restarts and preserve tool observations. `edit` and `set_model` are blocked in read-only mode. `set_model(path="...")` accepts a workspace-local GGUF.

To smoke-test transport, connect an MCP client, initialize the session, list tools, and call `get_model`. This tests discovery/configuration, not model inference. Running the script directly waits for MCP JSON-RPC on stdin.

Coding tools accept an optional `profile` naming an existing saved model configuration. The server keeps one model loaded at a time, releases it on profile changes, and uses that profile’s context limit. Profile selection never changes executor permissions.

Set `LOCAL_CODER_PROCESS_WAIT_SECONDS` at launch to control runtime-managed command waiting (default 2 seconds, range 0–30). Zero returns command handles immediately. The runtime polls during the wait without additional model calls; long jobs still return handles for explicit polling or cancellation.

`LOCAL_CODER_TOOL_WORKERS` controls independent read concurrency (default 4, range 1–8); mutations and unknown tools stay serial.
