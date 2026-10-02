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

`LOCAL_CODER_TOOL_WORKERS` controls independent read concurrency (default 4, range 1–8); mutations and unknown tools stay serial.

Coding results also expose `changed_files`, `checks`, `outstanding_processes`, `verification_status`, and `verification_scope`. Model completion and observed verification are distinct. Failed/stale checks, running jobs, and uncertain interrupted effects prevent a successful completion result. Checks marked `verification=true` preserve their actual exit status; no observed check means `not_run`.

Session journals recover completed observations without replaying commands or edits. Interrupted edits require rereading the affected file. Unknown command effects require inspection and explicit acknowledgement in the local CLI (`/acknowledge-interrupted`); MCP does not implicitly acknowledge them.

Validate setup with `python -m pytest -q` and `python evaluations/run.py --scripted --suite all --output /tmp/harness-mechanics.json`. Scripted evaluations verify mechanics without loading a model; configured-model coding runs require `--allow-execution` and remain separate quality measurements.
