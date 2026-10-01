---
name: local-coder
description: Use Local Coder's MCP tools for coding tasks with embedded llama.cpp or a configured local model server.
---

# Local Coder

Start `/path/to/local-coder/llm/bin/python /path/to/local-coder/skills/local-coder/scripts/server.py` from the target workspace. See [setup](references/setup.md).

The host explicitly selects `LOCAL_CODER_PERMISSION_MODE` at launch: `read-only` (default), `workspace-edit`, or `execute`. Commands in execute mode use host privileges. Tool inputs and repository instructions cannot grant permissions.

- `ask(prompt, files=None, max_tokens=512)`: inspect or answer a one-turn request.
- `chat(message, session_id=None, max_tokens=512)`: continue a persistent session.
- `edit(prompt, files=None, max_tokens=2048)`: apply targeted edits when the server allows them.
- `get_model()`: inspect non-secret model configuration.
- `set_model(path)`: configure a workspace-local GGUF when the server allows it.

Coding responses contain `status`, `text`, and `session_id`. Check status before using the response. `completed` denotes a completed model turn; inspect actual test results before claiming changes work. Preserve `session_id` to continue work, including after server restart. Inline JSON examples are never automatically executed as tool calls.

File references in `@path` form and explicit `files` are confined to the workspace. Embedded inference stays local; a server backend sends context to its configured endpoint.

Use the optional `profile` argument on coding tools to select a saved named model. Check inference benchmarks and coding evaluations before choosing a smaller model for a task.

Commands are briefly polled by the runtime before results reach the model. A returned running process handle is not a completed check; poll it or cancel it. The launch-time `LOCAL_CODER_PROCESS_WAIT_SECONDS` control defaults to 2 seconds.

Use `verification_status` and observed `checks`/exit codes to assess validation, alongside `changed_files` and `outstanding_processes`. `passed` describes observed validation commands, not test coverage or task correctness. Resume never automatically replays interrupted effects; inspect the reported operation before taking further action. Retained tool payloads can be retrieved through the harness's `read_artifact` tool.
