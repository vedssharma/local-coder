#!/usr/bin/env python3
"""
local-coder MCP server

Exposes local-coder's ask, chat, edit, and model-management capabilities
as MCP tools over stdio.

Run from the local-coder project root with its virtualenv:
    /path/to/local-coder/llm/bin/python local-coder/scripts/server.py

Or configure in an agent's MCP settings — see references/setup.md.
"""

import sys
import os
import uuid

# Ensure local-coder modules are importable (server runs from project root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Move up one more level to the local-coder project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
os.chdir(project_root)

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("local-coder")

# ---------------------------------------------------------------------------
# Lazy singletons — loaded once per server process
# ---------------------------------------------------------------------------

_llm = None
_mcp_client = None
_console = None

# In-memory chat sessions: {session_id: [message_dict, ...]}
_sessions: dict[str, list] = {}


def _get_llm():
    global _llm
    if _llm is None:
        import config
        from llama_cpp import Llama
        model_config = config.get_model_config()
        _llm = Llama(
            model_path=model_config["model_path"],
            n_ctx=model_config["n_ctx"],
            n_gpu_layers=model_config["n_gpu_layers"],
            verbose=False,
        )
    return _llm


def _get_console():
    global _console
    if _console is None:
        from rich.console import Console
        _console = Console(stderr=True)
    return _console


def _get_mcp_client(disable: bool = False):
    global _mcp_client
    if disable:
        return None
    if _mcp_client is None:
        from mcp_client import MCPClient
        _mcp_client = MCPClient()
        _mcp_client.connect()
    return _mcp_client


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def ask(
    prompt: str,
    files: list[str] | None = None,
    max_tokens: int = 512,
    disable_filesystem: bool = False,
) -> str:
    """
    Send a one-off coding question to the local LLM and get an answer.

    Args:
        prompt: The question or task. Use @filepath syntax to reference files
                (e.g. "explain @main.py"), OR pass explicit file paths in `files`.
        files: Optional list of file paths whose contents should be injected
               as context (alternative to @filepath in the prompt).
        max_tokens: Maximum tokens to generate (default 512).
        disable_filesystem: Set True to skip MCP filesystem tools.
    """
    from prompt_builder import build_messages
    from agent import run_agent_loop
    from helpers import parse_file_references

    original_prompt, file_contents = parse_file_references(prompt)

    # Merge explicitly passed files
    if files:
        for fpath in files:
            if fpath not in file_contents:
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        file_contents[fpath] = f.read()
                except Exception as e:
                    file_contents[fpath] = f"[Error reading file: {e}]"

    messages = build_messages(original_prompt, file_contents)
    fs = _get_mcp_client(disable=disable_filesystem)

    result = run_agent_loop(
        llm=_get_llm(),
        messages=messages,
        console=_get_console(),
        max_tokens=max_tokens,
        mcp_client=fs,
    )
    return result or "(no response)"


@mcp.tool()
def chat(
    message: str,
    session_id: str | None = None,
    max_tokens: int = 512,
    disable_filesystem: bool = False,
) -> dict:
    """
    Send a message in a stateful multi-turn chat session.

    Returns a dict with keys:
      - reply (str): the assistant's response
      - session_id (str): pass this back on the next call to continue the conversation
      - turn (int): turn number in the session

    Args:
        message: The user's message.
        session_id: Omit (or pass None) to start a new session. Include the
                    returned session_id in subsequent calls to continue.
        max_tokens: Maximum tokens to generate (default 512).
        disable_filesystem: Set True to skip MCP filesystem tools.
    """
    from prompt_builder import build_messages
    from agent import run_agent_loop
    from helpers import parse_file_references

    # Resolve or create session
    if not session_id or session_id not in _sessions:
        session_id = str(uuid.uuid4())
        _sessions[session_id] = []

    history = _sessions[session_id]

    original_prompt, file_contents = parse_file_references(message)
    messages = build_messages(original_prompt, file_contents, history=history)
    fs = _get_mcp_client(disable=disable_filesystem)

    reply = run_agent_loop(
        llm=_get_llm(),
        messages=messages,
        console=_get_console(),
        max_tokens=max_tokens,
        mcp_client=fs,
    )
    reply = reply or "(no response)"

    # Persist turn in history (cap at last 20 messages = 10 turns)
    history.append({"role": "user", "content": original_prompt})
    history.append({"role": "assistant", "content": reply})
    if len(history) > 20:
        _sessions[session_id] = history[-20:]

    return {
        "reply": reply,
        "session_id": session_id,
        "turn": len(_sessions[session_id]) // 2,
    }


@mcp.tool()
def edit(
    prompt: str,
    files: list[str] | None = None,
    max_tokens: int = 2048,
) -> str:
    """
    Request code changes to one or more files.

    The local LLM will read files via MCP filesystem tools, apply edits, and
    return a summary of what was changed. Filesystem tools are always enabled
    for edit so the model can read and write files.

    Args:
        prompt: Natural-language description of the change. Use @filepath to
                reference specific files (e.g. "refactor @helpers.py to use pathlib").
        files: Optional list of file paths to pre-load as context.
        max_tokens: Maximum tokens to generate (default 2048).
    """
    from prompt_builder import build_edit_system_message, build_user_message
    from agent import run_agent_loop
    from helpers import parse_file_references

    original_prompt, file_contents = parse_file_references(prompt)

    if files:
        for fpath in files:
            if fpath not in file_contents:
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        file_contents[fpath] = f.read()
                except Exception as e:
                    file_contents[fpath] = f"[Error reading file: {e}]"

    messages = [build_edit_system_message(), build_user_message(original_prompt, file_contents)]
    fs = _get_mcp_client(disable=False)

    result = run_agent_loop(
        llm=_get_llm(),
        messages=messages,
        console=_get_console(),
        max_tokens=max_tokens,
        mcp_client=fs,
    )
    return result or "(no response)"


@mcp.tool()
def get_model() -> dict:
    """
    Return current model configuration (path, context size, GPU layers, file status).
    """
    import config
    cfg = config.get_model_config()
    model_path = cfg.get("model_path", "")
    exists = os.path.exists(model_path)
    size_gb = None
    if exists:
        size_gb = round(os.path.getsize(model_path) / (1024 ** 3), 2)
    return {
        "model_path": model_path,
        "n_ctx": cfg.get("n_ctx"),
        "n_gpu_layers": cfg.get("n_gpu_layers"),
        "file_exists": exists,
        "size_gb": size_gb,
    }


@mcp.tool()
def set_model(model_path: str) -> dict:
    """
    Switch the active GGUF model. The new model will be used on the next
    LLM call (the current process will reload it automatically).

    Args:
        model_path: Absolute path to a .gguf model file.
    """
    global _llm
    import config

    if not os.path.isfile(model_path):
        return {"success": False, "error": f"File not found: {model_path}"}
    if not model_path.lower().endswith(".gguf"):
        return {"success": False, "error": "Path must point to a .gguf file"}

    success = config.set_model_path(os.path.abspath(model_path))
    if success:
        _llm = None  # Force reload on next call
        return {"success": True, "model_path": os.path.abspath(model_path)}
    return {"success": False, "error": "Failed to save config"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
