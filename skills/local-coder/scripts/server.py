#!/usr/bin/env python3
"""Expose the shared local-coder runtime over MCP stdio.

Launch from the workspace to operate on. LOCAL_CODER_PERMISSION_MODE controls
capabilities for the entire server; tool arguments cannot elevate permissions.
"""
import os
from pathlib import Path
import sys
import threading
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from mcp.server.fastmcp import FastMCP
import config
from helpers import parse_file_references
from model_backend import create_model
from runtime import Runtime
from workspace_tools import WorkspaceTools

mcp = FastMCP('local-coder')
_model = None
_model_signature = None
_lock = threading.RLock()


def get_model_instance(profile_name=None):
    global _model, _model_signature
    import json
    profile = config.get_model_config(profile_name)
    signature = json.dumps(profile, sort_keys=True)
    if _model is None or signature != _model_signature:
        if _model is not None:
            _model.close()
        _model = create_model(profile)
        _model_signature = signature
    return _model


def run_turn(prompt, files=None, session_id=None, max_tokens=512, profile_name=None):
    # One model/session write at a time. No shared mutable transcripts across callers.
    with _lock:
        with Runtime(get_model_instance(profile_name), os.getcwd(), config.CONFIG_DIR,
                     mode=os.environ.get('LOCAL_CODER_PERMISSION_MODE', 'read-only'),
                     context_window=config.get_model_config(profile_name)['n_ctx'],
                     process_wait_seconds=float(os.environ.get('LOCAL_CODER_PROCESS_WAIT_SECONDS', '2')),
                     tool_workers=int(os.environ.get('LOCAL_CODER_TOOL_WORKERS', '4')),
                     subagents=int(os.environ.get('LOCAL_CODER_SUBAGENTS', '4'))) as runtime:
            if session_id:
                runtime.resume(session_id)
            original, contents = parse_file_references(prompt, root=runtime.tools.root)
            for name in files or []:
                path = runtime.tools.path(name)
                with path.open() as f:
                    contents[name] = f.read(32000)
            result = runtime.turn(original, contents, max_tokens)
            return {**asdict(result), 'session_id': runtime.session_id}


@mcp.tool()
def ask(prompt: str, files: list[str] | None = None, max_tokens: int = 512, profile: str | None = None) -> dict:
    """Ask a question; returns text, explicit run status, and a resumable session ID."""
    return run_turn(prompt, files, max_tokens=max_tokens, profile_name=profile)


@mcp.tool()
def chat(message: str, session_id: str | None = None, max_tokens: int = 512, profile: str | None = None) -> dict:
    """Continue a saved session, retaining tool observations and decisions."""
    return run_turn(message, session_id=session_id, max_tokens=max_tokens, profile_name=profile)


@mcp.tool()
def edit(prompt: str, files: list[str] | None = None, max_tokens: int = 2048, profile: str | None = None) -> dict:
    """Edit using apply_patch; server must be launched in workspace-edit or execute mode."""
    if os.environ.get('LOCAL_CODER_PERMISSION_MODE', 'read-only') == 'read-only':
        return {'status': 'blocked', 'text': 'Server is read-only; restart with LOCAL_CODER_PERMISSION_MODE=workspace-edit or execute.'}
    return run_turn(prompt, files, max_tokens=max_tokens, profile_name=profile)


@mcp.tool()
def get_model() -> dict:
    """Return model configuration; credentials are never included."""
    profile = config.get_model_config()
    return {k: profile[k] for k in ('backend', 'model_path', 'n_ctx', 'n_gpu_layers',
            'model', 'base_url', 'supports_tools', 'chat_format') if k in profile}


@mcp.tool()
def set_model(path: str) -> dict:
    """Set a workspace-local GGUF model; disabled in read-only mode."""
    global _model
    if os.environ.get('LOCAL_CODER_PERMISSION_MODE', 'read-only') == 'read-only':
        return {'status': 'blocked', 'text': 'Model configuration is read-only.'}
    with _lock:
        safe_path = WorkspaceTools().path(path)
        success = config.set_model_path(str(safe_path))
        if success:
            profile = config.get_model_config()
            profile['backend'] = 'embedded'
            config.update_model_config(profile)
            _model = None
        return {'status': 'completed' if success else 'blocked'}


if __name__ == '__main__':
    mcp.run(transport='stdio')
