"""Shared fixtures for the local-coder test suite."""

import sys
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Stub out heavy / unavailable native modules before any app code is imported.
# llama_cpp requires a compiled binary and cannot be pip-installed in test CI.
# ---------------------------------------------------------------------------
_llama_stub = MagicMock()
_llama_stub.Llama = MagicMock
sys.modules.setdefault("llama_cpp", _llama_stub)


@pytest.fixture
def tmp_dir(tmp_path):
    """Return a temporary directory path object."""
    return tmp_path


@pytest.fixture
def sample_gguf_file(tmp_path):
    """Create a fake .gguf file for model path tests."""
    model_file = tmp_path / "model.gguf"
    model_file.write_text("fake model content")
    return str(model_file)


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    """
    Redirect config storage to a temporary directory.

    Patches config.CONFIG_DIR and config.CONFIG_FILE to point at tmp_path
    so tests don't touch ~/.local-coder.
    """
    import config as cfg

    fake_dir = tmp_path / ".local-coder"
    fake_file = fake_dir / "config.json"

    monkeypatch.setattr(cfg, "CONFIG_DIR", fake_dir)
    monkeypatch.setattr(cfg, "CONFIG_FILE", fake_file)

    return fake_dir


@pytest.fixture
def mock_llm():
    """Return a mock LLM that returns a simple text response by default."""
    llm = MagicMock()
    llm.create_chat_completion.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello, world!", "tool_calls": None},
                "finish_reason": "stop",
            }
        ]
    }
    return llm


@pytest.fixture
def mock_mcp_client():
    """Return a mock MCPClient that is connected and exposes no tools."""
    client = MagicMock()
    client.is_connected = True
    client.get_openai_tool_schemas.return_value = [
        {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
        for name in ("read_file", "list_directory")
    ]
    client.call_tool.return_value = "tool result"
    from tool_registry import ToolRegistry, ToolSpec
    client.registry = ToolRegistry()
    for definition in client.get_openai_tool_schemas.return_value:
        client.registry.register(ToolSpec(definition, handler=None, side_effects='none',
                                         cacheable=True, compact_observation=True))
    return client


@pytest.fixture
def mock_console():
    """Return a mock Rich Console."""
    return MagicMock()
