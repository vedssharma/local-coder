"""Tests for main.py — CLI commands and helper functions."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

# Import the Typer app — avoid actually loading the LLM or connecting MCP
import main as app_module
from main import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers used across tests
# ---------------------------------------------------------------------------

def _patch_llm(text="mocked answer"):
    """Return a context manager that replaces get_llm() with a mock."""
    mock = MagicMock()
    mock.create_chat_completion.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": text, "tool_calls": None},
                "finish_reason": "stop",
            }
        ]
    }
    return patch("main.get_llm", return_value=mock)


def _patch_mcp(connected=True):
    """Return a context manager that replaces get_mcp_client() with a mock."""
    mock = MagicMock()
    mock.is_connected = connected
    mock.tool_names = set()
    mock.get_openai_tool_schemas.return_value = []
    mock.call_tool.return_value = "tool result"
    return patch("main.get_mcp_client", return_value=mock)


# ---------------------------------------------------------------------------
# models command — display
# ---------------------------------------------------------------------------

class TestModelsCommandDisplay:
    def test_shows_model_path(self, config_dir):
        result = runner.invoke(app, ["models"])
        assert result.exit_code == 0
        assert "model_path" in result.output.lower() or "Model path" in result.output

    def test_shows_context_size(self, config_dir):
        result = runner.invoke(app, ["models"])
        assert "n_ctx" in result.output.lower() or "Context size" in result.output

    def test_shows_gpu_layers(self, config_dir):
        result = runner.invoke(app, ["models"])
        assert "n_gpu_layers" in result.output.lower() or "GPU layers" in result.output

    def test_shows_not_found_when_model_missing(self, config_dir):
        result = runner.invoke(app, ["models"])
        # Default model path doesn't exist in test env
        assert "Not found" in result.output or "✗" in result.output

    def test_shows_available_when_model_exists(self, config_dir, sample_gguf_file):
        import config as cfg
        cfg.save_config({
            "model_path": sample_gguf_file,
            "n_ctx": 8192,
            "n_gpu_layers": -1,
        })
        result = runner.invoke(app, ["models"])
        assert "Available" in result.output or "✓" in result.output


# ---------------------------------------------------------------------------
# models command — --set flag
# ---------------------------------------------------------------------------

class TestModelsCommandSet:
    def test_set_nonexistent_path_exits_with_error(self, config_dir):
        result = runner.invoke(app, ["models", "--set", "/nonexistent/model.gguf"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "Error" in result.output

    def test_set_non_gguf_file_exits_with_error(self, config_dir, tmp_path):
        bad = tmp_path / "model.bin"
        bad.write_text("data")
        result = runner.invoke(app, ["models", "--set", str(bad)])
        assert result.exit_code != 0
        assert ".gguf" in result.output.lower() or "Error" in result.output

    def test_set_valid_gguf_updates_config(self, config_dir, sample_gguf_file):
        result = runner.invoke(app, ["models", "--set", sample_gguf_file])
        assert result.exit_code == 0
        assert "updated" in result.output.lower() or "✓" in result.output

    def test_set_valid_gguf_persists_path(self, config_dir, sample_gguf_file):
        runner.invoke(app, ["models", "--set", sample_gguf_file])
        import config as cfg
        stored = cfg.get_model_path()
        assert os.path.abspath(sample_gguf_file) == stored


# ---------------------------------------------------------------------------
# ask command
# ---------------------------------------------------------------------------

class TestAskCommand:
    def test_ask_basic_question(self, config_dir, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with _patch_llm("The answer is 42."), _patch_mcp():
            result = runner.invoke(app, ["ask", "What is 6 times 7?"])
        assert result.exit_code == 0

    def test_ask_with_no_mcp_flag(self, config_dir, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with _patch_llm("answer"):
            result = runner.invoke(app, ["ask", "--no-mcp", "What is 2+2?"])
        assert result.exit_code == 0

    def test_ask_respects_max_tokens_option(self, config_dir, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        captured_kwargs = {}

        mock_llm = MagicMock()
        mock_llm.create_chat_completion.side_effect = lambda **kw: (
            captured_kwargs.update(kw) or {
                "choices": [{"message": {"content": "ok", "tool_calls": None}, "finish_reason": "stop"}]
            }
        )

        with patch("main.get_llm", return_value=mock_llm), _patch_mcp():
            runner.invoke(app, ["ask", "--max-tokens", "256", "Question"])

        assert captured_kwargs.get("max_tokens") == 256


# ---------------------------------------------------------------------------
# _gather_project_context
# ---------------------------------------------------------------------------

class TestGatherProjectContext:
    def test_includes_directory_listing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "myfile.py").write_text("pass")
        ctx = app_module._gather_project_context()
        assert "myfile.py" in ctx

    def test_includes_readme_when_present(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "README.md").write_text("# My Project\n")
        ctx = app_module._gather_project_context()
        assert "README.md" in ctx
        assert "My Project" in ctx

    def test_includes_requirements_when_present(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "requirements.txt").write_text("typer\nrich\n")
        ctx = app_module._gather_project_context()
        assert "requirements.txt" in ctx

    def test_truncates_large_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "README.md").write_text("x" * 5000)
        ctx = app_module._gather_project_context()
        assert "truncated" in ctx

    def test_skips_hidden_dirs(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".git").mkdir()
        ctx = app_module._gather_project_context()
        assert ".git" not in ctx

    def test_returns_string(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ctx = app_module._gather_project_context()
        assert isinstance(ctx, str)


# ---------------------------------------------------------------------------
# get_mcp_client — lazy initialisation
# ---------------------------------------------------------------------------

class TestGetMCPClient:
    def test_returns_same_instance_on_second_call(self, monkeypatch):
        # Reset module-level singleton
        monkeypatch.setattr(app_module, "_mcp_client", None)

        fake_client = MagicMock()
        fake_client.is_connected = True
        fake_client.tool_names = {"read_file"}

        with patch("main.MCPClient", return_value=fake_client):
            c1 = app_module.get_mcp_client()
            c2 = app_module.get_mcp_client()

        assert c1 is c2
        fake_client.connect.assert_called_once()

    def test_reuses_existing_client(self, monkeypatch):
        existing = MagicMock()
        monkeypatch.setattr(app_module, "_mcp_client", existing)
        result = app_module.get_mcp_client()
        assert result is existing


# ---------------------------------------------------------------------------
# handle_model_command
# ---------------------------------------------------------------------------

class TestHandleModelCommand:
    def test_keep_current_model_on_empty_input(self, config_dir, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "")
        # Should not raise
        app_module.handle_model_command()

    def test_error_on_nonexistent_file(self, config_dir, capsys, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "/nonexistent/model.gguf")
        app_module.handle_model_command()
        # Should print an error message
        captured = capsys.readouterr()
        assert "Error" in captured.out or "not found" in captured.out.lower()

    def test_error_on_non_gguf_extension(self, config_dir, tmp_path, capsys, monkeypatch):
        bad = tmp_path / "model.bin"
        bad.write_text("data")
        monkeypatch.setattr("builtins.input", lambda _: str(bad))
        app_module.handle_model_command()
        captured = capsys.readouterr()
        assert "Error" in captured.out or ".gguf" in captured.out.lower()

    def test_numeric_selection_switches_model(self, config_dir, tmp_path, monkeypatch, capsys):
        gguf = tmp_path / "alt.gguf"
        gguf.write_text("fake")

        # Patch glob.glob to return our fake gguf
        monkeypatch.setattr("main.glob.glob", lambda _: [str(gguf)])

        loaded = {}

        def fake_llama(model_path, **kwargs):
            loaded["path"] = model_path
            return MagicMock()

        monkeypatch.setattr("main.Llama", fake_llama)
        monkeypatch.setattr("builtins.input", lambda _: "1")
        app_module.handle_model_command()
        assert "path" in loaded


def test_configure_openai_profile(config_dir):
    result = runner.invoke(app, ['models', '--backend', 'openai', '--base-url', 'http://127.0.0.1:8080/v1', '--model-name', 'coder', '--no-tools'])
    assert result.exit_code == 0, result.output
    import config
    profile = config.get_model_config()
    assert profile['backend'] == 'openai' and profile['model'] == 'coder'
    assert profile['supports_tools'] is False


def test_profile_cli_preserves_named_settings(config_dir):
    result = runner.invoke(app, ['profiles', 'save', 'small'])
    assert result.exit_code == 0, result.output
    assert runner.invoke(app, ['profiles', 'use', 'small']).exit_code == 0
    assert runner.invoke(app, ['models', '--threads', '2']).exit_code == 0
    assert runner.invoke(app, ['profiles', 'route', 'answer', 'small']).exit_code == 0
    import config
    assert config.get_model_config('small')['n_threads'] == 2
    assert config.load_config()['routes']['answer'] == 'small'
