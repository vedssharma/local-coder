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
        with _patch_llm("The answer is 42."):
            result = runner.invoke(app, ["ask", "What is 6 times 7?"])
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

        with patch("main.get_llm", return_value=mock_llm):
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


def test_configure_hosted_provider_stores_key_privately(config_dir, monkeypatch):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    result = runner.invoke(app, ['models', '--provider', 'anthropic', '--model-name', 'claude-sonnet-5-5'], input='sk-test\n')
    assert result.exit_code == 0, result.output
    import config, providers, os, stat
    profile = config.get_model_config()
    assert profile['provider'] == 'anthropic' and profile['model'] == 'claude-sonnet-5-5'
    assert profile['base_url'] == 'https://api.anthropic.com/v1'
    assert 'sk-test' not in config.CONFIG_FILE.read_text()
    assert stat.S_IMODE(os.stat(providers.keys_file()).st_mode) == 0o600
    headers = providers.auth_headers(profile)
    assert headers['x-api-key'] == 'sk-test' and headers['Authorization'] == 'Bearer sk-test'


def test_provider_env_key_overrides_stored(config_dir, monkeypatch):
    import providers
    providers.save_key('openai', 'stored')
    monkeypatch.setenv('OPENAI_API_KEY', 'fromenv')
    assert providers.resolve_key({'provider': 'openai'}) == 'fromenv'


def test_model_command_api_flow(config_dir, monkeypatch):
    answers = iter(['api', '1', '2', ''])
    monkeypatch.setattr('builtins.input', lambda _: next(answers))
    monkeypatch.setattr('getpass.getpass', lambda _: 'sk-abc')
    app_module.handle_model_command()
    import config
    profile = config.get_model_config()
    assert profile['provider'] == 'openai' and profile['model'] == 'gpt-5-mini'
