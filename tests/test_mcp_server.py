"""MCP server budget and model lifecycle (skills/local-coder/scripts/server.py)."""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip('mcp')


def _server():
    spec = importlib.util.spec_from_file_location(
        'local_coder_server_under_test', Path(__file__).resolve().parents[1] / 'skills/local-coder/scripts/server.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_budget_matches_cli_defaults_and_reads_launch_environment(monkeypatch):
    server = _server()
    for name in ('LOCAL_CODER_MAX_STEPS', 'LOCAL_CODER_MAX_SECONDS', 'LOCAL_CODER_TOKEN_BUDGET'):
        monkeypatch.delenv(name, raising=False)
    budget = server._budget()
    assert (budget.max_steps, budget.max_seconds, budget.max_generated_tokens) == (30, 300.0, 8192)
    monkeypatch.setenv('LOCAL_CODER_MAX_STEPS', '60')
    monkeypatch.setenv('LOCAL_CODER_MAX_SECONDS', '600')
    monkeypatch.setenv('LOCAL_CODER_TOKEN_BUDGET', '16384')
    budget = server._budget()
    assert (budget.max_steps, budget.max_seconds, budget.max_generated_tokens) == (60, 600.0, 16384)
    monkeypatch.setenv('LOCAL_CODER_MAX_STEPS', 'many')
    with pytest.raises(ValueError, match='LOCAL_CODER_MAX_STEPS'):
        server._budget()
    monkeypatch.setenv('LOCAL_CODER_MAX_STEPS', '0')
    with pytest.raises(ValueError):
        server._budget()


def test_set_model_releases_the_loaded_model(config_dir, tmp_path, monkeypatch):
    server = _server()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('LOCAL_CODER_PERMISSION_MODE', 'workspace-edit')
    (tmp_path / 'small.gguf').write_bytes(b'GGUF')

    class Loaded:
        closed = False

        def close(self):
            self.closed = True

    old = Loaded()
    server._model, server._model_signature = old, 'old-profile'
    assert server.set_model('small.gguf') == {'status': 'completed'}
    assert old.closed
    assert server._model is None and server._model_signature is None
