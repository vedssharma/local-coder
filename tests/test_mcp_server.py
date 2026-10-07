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


# ---------------------------------------------------------------------------
# MCP tools end to end with a fake model
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock


def _fake_model(*texts):
    model = MagicMock()
    model.create_chat_completion.side_effect = [
        {'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}]} for text in texts]
    return model


def test_env_flag_parses_booleans(monkeypatch):
    server = _server()
    monkeypatch.delenv('LOCAL_CODER_WEB', raising=False)
    assert server._env_flag('LOCAL_CODER_WEB') is None
    monkeypatch.setenv('LOCAL_CODER_WEB', 'Yes')
    assert server._env_flag('LOCAL_CODER_WEB') is True
    monkeypatch.setenv('LOCAL_CODER_WEB', 'off')
    assert server._env_flag('LOCAL_CODER_WEB') is False
    monkeypatch.setenv('LOCAL_CODER_WEB', 'maybe')
    with pytest.raises(ValueError, match='LOCAL_CODER_WEB must be 1 or 0'):
        server._env_flag('LOCAL_CODER_WEB')


def test_model_instance_is_cached_per_profile_and_replaced_on_change(config_dir, monkeypatch):
    import config
    server = _server()
    created = []

    def create(profile):
        created.append(MagicMock())
        return created[-1]
    monkeypatch.setattr(server, 'create_model', create)
    first = server.get_model_instance()
    assert server.get_model_instance() is first
    config.update_model_config({**config.get_model_config(), 'n_threads': 3})
    second = server.get_model_instance()
    assert second is not first and first.close.called
    assert len(created) == 2


def test_ask_and_chat_share_a_resumable_session(config_dir, tmp_path, monkeypatch):
    server = _server()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('LOCAL_CODER_PERMISSION_MODE', raising=False)
    (tmp_path / 'notes.txt').write_text('the secret word is plum')
    model = _fake_model('first', 'second')
    monkeypatch.setattr(server, 'get_model_instance', lambda profile_name=None: model)
    result = server.ask('What does @notes.txt say?', files=['notes.txt'], max_tokens=64)
    assert result['status'] == 'completed' and result['text'] == 'first'
    sent = model.create_chat_completion.call_args.kwargs['messages']
    assert any('the secret word is plum' in (m.get('content') or '') for m in sent)
    followup = server.chat('And then?', session_id=result['session_id'])
    assert followup['session_id'] == result['session_id'] and followup['text'] == 'second'
    assert any(m.get('content') == 'first' for m in model.create_chat_completion.call_args.kwargs['messages'])


def test_edit_and_set_model_are_blocked_in_read_only_mode(config_dir, tmp_path, monkeypatch):
    server = _server()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('LOCAL_CODER_PERMISSION_MODE', raising=False)
    assert server.edit('change things')['status'] == 'blocked'
    assert server.set_model('model.gguf')['status'] == 'blocked'


def test_edit_runs_with_edit_tools_when_permitted(config_dir, tmp_path, monkeypatch):
    server = _server()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('LOCAL_CODER_PERMISSION_MODE', 'workspace-edit')
    model = _fake_model('edited')
    monkeypatch.setattr(server, 'get_model_instance', lambda profile_name=None: model)
    assert server.edit('change things')['status'] == 'completed'
    names = {t['function']['name'] for t in model.create_chat_completion.call_args.kwargs['tools']}
    assert 'edit' in names


def test_get_model_omits_credentials(config_dir, monkeypatch):
    import config, providers
    server = _server()
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-secret')
    config.update_model_config({**config.get_model_config(), **providers.provider_profile('openai', 'gpt-5-mini')})
    described = server.get_model()
    assert described['backend'] == 'openai' and described['model'] == 'gpt-5-mini'
    assert 'sk-secret' not in str(described) and 'provider' not in described
