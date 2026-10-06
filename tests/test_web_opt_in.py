"""Web tools are opt-in for local models: they could send workspace content to any public host."""
import importlib.util
from pathlib import Path

import pytest

import config as cfg
import prompt_builder
from workspace_tools import WorkspaceTools


@pytest.mark.parametrize('profile, override, expected', [
    ({'backend': 'embedded'}, None, False),
    ({'backend': 'openai', 'base_url': 'http://127.0.0.1:8080/v1'}, None, False),
    ({'backend': 'openai', 'provider': 'anthropic'}, None, True),
    ({'backend': 'embedded', 'web': True}, None, True),
    ({'backend': 'openai', 'provider': 'openai', 'web': False}, None, False),
    ({'backend': 'embedded', 'web': False}, True, True),
    ({'backend': 'openai', 'provider': 'openai'}, False, False),
])
def test_web_enabled_defaults_to_hosted_only(profile, override, expected):
    assert cfg.web_enabled(profile, override) is expected


def test_disabled_web_tools_are_not_registered_or_callable(tmp_path):
    tools = WorkspaceTools(tmp_path, mode='execute', web=False)
    assert tools.tool_names == {'read', 'list', 'search', 'write', 'edit', 'bash'}
    with pytest.raises(PermissionError):
        tools.authorize('web_fetch', {'url': 'https://example.org/?q=secret'})
    tools.close()


def test_system_prompt_mentions_web_tools_only_when_offered(tmp_path):
    without = prompt_builder.build_system_message(tmp_path, tools={'read', 'list', 'search'})['content']
    assert 'web_search' not in without and 'Web content' not in without
    offered = prompt_builder.build_system_message(tmp_path, tools={'read', 'web_fetch'})['content']
    assert 'web_search and web_fetch' in offered and 'untrusted evidence' in offered


def _server():
    spec = importlib.util.spec_from_file_location(
        'local_coder_server', Path(__file__).resolve().parents[1] / 'skills/local-coder/scripts/server.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_server_web_flag(monkeypatch):
    pytest.importorskip('mcp')
    server = _server()
    monkeypatch.delenv('LOCAL_CODER_WEB', raising=False)
    assert server._env_flag('LOCAL_CODER_WEB') is None
    monkeypatch.setenv('LOCAL_CODER_WEB', '1')
    assert server._env_flag('LOCAL_CODER_WEB') is True
    monkeypatch.setenv('LOCAL_CODER_WEB', 'off')
    assert server._env_flag('LOCAL_CODER_WEB') is False
    monkeypatch.setenv('LOCAL_CODER_WEB', 'maybe')
    with pytest.raises(ValueError):
        server._env_flag('LOCAL_CODER_WEB')
