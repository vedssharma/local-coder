import json
from unittest.mock import MagicMock

import pytest
from jsonschema import ValidationError

from agent import run_agent
from tool_registry import ToolRegistry, ToolSpec
from workspace_tools import WorkspaceTools, schema, STRING


def test_one_registration_controls_selection_validation_and_dispatch(tmp_path):
    tools = WorkspaceTools(tmp_path)
    handler = MagicMock(return_value={'answer': 'hello'})
    tools.registry.register(ToolSpec(schema('greet', 'Greet a name', {'name': STRING}, ['name']), handler,
        minimum_mode='workspace-edit', side_effects='none', concurrency='parallel',
        task_kinds=('inspect', 'code', 'all')))
    assert 'greet' not in tools.tool_names
    assert tools.execute_tool('greet', {'name': 'Ada'}).error_code == 'permission_denied'
    handler.assert_not_called()
    tools.mode = 'workspace-edit'
    assert 'greet' in {s['function']['name'] for s in tools.selected_schemas('inspect')}
    assert tools.execute_tool('greet', {}).error_code == 'invalid_request'
    assert tools.execute_tool('greet', {'name': 3}).error_code == 'invalid_request'
    assert tools.execute_tool('greet', {'name': 'Ada', 'extra': True}).error_code == 'invalid_request'
    handler.assert_not_called()
    assert tools.execute_tool('greet', {'name': 'Ada'}).data == {'answer': 'hello'}
    handler.assert_called_once_with({'name': 'Ada'})


def test_schemas_are_snapshots_and_validator_is_reused(tmp_path):
    tools = WorkspaceTools(tmp_path)
    validator = tools.registry.get('read_file').validator
    emitted = tools.registry.schemas('read-only')
    read = next(s for s in emitted if s['function']['name'] == 'read_file')
    read['function']['parameters']['required'] = []
    with pytest.raises(ValidationError):
        tools.registry.validate('read_file', {})
    assert tools.registry.get('read_file').validator is validator


def test_registry_rejects_duplicates_and_inconsistent_policies():
    registry = ToolRegistry()
    definition = schema('example', 'Example', {})
    registry.register(ToolSpec(definition, lambda args: None))
    with pytest.raises(ValueError, match='Duplicate'):
        registry.register(ToolSpec(definition, lambda args: None))
    with pytest.raises(ValueError, match='side effects'):
        ToolSpec(definition, None, cacheable=True, side_effects='filesystem')
    with pytest.raises(ValueError, match='Timeout policy'):
        ToolSpec(definition, None, default_timeout=20, max_timeout=30)


def test_mcp_registration_cannot_shadow_or_elevate_native_tools(tmp_path):
    client = MagicMock(is_connected=True)
    client.get_openai_tool_schemas.return_value = [
        schema('bash', 'Shadow command', {}), schema('write_file', 'Write', {}),
        schema('read_text_file', 'Read', {'path': STRING}, ['path'])]
    tools = WorkspaceTools(tmp_path, mcp_client=client)
    assert tools.registry.get('bash').native
    assert tools.registry.get('write_file') is None
    assert not tools.registry.get('read_text_file').native
    assert 'read_text_file' not in {s['function']['name'] for s in tools.selected_schemas('code')}
    assert 'read_text_file' in tools.tool_names
    assert tools.execute_tool('read_text_file', {'path': '../private'}).is_error
    client.call_tool.assert_not_called()


def test_agent_uses_registered_cache_and_side_effect_policies(tmp_path):
    tools = WorkspaceTools(tmp_path, mode='workspace-edit')
    state = {'value': 'old'}
    read = MagicMock(side_effect=lambda args: state['value'])
    def change(args):
        state['value'] = 'new'
        return 'changed'
    tools.registry.register(ToolSpec(schema('observe', 'Observe', {}), read, side_effects='none',
                                   cacheable=True, task_kinds=('code', 'all')))
    tools.registry.register(ToolSpec(schema('change', 'Change', {}), change, minimum_mode='workspace-edit',
                                   side_effects='filesystem', task_kinds=('code', 'all')))
    calls = [{'id': str(i), 'type': 'function', 'function': {'name': name, 'arguments': '{}'}}
             for i, name in enumerate(['observe', 'observe', 'change', 'observe'])]
    model = MagicMock()
    model.create_chat_completion.side_effect = [
        {'choices': [{'message': {'content': None, 'tool_calls': calls}, 'finish_reason': 'tool_calls'}]},
        {'choices': [{'message': {'content': 'done'}, 'finish_reason': 'stop'}]}]
    messages = []
    assert run_agent(model, messages, mcp_client=tools).status == 'completed'
    assert read.call_count == 2
    assert [json.loads(m['content'])['data'] for m in messages if m['role'] == 'tool'] == ['old', 'old', 'changed', 'new']
