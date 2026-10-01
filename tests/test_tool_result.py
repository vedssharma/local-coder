import json
import sys
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent import run_agent
from mcp_client import MCPClient
from session import ContextManager
from tool_result import ToolResult
from workspace_tools import WorkspaceTools


def completed(tools, result):
    while result.status == 'running':
        time.sleep(0.01)
        result = tools.execute_tool('poll_process', {'process_id': result.data['process_id']})
    return result


def test_command_failure_is_distinct_from_launch_error(tmp_path):
    with_tools = WorkspaceTools(tmp_path, mode='execute')
    try:
        failed = completed(with_tools, with_tools.execute_tool('run_command', {
            'argv': [sys.executable, '-c', 'print("failed check"); raise SystemExit(7)']}))
        assert failed.status == 'failed' and failed.error_code == 'command_failed'
        assert failed.data['exit_code'] == 7 and 'failed check' in failed.data['output']
        launch = with_tools.execute_tool('run_command', {'argv': [str(tmp_path / 'missing')]})
        assert launch.status == 'error' and launch.error_code == 'not_found'
        assert launch.data is None and not launch.retryable
        assert failed.duration_seconds >= 0
        denied = WorkspaceTools(tmp_path).execute_tool('bash', {'command': 'true'})
        assert denied.error_code == 'permission_denied'
    finally:
        with_tools.close()


def test_bounded_envelope_retains_status_and_artifact(tmp_path):
    context = ContextManager(artifact_dir=tmp_path)
    result = ToolResult(status='failed', data={'output': 'long output\n' * 1000, 'exit_code': 3},
                        error_code='command_failed')
    model = json.loads(result.to_model(context))
    assert model['status'] == 'failed' and model['error_code'] == 'command_failed'
    assert model['data_truncated'] and len(model['data']) < 4000
    assert model['artifacts'] == result.artifacts
    from pathlib import Path
    assert json.loads(Path(model['artifacts'][0]).read_text())['exit_code'] == 3


def test_successful_error_prefixed_text_does_not_block_agent(tmp_path):
    (tmp_path / 'log').write_text('Error: this is content, not a tool failure')
    tools = WorkspaceTools(tmp_path)
    model = MagicMock()
    call = {'choices': [{'message': {'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': 'read', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{"path":"log"}'}}]},
        'finish_reason': 'tool_calls'}]}
    model.create_chat_completion.side_effect = [call, call, call,
        {'choices': [{'message': {'content': 'done'}, 'finish_reason': 'stop'}]}]
    events = []
    assert run_agent(model, [], mcp_client=tools, emit=events.append).status == 'completed'
    assert all(e['result']['status'] == 'success' for e in events if e['type'] == 'tool_finished')


@pytest.mark.asyncio
async def test_mcp_uses_error_flag_not_text_prefix(tmp_path):
    client = MCPClient(tmp_path)
    client._session = AsyncMock()
    response = MagicMock()
    response.content = [MagicMock(text='Error: ordinary document text')]
    response.isError = False
    client._session.call_tool.return_value = response
    assert (await client._execute_tool_async('read_file', {})).status == 'success'
    response.isError = True
    assert (await client._execute_tool_async('read_file', {})).error_code == 'mcp_tool_error'


def test_batch_preserves_child_failure_metadata(tmp_path):
    (tmp_path / 'ok').write_text('ok')
    result = WorkspaceTools(tmp_path).execute_tool('batch_read', {'requests': [{'path': 'ok'}, {'path': 'missing'}]})
    assert result.status == 'success'
    assert result.data[0]['result']['status'] == 'success'
    assert result.data[1]['result']['error_code'] == 'not_found'
