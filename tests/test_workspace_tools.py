import json
import sys
import time
import pytest
from workspace_tools import WorkspaceTools


def wait(tools, result):
    while result['running']:
        time.sleep(0.01)
        result = json.loads(tools.call_tool('poll_process', {'process_id': result['process_id']}))
    return result


def test_bash_pipeline_redirect_cwd_and_exit_status(tmp_path):
    (tmp_path / 'subdir').mkdir()
    tools = WorkspaceTools(tmp_path, mode='execute')
    try:
        result = wait(tools, json.loads(tools.call_tool('bash', {
            'command': "values=(alpha beta); printf '%s\\n' \"${values[@]}\" | tr a-z A-Z > result.txt; cat result.txt; printf 'stderr\\n' >&2; exit 7",
            'cwd': 'subdir'})))
        assert result['exit_code'] == 7
        assert 'ALPHA\nBETA\n' in result['output'] and 'stderr' in result['output']
        assert (tmp_path / 'subdir' / 'result.txt').read_text() == 'ALPHA\nBETA\n'
    finally:
        tools.close()


@pytest.mark.parametrize('mode', ['read-only', 'workspace-edit'])
def test_bash_requires_execute_mode(tmp_path, mode):
    tools = WorkspaceTools(tmp_path, mode=mode)
    assert 'bash' not in tools.tool_names
    assert tools.call_tool('bash', {'command': 'touch forbidden'}).startswith('Error:')
    assert not (tmp_path / 'forbidden').exists()


def test_bash_validation_and_missing_executable(tmp_path, monkeypatch):
    tools = WorkspaceTools(tmp_path, mode='execute')
    assert 'bash' in {s['function']['name'] for s in tools.selected_schemas('code')}
    assert 'bash' not in {s['function']['name'] for s in tools.selected_schemas('inspect')}
    for args in ({'command': ''}, {'command': ' '}, {'command': []},
                 {'command': 'touch forbidden', 'cwd': '..'},
                 {'command': 'touch forbidden', 'timeout_seconds': 301}):
        assert tools.call_tool('bash', args).startswith('Error:')
    monkeypatch.setattr('workspace_tools.shutil.which', lambda _: None)
    assert 'Bash is not installed' in tools.call_tool('bash', {'command': 'true'})
    assert not tools.processes


def test_bash_timeout_cancel_and_noninteractive_input(tmp_path):
    tools = WorkspaceTools(tmp_path, mode='execute')
    try:
        eof = wait(tools, json.loads(tools.call_tool('bash', {'command': 'read value'})))
        assert eof['exit_code'] == 1
        timed = wait(tools, json.loads(tools.call_tool('bash', {
            'command': 'sleep 30', 'timeout_seconds': 1})))
        assert timed['timed_out'] and timed['exit_code'] != 0
        running = json.loads(tools.call_tool('bash', {'command': 'sleep 30'}))
        cancelled = json.loads(tools.call_tool('cancel_process', {'process_id': running['process_id']}))
        assert not cancelled['running'] and cancelled['exit_code'] != 0
    finally:
        tools.close()


def test_patch_rejects_stale_and_ambiguous_text(tmp_path):
    p = tmp_path / 'a.py'
    p.write_text('x = 1\nx = 1\n')
    tools = WorkspaceTools(tmp_path, mode="execute")
    assert tools.call_tool('apply_patch', {'path': 'a.py', 'old_text': 'x = 1', 'new_text': 'x = 2'}).startswith('Error')
    assert p.read_text() == 'x = 1\nx = 1\n'
    assert tools.call_tool('apply_patch', {'path': 'a.py', 'old_text': 'missing', 'new_text': 'x'}).startswith('Error')


def test_search_patch_and_real_command(tmp_path):
    (tmp_path / 'a.py').write_text('assert 1 == 2\n')
    tools = WorkspaceTools(tmp_path, mode="execute")
    assert 'a.py:1:' in tools.call_tool('search_code', {'pattern': 'assert'})
    first = wait(tools, json.loads(tools.call_tool('run_command', {'argv': [sys.executable, 'a.py']})))
    assert first['exit_code'] == 1
    tools.call_tool('apply_patch', {'path': 'a.py', 'old_text': '1 == 2', 'new_text': '1 == 1'})
    second = wait(tools, json.loads(tools.call_tool('run_command', {'argv': [sys.executable, 'a.py']})))
    assert second['exit_code'] == 0
    assert tools.call_tool('read_file', {'path': 'a.py', 'start_line': 1, 'end_line': 1}) == '1: assert 1 == 1\n'
    tools.close()


def test_process_timeout_and_cancel(tmp_path):
    tools = WorkspaceTools(tmp_path, mode="execute")
    result = wait(tools, json.loads(tools.call_tool('run_command', {
        'argv': [sys.executable, '-c', 'import time; time.sleep(30)'], 'timeout_seconds': 1})))
    assert result['timed_out'] and result['exit_code'] != 0
    result = json.loads(tools.call_tool('run_command', {'argv': [sys.executable, '-c', 'import time; time.sleep(30)']}))
    result = json.loads(tools.call_tool('cancel_process', {'process_id': result['process_id']}))
    assert not result['running']
    tools.close()


def test_permission_modes_and_symlink_escape(tmp_path):
    outside = tmp_path.parent / 'outside.txt'
    outside.write_text('private')
    (tmp_path / 'escape').symlink_to(outside)
    tools = WorkspaceTools(tmp_path)
    assert tools.call_tool('read_file', {'path': 'escape'}).startswith('Error')
    assert tools.call_tool('apply_patch', {'path': 'new', 'old_text': '', 'new_text': 'x'}).startswith('Error')
    assert tools.call_tool('run_command', {'argv': [sys.executable, '-c', 'print(1)']}).startswith('Error')
    assert not (tmp_path / 'new').exists()


def test_undo_preserves_preexisting_and_subsequent_user_changes(tmp_path):
    p = tmp_path / 'a.py'
    p.write_text('user original\n')
    tools = WorkspaceTools(tmp_path, mode='workspace-edit')
    tools.call_tool('apply_patch', {'path': 'a.py', 'old_text': 'user original', 'new_text': 'agent change'})
    p.write_text('later user change\n')
    import pytest
    with pytest.raises(ValueError, match='preserve your changes'):
        tools.undo_last()
    p.write_text('agent change\n')
    assert 'Undid' in WorkspaceTools(tmp_path, mode='workspace-edit').undo_last()
    assert p.read_text() == 'user original\n'


def test_mcp_cannot_bypass_permissions(tmp_path):
    from unittest.mock import MagicMock
    client = MagicMock()
    client.is_connected = True
    client.get_openai_tool_schemas.return_value = [
        {'type': 'function', 'function': {'name': 'write_file'}},
        {'type': 'function', 'function': {'name': 'read_text_file'}}]
    tools = WorkspaceTools(tmp_path, mcp_client=client)
    assert tools.call_tool('write_file', {'path': 'a', 'content': 'bad'}).startswith('Error')
    assert tools.call_tool('read_text_file', {'path': '../outside'}).startswith('Error')
    client.call_tool.assert_not_called()


def test_diff_includes_new_files_and_output_marks_truncation(tmp_path):
    import subprocess
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '--allow-empty', '-qm', 'initial'], cwd=tmp_path, check=True)
    tools = WorkspaceTools(tmp_path, mode='execute')
    assert tools.call_tool('apply_patch', {'path': 'new.py', 'old_text': '', 'new_text': 'x = 1\n'}).startswith('Patched')
    diff = json.loads(tools.call_tool('git_diff', {}))
    assert diff['untracked_files'] == ['new.py']
    result = wait(tools, json.loads(tools.call_tool('run_command', {'argv': [sys.executable, '-c', 'print("x" * 100000)']})))
    assert result['output_truncated'] and len(result['output']) <= 32000
    tools.close()


def test_finished_command_does_not_leave_background_pipe_open(tmp_path):
    tools = WorkspaceTools(tmp_path, mode='execute')
    try:
        result = wait(tools, json.loads(tools.call_tool('run_command', {'argv': [sys.executable, '-c',
            'import subprocess, sys; subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])']})))
        state = tools.processes[result['process_id']]
        state['reader'].join(timeout=1)
        assert not state['reader'].is_alive()
    finally:
        tools.close()


def test_batches_preserve_order_and_scope(tmp_path):
    (tmp_path / 'a.py').write_text('alpha\n')
    (tmp_path / 'b.py').write_text('beta\n')
    tools = WorkspaceTools(tmp_path)
    result = json.loads(tools.call_tool('batch_read', {'requests': [{'path': 'b.py'}, {'path': 'a.py'}, {'path': '../outside'}]}))
    assert 'beta' in result[0]['output'] and 'alpha' in result[1]['output']
    assert result[2]['output'].startswith('Error:')
    searches = json.loads(tools.call_tool('batch_search', {'requests': [{'pattern': 'alpha'}, {'pattern': 'beta'}]}))
    assert 'a.py' in searches[0]['output'] and 'b.py' in searches[1]['output']
