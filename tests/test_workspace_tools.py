import json
import sys
import time
import pytest
from workspace_tools import WorkspaceTools


def test_bash_pipeline_redirect_cwd_and_exit_status(tmp_path):
    (tmp_path / 'subdir').mkdir()
    tools = WorkspaceTools(tmp_path, mode='execute')
    try:
        result = (json.loads(tools.call_tool('bash', {
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


def test_bash_timeout_and_noninteractive_input(tmp_path):
    tools = WorkspaceTools(tmp_path, mode='execute')
    try:
        eof = json.loads(tools.call_tool('bash', {'command': 'read value'}))
        assert eof['exit_code'] == 1
        started = time.monotonic()
        timed = json.loads(tools.call_tool('bash', {'command': 'sleep 30', 'timeout_seconds': 1}))
        assert timed['timed_out'] and timed['exit_code'] != 0 and time.monotonic() - started < 10
        assert not tools.processes
    finally:
        tools.close()


def test_patch_rejects_stale_and_ambiguous_text(tmp_path):
    p = tmp_path / 'a.py'
    p.write_text('x = 1\nx = 1\n')
    tools = WorkspaceTools(tmp_path, mode="execute")
    assert tools.call_tool('edit', {'path': 'a.py', 'old_text': 'x = 1', 'new_text': 'x = 2'}).startswith('Error')
    assert p.read_text() == 'x = 1\nx = 1\n'
    assert tools.call_tool('edit', {'path': 'a.py', 'old_text': 'missing', 'new_text': 'x'}).startswith('Error')


def test_read_edit_write_and_real_command(tmp_path):
    (tmp_path / 'a.py').write_text('assert 1 == 2\n')
    tools = WorkspaceTools(tmp_path, mode="execute")
    first = json.loads(tools.call_tool('bash', {'command': f'{sys.executable} a.py'}))
    assert first['exit_code'] == 1
    tools.call_tool('edit', {'path': 'a.py', 'old_text': '1 == 2', 'new_text': '1 == 1'})
    second = json.loads(tools.call_tool('bash', {'command': f'{sys.executable} a.py'}))
    assert second['exit_code'] == 0
    assert tools.call_tool('read', {'path': 'a.py', 'start_line': 1, 'end_line': 1}) == '1: assert 1 == 1\n'
    assert tools.call_tool('write', {'path': 'pkg/b.py', 'content': 'y = 1\n'}).startswith('Wrote')
    assert (tmp_path / 'pkg' / 'b.py').read_text() == 'y = 1\n'
    assert tools.call_tool('write', {'path': 'pkg/b.py', 'content': 'y = 2\n'}).startswith('Wrote')
    assert (tmp_path / 'pkg' / 'b.py').read_text() == 'y = 2\n'
    # edit cannot create files, and neither tool can touch harness metadata.
    assert tools.call_tool('edit', {'path': 'new.py', 'old_text': 'a', 'new_text': 'b'}).startswith('Error')
    assert tools.call_tool('write', {'path': '.git/config', 'content': 'x'}).startswith('Error')
    assert tools.call_tool('write', {'path': '.local-coder/undo/x', 'content': 'x'}).startswith('Error')
    tools.close()


def test_registered_tools_are_the_minimal_set(tmp_path):
    tools = WorkspaceTools(tmp_path, mode='execute')
    assert tools.tool_names == {'read', 'write', 'edit', 'bash', 'web_search', 'web_fetch'}
    tools.close()


def test_permission_modes_and_symlink_escape(tmp_path):
    outside = tmp_path.parent / 'outside.txt'
    outside.write_text('private')
    (tmp_path / 'escape').symlink_to(outside)
    tools = WorkspaceTools(tmp_path)
    assert tools.call_tool('read', {'path': 'escape'}).startswith('Error')
    assert tools.call_tool('write', {'path': 'new', 'content': 'x'}).startswith('Error')
    assert tools.call_tool('bash', {'command': 'echo 1'}).startswith('Error')
    assert not (tmp_path / 'new').exists()


def test_undo_preserves_preexisting_and_subsequent_user_changes(tmp_path):
    p = tmp_path / 'a.py'
    p.write_text('user original\n')
    tools = WorkspaceTools(tmp_path, mode='workspace-edit')
    tools.call_tool('edit', {'path': 'a.py', 'old_text': 'user original', 'new_text': 'agent change'})
    p.write_text('later user change\n')
    import pytest
    with pytest.raises(ValueError, match='preserve your changes'):
        tools.undo_last()
    p.write_text('agent change\n')
    assert 'Undid' in WorkspaceTools(tmp_path, mode='workspace-edit').undo_last()
    assert p.read_text() == 'user original\n'


def test_output_marks_truncation(tmp_path):
    tools = WorkspaceTools(tmp_path, mode='execute')
    try:
        result = json.loads(tools.call_tool('bash', {'command': f"{sys.executable} -c 'print(\"x\" * 100000)'"}))
        assert result['output_truncated'] and len(result['output']) <= 32000
    finally:
        tools.close()


def test_finished_command_does_not_leave_background_process_running(tmp_path):
    tools = WorkspaceTools(tmp_path, mode='execute')
    try:
        started = time.monotonic()
        result = json.loads(tools.call_tool('bash', {'command': 'sleep 30 & echo started'}))
        assert result['exit_code'] == 0 and 'started' in result['output']
        assert time.monotonic() - started < 10
    finally:
        tools.close()


def test_finished_command_cleanup_is_signalled_once(tmp_path, monkeypatch):
    import os
    signals = []
    real_killpg = os.killpg
    def track(group, signal):
        signals.append(group)
        return real_killpg(group, signal)
    monkeypatch.setattr(os, 'killpg', track)
    tools = WorkspaceTools(tmp_path, mode='execute')
    try:
        assert tools.execute_tool('bash', {'command': 'echo ok'}).status == 'success'
    finally:
        tools.close()
    assert len(signals) == 1
