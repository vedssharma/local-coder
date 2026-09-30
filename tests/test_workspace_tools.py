import json
import sys
import time
from workspace_tools import WorkspaceTools


def wait(tools, result):
    while result['running']:
        time.sleep(0.01)
        result = json.loads(tools.call_tool('poll_process', {'process_id': result['process_id']}))
    return result


def test_patch_rejects_stale_and_ambiguous_text(tmp_path):
    p = tmp_path / 'a.py'
    p.write_text('x = 1\nx = 1\n')
    tools = WorkspaceTools(tmp_path)
    assert tools.call_tool('apply_patch', {'path': 'a.py', 'old_text': 'x = 1', 'new_text': 'x = 2'}).startswith('Error')
    assert p.read_text() == 'x = 1\nx = 1\n'
    assert tools.call_tool('apply_patch', {'path': 'a.py', 'old_text': 'missing', 'new_text': 'x'}).startswith('Error')


def test_search_patch_and_real_command(tmp_path):
    (tmp_path / 'a.py').write_text('assert 1 == 2\n')
    tools = WorkspaceTools(tmp_path)
    assert 'a.py:1:' in tools.call_tool('search_code', {'pattern': 'assert'})
    first = wait(tools, json.loads(tools.call_tool('run_command', {'argv': [sys.executable, 'a.py']})))
    assert first['exit_code'] == 1
    tools.call_tool('apply_patch', {'path': 'a.py', 'old_text': '1 == 2', 'new_text': '1 == 1'})
    second = wait(tools, json.loads(tools.call_tool('run_command', {'argv': [sys.executable, 'a.py']})))
    assert second['exit_code'] == 0
    assert tools.call_tool('read_file', {'path': 'a.py', 'start_line': 1, 'end_line': 1}) == '1: assert 1 == 1\n'
    tools.close()


def test_process_timeout_and_cancel(tmp_path):
    tools = WorkspaceTools(tmp_path)
    result = wait(tools, json.loads(tools.call_tool('run_command', {
        'argv': [sys.executable, '-c', 'import time; time.sleep(30)'], 'timeout_seconds': 1})))
    assert result['timed_out'] and result['exit_code'] != 0
    result = json.loads(tools.call_tool('run_command', {'argv': [sys.executable, '-c', 'import time; time.sleep(30)']}))
    result = json.loads(tools.call_tool('cancel_process', {'process_id': result['process_id']}))
    assert not result['running']
    tools.close()
