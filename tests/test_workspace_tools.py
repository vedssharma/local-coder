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
    assert tools.tool_names == {'read', 'list', 'search', 'write', 'edit', 'bash', 'web_search', 'web_fetch'}
    tools.close()


@pytest.mark.parametrize('mode', ['read-only', 'workspace-edit'])
def test_list_and_search_work_without_execute_mode(tmp_path, mode):
    (tmp_path / 'pkg').mkdir()
    (tmp_path / 'pkg' / 'core.py').write_text('def add(a, b):\n    return a + b\n')
    (tmp_path / 'README.md').write_text('Call add() to sum.\n')
    (tmp_path / 'node_modules').mkdir()
    (tmp_path / 'node_modules' / 'dep.py').write_text('def add(): pass\n')
    (tmp_path / 'blob.bin').write_bytes(b'add\0\x01')
    tools = WorkspaceTools(tmp_path, mode=mode)
    assert {'list', 'search'} <= {s['function']['name'] for s in tools.selected_schemas('auto')}
    assert tools.call_tool('list', {}).splitlines() == ['README.md', 'blob.bin', 'pkg/core.py']
    assert tools.call_tool('list', {'pattern': '**/*.py'}) == 'pkg/core.py'
    assert tools.call_tool('list', {'path': 'pkg'}) == 'pkg/core.py'
    assert tools.call_tool('list', {'max_entries': 1}).splitlines() == ['README.md', '[2 more files not shown; narrow path or pattern]']
    assert tools.call_tool('search', {'pattern': r'def add'}) == 'pkg/core.py:1: def add(a, b):'
    assert tools.call_tool('search', {'pattern': 'CALL', 'case_sensitive': False}) == 'README.md:1: Call add() to sum.'
    assert tools.call_tool('search', {'pattern': 'add', 'glob': '*.md'}) == 'README.md:1: Call add() to sum.'
    assert tools.call_tool('search', {'pattern': 'nothing here'}) == '(no matches)'
    assert tools.execute_tool('search', {'pattern': '('}).error_code == 'invalid_request'
    assert tools.execute_tool('list', {'path': 'missing'}).error_code == 'not_found'
    assert tools.execute_tool('list', {'path': '..'}).error_code == 'invalid_request'
    tools.close()


def test_list_respects_gitignore_and_skips_escaping_symlinks(tmp_path):
    import shutil, subprocess
    if not shutil.which('git'):
        pytest.skip('git is not installed')
    subprocess.run(['git', 'init', '-q'], cwd=tmp_path, check=True)
    (tmp_path / '.gitignore').write_text('build/\n')
    (tmp_path / 'build').mkdir()
    (tmp_path / 'build' / 'out.py').write_text('x = 1\n')
    (tmp_path / 'src.py').write_text('x = 1\n')
    outside = tmp_path.parent / f'{tmp_path.name}-outside.txt'
    outside.write_text('x = 1\n')
    (tmp_path / 'link.txt').symlink_to(outside)
    tools = WorkspaceTools(tmp_path)
    assert tools.call_tool('list', {}).splitlines() == ['.gitignore', 'src.py']
    assert tools.call_tool('search', {'pattern': 'x = 1'}) == 'src.py:1: x = 1'
    tools.close()


def test_search_limits_results(tmp_path):
    (tmp_path / 'a.txt').write_text('hit\n' * 5)
    tools = WorkspaceTools(tmp_path)
    lines = tools.call_tool('search', {'pattern': 'hit', 'max_results': 2}).splitlines()
    assert lines == ['a.txt:1: hit', 'a.txt:2: hit', '[more matches not shown; narrow pattern, path, or glob]']
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


# ---------------------------------------------------------------------------
# Error mapping, ranges, listing fallbacks, undo safety and process cleanup
# ---------------------------------------------------------------------------

import errno
import subprocess
import threading
from pathlib import Path

import web_tools
import workspace_tools
from execution_context import ExecutionContext
from workspace_tools import PatchConflict, edit_text


def _context():
    return ExecutionContext(time.monotonic() + 60, threading.Event())


def test_unknown_modes_and_task_kinds_are_rejected(tmp_path):
    with pytest.raises(ValueError, match='Unknown permission mode'):
        WorkspaceTools(tmp_path, mode='root')
    with pytest.raises(ValueError, match='task_kind'):
        WorkspaceTools(tmp_path).selected_schemas('everything')


def test_authorize_checks_every_listed_path(tmp_path):
    tools = WorkspaceTools(tmp_path)
    with pytest.raises(ValueError, match='outside the workspace'):
        tools.authorize('read', {'path': 'a', 'paths': ['ok', '../escape']})


def test_read_ranges(tmp_path):
    (tmp_path / 'a.txt').write_text(''.join(f'line {n}\n' for n in range(1, 11)))
    tools = WorkspaceTools(tmp_path)
    assert tools.execute_tool('read', {'path': 'a.txt', 'start_line': 3, 'end_line': 4}).data == '3: line 3\n4: line 4\n'
    for bad in ({'start_line': 5, 'end_line': 4}, {'start_line': 1, 'end_line': 2000}):
        result = tools.execute_tool('read', {'path': 'a.txt', **bad})
        assert result.error_code == 'invalid_request' and 'at most 1001 lines' in result.error_message
    assert 'minimum of 1' in tools.execute_tool('read', {'path': 'a.txt', 'start_line': 0}).error_message


@pytest.mark.parametrize('error, code, retryable', [
    (web_tools.WebRequestError('HTTP 429', 'http_429', True), 'http_429', True),
    (TimeoutError('slow'), 'tool_timeout', True),
    (OSError(errno.ECONNRESET, 'reset'), 'io_error', True),
    (OSError(errno.ENOSPC, 'disk full'), 'io_error', False),
])
def test_tool_failures_map_to_error_codes(tmp_path, monkeypatch, error, code, retryable):
    def fail(**kwargs):
        raise error
    monkeypatch.setattr(web_tools, 'fetch', fail)
    result = WorkspaceTools(tmp_path).execute_tool('web_fetch', {'url': 'https://example.com'})
    assert (result.status, result.error_code, result.retryable) == ('error', code, retryable)


def test_web_search_gets_the_default_timeout(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(web_tools, 'search', lambda **kwargs: seen.update(kwargs) or 'results')
    assert WorkspaceTools(tmp_path).execute_tool('web_search', {'query': 'python'}).data == 'results'
    assert seen == {'query': 'python', 'timeout_seconds': 20}


def test_list_of_a_single_file_and_bound_context(tmp_path):
    (tmp_path / 'pkg').mkdir()
    (tmp_path / 'pkg' / 'a.py').write_text('needle\n')
    tools = WorkspaceTools(tmp_path)
    assert tools.execute_tool('list', {'path': 'pkg/a.py'}).data == 'pkg/a.py'
    assert tools.execute_tool('list', {'path': 'pkg'}, context=_context()).data == 'pkg/a.py'
    assert tools.execute_tool('search', {'pattern': 'needle'}, context=_context()).data == 'pkg/a.py:1: needle'


def _git(root, *args):
    subprocess.run(['git', *args], cwd=root, check=True, capture_output=True)


def test_git_listing_still_skips_dependency_directories(tmp_path):
    _git(tmp_path, 'init', '-q')
    (tmp_path / 'node_modules').mkdir()
    (tmp_path / 'node_modules' / 'dep.js').write_text('x')
    (tmp_path / 'app.js').write_text('x')
    assert WorkspaceTools(tmp_path).execute_tool('list', {}).data == 'app.js'


def test_listing_falls_back_to_walking_when_git_fails(tmp_path, monkeypatch):
    (tmp_path / '.git').mkdir()  # Not a real repository: git exits nonzero.
    (tmp_path / 'a.txt').write_text('x')
    tools = WorkspaceTools(tmp_path)
    assert tools.execute_tool('list', {}).data == 'a.txt'

    def hang(*args, **kwargs):
        raise subprocess.TimeoutExpired('git', 20)
    monkeypatch.setattr(workspace_tools.subprocess, 'run', hang)
    assert tools.execute_tool('list', {}).data == 'a.txt'


def test_search_skips_oversized_files(tmp_path, monkeypatch):
    (tmp_path / 'big.txt').write_text('needle ' * 100)
    (tmp_path / 'small.txt').write_text('needle\n')
    monkeypatch.setattr(workspace_tools, 'MAX_SEARCH_FILE_BYTES', 100)
    assert WorkspaceTools(tmp_path).execute_tool('search', {'pattern': 'needle'}).data == 'small.txt:1: needle'


def test_search_skips_files_that_cannot_be_read(tmp_path, monkeypatch):
    (tmp_path / 'a.txt').write_text('needle\n')
    (tmp_path / 'b.txt').write_text('needle\n')
    read_bytes = Path.read_bytes

    def flaky(self):
        if self.name == 'a.txt':
            raise PermissionError('denied')
        return read_bytes(self)
    monkeypatch.setattr(Path, 'read_bytes', flaky)
    assert WorkspaceTools(tmp_path).execute_tool('search', {'pattern': 'needle'}).data == 'b.txt:1: needle'


def test_write_refuses_directories_and_concurrent_changes(tmp_path, monkeypatch):
    (tmp_path / 'folder').mkdir()
    tools = WorkspaceTools(tmp_path, mode='workspace-edit')
    result = tools.execute_tool('write', {'path': 'folder', 'content': 'x'})
    assert result.error_code == 'invalid_request' and 'directory' in result.error_message
    target = tmp_path / 'a.txt'
    target.write_text('original')
    chmod = Path.chmod

    def interfere(self, mode):
        target.write_text('someone else')
        return chmod(self, mode)
    monkeypatch.setattr(Path, 'chmod', interfere)
    result = tools.execute_tool('write', {'path': 'a.txt', 'content': 'mine'})
    assert result.error_code == 'stale_patch'
    assert target.read_text() == 'someone else'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['a.txt', 'folder']


def test_undo_requires_edit_mode_and_rejects_symlinked_records(tmp_path):
    with pytest.raises(PermissionError):
        WorkspaceTools(tmp_path).undo_last()
    tools = WorkspaceTools(tmp_path, mode='workspace-edit')
    assert tools.undo_last() == 'No harness edits to undo.'
    undo = tmp_path / '.local-coder' / 'undo'
    undo.mkdir(parents=True)
    (tmp_path / 'elsewhere.json').write_text('{}')
    (undo / '1.json').symlink_to(tmp_path / 'elsewhere.json')
    with pytest.raises(ValueError, match='Invalid undo record'):
        tools.undo_last()


def test_undo_refuses_an_inconsistent_chain(tmp_path):
    tools = WorkspaceTools(tmp_path, mode='workspace-edit')
    tools.turn_id = 'turn'
    tools.execute_tool('write', {'path': 'a.txt', 'content': 'one'})
    tools.execute_tool('write', {'path': 'a.txt', 'content': 'two'})
    first = sorted((tmp_path / '.local-coder' / 'undo').glob('*.json'))[0]
    record = json.loads(first.read_text())
    first.write_text(json.dumps({**record, 'after': 'tampered'}))
    with pytest.raises(ValueError, match='inconsistent'):
        tools.undo_turn()
    assert (tmp_path / 'a.txt').read_text() == 'two'


def test_undo_stops_if_the_file_changes_while_undoing(tmp_path, monkeypatch):
    tools = WorkspaceTools(tmp_path, mode='workspace-edit')
    (tmp_path / 'a.txt').write_text('before')
    tools.execute_tool('write', {'path': 'a.txt', 'content': 'after'})
    chmod = Path.chmod

    def interfere(self, mode):
        (tmp_path / 'a.txt').write_text('user edit')
        return chmod(self, mode)
    monkeypatch.setattr(Path, 'chmod', interfere)
    with pytest.raises(ValueError, match='File changed during undo'):
        tools.undo_last()
    assert (tmp_path / 'a.txt').read_text() == 'user edit'


def test_process_arguments_are_validated(tmp_path):
    tools = WorkspaceTools(tmp_path, mode='execute')
    with pytest.raises(ValueError, match='argv'):
        tools._start([], tmp_path, 10)
    with pytest.raises(ValueError, match='argv'):
        tools._start(['echo', 1], tmp_path, 10)
    with pytest.raises(ValueError, match='timeout_seconds'):
        tools._start(['true'], tmp_path, 301)


def test_cancel_all_processes_kills_running_commands(tmp_path):
    tools = WorkspaceTools(tmp_path, mode='execute')
    started = tools._start(['sleep', '30'], tmp_path, 60)
    assert started['running']
    tools.cancel_all_processes()
    state = tools.processes[started['process_id']]
    assert state['cancelled'] and state['proc'].poll() is not None
    tools.close()


# edit_text matching corner cases ---------------------------------------------

def test_edit_ignores_blank_line_whitespace_and_shifts_indentation():
    original = 'def f():\n    a = 1\n   \n    return a\n'
    updated, note = edit_text(original, 'a = 1\n\nreturn a', 'a = 2\n\nreturn a')
    assert updated == 'def f():\n    a = 2\n\n    return a\n'
    assert 'added 4 characters of indentation' in note


@pytest.mark.parametrize('original, old', [
    ('\tvalue = 1\n', '  value = 1'),               # tab vs spaces: no uniform shift
    ('    a = 1\n  b = 2\n', 'a = 1\nb = 2'),       # different shifts per line
])
def test_edit_rejects_non_uniform_indentation(original, old):
    with pytest.raises(PatchConflict):
        edit_text(original, old, 'x')


def test_edit_cannot_dedent_a_replacement_line_lacking_the_prefix():
    original = 'a = 1\nb = 2\n'
    with pytest.raises(PatchConflict):
        edit_text(original, '    a = 1\n    b = 2', '    a = 3\nb = 4')


def test_edit_with_blank_old_text_asks_for_a_reread():
    with pytest.raises(PatchConflict, match='Reread the file'):
        edit_text('content\n', '\n\n', 'x')
