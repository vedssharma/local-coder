"""edit: exact matches, start_line for duplicates, whitespace-tolerant matches, and useful errors."""
import pytest

from workspace_tools import PatchConflict, WorkspaceTools, edit_text

SOURCE = 'def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n'


def test_exact_unique_match_is_unchanged_behavior():
    assert edit_text(SOURCE, 'return a - b', 'return b - a') == (SOURCE.replace('a - b', 'b - a'), None)


def test_duplicates_report_line_numbers_and_start_line_picks_one():
    text = 'x = 1\nprint(x)\nx = 1\nprint(x)\n'
    with pytest.raises(PatchConflict, match=r'matches 2 times, starting at lines 1, 3'):
        edit_text(text, 'x = 1', 'x = 2')
    assert edit_text(text, 'x = 1', 'x = 2', start_line=3)[0] == 'x = 1\nprint(x)\nx = 2\nprint(x)\n'
    with pytest.raises(PatchConflict, match='start_line 2 is not one of them'):
        edit_text(text, 'x = 1', 'x = 2', start_line=2)


def test_trailing_whitespace_difference_is_tolerated_and_reported():
    text = 'def f():  \n    return 1\n'
    updated, note = edit_text(text, 'def f():\n    return 1', 'def f():\n    return 2')
    assert updated == 'def f():\n    return 2\n'
    assert note == 'matched ignoring whitespace differences, lines 1-2'


def test_uniform_indentation_shift_is_applied_to_new_text():
    text = 'class A:\n    def f(self):\n        return 1\n'
    updated, note = edit_text(text, 'def f(self):\n    return 1\n', 'def f(self):\n    value = 2\n    return value\n')
    assert updated == 'class A:\n    def f(self):\n        value = 2\n        return value\n'
    assert 'added 4 characters of indentation' in note
    updated, note = edit_text('if x:\n  y()\n', '    if x:\n      y()', '    if x:\n      z()')
    assert updated == 'if x:\n  z()\n' and 'removed 4 characters' in note


def test_crlf_files_keep_their_line_endings():
    text = 'a = 1\r\nb = 2\r\n'
    updated, _ = edit_text(text, 'a = 1\nb = 2', 'a = 1\nb = 3')
    assert updated == 'a = 1\r\nb = 3\r\n'


def test_ambiguous_whitespace_match_is_refused():
    text = 'if a:\n    go()  \nif b:\n    go()\t\n'
    with pytest.raises(PatchConflict, match='several places when whitespace is ignored'):
        edit_text(text, 'go()   ', 'stop()')


def test_no_match_shows_the_closest_block_with_line_numbers():
    with pytest.raises(PatchConflict) as error:
        edit_text(SOURCE, 'def sub(a, b):\n    return a - c', 'x')
    message = str(error.value)
    assert 'Closest block' in message and 'lines 5-6' in message and '    5| def sub(a, b):' in message
    with pytest.raises(PatchConflict, match='Nothing similar'):
        edit_text(SOURCE, 'completely unrelated text here', 'x')


def test_edit_tool_reports_the_note_and_accepts_start_line(tmp_path):
    (tmp_path / 'a.py').write_text('    pass\npass\n')
    tools = WorkspaceTools(tmp_path, mode='workspace-edit')
    result = tools.execute_tool('edit', {'path': 'a.py', 'old_text': 'pass', 'new_text': 'return', 'start_line': 2})
    assert result.status == 'success' and (tmp_path / 'a.py').read_text() == '    pass\nreturn\n'
    result = tools.execute_tool('edit', {'path': 'a.py', 'old_text': 'pass', 'new_text': 'x'})
    assert result.status == 'success'
    (tmp_path / 'b.py').write_text('def f():\n        return 1\n')
    result = tools.execute_tool('edit', {'path': 'b.py', 'old_text': 'return 1 ', 'new_text': 'return 2'})
    assert 'matched ignoring whitespace differences' in result.data
    assert (tmp_path / 'b.py').read_text() == 'def f():\n        return 2\n'
    result = tools.execute_tool('edit', {'path': 'b.py', 'old_text': 'return 3', 'new_text': 'x'})
    assert result.error_code == 'stale_patch' and 'Closest block' in result.error_message
    tools.close()
