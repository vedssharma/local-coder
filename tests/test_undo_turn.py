"""Undoing every patch from the last turn."""

import pytest

from workspace_tools import WorkspaceTools


def _turn(tools, turn_id, *calls):
    tools.turn_id = turn_id
    for name, args in calls:
        assert not tools.call_tool(name, args).startswith('Error')


def test_undo_turn_reverts_every_patch_from_the_last_turn_only(tmp_path):
    (tmp_path / 'a.py').write_text('a = 1\n')
    (tmp_path / 'b.py').write_text('b = 1\n')
    tools = WorkspaceTools(tmp_path, mode='workspace-edit')
    _turn(tools, 'first', ('edit', {'path': 'a.py', 'old_text': 'a = 1', 'new_text': 'a = 2'}))
    _turn(tools, 'second',
          ('edit', {'path': 'a.py', 'old_text': 'a = 2', 'new_text': 'a = 3'}),
          ('edit', {'path': 'a.py', 'old_text': 'a = 3', 'new_text': 'a = 4'}),
          ('edit', {'path': 'b.py', 'old_text': 'b = 1', 'new_text': 'b = 2'}),
          ('write', {'path': 'new.py', 'content': 'x = 1\n'}),
          ('edit', {'path': 'new.py', 'old_text': 'x = 1', 'new_text': 'x = 2'}))
    assert tools.undo_turn() == 'Undid 5 harness edits to a.py, b.py, new.py'
    assert (tmp_path / 'a.py').read_text() == 'a = 2\n'
    assert (tmp_path / 'b.py').read_text() == 'b = 1\n'
    assert not (tmp_path / 'new.py').exists()
    assert tools.undo_turn() == 'Undid 1 harness edit to a.py'
    assert (tmp_path / 'a.py').read_text() == 'a = 1\n'
    assert tools.undo_turn() == 'No harness edits to undo.'


def test_undo_turn_changes_nothing_when_any_file_changed_since(tmp_path):
    (tmp_path / 'a.py').write_text('a = 1\n')
    (tmp_path / 'b.py').write_text('b = 1\n')
    tools = WorkspaceTools(tmp_path, mode='workspace-edit')
    _turn(tools, 'only',
          ('edit', {'path': 'a.py', 'old_text': 'a = 1', 'new_text': 'a = 2'}),
          ('edit', {'path': 'b.py', 'old_text': 'b = 1', 'new_text': 'b = 2'}))
    (tmp_path / 'a.py').write_text('a = 99\n')
    with pytest.raises(ValueError, match='preserve your changes'):
        tools.undo_turn()
    assert (tmp_path / 'b.py').read_text() == 'b = 2\n'
    assert len(list((tmp_path / '.local-coder/undo').glob('*.json'))) == 2


def test_records_without_a_turn_are_undone_one_at_a_time(tmp_path):
    (tmp_path / 'a.py').write_text('a = 1\n')
    tools = WorkspaceTools(tmp_path, mode='workspace-edit')
    tools.call_tool('edit', {'path': 'a.py', 'old_text': 'a = 1', 'new_text': 'a = 2'})
    tools.call_tool('edit', {'path': 'a.py', 'old_text': 'a = 2', 'new_text': 'a = 3'})
    assert tools.undo_turn() == 'Undid 1 harness edit to a.py'
    assert (tmp_path / 'a.py').read_text() == 'a = 2\n'


def test_runtime_turns_tag_their_patches(tmp_path):
    import json
    from unittest.mock import MagicMock
    from runtime import Runtime
    (tmp_path / 'a.py').write_text('a = 1\n')
    edit = {'id': 'c1', 'type': 'function', 'function': {
        'name': 'edit', 'arguments': json.dumps({'path': 'a.py', 'old_text': 'a = 1', 'new_text': 'a = 2'})}}
    model = MagicMock()
    model.create_chat_completion.side_effect = [
        {'choices': [{'message': {'content': None, 'tool_calls': [edit]}, 'finish_reason': 'tool_calls'}]},
        {'choices': [{'message': {'content': 'done'}, 'finish_reason': 'stop'}]}]
    with Runtime(model, tmp_path, tmp_path / 'state', mode='workspace-edit') as runtime:
        runtime.turn('change a')
        turn = runtime.tools.turn_id
        records = [json.loads(p.read_text()) for p in (tmp_path / '.local-coder/undo').glob('*.json')]
        assert turn and [r['turn'] for r in records] == [turn]
        assert runtime.tools.undo_turn() == 'Undid 1 harness edit to a.py'
