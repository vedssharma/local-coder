from unittest.mock import MagicMock
from runtime import Runtime


def test_resume_keeps_observations_and_does_not_restore_permissions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = MagicMock()
    model.create_chat_completion.return_value = {'choices': [{'message': {'content': 'done'}, 'finish_reason': 'stop'}]}
    events = []
    with Runtime(model, tmp_path, tmp_path / 'state', mode='execute', emit=events.append) as runtime:
        result = runtime.turn('task')
        key = runtime.session_id
        assert result.status == 'completed'
    with Runtime(model, tmp_path, tmp_path / 'state') as runtime:
        runtime.resume(key)
        assert runtime.tools.mode == 'read-only'
        assert runtime.messages[-1]['content'] == 'done'
    assert events[-1]['type'] == 'turn_result'


def test_minimal_tools_and_unchanged_file_context(tmp_path):
    model = MagicMock()
    model.create_chat_completion.return_value = {'choices': [{'message': {'content': 'done'}, 'finish_reason': 'stop'}]}
    with Runtime(model, tmp_path, tmp_path / 'state') as runtime:
        runtime.turn('explain', {'a.py': 'unique file content'})
        runtime.turn('explain again', {'a.py': 'unique file content'})
        assert sum('unique file content' in m.get('content', '') for m in runtime.messages) == 1
        names = {s['function']['name'] for s in model.create_chat_completion.call_args.kwargs['tools']}
        assert names == {'read', 'list', 'search', 'web_search', 'web_fetch'}
        runtime.turn('changed', {'a.py': 'updated file content'})
        assert 'updated file content' in runtime.messages[-2]['content']
    with Runtime(model, tmp_path, tmp_path / 'state', task_kind='answer') as runtime:
        runtime.turn('hello')
        assert 'tools' not in model.create_chat_completion.call_args.kwargs
