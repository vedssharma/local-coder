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
