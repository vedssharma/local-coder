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


def test_default_max_tokens_follow_the_context_window(tmp_path):
    model = MagicMock()
    model.create_chat_completion.return_value = {'choices': [{'message': {'content': 'done'}, 'finish_reason': 'stop'}]}
    with Runtime(model, tmp_path, tmp_path / 'state', context_window=16384) as runtime:
        runtime.turn('explain')
        assert model.create_chat_completion.call_args.kwargs['max_tokens'] == 4096
        runtime.turn('explain', max_tokens=700)
        assert model.create_chat_completion.call_args.kwargs['max_tokens'] == 700


# ---------------------------------------------------------------------------
# Construction, housekeeping, traces and model adapters
# ---------------------------------------------------------------------------

import json
import pytest
from model_backend import ModelAdapter


def _answer():
    return {'choices': [{'message': {'content': 'done'}, 'finish_reason': 'stop'}]}


def test_runtime_validates_workers_and_max_tokens(tmp_path):
    with pytest.raises(ValueError, match='tool_workers'):
        Runtime(MagicMock(), tmp_path, tmp_path / 'state', tool_workers=0)
    model = MagicMock()
    model.create_chat_completion.return_value = _answer()
    with Runtime(model, tmp_path, tmp_path / 'state') as runtime:
        with pytest.raises(ValueError, match='max_tokens must be positive'):
            runtime.turn('hi', max_tokens=0)


def test_housekeeping_failures_never_block_a_runtime(tmp_path):
    with Runtime(MagicMock(), tmp_path, tmp_path / 'state', retention={'nonsense': 1}) as runtime:
        assert runtime.session_id is None


def test_acknowledging_without_a_session_is_a_no_op(tmp_path):
    with Runtime(MagicMock(), tmp_path, tmp_path / 'state') as runtime:
        runtime.acknowledge_interrupted()
        assert runtime.checkpoint == {}


def test_new_session_resets_transcript_and_memory(tmp_path):
    model = MagicMock()
    model.create_chat_completion.return_value = _answer()
    with Runtime(model, tmp_path, tmp_path / 'state') as runtime:
        runtime.turn('first')
        old = runtime.session_id
        runtime.context.memory = ['note']
        runtime.new()
        assert (runtime.session_id, runtime.messages, runtime.context.memory, runtime.context.task) == (None, [], [], None)
        runtime.turn('second')
        assert runtime.session_id != old


def test_model_adapters_receive_events_and_lose_the_context_after_a_turn(tmp_path):
    class Adapter(ModelAdapter):
        def _complete(self, **kwargs):
            self.emit({'type': 'assistant_delta', 'text': 'done'})
            assert self.execution_context is not None
            return _answer()
    model = Adapter({})
    events = []
    with Runtime(model, tmp_path, tmp_path / 'state', emit=events.append, trace=True) as runtime:
        runtime.turn('hi')
        assert model.cancel_event is runtime.cancel_event
        assert model.execution_context is None
        trace = tmp_path / '.local-coder' / 'traces' / (runtime.session_id + '.jsonl')
    assert 'assistant_delta' in [e['type'] for e in events]
    assert {'type': 'assistant_delta', 'text': 'done'} in [json.loads(line) for line in trace.read_text().splitlines()]


def test_the_tool_executor_refuses_work_once_the_turn_is_cancelled(tmp_path):
    (tmp_path / 'a.txt').write_text('x')
    model = MagicMock()
    seen = []
    with Runtime(model, tmp_path, tmp_path / 'state') as runtime:
        def respond(**kwargs):
            runtime.cancel_event.set()
            seen.append(runtime.tool_executor('read', {'path': 'a.txt'}))
            return _answer()
        model.create_chat_completion.side_effect = respond
        result = runtime.turn('read it')
    assert result.status == 'cancelled'
    assert (seen[0].status, seen[0].error_code) == ('cancelled', 'interrupted')


def test_failed_side_effects_require_inspection(tmp_path):
    from tool_result import ToolResult
    with Runtime(MagicMock(), tmp_path, tmp_path / 'state', mode='workspace-edit') as runtime:
        runtime.session_id = runtime.store.save([])
        call = {'id': 'w', 'function': {'name': 'write', 'arguments': '{}'}}
        runtime.checkpoint_call('completed', call, {'path': 'a.txt'}, ToolResult.error('io_error', 'disk full'))
        assert runtime.interrupted_operations['w']['state'] == 'interrupted'
        runtime.acknowledge_interrupted()
        assert runtime.interrupted_operations == {}
