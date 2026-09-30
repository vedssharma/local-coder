import pytest
from session import ContextManager, SessionStore, repository_instructions


def test_session_roundtrip_and_workspace_binding(tmp_path):
    store = SessionStore(tmp_path / 'sessions', tmp_path)
    messages = [{'role': 'user', 'content': 'fix'}, {'role': 'assistant', 'content': 'done'}]
    key = store.save(messages)
    assert store.load(key) == messages
    assert store.list() == [key]
    with pytest.raises(ValueError):
        SessionStore(tmp_path / 'sessions', tmp_path / 'other').load(key)
    with pytest.raises(ValueError):
        store.load('../escape')


def test_context_compacts_complete_turns_and_retains_current_tools():
    messages = [{'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': 'old'},
                {'role': 'assistant', 'content': 'x' * 2000}, {'role': 'user', 'content': 'current'},
                {'role': 'assistant', 'tool_calls': [{'id': 'c', 'function': {'name': 'read_file'}}]},
                {'role': 'tool', 'tool_call_id': 'c', 'content': 'observed'}]
    context = ContextManager(2500)
    context.fit(messages, [], 200)
    assert context.size(messages, []) <= 2300
    assert messages[-1]['tool_call_id'] == 'c'
    assert any(m.get('name') == 'working_memory' for m in messages)
    assert any(m.get('content') == 'current' for m in messages)


def test_oversized_active_request_is_rejected_and_output_retrievable(tmp_path):
    context = ContextManager(1000, artifact_dir=tmp_path)
    with pytest.raises(ValueError):
        context.fit([{'role': 'user', 'content': 'x' * 2000}], [], 100)
    result = context.bound_output('x' * 10000)
    assert 'output truncated' in result
    assert next(tmp_path.glob('*.txt')).read_text() == 'x' * 10000


def test_scoped_instructions(tmp_path):
    (tmp_path / 'AGENTS.md').write_text('root rules')
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / 'AGENTS.md').write_text('src rules')
    (tmp_path / 'other').mkdir()
    (tmp_path / 'other' / 'AGENTS.md').write_text('other rules')
    content = repository_instructions(tmp_path, tmp_path / 'src' / 'a.py')
    assert 'root rules' in content and 'src rules' in content and 'other rules' not in content
