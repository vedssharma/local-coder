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


def test_compaction_preserves_original_task_across_restart():
    messages = [{'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': 'Fix the payment bug'},
                {'role': 'assistant', 'content': 'x' * 2000}, {'role': 'user', 'content': 'continue'}]
    context = ContextManager(2200)
    context.fit(messages, [], 100)
    assert any('Original task: Fix the payment bug' in m.get('content', '') for m in messages)
    messages += [{'role': 'assistant', 'content': 'y' * 2000}, {'role': 'user', 'content': 'continue again'}]
    restored = ContextManager(2200)
    restored.fit(messages, [], 100)
    assert any('Original task: Fix the payment bug' in m.get('content', '') for m in messages)


def test_output_truncation_handles_unwritable_artifact_location(tmp_path):
    location = tmp_path / 'file-not-directory'
    location.write_text('existing user file')
    context = ContextManager(1000, artifact_dir=location)
    output = context.bound_output('x' * 10000)
    assert 'Could not retain full output' in output
    assert location.read_text() == 'existing user file'


def test_token_cache_reuses_unchanged_fragments_and_invalidates_on_mutation():
    calls = []
    def count(text):
        calls.append(text)
        return len(text)
    context = ContextManager(10000, count_tokens=count, cache_entries=4)
    messages = [{'role': 'system', 'content': 'instructions'}, {'role': 'user', 'content': 'question'}]
    schemas = [{'name': 'tool'}]
    first = context.size(messages, schemas)
    assert len(calls) == 3
    assert context.size(messages, schemas) == first and len(calls) == 3
    messages.append({'role': 'assistant', 'content': 'answer'})
    context.size(messages, schemas)
    assert len(calls) == 4
    messages[1]['content'] = 'changed'
    context.size(messages, schemas)
    assert len(calls) > 4
    context.count_tokens = lambda text: 2 * len(text)
    assert context.size(messages, schemas) > first
    assert context.cache_misses > 0
    assert len(context._counts) <= 4


def test_cached_and_uncached_context_budgets_agree():
    cached = ContextManager(8192)
    uncached = ContextManager(8192, cache_entries=0)
    messages = [{'role': 'user', 'content': 'hello'}]
    for index in range(20):
        messages.append({'role': 'assistant', 'content': str(index)})
        assert cached.size(messages, []) == uncached.size(messages, [])
