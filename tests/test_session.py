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


# ---------------------------------------------------------------------------
# Storage validation and compaction edge cases
# ---------------------------------------------------------------------------

import json


def test_saving_without_a_checkpoint_keeps_the_stored_one(tmp_path):
    store = SessionStore(tmp_path / 'sessions', tmp_path)
    key = store.save([], checkpoint={'calls': {'a': {'state': 'completed'}}})
    store.save([{'role': 'user', 'content': 'next'}], key)
    assert store.load_state(key) == ([{'role': 'user', 'content': 'next'}], {'calls': {'a': {'state': 'completed'}}})


@pytest.mark.parametrize('data, message', [
    ({'messages': [{'role': 'wizard'}]}, 'Invalid session transcript'),
    ({'messages': 'not a list'}, 'Invalid session transcript'),
    ({'messages': [], 'checkpoint': {'calls': []}}, 'Invalid session checkpoint'),
    ({'messages': [], 'checkpoint': []}, 'Invalid session checkpoint'),
])
def test_corrupt_sessions_are_rejected(tmp_path, data, message):
    store = SessionStore(tmp_path / 'sessions', tmp_path)
    key = store.save([])
    path = tmp_path / 'sessions' / (key + '.json')
    path.write_text(json.dumps({'version': 2, 'workspace': store.workspace, **data}))
    with pytest.raises(ValueError, match=message):
        store.load_state(key)


def test_summaries_skip_unreadable_sessions(tmp_path):
    store = SessionStore(tmp_path / 'sessions', tmp_path)
    good = store.save([{'role': 'user', 'content': 'hello'}])
    (tmp_path / 'sessions' / ('a' * 32 + '.json')).write_text('{broken')
    assert [s['id'] for s in store.summaries()] == [good]


def test_context_manager_validates_its_token_counter():
    with pytest.raises(ValueError, match='cache_entries'):
        ContextManager(cache_entries=-1)
    context = ContextManager()
    with pytest.raises(ValueError, match='must be callable'):
        context.count_tokens = 42
    context.count_tokens = lambda text: -1
    with pytest.raises(ValueError, match='invalid count'):
        context.count_tokens('text')


def test_digest_keeps_the_tail_of_a_single_oversized_note():
    context = ContextManager(1000)
    context.memory = ['x' * 5000 + 'END']
    digest = context._digest()
    assert len(digest) == context.summary_chars and digest.endswith('END')


def test_summaries_describe_tool_calls_and_observations():
    notes = ContextManager._summarize([
        {'role': 'assistant', 'tool_calls': [{'id': 'c', 'function': {'name': 'read', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'c', 'content': 'file body'}])
    assert notes == ['Tool: {"name": "read", "arguments": "{}"}', 'Observation: file body']


def test_compaction_without_a_user_request_does_nothing():
    assert ContextManager()._compact_active([{'role': 'system', 'content': 'rules'}]) is False


def _call(key, name='read'):
    return {'role': 'assistant', 'content': None,
            'tool_calls': [{'id': key, 'type': 'function', 'function': {'name': name, 'arguments': '{}'}}]}


def test_newest_exchange_skips_non_envelope_tool_output():
    messages = [{'role': 'user', 'content': 'task'}, _call('a'),
                {'role': 'tool', 'tool_call_id': 'a', 'content': 'plain ' * 400}]
    assert ContextManager()._compact_active(messages) is False
    messages[-1]['content'] = json.dumps(['a', 'list'])
    assert ContextManager()._compact_active(messages) is False


def test_older_exchanges_with_plain_outputs_are_summarized(tmp_path):
    context = ContextManager(artifact_dir=tmp_path / 'artifacts')
    messages = [{'role': 'user', 'content': 'task'},
                _call('a'), {'role': 'tool', 'tool_call_id': 'a', 'content': 'not json'},
                _call('b'), {'role': 'tool', 'tool_call_id': 'b', 'content': '[1, 2]'},
                _call('c'), {'role': 'tool', 'tool_call_id': 'c', 'content': '{"status": "success", "data": "ok"}'}]
    assert context._compact_active(messages) is True
    outcomes = context.active_notes[0]['outcomes']
    assert outcomes[0]['data'] == 'not json' and outcomes[0]['status'] is None
    assert context.active_notes[0]['artifact']
    assert context._compact_active(messages) is True
    assert context.active_notes[1]['outcomes'][0]['data'] == '[1, 2]'


@pytest.mark.parametrize('stored, restored', [
    ('Compacted tool evidence:\n[{"calls": [{"id": "z", "name": "read", "arguments": ""}], "outcomes": []}]', 1),
    ('Compacted tool evidence:\nnot json', 0),
    ('no newline at all', 0),
])
def test_resumed_tool_memory_is_restored_when_readable(stored, restored):
    context = ContextManager()
    messages = [{'role': 'user', 'content': 'task'},
                {'role': 'user', 'name': 'active_tool_memory', 'content': stored},
                _call('a'), {'role': 'tool', 'tool_call_id': 'a', 'content': '{}'},
                _call('b'), {'role': 'tool', 'tool_call_id': 'b', 'content': '{}'}]
    assert context._compact_active(messages) is True
    assert len(context.active_notes) == restored + 1


def test_instructions_target_must_be_inside_the_workspace(tmp_path):
    (tmp_path / 'work').mkdir()
    with pytest.raises(ValueError, match='outside workspace'):
        repository_instructions(tmp_path / 'work', tmp_path)
