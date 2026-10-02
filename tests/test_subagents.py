import json
import threading
from unittest.mock import MagicMock

import pytest

from runtime import Runtime


def call(name, **arguments):
    return {'choices': [{'message': {'content': None, 'tool_calls': [
        {'id': f'c_{name}', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments)}}]},
        'finish_reason': 'tool_calls'}]}


def text(value):
    return {'choices': [{'message': {'content': value}, 'finish_reason': 'stop'}]}


def scripted(spawn, child, parents=None):
    """Route model calls by transcript: coordinator vs. subagent, first vs. later step."""
    seen_tools = []
    def respond(**kwargs):
        messages = kwargs['messages']
        is_child = any(m.get('name') == 'subagent_guidance' for m in messages)
        stepped = any(m.get('role') == 'tool' for m in messages)
        if is_child:
            seen_tools.append({s['function']['name'] for s in kwargs.get('tools', [])})
            return child(messages, stepped)
        return text('coordinator done') if stepped else spawn
    model = MagicMock()
    model.create_chat_completion.side_effect = respond
    return model, seen_tools


def tool_data(runtime):
    message = next(m for m in runtime.messages if m.get('role') == 'tool')
    return json.loads(message['content'])


def user_prompt(messages):
    return next(m['content'] for m in messages if m.get('role') == 'user')


def test_subagents_run_in_parallel_and_report_in_order(tmp_path):
    barrier = threading.Barrier(2)
    def child(messages, stepped):
        if stepped:
            return text('finished ' + user_prompt(messages))
        barrier.wait(timeout=5)  # only passes if both subagents are inside the model call together
        name = user_prompt(messages)
        return call('apply_patch', path=f'{name}.txt', old_text='', new_text=name)
    spawn = call('spawn_subagents', tasks=[{'prompt': 'a'}, {'prompt': 'b', 'name': 'bee'}])
    model, _ = scripted(spawn, child)
    with Runtime(model, tmp_path, tmp_path / 'state', mode='workspace-edit', subagents=2) as runtime:
        result = runtime.turn('split it')
    data = tool_data(runtime)['data']
    assert result.status == 'completed'
    assert [r['name'] for r in data['subagents']] == ['sub-1', 'bee']
    assert [r['report'] for r in data['subagents']] == ['finished a', 'finished b']
    assert data['all_completed'] and data['changed_files'] == ['a.txt', 'b.txt']
    assert (tmp_path / 'a.txt').read_text() == 'a' and (tmp_path / 'b.txt').read_text() == 'b'
    assert sorted(result.changed_files) == ['a.txt', 'b.txt']


def test_subagents_cannot_nest_or_exceed_parent_permissions(tmp_path):
    def child(messages, stepped):
        if stepped:
            return text('blocked as expected')
        return call('apply_patch', path='x.txt', old_text='', new_text='x')
    spawn = call('spawn_subagents', tasks=[{'prompt': 'try writing', 'mode': 'read-only'}])
    model, seen = scripted(spawn, child)
    with Runtime(model, tmp_path, tmp_path / 'state', mode='workspace-edit', subagents=2) as runtime:
        runtime.turn('delegate')
    assert not (tmp_path / 'x.txt').exists()
    assert 'spawn_subagents' not in seen[0] and 'apply_patch' not in seen[0] and 'read_file' in seen[0]
    assert 'apply_patch' in {s['function']['name'] for s in model.create_chat_completion.call_args_list[0].kwargs['tools']}

    spawn = call('spawn_subagents', tasks=[{'prompt': 'escalate', 'mode': 'execute'}])
    model, seen = scripted(spawn, child)
    with Runtime(model, tmp_path, tmp_path / 'state2', mode='workspace-edit', subagents=2) as runtime:
        runtime.turn('delegate')
    assert tool_data(runtime)['error_code'] == 'permission_denied' and not seen


def test_disabled_by_default_and_with_answer_tasks(tmp_path):
    model = MagicMock()
    model.create_chat_completion.return_value = text('ok')
    with Runtime(model, tmp_path, tmp_path / 'state') as runtime:
        runtime.turn('hi')
    assert 'spawn_subagents' not in {s['function']['name'] for s in model.create_chat_completion.call_args.kwargs['tools']}
    with Runtime(model, tmp_path, tmp_path / 'state', subagents=3) as runtime:
        runtime.turn('hi')
        schema = next(s for s in model.create_chat_completion.call_args.kwargs['tools'] if s['function']['name'] == 'spawn_subagents')
        assert schema['function']['parameters']['properties']['tasks']['maxItems'] == 3
    with Runtime(model, tmp_path, tmp_path / 'state', subagents=3, task_kind='answer') as runtime:
        runtime.turn('hi')
        assert 'tools' not in model.create_chat_completion.call_args.kwargs
    with pytest.raises(ValueError):
        Runtime(model, tmp_path, tmp_path / 'state', subagents=9)


def test_subagent_checks_count_as_verification_evidence(tmp_path):
    def child(messages, stepped):
        if stepped:
            return text('ran the check')
        return call('bash', command='true', verification=True)
    spawn = call('spawn_subagents', tasks=[{'prompt': 'check'}])
    model, _ = scripted(spawn, child)
    with Runtime(model, tmp_path, tmp_path / 'state', mode='execute', subagents=1) as runtime:
        result = runtime.turn('verify')
    assert result.status == 'completed'
    assert result.verification_status == 'passed' and len(result.checks) == 1


def test_spawn_count_is_bounded_per_turn_and_reset_each_turn(tmp_path):
    from execution_context import ExecutionContext
    import time
    model = MagicMock()
    model.create_chat_completion.return_value = text('ok')
    with Runtime(model, tmp_path, tmp_path / 'state', subagents=4) as runtime:
        runtime.tool_executor = lambda name, args: None
        runtime.subagents.runs = 14
        with ExecutionContext(time.monotonic() + 5, threading.Event()).bind():
            result = runtime.subagents._handler({'tasks': [{'prompt': 'x'}] * 4})
        assert result.error_code == 'subagent_limit'
        runtime.turn('new turn')
        assert runtime.subagents.runs == 0
