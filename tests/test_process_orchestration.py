import json
import sys
import threading
from unittest.mock import MagicMock

import pytest

from agent import RunBudget
from runtime import Runtime


def response(name, args):
    return {'choices': [{'message': {'content': None, 'tool_calls': [
        {'id': 'process', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]},
        'finish_reason': 'tool_calls'}]}


def answer():
    return {'choices': [{'message': {'content': 'done'}, 'finish_reason': 'stop'}]}


def test_command_waits_without_model_polls_and_emits_incremental_output(tmp_path):
    model = MagicMock()
    model.create_chat_completion.side_effect = [response('run_command', {'argv': [sys.executable, '-u', '-c',
        'import time; print("one"); time.sleep(.15); print("two"); time.sleep(.15)']}), answer()]
    events = []
    with Runtime(model, tmp_path, tmp_path / 'state', mode='execute', emit=events.append) as runtime:
        assert runtime.turn('run checks').status == 'completed'
        result = json.loads(next(m['content'] for m in runtime.messages if m['role'] == 'tool'))
        assert result['status'] == 'success' and result['data']['exit_code'] == 0
        assert result['duration_seconds'] > 0
        assert model.create_chat_completion.call_count == 2
    output = ''.join(e['text'] for e in events if e['type'] == 'process_output')
    assert output == 'one\ntwo\n'
    assert len([e for e in events if e['type'] == 'process_output']) >= 2


@pytest.mark.parametrize('wait_seconds', [0, .05])
def test_wait_limit_returns_live_process_handle(tmp_path, wait_seconds):
    model = MagicMock()
    model.create_chat_completion.side_effect = [response('bash', {'command': 'sleep 30'}), answer()]
    with Runtime(model, tmp_path, tmp_path / 'state', mode='execute', process_wait_seconds=wait_seconds) as runtime:
        assert runtime.turn('start command').status == 'completed'
        result = json.loads(next(m['content'] for m in runtime.messages if m['role'] == 'tool'))
        assert result['status'] == 'running' and result['data']['running']
        state = runtime.tools.processes[result['data']['process_id']]
        assert state['proc'].poll() is None
    assert state['proc'].poll() is not None


def test_explicit_poll_also_uses_bounded_runtime_wait(tmp_path):
    model = MagicMock()
    seen = []
    def generate(**kwargs):
        if not seen:
            seen.append('launch')
            return response('run_command', {'argv': [sys.executable, '-c', 'import time; time.sleep(.2)']})
        observation = json.loads(kwargs['messages'][-1]['content'])
        seen.append(observation['status'])
        if observation['status'] == 'running':
            return response('poll_process', {'process_id': observation['data']['process_id']})
        return answer()
    model.create_chat_completion.side_effect = generate
    with Runtime(model, tmp_path, tmp_path / 'state', mode='execute', process_wait_seconds=.15) as runtime:
        assert runtime.turn('run checks').status == 'completed'
    assert seen == ['launch', 'running', 'success']


def test_cancellation_kills_process_group_and_fills_pending_results(tmp_path):
    model = MagicMock()
    call = response('run_command', {'argv': [sys.executable, '-u', '-c',
        'import subprocess, sys, time; subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"]); print("started"); time.sleep(30)']})
    call['choices'][0]['message']['tool_calls'].append({'id': 'edit', 'type': 'function', 'function': {
        'name': 'apply_patch', 'arguments': '{"path":"forbidden","old_text":"","new_text":"bad"}'}})
    model.create_chat_completion.return_value = call
    with Runtime(model, tmp_path, tmp_path / 'state', mode='execute') as runtime:
        timer = threading.Timer(.2, runtime.cancel_event.set)
        timer.start()
        try:
            assert runtime.turn('run command').status == 'cancelled'
        finally:
            timer.cancel()
        assert not (tmp_path / 'forbidden').exists()
        outcomes = [json.loads(m['content']) for m in runtime.messages if m['role'] == 'tool']
        assert len(outcomes) == 2 and outcomes[0]['status'] == 'cancelled'
        assert model.create_chat_completion.call_count == 1
        state = next(iter(runtime.tools.processes.values()))
        assert state['proc'].poll() is not None and not state['reader'].is_alive()


def test_run_deadline_stops_command_before_its_native_timeout(tmp_path):
    model = MagicMock()
    model.create_chat_completion.return_value = response('bash', {'command': 'sleep 30', 'timeout_seconds': 30})
    with Runtime(model, tmp_path, tmp_path / 'state', mode='execute', budget=RunBudget(max_seconds=.2)) as runtime:
        assert runtime.turn('run checks').status == 'budget_exhausted'
        observation = json.loads(next(m['content'] for m in runtime.messages if m['role'] == 'tool'))
        assert observation['status'] == 'timed_out' and observation['data']['timed_out']
        assert next(iter(runtime.tools.processes.values()))['proc'].poll() is not None
        assert model.create_chat_completion.call_count == 1


def test_cancellation_during_next_model_call_cleans_outstanding_process(tmp_path):
    model = MagicMock()
    with Runtime(model, tmp_path, tmp_path / 'state', mode='execute', process_wait_seconds=0) as runtime:
        def generate(**kwargs):
            if kwargs['messages'][-1]['role'] != 'tool':
                return response('bash', {'command': 'sleep 30'})
            runtime.cancel_event.set()
            return answer()
        model.create_chat_completion.side_effect = generate
        assert runtime.turn('start command').status == 'cancelled'
        assert next(iter(runtime.tools.processes.values()))['proc'].poll() is not None


def test_incremental_output_preserves_split_utf8(tmp_path):
    model = MagicMock()
    model.create_chat_completion.side_effect = [response('run_command', {'argv': [sys.executable, '-c',
        'import os,time; os.write(1,b"\\xe2"); time.sleep(.15); os.write(1,b"\\x82\\xac\\n")']}), answer()]
    events = []
    with Runtime(model, tmp_path, tmp_path / 'state', mode='execute', emit=events.append) as runtime:
        assert runtime.turn('print unicode').status == 'completed'
    assert ''.join(e['text'] for e in events if e['type'] == 'process_output') == '€\n'


@pytest.mark.parametrize('wait_seconds', [-1, 31, float('nan'), float('inf'), True])
def test_invalid_wait_control_rejected(tmp_path, wait_seconds):
    with pytest.raises(ValueError, match='process_wait_seconds'):
        Runtime(MagicMock(), tmp_path, tmp_path / 'state', process_wait_seconds=wait_seconds)
