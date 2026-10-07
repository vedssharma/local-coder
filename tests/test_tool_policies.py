"""Tool specs, the scheduler's failure handling, and verification evidence."""
import threading
import time
from types import SimpleNamespace

import pytest

from execution_context import DeadlineExceeded, ExecutionCancelled, ExecutionContext
from tool_registry import ToolRegistry, ToolSpec
from tool_result import ToolResult
from tool_scheduler import ToolScheduler
from verification import VerificationLedger, is_check


def _schema(name='probe', **properties):
    return {'type': 'function', 'function': {'name': name, 'parameters': {'type': 'object', 'properties': properties}}}


# ---------------------------------------------------------------------------
# ToolSpec and ToolRegistry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('options, message', [
    ({'minimum_mode': 'root'}, 'Unknown tool permission mode'),
    ({'side_effects': 'magic'}, 'Unknown side-effect policy'),
    ({'concurrency': 'sometimes'}, 'Unknown concurrency policy'),
    ({'side_effects': 'process', 'cacheable': True}, 'Only tools without side effects'),
    ({'side_effects': 'none', 'compact_observation': True}, 'requires a cacheable tool'),
    ({'side_effects': 'filesystem', 'retry_safe': True}, 'Only explicitly safe reads'),
    ({'default_timeout': 10}, 'Invalid timeout policy'),
    ({'default_timeout': 500, 'max_timeout': 300}, 'Invalid timeout policy'),
])
def test_inconsistent_tool_policies_are_rejected(options, message):
    with pytest.raises(ValueError, match=message):
        ToolSpec(_schema(), handler=None, **options)


def test_registry_executes_with_permission_and_validation():
    registry = ToolRegistry()
    registry.register(ToolSpec({'type': 'function', 'function': {'name': 'echo', 'parameters': {
        'type': 'object', 'properties': {'text': {'type': 'string'}}, 'required': ['text']}}},
        handler=lambda args: args['text'], minimum_mode='workspace-edit'))
    assert registry.execute('echo', {'text': 'hi'}, 'execute') == 'hi'
    with pytest.raises(PermissionError):
        registry.execute('echo', {'text': 'hi'}, 'read-only')
    from jsonschema import ValidationError
    with pytest.raises(ValidationError):
        registry.execute('echo', {}, 'workspace-edit')
    with pytest.raises(ValueError, match='Duplicate tool'):
        registry.register(ToolSpec(_schema('echo'), handler=None))


# ---------------------------------------------------------------------------
# ToolScheduler
# ---------------------------------------------------------------------------

def _scheduler(executor, emit=None):
    registry = ToolRegistry()
    registry.register(ToolSpec(_schema(), handler=None, side_effects='none', concurrency='parallel'))
    context = ExecutionContext(time.monotonic() + 30, threading.Event())
    return ToolScheduler(registry, {'probe'}, executor, context, emit or (lambda event: None))


def _call(key='a'):
    return {'id': key, 'function': {'name': 'probe', 'arguments': '{}'}}


def test_scheduler_rejects_invalid_worker_counts():
    with pytest.raises(ValueError, match='tool_workers'):
        ToolScheduler(ToolRegistry(), set(), None, None, None, workers=9)


@pytest.mark.parametrize('error, status, code', [
    (ExecutionCancelled('stop'), 'cancelled', 'interrupted'),
    (KeyboardInterrupt(), 'cancelled', 'interrupted'),
    (DeadlineExceeded('late'), 'timed_out', 'run_deadline'),
    (RuntimeError('tool crashed'), 'error', 'execution_error'),
])
def test_scheduler_turns_executor_failures_into_results(error, status, code):
    def executor(name, args):
        raise error
    [(call, args, result)] = list(_scheduler(executor).run([_call()]))
    assert (result.status, result.error_code) == (status, code)


def test_an_interrupt_outside_the_tool_cancels_the_run():
    def emit(event):
        if event['type'] == 'tool_started':
            raise KeyboardInterrupt
    scheduler = _scheduler(lambda name, args: ToolResult(data='ok'), emit)
    [(call, args, result)] = list(scheduler.run([_call()]))
    assert (result.status, result.error_code) == ('cancelled', 'interrupted')
    assert scheduler.stopped and scheduler.context.cancel_event.is_set()


# ---------------------------------------------------------------------------
# Verification evidence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name, arguments, expected', [
    ('read', {'verification': True}, True),
    ('read', {'path': 'a'}, False),
    ('bash', {'command': 'echo "unterminated'}, False),
    ('bash', {'command': 'CI=1 PYTHONPATH=src'}, False),
    ('bash', {'command': 'CI=1 pytest -q'}, True),
    ('bash', {'command': '.venv/bin/pytest-3'}, True),
    ('bash', {'command': 'python3 -m pytest'}, True),
    ('bash', {'command': 'python -m unittest discover'}, True),
    ('bash', {'command': 'python -m http.server'}, False),
    ('bash', {'command': 'python -m'}, False),
    ('bash', {'command': 'npm run test'}, True),
    ('bash', {'command': 'cargo test'}, True),
    ('bash', {'command': 'cargo build'}, False),
    ('bash', {'command': 'make test'}, False),
])
def test_is_check(name, arguments, expected):
    assert is_check(name, arguments) is expected


def _process(key, running, exit_code=None, output=''):
    return ToolResult(status='running' if running else ('success' if exit_code == 0 else 'failed'),
                      data={'process_id': key, 'running': running, 'exit_code': exit_code, 'output': output})


def test_each_call_is_observed_once():
    ledger = VerificationLedger()
    ledger.observe('c1', 'bash', {'command': 'pytest'}, _process('p1', False, 1))
    ledger.observe('c1', 'bash', {'command': 'pytest'}, _process('p1', False, 0))
    assert ledger.state['commands']['p1']['exit_code'] == 1


def test_running_checks_are_polled_or_marked_interrupted():
    ledger = VerificationLedger()
    ledger.observe('c1', 'bash', {'command': 'pytest'}, _process('p1', True))
    ledger.observe('c2', 'bash', {'command': 'pytest -x'}, _process('p2', True))
    tools = SimpleNamespace(processes={'p1': object()},
                            _poll=lambda key: {'running': False, 'exit_code': 0, 'output': 'passed', 'process_id': key})
    summary = VerificationLedger(ledger.state).summarize(tools)
    commands = ledger.state['commands']
    assert (commands['p1']['status'], commands['p1']['exit_code'], commands['p1']['output_preview']) == ('success', 0, 'passed')
    assert (commands['p2']['status'], commands['p2']['running']) == ('interrupted', False)
    assert summary['verification_status'] == 'failed'


def test_a_check_still_running_is_in_progress():
    ledger = VerificationLedger()
    ledger.observe('c1', 'bash', {'command': 'pytest'}, _process('p1', True))
    tools = SimpleNamespace(processes={'p1': object()}, _poll=lambda key: {'running': True})
    summary = ledger.summarize(tools)
    assert summary['verification_status'] == 'in_progress'
    assert [r['process_id'] for r in summary['outstanding_processes']] == ['p1']
