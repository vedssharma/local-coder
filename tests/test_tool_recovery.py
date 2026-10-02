import json
import threading
import time
from unittest.mock import MagicMock

from agent import run_agent
from execution_context import ExecutionContext
from tool_recovery import execute_with_recovery, ProgressTracker
from tool_registry import ToolSpec
from tool_result import ToolResult
from workspace_tools import WorkspaceTools, schema


def test_transient_read_retries_but_commands_do_not(tmp_path):
    tools = WorkspaceTools(tmp_path, mode='execute')
    execute = MagicMock(side_effect=[ToolResult.error('network_error', 'temporary', True), ToolResult(data='docs')])
    events = []
    result = execute_with_recovery(tools.registry.get('web_fetch'), execute,
        ExecutionContext(time.monotonic() + 2, threading.Event()), events.append)
    assert result.status == 'success' and result.attempts == 2
    assert events[0]['type'] == 'tool_retry'
    for name in ('bash', 'edit', 'write'):
        execute = MagicMock(return_value=ToolResult.error('io_error', 'temporary', True))
        result = execute_with_recovery(tools.registry.get(name), execute,
            ExecutionContext(time.monotonic() + 2, threading.Event()), events.append)
        assert execute.call_count == 1 and result.is_error


def test_retry_attempts_are_bounded_and_permission_errors_are_not_retried(tmp_path):
    tools = WorkspaceTools(tmp_path)
    for result, attempts in [(ToolResult.error('network_error', 'temporary', True), 3),
                             (ToolResult.error('permission_denied', 'denied'), 1)]:
        execute = MagicMock(return_value=result)
        outcome = execute_with_recovery(tools.registry.get('web_fetch'), execute,
            ExecutionContext(time.monotonic() + 2, threading.Event()), lambda e: None)
        assert execute.call_count == attempts and outcome.attempts == attempts


def test_semantic_cycles_stop_but_advancing_polls_and_changed_reads_continue():
    tracker = ProgressTracker()
    observed = [tracker.observe(name, {}, ToolResult(data='unchanged')) for name in ['read', 'search'] * 3]
    assert observed[-1]
    tracker = ProgressTracker()
    for i in range(20):
        assert not tracker.observe('poll', {'process_id': 'same'}, ToolResult(status='running', data={'output': str(i)}))
        assert not tracker.observe('read', {}, ToolResult(data=f'new content {i}'))


def test_native_stale_patch_has_recoverable_error_code(tmp_path):
    (tmp_path / 'a').write_text('current')
    result = WorkspaceTools(tmp_path, mode='workspace-edit').execute_tool('edit', {
        'path': 'a', 'old_text': 'stale', 'new_text': 'new'})
    assert result.error_code == 'stale_patch' and not result.retryable
    assert (tmp_path / 'a').read_text() == 'current'
