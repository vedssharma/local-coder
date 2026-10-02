"""Parallel subagents: isolated tool loops that split a large task into independent slices."""
from concurrent.futures import ThreadPoolExecutor
import threading
import time

from agent import run_agent, RunBudget
from execution_context import CURRENT_CONTEXT
from prompt_builder import build_messages
from session import ContextManager
from tool_registry import ToolSpec, MODES
from tool_result import ToolResult
from workspace_tools import schema

TOOL_NAME = 'spawn_subagents'
DEFAULT_STEPS = 10
MAX_REPORT_CHARS = 3000
MAX_RUNS_PER_TURN = 16
_LOCAL = threading.local()

GUIDANCE = (
    'You are a subagent handling one slice of a larger task that a coordinator split up. '
    'You cannot see the coordinator\'s conversation; rely only on this request. Other subagents work in '
    'parallel on other slices: touch only the files your slice names and do not undo others\' edits. '
    'You cannot start further subagents. Finish with a concise report: what you did, files changed, '
    'checks run with their observed results, and anything unresolved. Do not claim unobserved results.'
)


def in_subagent():
    """True on a subagent worker thread, so UI streaming from other threads can be suppressed."""
    return getattr(_LOCAL, 'active', False)


class SubagentManager:
    def __init__(self, runtime, max_parallel):
        if type(max_parallel) is not int or not 0 <= max_parallel <= 8:
            raise ValueError('subagents must be between 0 and 8')
        self.runtime, self.max_parallel = runtime, max_parallel
        self.runs = 0
        self._lock = threading.Lock()
        if max_parallel:
            runtime.tools.registry.register(self._spec())

    def reset(self):
        self.runs = 0

    def _spec(self):
        task = {'type': 'object', 'additionalProperties': False, 'required': ['prompt'], 'properties': {
            'prompt': {'type': 'string', 'minLength': 1}, 'name': {'type': 'string', 'minLength': 1, 'maxLength': 40},
            'mode': {'type': 'string', 'enum': list(MODES)}}}
        return ToolSpec(
            schema(TOOL_NAME,
                   'Split large work into independent slices run in parallel by isolated subagents. They cannot see this '
                   'chat: each prompt must be self-contained and name its files; write-capable slices need disjoint files. '
                   'Returns each final report; verify the combined result. mode can only narrow permissions.',
                   {'tasks': {'type': 'array', 'minItems': 1, 'maxItems': self.max_parallel, 'items': task}}, ['tasks']),
            self._handler, side_effects='unknown', concurrency='serial', task_kinds=('inspect', 'code', 'all'))

    def _handler(self, args):
        runtime, context = self.runtime, CURRENT_CONTEXT.get()
        execute = runtime.tool_executor
        if context is None or execute is None:
            return ToolResult.error('invalid_request', 'Subagents can only run during an active turn')
        tasks = args['tasks']
        parent = MODES.index(runtime.tools.mode)
        for task in tasks:
            if MODES.index(task.get('mode', runtime.tools.mode)) > parent:
                raise PermissionError('A subagent cannot exceed the session permission mode')
        with self._lock:
            if self.runs + len(tasks) > MAX_RUNS_PER_TURN:
                return ToolResult.error('subagent_limit', f'At most {MAX_RUNS_PER_TURN} subagents may run per turn')
            first = self.runs
            self.runs += len(tasks)
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=min(self.max_parallel, len(tasks))) as pool:
            futures = [pool.submit(self._run, first + i + 1, task, context, execute) for i, task in enumerate(tasks)]
            reports = [f.result() for f in futures]
        data = {'subagents': reports, 'all_completed': all(r['status'] == 'completed' for r in reports),
                'changed_files': sorted({p for r in reports for p in r['changed_files']})}
        return ToolResult(data=data, changed_files=data['changed_files'], duration_seconds=time.monotonic() - started)

    def _run(self, number, task, context, execute):
        runtime = self.runtime
        ident = f'sub-{number}'
        name = task.get('name') or ident
        mode = task.get('mode', runtime.tools.mode)
        schemas = [s for s in runtime.tools.registry.schemas(mode, runtime.task_kind if runtime.task_kind != 'auto' else
                   ('inspect' if runtime.tools.mode == 'read-only' else 'code')) if s['function']['name'] != TOOL_NAME]
        allowed = {s['function']['name'] for s in schemas}
        changed = set()

        def guarded(tool, arguments):
            if tool not in allowed:
                return ToolResult.error('permission_denied', f'Tool {tool} is unavailable to this subagent')
            return execute(tool, arguments)

        def checkpoint(stage, call, arguments, result):
            if stage == 'completed':
                changed.update(result.changed_files)
                runtime.record_subagent_evidence(f'{ident}:{call["id"]}', call['function']['name'], arguments, result)

        def emit(event):
            if event['type'] == 'tool_started':
                runtime.emit({'type': 'subagent_tool', 'subagent': name, 'id': ident, 'name': event['name']})

        runtime.emit({'type': 'subagent_started', 'id': ident, 'name': name, 'mode': mode})
        messages = build_messages(task['prompt'], {}, root=runtime.tools.root)
        messages.insert(1, {'role': 'system', 'name': 'subagent_guidance', 'content': GUIDANCE})
        remaining = max(1.0, context.deadline - time.monotonic())
        budget = RunBudget(DEFAULT_STEPS, remaining,
                           runtime.budget.max_generated_tokens)
        child_context = ContextManager(runtime.context.window, count_tokens=runtime.context.count_tokens,
                                       artifact_dir=runtime.context.artifact_dir)
        _LOCAL.active = True
        try:
            result = run_agent(runtime.model, messages, runtime.max_tokens, runtime.tools, budget=budget,
                               cancel_event=context.cancel_event, emit=emit, context_manager=child_context,
                               tool_schemas=schemas, tool_executor=guarded, execution_context=context,
                               tool_workers=runtime.tool_workers, checkpoint=checkpoint)
            status, text, steps, reason = result.status, result.text, result.steps, result.reason
        except Exception as exc:
            status, text, steps, reason = 'blocked', f'Subagent failed: {exc}', 0, 'subagent_error'
        finally:
            _LOCAL.active = False
        if len(text) > MAX_REPORT_CHARS:
            text = text[:MAX_REPORT_CHARS] + '\n[report truncated]'
        runtime.emit({'type': 'subagent_finished', 'id': ident, 'name': name, 'status': status, 'steps': steps})
        return {'id': ident, 'name': name, 'mode': mode, 'status': status, 'reason': reason, 'steps': steps,
                'changed_files': sorted(changed), 'report': text}
