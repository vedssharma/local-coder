"""Bounded independent reads with ordered results and mutation barriers."""
from concurrent.futures import ThreadPoolExecutor
import json
import time

from jsonschema import ValidationError
from execution_context import ExecutionCancelled, DeadlineExceeded
from tool_recovery import execute_with_recovery
from tool_result import ToolResult


class ToolScheduler:
    def __init__(self, registry, registered, executor, context, emit, workers=4, checkpoint=None):
        if type(workers) is not int or not 1 <= workers <= 8:
            raise ValueError('tool_workers must be between 1 and 8')
        self.registry, self.registered, self.executor = registry, registered, executor
        self.context, self.emit, self.workers = context, emit, workers
        self.checkpoint = checkpoint or (lambda *a: None)
        self.stopped = False
        self.cache = {}
        self.elapsed = 0.0
        self.executions = 0

    def prepare(self, call):
        name, raw = call['function']['name'], call['function'].get('arguments')
        try:
            if name not in self.registered:
                raise ValueError(f'Unknown or unavailable tool: {name}')
            args = json.loads(raw) if isinstance(raw, str) else raw
            self.registry.validate(name, args)
            return name, args, None
        except (ValueError, TypeError, ValidationError) as exc:
            return name, raw, ToolResult.error('invalid_arguments', exc)

    def execute(self, item, call):
        name, args, invalid = item
        self.emit({'type': 'tool_started', 'name': name, 'call_id': call['id']})
        if invalid:
            return invalid
        if self.stopped:
            return ToolResult.error('run_stopped', 'Run stopped; tool was not executed')
        try:
            with self.context.bind():
                self.checkpoint('executing',call,args,None)
                result = execute_with_recovery(self.registry.get(name), lambda: self.executor(name, args),
                                             self.context, self.emit)
                self.checkpoint('completed',call,args,result)
                return result
        except (ExecutionCancelled, KeyboardInterrupt):
            return ToolResult(status='cancelled', error_code='interrupted')
        except DeadlineExceeded:
            return ToolResult(status='timed_out', error_code='run_deadline')
        except Exception as exc:
            return ToolResult.error('execution_error', exc)

    def run(self, calls):
        items = [self.prepare(call) for call in calls]
        index = 0
        while index < len(calls):
            def parallel(i):
                spec = self.registry.get(items[i][0])
                return not items[i][2] and spec and spec.concurrency == 'parallel' and spec.side_effects in ('none', 'network')
            end = index + 1
            if parallel(index):
                while end < len(calls) and parallel(end):
                    end += 1
            group = range(index, end)
            pending, results, keys = {}, {}, {}
            started = time.monotonic()
            # The pool is drained before crossing a serial/mutation barrier.
            with ThreadPoolExecutor(max_workers=min(self.workers, end - index)) as pool:
                for i in group:
                    name, args, invalid = items[i]
                    spec = self.registry.get(name)
                    key = json.dumps([name, args], sort_keys=True)
                    keys[i] = key
                    if not invalid and spec and spec.cacheable and key in self.cache:
                        results[i] = self.cache[key]
                        self.emit({'type': 'tool_reused', 'name': name, 'call_id': calls[i]['id']})
                    elif not invalid and spec and spec.cacheable and key in pending:
                        results[i] = pending[key]
                        self.emit({'type': 'tool_reused', 'name': name, 'call_id': calls[i]['id']})
                    else:
                        if spec and spec.side_effects not in ('none', 'network'):
                            self.cache.clear()
                        future = pool.submit(self.execute, items[i], calls[i])
                        results[i] = future
                        if spec and spec.cacheable:
                            pending[key] = future
                        self.executions += 1
                for i in group:
                    if not isinstance(results[i], ToolResult):
                        try:
                            results[i] = results[i].result()
                        except KeyboardInterrupt:
                            self.context.cancel_event.set()
                            self.stopped = True
                            results[i] = ToolResult(status='cancelled', error_code='interrupted')
                    spec = self.registry.get(items[i][0])
                    if spec and spec.cacheable and results[i].status == 'success':
                        self.cache[keys[i]] = results[i]
            self.elapsed += time.monotonic() - started
            for i in group:
                yield calls[i], items[i][1], results[i]
            index = end
