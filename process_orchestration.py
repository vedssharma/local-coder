"""Wait for command observations without spending a model call on each poll."""
import math
import time
import codecs

from tool_result import ToolResult, invoke_tool
from execution_context import ExecutionContext, ExecutionCancelled, DeadlineExceeded


def validate_wait_seconds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 30:
        raise ValueError('process_wait_seconds must be between 0 and 30')
    return float(value)


class ProcessOrchestrator:
    def __init__(self, tools, wait_seconds, deadline, cancel_event, emit):
        self.tools = tools
        self.wait_seconds = validate_wait_seconds(wait_seconds)
        self.deadline = deadline
        self.cancel_event = cancel_event
        self.emit = emit
        self.context = ExecutionContext(deadline, cancel_event)
        self.output_positions = {}
        self.decoders = {}

    def output_update(self, key):
        state = self.tools.processes[key]
        with state['lock']:
            total = state['output_bytes']
            count = total - self.output_positions.get(key, 0)
            chunk = state['output'][-count:] if count else b''
            lost = count > len(state['output'])
        if count:
            self.output_positions[key] = total
        decoder = self.decoders.setdefault(key, codecs.getincrementaldecoder('utf-8')(errors='replace'))
        if lost:
            decoder.reset()
        text = decoder.decode(chunk, final=state['proc'].poll() is not None and not state['reader'].is_alive())
        if text:
            self.emit({'type': 'process_output', 'process_id': key,
                       'text': text, 'output_truncated': lost})

    def __call__(self, name, arguments):
        started = time.monotonic()
        if self.cancel_event.is_set():
            return ToolResult(status='cancelled', error_code='interrupted')
        if started >= self.deadline:
            return ToolResult(status='timed_out', error_code='run_deadline')
        with self.context.bind():
            result = invoke_tool(self.tools, name, arguments)
        spec = self.tools.registry.get(name)
        if not spec or not spec.wait_for_process or not isinstance(result.data, dict):
            return result
        key = result.data.get('process_id')
        if key not in self.tools.processes:
            return result
        self.output_update(key)
        wait_until = min(time.monotonic() + self.wait_seconds, self.deadline)
        while result.status == 'running':
            state = self.tools.processes[key]
            if self.cancel_event.is_set() or time.monotonic() >= self.deadline:
                if self.cancel_event.is_set():
                    state['cancelled'] = True
                else:
                    state['timed_out'] = True
                self.tools._kill(state)
                result = ToolResult.process(self.tools._poll(key))
                self.output_update(key)
                break
            remaining = wait_until - time.monotonic()
            if remaining <= 0:
                break
            self.cancel_event.wait(min(0.05, remaining))
            result = ToolResult.process(self.tools._poll(key))
            self.output_update(key)
        result.duration_seconds = time.monotonic() - started
        return result
