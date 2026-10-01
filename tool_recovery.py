"""Bounded safe retries and semantic no-progress detection."""
import json
import time


def execute_with_recovery(spec, execute, context, emit, max_retries=2):
    started = time.monotonic()
    for attempt in range(max_retries + 1):
        context.check()
        result = execute()
        result.attempts = attempt + 1
        if not (spec and spec.retry_safe and result.status == 'error' and result.retryable and attempt < max_retries):
            result.duration_seconds = time.monotonic() - started
            return result
        delay = .1 * (2 ** attempt)
        emit({'type': 'tool_retry', 'name': spec.name, 'attempt': attempt + 2,
              'error_code': result.error_code, 'delay_seconds': delay})
        context.wait(delay)


class ProgressTracker:
    def __init__(self):
        self.history = []

    def observe(self, name, arguments, result):
        if result.status == 'running':
            # A silent live process is not proof of a stuck model. Its native
            # timeout and the shared deadline bound waiting independently.
            self.history.clear()
            return False
        data = result.data
        if isinstance(data, dict):
            data = {k: v for k, v in data.items() if k not in ('process_id', 'output_bytes')}
        key = json.dumps([name, arguments, result.status, result.error_code, data], sort_keys=True)
        self.history.append(key)
        self.history = self.history[-12:]
        for period in (1, 2, 3):
            needed = period * 3
            if len(self.history) >= max(6, needed):
                tail = self.history[-needed:]
                if all(tail[i] == tail[i % period] for i in range(needed)):
                    return True
        return False
