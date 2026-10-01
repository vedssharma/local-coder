"""Structured outcomes shared by native tools, MCP, and the agent loop."""
from dataclasses import asdict, dataclass, field
import json
from typing import Literal


@dataclass
class ToolResult:
    status: Literal['success', 'running', 'failed', 'error', 'timed_out', 'cancelled'] = 'success'
    data: object = None
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False
    duration_seconds: float = 0.0
    artifacts: list[str] = field(default_factory=list)

    @property
    def is_error(self):
        return self.status in ('error', 'failed', 'timed_out')

    @classmethod
    def error(cls, code, message, retryable=False):
        return cls(status='error', error_code=code, error_message=str(message), retryable=retryable)

    @classmethod
    def process(cls, data):
        if data.get('running'):
            return cls(status='running', data=data)
        if data.get('timed_out'):
            return cls(status='timed_out', data=data, error_code='process_timeout')
        if data.get('cancelled'):
            return cls(status='cancelled', data=data)
        if data.get('exit_code') != 0:
            return cls(status='failed', data=data, error_code='command_failed')
        return cls(data=data)

    def to_dict(self):
        return asdict(self)

    def to_legacy(self):
        """Keep the public string API while internal execution uses typed outcomes."""
        if self.status == 'error':
            return f'Error: {self.error_message}'
        return self.data if isinstance(self.data, str) else json.dumps(self.data)

    def to_model(self, context):
        # Bound only the payload; status and error metadata remain valid JSON.
        envelope = self.to_dict()
        text = json.dumps(self.data, ensure_ascii=False)
        bounded = context.bound_output(text)
        if bounded != text:
            envelope['data'] = bounded
            envelope['data_truncated'] = True
            if context.artifact_dir:
                import hashlib
                path = context.artifact_dir / (hashlib.sha256(text.encode()).hexdigest() + '.txt')
                if path.exists() and str(path) not in self.artifacts:
                    self.artifacts.append(str(path))
                envelope['artifacts'] = list(self.artifacts)
        return json.dumps(envelope, ensure_ascii=False)


def invoke_tool(client, name, arguments):
    """Use the typed protocol when implemented, adapting older string clients."""
    if callable(getattr(type(client), 'execute_tool', None)):
        return client.execute_tool(name, arguments)
    return ToolResult(data=client.call_tool(name, arguments) or '(empty result)')
