"""Bounded agent execution with explicit, inspectable outcomes."""
from dataclasses import dataclass, field
import json
import re
import time
from typing import Literal

from jsonschema import ValidationError
from session import ContextManager
from tool_result import ToolResult, invoke_tool
from tool_registry import ToolRegistry, ToolSpec
from execution_context import ExecutionContext, ExecutionCancelled, DeadlineExceeded
import threading
from tool_recovery import ProgressTracker
from tool_scheduler import ToolScheduler

MAX_AGENT_ITERATIONS = 10
TRUNCATED_CALL_RECOVERY = ('Your previous response hit the {limit}-token output limit before its tool call was complete, '
                           'so nothing was executed. Make the change in smaller steps: use edit for targeted '
                           'replacements, or write a short file and extend it with further edit calls.')


def default_output_tokens(context_window):
    """Per-call output limit: room for whole-file writes without starving a small context of prompt space."""
    return max(512, min(4096, context_window // 4))


@dataclass
class RunBudget:
    max_steps: int = MAX_AGENT_ITERATIONS
    max_seconds: float = 300
    max_generated_tokens: int = 8192

    def __post_init__(self):
        if min(self.max_steps, self.max_seconds, self.max_generated_tokens) <= 0:
            raise ValueError('Run budgets must be positive')


@dataclass
class RunResult:
    status: Literal['completed', 'blocked', 'cancelled', 'budget_exhausted']
    text: str
    steps: int = 0
    generated_tokens: int = 0
    reason: str = ''
    performance: dict = field(default_factory=dict)
    changed_files: list[str] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)
    outstanding_processes: list[dict] = field(default_factory=list)
    verification_status: str = 'not_run'
    verification_scope: str = 'observed_commands_and_changes'


def _build_tool_schemas(mcp_client=None):
    if not mcp_client or not mcp_client.is_connected:
        return []
    return mcp_client.get_openai_tool_schemas()


def run_agent(llm, messages, max_tokens=512, mcp_client=None, budget=None,
              cancel_event=None, emit=None, inline_tool_calls=False, context_manager=None, tool_schemas=None,
              tool_executor=None, execution_context=None, tool_workers=4, checkpoint=None, reserved_call_ids=()):
    """Run a turn. Completion means the model finished, not that its claims were verified.

    Time and cancellation are checked between model/tool operations. Blocking model
    calls require backend timeouts; this engine never promises to interrupt them.
    """
    checkpoint = checkpoint or (lambda *a: None)
    reserved_call_ids = set(reserved_call_ids)
    budget = budget or RunBudget()
    emit = emit or (lambda event: None)
    schemas = _build_tool_schemas(mcp_client) if tool_schemas is None else tool_schemas
    from model_backend import ModelAdapter
    if isinstance(llm, ModelAdapter) and not llm.profile.get("supports_tools", True):
        schemas = []
    schemas = sorted(schemas, key=lambda s: s['function']['name'])
    registered = {s['function']['name'] for s in schemas}
    registry = getattr(mcp_client, 'registry', None)
    if not isinstance(registry, ToolRegistry):
        # Unknown external tools retain conservative execution metadata.
        registry = ToolRegistry()
        for definition in schemas:
            registry.register(ToolSpec(definition, handler=None))
    context_manager = context_manager or ContextManager(window=32768)
    started, generated, failures, steps = time.monotonic(), 0, {}, 0
    truncated_calls = 0

    execution_context = execution_context or ExecutionContext(started + budget.max_seconds, cancel_event or threading.Event())
    if isinstance(llm, ModelAdapter):
        llm.execution_context = execution_context

    progress = ProgressTracker()
    observations = {}
    metrics = {'model_seconds': 0.0, 'context_seconds': 0.0, 'tool_seconds': 0.0, 'model_calls': 0, 'tool_calls': 0}

    def finish(status, text, reason=''):
        result = RunResult(status, text, steps, generated, reason, {**metrics, 'total_seconds': time.monotonic() - started,
            'token_cache_hits': context_manager.cache_hits, 'token_cache_misses': context_manager.cache_misses})
        emit({'type': 'run_finished', 'status': status, 'reason': reason})
        return result

    def stopped(check_tokens=True):
        if cancel_event is not None and cancel_event.is_set():
            return finish('cancelled', 'Run cancelled.')
        if time.monotonic() - started >= budget.max_seconds or (check_tokens and generated >= budget.max_generated_tokens):
            return finish('budget_exhausted', 'Run budget exhausted; work may be incomplete.')
        return None

    for step in range(budget.max_steps):
        if result := stopped():
            return result
        steps = step + 1
        emit({'type': 'model_started', 'step': steps})
        kwargs = {'messages': messages,
                  'max_tokens': min(max_tokens, budget.max_generated_tokens - generated), 'stream': False}
        if schemas:
            kwargs['tools'] = schemas
        try:
            phase = time.monotonic()
            context_manager.fit(messages, schemas, kwargs['max_tokens'])
            metrics['context_seconds'] += time.monotonic() - phase
            phase = time.monotonic()
            metrics['model_calls'] += 1
            response = llm.create_chat_completion(**kwargs)
            metrics['model_seconds'] += time.monotonic() - phase
            choice = response['choices'][0]
            message = choice['message']
            if not isinstance(message, dict):
                raise ValueError('Model returned an invalid message')
            message['role'] = 'assistant'
            if message.get('content') is not None and not isinstance(message['content'], str):
                raise ValueError('Model returned invalid text content')
        except (KeyboardInterrupt, ExecutionCancelled):
            return finish('cancelled', 'Run cancelled.')
        except DeadlineExceeded:
            return finish('budget_exhausted', 'Run deadline exceeded.', 'deadline')
        except Exception as exc:
            return finish('blocked', f'Model request failed: {exc}', 'model_error')
        usage = response.get('usage', {}).get('completion_tokens')
        generated += usage if isinstance(usage, int) and usage >= 0 else max(1, context_manager.count_tokens(json.dumps(message)))
        if result := stopped(check_tokens=False):
            return result
        if choice.get('finish_reason') == 'length':
            # A cut-off tool call is never executed. Ask once for smaller steps instead of ending the run.
            if message.get('tool_calls') and not truncated_calls:
                truncated_calls += 1
                messages.append({'role': 'user', 'name': 'agent_recovery',
                                 'content': TRUNCATED_CALL_RECOVERY.format(limit=kwargs['max_tokens'])})
                emit({'type': 'output_truncated', 'step': steps, 'recovering': True})
                continue
            return finish('budget_exhausted', message.get('content') or 'Model output was truncated.', 'output_truncated')
        truncated_calls = 0
        if choice.get('finish_reason') in ('content_filter', 'error'):
            return finish('blocked', 'Model could not complete this request.', 'model_rejected')
        calls = message.get('tool_calls') or []
        if not calls and inline_tool_calls:
            calls = _parse_inline_tool_calls(message.get('content'))
            if calls:
                message = {'role': 'assistant', 'content': None, 'tool_calls': calls}
        if not calls:
            text = message.get('content')
            if text:
                messages.append({'role': 'assistant', 'content': text})
                emit({'type': 'assistant_text', 'text': text})
                return finish('completed', text)
            messages.append({'role': 'user', 'name': 'agent_recovery', 'content': 'You must respond with an answer or a valid tool call.'})
            continue
        if not isinstance(calls, list):
            return finish('blocked', 'Model returned invalid tool calls.', 'invalid_protocol')
        # Validate the envelope before appending it so transcripts remain resumable.
        ids = set()
        prior_ids = {m['tool_call_id'] for m in messages if m.get('role') == 'tool'} | set(reserved_call_ids)
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get('function'), dict):
                return finish('blocked', 'Model returned invalid tool calls.', 'invalid_protocol')
            if not isinstance(call['function'].get('name'), str):
                return finish('blocked', 'Model returned an invalid tool name.', 'invalid_protocol')
            key = call.get('id') or f'call_{step}_{len(ids)}'
            if not isinstance(key, str):
                return finish('blocked', 'Model returned an invalid tool call ID.', 'invalid_protocol')
            if key in ids:
                return finish('blocked', 'Model returned duplicate tool call IDs.', 'invalid_protocol')
            while key in prior_ids:
                key = f'{key}_{step}_{len(ids)}'
            call['id'] = key
            ids.add(key)
        reserved_call_ids.update(ids)
        messages.append(message)
        checkpoint('batch', message, None, None)
        halt = None
        scheduler = ToolScheduler(registry, registered,
            tool_executor or (lambda name, args: invoke_tool(mcp_client, name, args)),
            execution_context, emit, tool_workers, checkpoint)
        for call, args, tool_result in scheduler.run(calls):
            name = call['function']['name']
            raw = call['function'].get('arguments')
            if tool_result.status == 'cancelled' and not halt:
                halt = finish('cancelled', 'Run cancelled.')
            if result := stopped():
                halt = halt or result
            # Retain one full identical observation in the active transcript.
            # Fresh execution across steps observes external edits before deduplication.
            no_progress = progress.observe(name, args, tool_result)
            original_output = json.dumps(tool_result.data, sort_keys=True)
            spec = registry.get(name)
            if spec and spec.compact_observation and tool_result.status == 'success':
                observation_key = json.dumps([name, args], sort_keys=True)
                previous = observations.get(observation_key)
                if previous and previous[1] == original_output and any(m.get('tool_call_id') == previous[0] for m in messages):
                    tool_result = ToolResult(data=f"Unchanged observation; see tool result {previous[0]}.")
                else:
                    observations[observation_key] = (call['id'], original_output)
            output = tool_result.to_model(context_manager)
            messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': output})
            checkpoint('observed',call,args,tool_result)
            emit({'type': 'tool_finished', 'name': name, 'call_id': call['id'], 'output': output, 'result': tool_result.to_dict()})
            fingerprint = json.dumps([name, raw], sort_keys=True)
            if no_progress and not halt:
                halt = finish('blocked', 'Repeated tool calls are making no observable progress.', 'no_progress')
            if tool_result.is_error:
                failures[fingerprint] = failures.get(fingerprint, 0) + 1
                if failures[fingerprint] >= 3:
                    halt = finish('blocked', f'Repeated tool failure: {tool_result.error_message or tool_result.error_code}', 'repeated_tool_failure')
            else:
                failures.pop(fingerprint, None)
            if halt:
                scheduler.stopped = True
        metrics['tool_seconds'] += scheduler.elapsed
        metrics['tool_calls'] += scheduler.executions
        if halt:
            halt.performance.update(metrics)
            return halt
    return finish('budget_exhausted', 'Step budget exhausted; work may be incomplete.', 'step_limit')


def run_agent_loop(llm, messages, console, max_tokens=512, mcp_client=None, **kwargs):
    """Compatibility text interface. New integrations should consume run_agent's result."""
    def emit(event):
        if event['type'] == 'tool_started':
            console.print(f"[dim]tool: {event['name']}[/dim]")
    result = run_agent(llm, messages, max_tokens, mcp_client, emit=emit, **kwargs)
    if result.status != 'completed':
        console.print(f'[{result.status}] {result.text}', markup=False)
    return result.text


def _parse_inline_tool_calls(content):
    """
    Parse tool calls the model emitted as a markdown JSON block, e.g.:

        ```json
        {"name": "read", "arguments": {"path": "main.py"}}
        ```

    Returns a list of tool_call dicts in OpenAI format, or an empty list.
    """
    if not content:
        return []
    calls = []
    for i, raw in enumerate(re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj.get("name"), str) and "arguments" in obj:
            args = obj["arguments"]
            calls.append({
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": obj["name"],
                    "arguments": args if isinstance(args, str) else json.dumps(args),
                },
            })
    return calls


def _format_args(args):
    """Format tool arguments for display, truncating long values."""
    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > 60:
            s = s[:57] + "..."
        parts.append(f"{k}={s!r}")
    return ", ".join(parts)
