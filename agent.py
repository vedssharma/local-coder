"""Bounded agent execution with explicit, inspectable outcomes."""
from dataclasses import dataclass
import json
import re
import time
from typing import Literal

from jsonschema import validate, ValidationError
from session import ContextManager

MAX_AGENT_ITERATIONS = 10


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


def _build_tool_schemas(mcp_client=None):
    if not mcp_client or not mcp_client.is_connected:
        return []
    return mcp_client.get_openai_tool_schemas()


def run_agent(llm, messages, max_tokens=512, mcp_client=None, budget=None,
              cancel_event=None, emit=None, inline_tool_calls=False, context_manager=None):
    """Run a turn. Completion means the model finished, not that its claims were verified.

    Time and cancellation are checked between model/tool operations. Blocking model
    calls require backend timeouts; this engine never promises to interrupt them.
    """
    budget = budget or RunBudget()
    emit = emit or (lambda event: None)
    schemas = _build_tool_schemas(mcp_client)
    from model_backend import ModelAdapter
    if isinstance(llm, ModelAdapter) and not llm.profile.get("supports_tools", True):
        schemas = []
    registered = {s['function']['name']: s['function'].get('parameters', {'type': 'object'})
                  for s in schemas}
    context_manager = context_manager or ContextManager(window=32768)
    started, generated, failures, steps = time.monotonic(), 0, {}, 0

    def finish(status, text, reason=''):
        result = RunResult(status, text, steps, generated, reason)
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
            context_manager.fit(messages, schemas, kwargs['max_tokens'])
            response = llm.create_chat_completion(**kwargs)
            choice = response['choices'][0]
            message = choice['message']
            if not isinstance(message, dict):
                raise ValueError('Model returned an invalid message')
        except KeyboardInterrupt:
            return finish('cancelled', 'Run cancelled.')
        except Exception as exc:
            return finish('blocked', f'Model request failed: {exc}', 'model_error')
        usage = response.get('usage', {}).get('completion_tokens')
        generated += usage if isinstance(usage, int) and usage >= 0 else max(1, context_manager.count_tokens(json.dumps(message)))
        if result := stopped(check_tokens=False):
            return result
        if choice.get('finish_reason') == 'length':
            return finish('budget_exhausted', message.get('content') or 'Model output was truncated.', 'output_truncated')
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
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get('function'), dict):
                return finish('blocked', 'Model returned invalid tool calls.', 'invalid_protocol')
            key = call.get('id') or f'call_{step}_{len(ids)}'
            if key in ids:
                return finish('blocked', 'Model returned duplicate tool call IDs.', 'invalid_protocol')
            call['id'] = key
            ids.add(key)
        messages.append(message)
        halt = None
        for call in calls:
            name = call['function'].get('name')
            raw = call['function'].get('arguments')
            emit({'type': 'tool_started', 'name': name, 'call_id': call['id']})
            try:
                if halt:
                    raise ValueError('Run stopped; tool was not executed')
                if result := stopped():
                    halt = result
                    raise ValueError('Run stopped; tool was not executed')
                if name not in registered:
                    raise ValueError(f'Unknown or unavailable tool: {name}')
                args = json.loads(raw) if isinstance(raw, str) else raw
                validate(args, registered[name])
                output = mcp_client.call_tool(name, args) or '(empty result)'
                output = str(output)
            except (ValueError, TypeError, ValidationError) as exc:
                output = f'Error: invalid tool call: {exc}'
            except KeyboardInterrupt:
                halt = finish('cancelled', 'Run cancelled.')
                output = 'Error: tool interrupted'
            except Exception as exc:
                output = f'Error: tool failed: {exc}'
            output = context_manager.bound_output(output)
            messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': output})
            emit({'type': 'tool_finished', 'name': name, 'call_id': call['id'], 'output': output})
            fingerprint = json.dumps([name, raw], sort_keys=True)
            if output.startswith('Error:'):
                failures[fingerprint] = failures.get(fingerprint, 0) + 1
                if failures[fingerprint] >= 3:
                    halt = finish('blocked', f'Repeated tool failure: {output}', 'repeated_tool_failure')
            else:
                failures.pop(fingerprint, None)
        if halt:
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
        {"name": "list_directory", "arguments": {"path": "."}}
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
