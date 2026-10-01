"""Bounded agent execution with explicit, inspectable outcomes."""
from dataclasses import dataclass, field
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
    performance: dict = field(default_factory=dict)


def _build_tool_schemas(mcp_client=None):
    if not mcp_client or not mcp_client.is_connected:
        return []
    return mcp_client.get_openai_tool_schemas()


def run_agent(llm, messages, max_tokens=512, mcp_client=None, budget=None,
              cancel_event=None, emit=None, inline_tool_calls=False, context_manager=None, tool_schemas=None):
    """Run a turn. Completion means the model finished, not that its claims were verified.

    Time and cancellation are checked between model/tool operations. Blocking model
    calls require backend timeouts; this engine never promises to interrupt them.
    """
    budget = budget or RunBudget()
    emit = emit or (lambda event: None)
    schemas = _build_tool_schemas(mcp_client) if tool_schemas is None else tool_schemas
    from model_backend import ModelAdapter
    if isinstance(llm, ModelAdapter) and not llm.profile.get("supports_tools", True):
        schemas = []
    schemas = sorted(schemas, key=lambda s: s['function']['name'])
    registered = {s['function']['name']: s['function'].get('parameters', {'type': 'object'})
                  for s in schemas}
    context_manager = context_manager or ContextManager(window=32768)
    started, generated, failures, steps = time.monotonic(), 0, {}, 0

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
        prior_ids = {m['tool_call_id'] for m in messages if m.get('role') == 'tool'}
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get('function'), dict):
                return finish('blocked', 'Model returned invalid tool calls.', 'invalid_protocol')
            key = call.get('id') or f'call_{step}_{len(ids)}'
            if not isinstance(key, str):
                return finish('blocked', 'Model returned an invalid tool call ID.', 'invalid_protocol')
            if key in ids:
                return finish('blocked', 'Model returned duplicate tool call IDs.', 'invalid_protocol')
            if key in prior_ids:
                key = f'{key}_{step}_{len(ids)}'
            call['id'] = key
            ids.add(key)
        messages.append(message)
        halt = None
        read_cache = {}
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
                phase = time.monotonic()
                metrics['tool_calls'] += 1
                cache_key = json.dumps([name, args], sort_keys=True)
                read_names = {'read_file', 'search_code', 'list_directory', 'batch_read', 'batch_search'}
                if name in read_names and cache_key in read_cache:
                    output = read_cache[cache_key]
                    emit({'type': 'tool_reused', 'name': name, 'call_id': call['id']})
                else:
                    output = mcp_client.call_tool(name, args) or '(empty result)'
                    if name in read_names and not str(output).startswith('Error:'):
                        read_cache[cache_key] = output
                    if name not in read_names:
                        read_cache.clear()
                metrics['tool_seconds'] += time.monotonic() - phase
                output = str(output)
            except (ValueError, TypeError, ValidationError) as exc:
                output = f'Error: invalid tool call: {exc}'
            except KeyboardInterrupt:
                halt = finish('cancelled', 'Run cancelled.')
                output = 'Error: tool interrupted'
            except Exception as exc:
                output = f'Error: tool failed: {exc}'
            # Retain one full identical observation in the active transcript.
            # Fresh execution across steps observes external edits before deduplication.
            original_output = output
            if name in ('read_file', 'list_directory', 'batch_read') and not output.startswith('Error:'):
                observation_key = json.dumps([name, args], sort_keys=True)
                previous = observations.get(observation_key)
                if previous and previous[1] == output and any(m.get('tool_call_id') == previous[0] for m in messages):
                    output = f"Unchanged observation; see tool result {previous[0]}."
                else:
                    observations[observation_key] = (call['id'], original_output)
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
