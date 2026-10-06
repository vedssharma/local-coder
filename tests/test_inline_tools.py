"""Tool calling for models whose profile says they have no native tool calls."""

import json

from agent import (INLINE_TOOLS_MARKER, _parse_inline_tool_calls, describe_inline_tools,
                   inline_transcript, run_agent)
from model_backend import ModelAdapter
from workspace_tools import WorkspaceTools

SCHEMAS = [{'type': 'function', 'function': {'name': 'read', 'description': 'Read a file',
                                             'parameters': {'type': 'object'}}}]


def test_tool_call_blocks_are_parsed_before_fenced_json():
    content = ('Let me look.\n<tool_call>\n{"name": "read", "arguments": {"path": "a.py"}}\n</tool_call>\n'
               '```json\n{"name": "ignored", "arguments": {}}\n```')
    calls = _parse_inline_tool_calls(content)
    assert [c['function']['name'] for c in calls] == ['read']
    assert json.loads(calls[0]['function']['arguments']) == {'path': 'a.py'}
    assert _parse_inline_tool_calls('<tool_call>["read"]</tool_call>') == []


def test_tools_are_described_once_in_the_system_message():
    messages = [{'role': 'system', 'content': 'Be brief.'}, {'role': 'user', 'content': 'hi'}]
    describe_inline_tools(messages, SCHEMAS)
    describe_inline_tools(messages, SCHEMAS)
    content = messages[0]['content']
    assert content.startswith('Be brief.') and content.count(INLINE_TOOLS_MARKER) == 1
    assert '- read: Read a file' in content and '<tool_call>' in content
    bare = [{'role': 'user', 'content': 'hi'}]
    describe_inline_tools(bare, SCHEMAS)
    assert bare[0]['role'] == 'system' and len(bare) == 2


def test_transcript_turns_tool_roles_into_text():
    call = {'id': 'call_0', 'type': 'function', 'function': {'name': 'read', 'arguments': '{"path": "a.py"}'}}
    messages = [{'role': 'user', 'content': 'go'},
                {'role': 'assistant', 'content': None, 'tool_calls': [call]},
                {'role': 'tool', 'tool_call_id': 'call_0', 'content': 'print(1)'}]
    converted = inline_transcript(messages)
    assert converted[0] == messages[0]
    assert converted[1]['role'] == 'assistant' and 'tool_calls' not in converted[1]
    assert _parse_inline_tool_calls(converted[1]['content'])[0]['function']['name'] == 'read'
    assert converted[2] == {'role': 'user', 'content': 'Tool result (read, call_0):\nprint(1)'}
    assert messages[1]['tool_calls'] == [call]  # The real transcript is untouched.


def test_model_without_native_tools_runs_inline_calls(tmp_path):
    (tmp_path / 'a.py').write_text('print(1)\n')

    class Inline(ModelAdapter):
        requests = []
        replies = ['<tool_call>{"name": "read", "arguments": {"path": "a.py"}}</tool_call>', 'It prints 1.']

        def _complete(self, **kwargs):
            self.requests.append(kwargs)
            return {'choices': [{'message': {'content': self.replies[len(self.requests) - 1]},
                                 'finish_reason': 'stop'}]}

    model = Inline({'supports_tools': False})
    messages = [{'role': 'system', 'content': 'Help.'}, {'role': 'user', 'content': 'What does a.py do?'}]
    result = run_agent(model, messages, mcp_client=WorkspaceTools(tmp_path))
    assert result.status == 'completed' and result.text == 'It prints 1.'
    assert all('tools' not in request for request in model.requests)
    assert INLINE_TOOLS_MARKER in model.requests[0]['messages'][0]['content']
    assert {m['role'] for m in model.requests[1]['messages']} == {'system', 'user', 'assistant'}
    assert 'print(1)' in model.requests[1]['messages'][-1]['content']
    assert any(m['role'] == 'tool' for m in messages)
