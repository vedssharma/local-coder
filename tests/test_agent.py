"""Tests for agent.py — agentic tool-calling loop and helpers."""

import json
from unittest.mock import MagicMock

import pytest

import agent



# ---------------------------------------------------------------------------
# _parse_inline_tool_calls
# ---------------------------------------------------------------------------

class TestParseInlineToolCalls:
    def test_empty_content_returns_empty(self):
        assert agent._parse_inline_tool_calls("") == []
        assert agent._parse_inline_tool_calls(None) == []

    def test_parses_json_block_with_json_tag(self):
        content = '```json\n{"name": "list_directory", "arguments": {"path": "."}}\n```'
        calls = agent._parse_inline_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "list_directory"

    def test_parses_json_block_without_language_tag(self):
        content = '```\n{"name": "read_file", "arguments": {"path": "main.py"}}\n```'
        calls = agent._parse_inline_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "read_file"

    def test_ignores_json_block_without_name_field(self):
        content = '```json\n{"action": "do_something", "arguments": {}}\n```'
        calls = agent._parse_inline_tool_calls(content)
        assert calls == []

    def test_ignores_json_block_without_arguments_field(self):
        content = '```json\n{"name": "read_file"}\n```'
        calls = agent._parse_inline_tool_calls(content)
        assert calls == []

    def test_invalid_json_block_skipped(self):
        content = '```json\n{not valid json}\n```'
        calls = agent._parse_inline_tool_calls(content)
        assert calls == []

    def test_multiple_blocks_parsed(self):
        block = '{"name": "list_directory", "arguments": {"path": "."}}'
        content = f'```json\n{block}\n```\n\n```json\n{block}\n```'
        calls = agent._parse_inline_tool_calls(content)
        assert len(calls) == 2

    def test_result_has_openai_tool_call_shape(self):
        content = '```json\n{"name": "read_file", "arguments": {"path": "x.py"}}\n```'
        calls = agent._parse_inline_tool_calls(content)
        c = calls[0]
        assert "id" in c
        assert c["type"] == "function"
        assert "name" in c["function"]
        assert "arguments" in c["function"]

    def test_dict_arguments_serialised_to_string(self):
        content = '```json\n{"name": "read_file", "arguments": {"path": "y.py"}}\n```'
        calls = agent._parse_inline_tool_calls(content)
        # arguments must be a JSON string so it can be parsed later
        args = calls[0]["function"]["arguments"]
        assert isinstance(args, str)
        parsed = json.loads(args)
        assert parsed["path"] == "y.py"

    def test_string_arguments_left_as_string(self):
        content = '```json\n{"name": "read_file", "arguments": "{\\"path\\": \\"z.py\\"}"}\n```'
        calls = agent._parse_inline_tool_calls(content)
        assert isinstance(calls[0]["function"]["arguments"], str)


# ---------------------------------------------------------------------------
# _build_tool_schemas
# ---------------------------------------------------------------------------

class TestBuildToolSchemas:
    def test_returns_empty_when_no_client(self):
        assert agent._build_tool_schemas(None) == []

    def test_delegates_to_tools(self):
        client = MagicMock()
        client.get_openai_tool_schemas.return_value = [{"type": "function"}]
        result = agent._build_tool_schemas(client)
        assert result == [{"type": "function"}]
        client.get_openai_tool_schemas.assert_called_once()


# ---------------------------------------------------------------------------
# run_agent
# ---------------------------------------------------------------------------

def _run(llm, messages, **kwargs):
    return agent.run_agent(llm, messages, **kwargs).text

def _make_text_response(text="Final answer."):
    """Helper: build a mock LLM response with plain text content."""
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": text, "tool_calls": None},
                "finish_reason": "stop",
            }
        ]
    }


def _make_tool_call_response(tool_name, args_dict):
    """Helper: build a mock LLM response that requests a tool call."""
    tool_call = {
        "id": "call_test",
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": json.dumps(args_dict),
        },
    }
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }


class TestRunAgent:
    def test_returns_text_when_llm_answers_directly(self, mock_llm):
        mock_llm.create_chat_completion.return_value = _make_text_response("Hello!")
        messages = [{"role": "user", "content": "hi"}]
        result = _run(mock_llm, messages, max_tokens=64)
        assert result == "Hello!"

    def test_final_answer_appended_to_messages(self, mock_llm):
        mock_llm.create_chat_completion.return_value = _make_text_response("Done.")
        messages = [{"role": "user", "content": "hi"}]
        _run(mock_llm, messages, max_tokens=64)
        assert messages[-1] == {"role": "assistant", "content": "Done."}

    def test_calls_mcp_tool_and_loops(self, mock_llm, mock_tools):
        """LLM first returns a tool call, then returns text."""
        mock_llm.create_chat_completion.side_effect = [
            _make_tool_call_response("read_file", {"path": "x.py"}),
            _make_text_response("I read the file."),
        ]
        messages = [{"role": "user", "content": "read a file"}]
        result = _run(mock_llm, messages, max_tokens=64, tools=mock_tools
        )
        assert result == "I read the file."
        mock_tools.call_tool.assert_called_once_with("read_file", {"path": "x.py"})

    def test_tool_result_appended_to_messages(self, mock_llm, mock_tools):
        mock_tools.call_tool.return_value = "file contents here"
        mock_llm.create_chat_completion.side_effect = [
            _make_tool_call_response("read_file", {"path": "x.py"}),
            _make_text_response("Summary."),
        ]
        messages = [{"role": "user", "content": "q"}]
        _run(mock_llm, messages, max_tokens=64, tools=mock_tools
        )
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert any("file contents here" in m["content"] for m in tool_msgs)

    def test_empty_tool_result_replaced_with_placeholder(self, mock_llm, mock_tools):
        mock_tools.call_tool.return_value = ""
        mock_llm.create_chat_completion.side_effect = [
            _make_tool_call_response("list_directory", {"path": "."}),
            _make_text_response("Done."),
        ]
        messages = [{"role": "user", "content": "q"}]
        _run(mock_llm, messages, max_tokens=64, tools=mock_tools
        )
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert any("(empty result)" in m["content"] for m in tool_msgs)

    def test_unknown_tool_returns_error_in_tool_msg(self, mock_llm):
        disconnected = MagicMock()
        disconnected.get_openai_tool_schemas.return_value = []
        mock_llm.create_chat_completion.side_effect = [
            _make_tool_call_response("read_file", {"path": "x.py"}),
            _make_text_response("Fallback."),
        ]
        messages = [{"role": "user", "content": "q"}]
        _run(mock_llm, messages, max_tokens=64, tools=disconnected
        )
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert any(json.loads(m["content"])["error_code"] == "invalid_arguments" for m in tool_msgs)

    def test_inline_tool_call_fallback_parsed(self, mock_llm, mock_tools):
        """If the LLM embeds the tool call in content instead of tool_calls, it is parsed."""
        inline_content = '```json\n{"name": "list_directory", "arguments": {"path": "."}}\n```'
        inline_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": inline_content,
                        "tool_calls": None,
                    },
                    "finish_reason": "stop",
                }
            ]
        }
        mock_llm.create_chat_completion.side_effect = [
            inline_response,
            _make_text_response("Done via inline."),
        ]
        messages = [{"role": "user", "content": "q"}]
        result = _run(mock_llm, messages, max_tokens=64, tools=mock_tools, inline_tool_calls=True
        )
        assert result == "Done via inline."
        mock_tools.call_tool.assert_called_once()

    def test_forces_final_answer_after_max_iterations(self):
        """When every iteration produces a tool call, the loop forces a final answer."""
        llm = MagicMock()

        def always_tool_call(*args, **kwargs):
            # If "tools" not in kwargs this is the forced final call
            if "tools" not in kwargs:
                return _make_text_response("Forced final answer.")
            return _make_tool_call_response("list_directory", {"path": "."})

        llm.create_chat_completion.side_effect = always_tool_call

        client = MagicMock()
        client.get_openai_tool_schemas.return_value = [
            {"type": "function", "function": {"name": "list_directory"}}
        ]
        client.call_tool.return_value = "some result"

        messages = [{"role": "user", "content": "q"}]
        result = _run(llm, messages, max_tokens=64, tools=client
        )
        assert "no observable progress" in result.lower()
        assert llm.create_chat_completion.call_count == 6

    def test_nudge_sent_on_empty_response(self, mock_llm):
        """Empty content + no tool calls causes a nudge message to be appended."""
        empty_response = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "", "tool_calls": None},
                    "finish_reason": "stop",
                }
            ]
        }
        mock_llm.create_chat_completion.side_effect = [
            empty_response,
            _make_text_response("Here it is."),
        ]
        messages = [{"role": "user", "content": "q"}]
        _run(mock_llm, messages, max_tokens=64)
        nudge_msgs = [m for m in messages if m.get("role") == "user" and "must respond" in m.get("content", "")]
        assert nudge_msgs

    def test_tool_args_as_dict_parsed_correctly(self, mock_llm, mock_tools):
        """Tool arguments that arrive as a dict (not a JSON string) are handled."""
        tool_call = {
            "id": "call_dict",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": {"path": "x.py"},  # dict, not string
            },
        }
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
        mock_llm.create_chat_completion.side_effect = [
            response,
            _make_text_response("Done."),
        ]
        messages = [{"role": "user", "content": "q"}]
        result = _run(mock_llm, messages, max_tokens=64, tools=mock_tools
        )
        assert result == "Done."
        mock_tools.call_tool.assert_called_once_with("read_file", {"path": "x.py"})


def test_json_example_is_never_executed_by_default(mock_llm, mock_tools):
    text = '```json\n{"name":"read_file","arguments":{"path":"x"}}\n```'
    mock_llm.create_chat_completion.return_value = _make_text_response(text)
    result = agent.run_agent(mock_llm, [], tools=mock_tools)
    assert result.status == 'completed' and result.text == text
    mock_tools.call_tool.assert_not_called()


def test_invalid_call_recovery_and_repeated_failure(mock_llm, mock_tools):
    bad = _make_tool_call_response('read_file', {})
    bad['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = '{broken'
    mock_llm.create_chat_completion.side_effect = [bad, bad, bad]
    result = agent.run_agent(mock_llm, [], tools=mock_tools)
    assert result.status == 'blocked' and result.reason == 'repeated_tool_failure'
    mock_tools.call_tool.assert_not_called()


def test_cancelled_and_truncated_calls_do_not_execute(mock_llm, mock_tools):
    import threading
    cancel = threading.Event()
    cancel.set()
    assert agent.run_agent(mock_llm, [], cancel_event=cancel).status == 'cancelled'
    mock_llm.create_chat_completion.assert_not_called()
    response = _make_tool_call_response('read_file', {'path': 'x'})
    response['choices'][0]['finish_reason'] = 'length'
    mock_llm.create_chat_completion.return_value = response
    assert agent.run_agent(mock_llm, [], tools=mock_tools).status == 'budget_exhausted'
    mock_tools.call_tool.assert_not_called()


def test_truncated_tool_call_asks_for_smaller_steps_once(mock_llm, mock_tools):
    truncated = _make_tool_call_response('read_file', {'path': 'x'})
    truncated['choices'][0]['finish_reason'] = 'length'
    mock_llm.create_chat_completion.side_effect = [truncated, _make_text_response('done')]
    messages = []
    result = agent.run_agent(mock_llm, messages, max_tokens=300, tools=mock_tools)
    assert result.status == 'completed'
    mock_tools.call_tool.assert_not_called()
    # The cut-off call never enters the transcript; only the recovery note does.
    assert not any(m.get('tool_calls') for m in messages)
    recovery = [m for m in messages if m.get('name') == 'agent_recovery']
    assert len(recovery) == 1 and '300-token output limit' in recovery[0]['content']


def test_text_truncation_still_ends_the_run(mock_llm):
    response = _make_text_response('partial')
    response['choices'][0]['finish_reason'] = 'length'
    mock_llm.create_chat_completion.return_value = response
    result = agent.run_agent(mock_llm, [])
    assert (result.status, result.reason, result.text) == ('budget_exhausted', 'output_truncated', 'partial')
    assert mock_llm.create_chat_completion.call_count == 1


@pytest.mark.parametrize('window, expected', [(2048, 512), (8192, 2048), (16384, 4096), (200000, 4096)])
def test_default_output_tokens_scale_with_context(window, expected):
    assert agent.default_output_tokens(window) == expected


def test_duplicate_reads_execute_once_per_batch_and_keep_protocol(mock_llm, mock_tools):
    response = _make_tool_call_response('read_file', {'path': 'a'})
    second = json.loads(json.dumps(response['choices'][0]['message']['tool_calls'][0]))
    second['id'] = 'second'
    response['choices'][0]['message']['tool_calls'].append(second)
    mock_tools.call_tool.return_value = 'observed contents'
    mock_llm.create_chat_completion.side_effect = [response, _make_text_response('done')]
    messages = []
    result = agent.run_agent(mock_llm, messages, tools=mock_tools)
    assert result.status == 'completed'
    mock_tools.call_tool.assert_called_once()
    outputs = [m for m in messages if m['role'] == 'tool']
    assert len(outputs) == 2 and 'Unchanged observation' in outputs[1]['content']


# ---------------------------------------------------------------------------
# Protocol validation and stop conditions
# ---------------------------------------------------------------------------

from execution_context import DeadlineExceeded, ExecutionCancelled


def _reply(message, finish_reason='stop', usage=None):
    return {'choices': [{'message': message, 'finish_reason': finish_reason}], **({'usage': usage} if usage else {})}


def _run_once(message, finish_reason='tool_calls', **kwargs):
    llm = MagicMock()
    llm.create_chat_completion.return_value = _reply(message, finish_reason)
    return agent.run_agent(llm, [{'role': 'user', 'content': 'go'}], **kwargs)


@pytest.mark.parametrize('message, text', [
    ('not a dict', 'Model request failed: Model returned an invalid message'),
    ({'content': ['parts']}, 'Model request failed: Model returned invalid text content'),
])
def test_malformed_model_messages_block_the_run(message, text):
    result = _run_once(message, 'stop')
    assert (result.status, result.reason, result.text) == ('blocked', 'model_error', text)


@pytest.mark.parametrize('error, status, reason', [
    (KeyboardInterrupt, 'cancelled', ''),
    (ExecutionCancelled('stop'), 'cancelled', ''),
    (DeadlineExceeded('late'), 'budget_exhausted', 'deadline'),
])
def test_interrupted_model_requests_end_the_run(error, status, reason):
    llm = MagicMock()
    llm.create_chat_completion.side_effect = error
    result = agent.run_agent(llm, [{'role': 'user', 'content': 'go'}])
    assert (result.status, result.reason) == (status, reason)


@pytest.mark.parametrize('calls, text', [
    ('not a list', 'Model returned invalid tool calls.'),
    (['not a dict'], 'Model returned invalid tool calls.'),
    ([{'id': 'a', 'function': {'name': 7}}], 'Model returned an invalid tool name.'),
    ([{'id': 7, 'function': {'name': 'read'}}], 'Model returned an invalid tool call ID.'),
    ([{'id': 'a', 'function': {'name': 'read'}}, {'id': 'a', 'function': {'name': 'read'}}],
     'Model returned duplicate tool call IDs.'),
])
def test_invalid_tool_call_envelopes_are_rejected(mock_tools, calls, text):
    result = _run_once({'content': None, 'tool_calls': calls}, tools=mock_tools)
    assert (result.status, result.reason, result.text) == ('blocked', 'invalid_protocol', text)
    mock_tools.call_tool.assert_not_called()


def test_time_budget_is_checked_before_each_model_call():
    llm = MagicMock()
    result = agent.run_agent(llm, [], budget=agent.RunBudget(max_steps=3, max_seconds=1e-9))
    assert result.status == 'budget_exhausted' and result.steps == 0
    llm.create_chat_completion.assert_not_called()


def test_cancellation_during_a_model_call_is_honored_before_tools_run(mock_tools):
    import threading
    cancel = threading.Event()
    llm = MagicMock()

    def respond(**kwargs):
        cancel.set()
        return _reply({'content': None, 'tool_calls': [{'id': 'a', 'function': {'name': 'read_file', 'arguments': '{}'}}]},
                      'tool_calls')
    llm.create_chat_completion.side_effect = respond
    result = agent.run_agent(llm, [], tools=mock_tools, cancel_event=cancel)
    assert result.status == 'cancelled'
    mock_tools.call_tool.assert_not_called()


def test_step_budget_ends_a_run_that_keeps_calling_tools(mock_tools):
    llm = MagicMock()
    llm.create_chat_completion.side_effect = lambda **kwargs: _reply({'content': None, 'tool_calls': [
        {'id': f'c{len(kwargs["messages"])}', 'function': {'name': 'read_file',
                                                          'arguments': json.dumps({'n': len(kwargs['messages'])})}}]},
        'tool_calls')
    result = agent.run_agent(llm, [], tools=mock_tools, budget=agent.RunBudget(max_steps=2))
    assert (result.status, result.reason, result.steps) == ('budget_exhausted', 'step_limit', 2)


def test_unparseable_arguments_are_passed_through():
    assert agent._arguments({'function': {'arguments': '{not json'}}) == '{not json'
    assert agent._arguments({'function': {'arguments': {'a': 1}}}) == {'a': 1}
