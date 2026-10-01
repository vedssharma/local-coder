"""Tests for agent.py — agentic tool-calling loop and helpers."""

import json
from unittest.mock import MagicMock, call, patch

import pytest

import agent


# ---------------------------------------------------------------------------
# _format_args
# ---------------------------------------------------------------------------

class TestFormatArgs:
    def test_simple_key_value(self):
        result = agent._format_args({"path": "/tmp"})
        assert "path='/tmp'" in result

    def test_multiple_args_joined_by_comma(self):
        result = agent._format_args({"a": "1", "b": "2"})
        assert "a='1'" in result
        assert "b='2'" in result

    def test_long_value_truncated(self):
        long_val = "x" * 100
        result = agent._format_args({"key": long_val})
        assert len(result) < 100
        assert "..." in result

    def test_empty_args(self):
        result = agent._format_args({})
        assert result == ""

    def test_value_at_exactly_60_chars_not_truncated(self):
        val = "y" * 60
        result = agent._format_args({"k": val})
        assert "..." not in result


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

    def test_returns_empty_when_not_connected(self):
        client = MagicMock()
        client.is_connected = False
        assert agent._build_tool_schemas(client) == []

    def test_delegates_to_client_when_connected(self):
        client = MagicMock()
        client.is_connected = True
        client.get_openai_tool_schemas.return_value = [{"type": "function"}]
        result = agent._build_tool_schemas(client)
        assert result == [{"type": "function"}]
        client.get_openai_tool_schemas.assert_called_once()


# ---------------------------------------------------------------------------
# run_agent_loop
# ---------------------------------------------------------------------------

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


class TestRunAgentLoop:
    def test_returns_text_when_llm_answers_directly(self, mock_llm, mock_console):
        mock_llm.create_chat_completion.return_value = _make_text_response("Hello!")
        messages = [{"role": "user", "content": "hi"}]
        result = agent.run_agent_loop(mock_llm, messages, mock_console, max_tokens=64)
        assert result == "Hello!"

    def test_final_answer_appended_to_messages(self, mock_llm, mock_console):
        mock_llm.create_chat_completion.return_value = _make_text_response("Done.")
        messages = [{"role": "user", "content": "hi"}]
        agent.run_agent_loop(mock_llm, messages, mock_console, max_tokens=64)
        assert messages[-1] == {"role": "assistant", "content": "Done."}

    def test_calls_mcp_tool_and_loops(self, mock_llm, mock_mcp_client, mock_console):
        """LLM first returns a tool call, then returns text."""
        mock_llm.create_chat_completion.side_effect = [
            _make_tool_call_response("read_file", {"path": "x.py"}),
            _make_text_response("I read the file."),
        ]
        messages = [{"role": "user", "content": "read a file"}]
        result = agent.run_agent_loop(
            mock_llm, messages, mock_console, max_tokens=64, mcp_client=mock_mcp_client
        )
        assert result == "I read the file."
        mock_mcp_client.call_tool.assert_called_once_with("read_file", {"path": "x.py"})

    def test_tool_result_appended_to_messages(self, mock_llm, mock_mcp_client, mock_console):
        mock_mcp_client.call_tool.return_value = "file contents here"
        mock_llm.create_chat_completion.side_effect = [
            _make_tool_call_response("read_file", {"path": "x.py"}),
            _make_text_response("Summary."),
        ]
        messages = [{"role": "user", "content": "q"}]
        agent.run_agent_loop(
            mock_llm, messages, mock_console, max_tokens=64, mcp_client=mock_mcp_client
        )
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert any("file contents here" in m["content"] for m in tool_msgs)

    def test_empty_tool_result_replaced_with_placeholder(self, mock_llm, mock_mcp_client, mock_console):
        mock_mcp_client.call_tool.return_value = ""
        mock_llm.create_chat_completion.side_effect = [
            _make_tool_call_response("list_directory", {"path": "."}),
            _make_text_response("Done."),
        ]
        messages = [{"role": "user", "content": "q"}]
        agent.run_agent_loop(
            mock_llm, messages, mock_console, max_tokens=64, mcp_client=mock_mcp_client
        )
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert any("(empty result)" in m["content"] for m in tool_msgs)

    def test_mcp_not_connected_returns_error_in_tool_msg(self, mock_llm, mock_console):
        disconnected = MagicMock()
        disconnected.is_connected = False
        disconnected.get_openai_tool_schemas.return_value = []
        mock_llm.create_chat_completion.side_effect = [
            _make_tool_call_response("read_file", {"path": "x.py"}),
            _make_text_response("Fallback."),
        ]
        messages = [{"role": "user", "content": "q"}]
        agent.run_agent_loop(
            mock_llm, messages, mock_console, max_tokens=64, mcp_client=disconnected
        )
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert any("Error" in m["content"] for m in tool_msgs)

    def test_inline_tool_call_fallback_parsed(self, mock_llm, mock_mcp_client, mock_console):
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
        result = agent.run_agent_loop(
            mock_llm, messages, mock_console, max_tokens=64, mcp_client=mock_mcp_client, inline_tool_calls=True
        )
        assert result == "Done via inline."
        mock_mcp_client.call_tool.assert_called_once()

    def test_forces_final_answer_after_max_iterations(self, mock_console):
        """When every iteration produces a tool call, the loop forces a final answer."""
        llm = MagicMock()

        def always_tool_call(*args, **kwargs):
            # If "tools" not in kwargs this is the forced final call
            if "tools" not in kwargs:
                return _make_text_response("Forced final answer.")
            return _make_tool_call_response("list_directory", {"path": "."})

        llm.create_chat_completion.side_effect = always_tool_call

        client = MagicMock()
        client.is_connected = True
        client.get_openai_tool_schemas.return_value = [
            {"type": "function", "function": {"name": "list_directory"}}
        ]
        client.call_tool.return_value = "some result"

        messages = [{"role": "user", "content": "q"}]
        result = agent.run_agent_loop(
            llm, messages, mock_console, max_tokens=64, mcp_client=client
        )
        assert "budget exhausted" in result.lower()
        assert llm.create_chat_completion.call_count == agent.MAX_AGENT_ITERATIONS

    def test_nudge_sent_on_empty_response(self, mock_llm, mock_console):
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
        agent.run_agent_loop(mock_llm, messages, mock_console, max_tokens=64)
        nudge_msgs = [m for m in messages if m.get("role") == "user" and "must respond" in m.get("content", "")]
        assert nudge_msgs

    def test_tool_args_as_dict_parsed_correctly(self, mock_llm, mock_mcp_client, mock_console):
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
        result = agent.run_agent_loop(
            mock_llm, messages, mock_console, max_tokens=64, mcp_client=mock_mcp_client
        )
        assert result == "Done."
        mock_mcp_client.call_tool.assert_called_once_with("read_file", {"path": "x.py"})


def test_json_example_is_never_executed_by_default(mock_llm, mock_mcp_client):
    text = '```json\n{"name":"read_file","arguments":{"path":"x"}}\n```'
    mock_llm.create_chat_completion.return_value = _make_text_response(text)
    result = agent.run_agent(mock_llm, [], mcp_client=mock_mcp_client)
    assert result.status == 'completed' and result.text == text
    mock_mcp_client.call_tool.assert_not_called()


def test_invalid_call_recovery_and_repeated_failure(mock_llm, mock_mcp_client):
    bad = _make_tool_call_response('read_file', {})
    bad['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = '{broken'
    mock_llm.create_chat_completion.side_effect = [bad, bad, bad]
    result = agent.run_agent(mock_llm, [], mcp_client=mock_mcp_client)
    assert result.status == 'blocked' and result.reason == 'repeated_tool_failure'
    mock_mcp_client.call_tool.assert_not_called()


def test_cancelled_and_truncated_calls_do_not_execute(mock_llm, mock_mcp_client):
    import threading
    cancel = threading.Event()
    cancel.set()
    assert agent.run_agent(mock_llm, [], cancel_event=cancel).status == 'cancelled'
    mock_llm.create_chat_completion.assert_not_called()
    response = _make_tool_call_response('read_file', {'path': 'x'})
    response['choices'][0]['finish_reason'] = 'length'
    mock_llm.create_chat_completion.return_value = response
    assert agent.run_agent(mock_llm, [], mcp_client=mock_mcp_client).status == 'budget_exhausted'
    mock_mcp_client.call_tool.assert_not_called()


def test_duplicate_reads_execute_once_per_batch_and_keep_protocol(mock_llm, mock_mcp_client):
    response = _make_tool_call_response('read_file', {'path': 'a'})
    second = json.loads(json.dumps(response['choices'][0]['message']['tool_calls'][0]))
    second['id'] = 'second'
    response['choices'][0]['message']['tool_calls'].append(second)
    mock_mcp_client.call_tool.return_value = 'observed contents'
    mock_llm.create_chat_completion.side_effect = [response, _make_text_response('done')]
    messages = []
    result = agent.run_agent(mock_llm, messages, mcp_client=mock_mcp_client)
    assert result.status == 'completed'
    mock_mcp_client.call_tool.assert_called_once()
    outputs = [m for m in messages if m['role'] == 'tool']
    assert len(outputs) == 2 and 'Unchanged observation' in outputs[1]['content']
