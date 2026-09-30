"""Tests for mcp_client.py — MCPClient wrapper around the MCP filesystem server."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_client import MCPClient


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestMCPClientInit:
    def test_default_allowed_dir_is_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        client = MCPClient()
        assert client.allowed_dir == str(tmp_path)

    def test_custom_allowed_dir_is_stored_as_abspath(self, tmp_path):
        client = MCPClient(allowed_dir=str(tmp_path))
        assert client.allowed_dir == str(tmp_path.resolve())

    def test_initial_state_not_connected(self, tmp_path):
        client = MCPClient(allowed_dir=str(tmp_path))
        assert not client.is_connected

    def test_initial_tool_names_empty(self, tmp_path):
        client = MCPClient(allowed_dir=str(tmp_path))
        assert client.tool_names == set()


# ---------------------------------------------------------------------------
# is_connected property
# ---------------------------------------------------------------------------

class TestIsConnected:
    def test_false_before_connect(self, tmp_path):
        client = MCPClient(allowed_dir=str(tmp_path))
        assert client.is_connected is False

    def test_true_after_manual_set(self, tmp_path):
        client = MCPClient(allowed_dir=str(tmp_path))
        client._connected = True
        assert client.is_connected is True


# ---------------------------------------------------------------------------
# get_openai_tool_schemas
# ---------------------------------------------------------------------------

class TestGetOpenAIToolSchemas:
    def _make_tool(self, name, description="desc", input_schema=None):
        tool = MagicMock()
        tool.name = name
        tool.description = description
        tool.inputSchema = input_schema or {"type": "object", "properties": {}}
        return tool

    def test_empty_when_no_tools(self, tmp_path):
        client = MCPClient(allowed_dir=str(tmp_path))
        assert client.get_openai_tool_schemas() == []

    def test_converts_single_tool(self, tmp_path):
        client = MCPClient(allowed_dir=str(tmp_path))
        client._tools = [self._make_tool("read_file", "Read a file")]
        schemas = client.get_openai_tool_schemas()
        assert len(schemas) == 1
        s = schemas[0]
        assert s["type"] == "function"
        assert s["function"]["name"] == "read_file"
        assert s["function"]["description"] == "Read a file"

    def test_converts_multiple_tools(self, tmp_path):
        client = MCPClient(allowed_dir=str(tmp_path))
        client._tools = [
            self._make_tool("read_file"),
            self._make_tool("write_file"),
        ]
        schemas = client.get_openai_tool_schemas()
        names = {s["function"]["name"] for s in schemas}
        assert names == {"read_file", "write_file"}

    def test_uses_default_schema_when_input_schema_is_none(self, tmp_path):
        client = MCPClient(allowed_dir=str(tmp_path))
        tool = self._make_tool("no_schema")
        tool.inputSchema = None
        client._tools = [tool]
        schemas = client.get_openai_tool_schemas()
        params = schemas[0]["function"]["parameters"]
        assert params["type"] == "object"
        assert params["properties"] == {}

    def test_empty_description_allowed(self, tmp_path):
        client = MCPClient(allowed_dir=str(tmp_path))
        tool = self._make_tool("t", description=None)
        client._tools = [tool]
        schemas = client.get_openai_tool_schemas()
        assert schemas[0]["function"]["description"] == ""


# ---------------------------------------------------------------------------
# call_tool — when not connected
# ---------------------------------------------------------------------------

class TestCallToolNotConnected:
    def test_returns_error_string_when_not_connected(self, tmp_path):
        client = MCPClient(allowed_dir=str(tmp_path))
        result = client.call_tool("read_file", {"path": "x.py"})
        assert "Error" in result
        assert "not connected" in result.lower()

    def test_returns_error_when_loop_is_none(self, tmp_path):
        client = MCPClient(allowed_dir=str(tmp_path))
        client._connected = True  # force connected but loop is None
        result = client.call_tool("read_file", {"path": "x.py"})
        assert "Error" in result


# ---------------------------------------------------------------------------
# _call_tool_async — error and success handling
# ---------------------------------------------------------------------------

class TestCallToolAsync:
    def _make_text_block(self, text):
        block = MagicMock()
        block.text = text
        return block

    def _make_non_text_block(self):
        class BinaryBlock:
            def __str__(self):
                return "<binary>"
        return BinaryBlock()

    @pytest.mark.asyncio
    async def test_success_returns_concatenated_text(self, tmp_path):
        client = MCPClient(allowed_dir=str(tmp_path))
        client._session = AsyncMock()

        result_obj = MagicMock()
        result_obj.isError = False
        result_obj.content = [self._make_text_block("line1"), self._make_text_block("line2")]
        client._session.call_tool = AsyncMock(return_value=result_obj)

        output = await client._call_tool_async("read_file", {"path": "x.py"})
        assert output == "line1\nline2"

    @pytest.mark.asyncio
    async def test_success_empty_content_returns_placeholder(self, tmp_path):
        client = MCPClient(allowed_dir=str(tmp_path))
        client._session = AsyncMock()

        result_obj = MagicMock()
        result_obj.isError = False
        result_obj.content = []
        client._session.call_tool = AsyncMock(return_value=result_obj)

        output = await client._call_tool_async("list_directory", {"path": "."})
        assert output == "(empty result)"

    @pytest.mark.asyncio
    async def test_error_result_returns_error_string(self, tmp_path):
        client = MCPClient(allowed_dir=str(tmp_path))
        client._session = AsyncMock()

        result_obj = MagicMock()
        result_obj.isError = True
        result_obj.content = [self._make_text_block("file not found")]
        client._session.call_tool = AsyncMock(return_value=result_obj)

        output = await client._call_tool_async("read_file", {"path": "missing.py"})
        assert output.startswith("Error")
        assert "file not found" in output

    @pytest.mark.asyncio
    async def test_error_result_unknown_when_no_text(self, tmp_path):
        client = MCPClient(allowed_dir=str(tmp_path))
        client._session = AsyncMock()

        result_obj = MagicMock()
        result_obj.isError = True
        result_obj.content = []
        client._session.call_tool = AsyncMock(return_value=result_obj)

        output = await client._call_tool_async("read_file", {"path": "x.py"})
        assert "unknown MCP error" in output

    @pytest.mark.asyncio
    async def test_non_text_block_converted_via_str(self, tmp_path):
        client = MCPClient(allowed_dir=str(tmp_path))
        client._session = AsyncMock()

        result_obj = MagicMock()
        result_obj.isError = False
        result_obj.content = [self._make_non_text_block()]
        client._session.call_tool = AsyncMock(return_value=result_obj)

        output = await client._call_tool_async("read_file", {"path": "x.py"})
        # Should not raise, result is the str() of the block
        assert isinstance(output, str)


# ---------------------------------------------------------------------------
# close — does not raise when nothing is connected
# ---------------------------------------------------------------------------

class TestClose:
    def test_close_without_connect_does_not_raise(self, tmp_path):
        client = MCPClient(allowed_dir=str(tmp_path))
        client.close()  # must be a no-op
        assert not client.is_connected

    def test_close_resets_state(self, tmp_path):
        client = MCPClient(allowed_dir=str(tmp_path))
        # Manually set some state as if connected
        client._connected = True
        client._loop = MagicMock()
        client._loop.call_soon_threadsafe = MagicMock()
        client._thread = MagicMock()
        client._exit_stack = None  # no async cleanup needed
        client.close()
        assert not client.is_connected
        assert client._session is None
        assert client._exit_stack is None


def test_resource_scopes_close_on_their_owning_task(monkeypatch, tmp_path):
    from contextlib import asynccontextmanager
    import mcp_client
    tasks = []
    @asynccontextmanager
    async def transport(_):
        owner = asyncio.current_task()
        tasks.append('entered')
        try:
            yield (None, None)
        finally:
            assert asyncio.current_task() is owner
            tasks.append('closed')
    class Session:
        def __init__(self, *_):
            pass
        async def __aenter__(self):
            self.owner = asyncio.current_task()
            return self
        async def __aexit__(self, *_):
            assert asyncio.current_task() is self.owner
        async def initialize(self):
            pass
        async def list_tools(self):
            result = MagicMock()
            result.tools = []
            return result
    monkeypatch.setattr(mcp_client, 'stdio_client', transport)
    monkeypatch.setattr(mcp_client, 'ClientSession', Session)
    client = MCPClient(tmp_path)
    client.connect()
    assert client.is_connected
    client.close()
    assert tasks == ['entered', 'closed']
