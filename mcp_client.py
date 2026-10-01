"""
MCP (Model Context Protocol) client for the filesystem server.

Starts @modelcontextprotocol/server-filesystem as a subprocess via stdio,
discovers available tools, and provides sync wrappers for calling them.

Uses a background thread with a persistent event loop to keep the MCP
session alive across multiple tool calls.
"""

import asyncio
import os
import threading
import time
from tool_result import ToolResult
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    """Wraps an MCP filesystem server connection."""

    def __init__(self, allowed_dir=None):
        """
        Args:
            allowed_dir: Directory the filesystem server is confined to.
                         Defaults to the current working directory.
        """
        self.allowed_dir = os.path.abspath(allowed_dir or os.getcwd())
        self._session = None
        self._exit_stack = None
        self._tools = []
        self._tool_names = set()
        self._connected = False
        self._loop = None
        self._thread = None
        self._ready = threading.Event()
        self._stop = None
        self._future = None
        self._connect_error = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self):
        """Start a lifecycle task that owns both entry and exit of MCP scopes."""
        if self._connected:
            return
        self._ready.clear()
        self._connect_error = None
        try:
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
            self._thread.start()
            self._future = asyncio.run_coroutine_threadsafe(self._connect_async(), self._loop)
            if not self._ready.wait(timeout=30):
                raise TimeoutError('MCP initialization timed out')
            if self._connect_error:
                raise self._connect_error
        except Exception as exc:
            print(f'[MCP] Failed to connect: {exc}')
            self.close()

    async def _connect_async(self):
        params = StdioServerParameters(command='npx', args=[
            '-y', '@modelcontextprotocol/server-filesystem', self.allowed_dir])
        self._stop = asyncio.Event()
        try:
            async with AsyncExitStack() as stack:
                self._exit_stack = stack
                read, write = await stack.enter_async_context(stdio_client(params))
                self._session = await stack.enter_async_context(ClientSession(read, write))
                await self._session.initialize()
                tools = await self._session.list_tools()
                self._tools = tools.tools
                self._tool_names = {tool.name for tool in self._tools}
                self._connected = True
                self._ready.set()
                await self._stop.wait()
        except Exception as exc:
            self._connect_error = exc
        finally:
            self._connected = False
            self._ready.set()

    def close(self):
        """Close MCP resources on their owning task before stopping its loop."""
        if self._loop:
            if self._stop:
                self._loop.call_soon_threadsafe(self._stop.set)
            if self._future:
                try:
                    self._future.result(timeout=5)
                except Exception:
                    self._future.cancel()
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join(timeout=5)
            if not self._loop.is_running():
                self._loop.close()
        self._loop = self._thread = self._future = self._stop = None
        self._exit_stack = self._session = None
        self._connected = False

    # ------------------------------------------------------------------
    # Tool discovery
    # ------------------------------------------------------------------

    @property
    def is_connected(self):
        return self._connected

    @property
    def tool_names(self):
        """Set of tool names provided by the MCP server."""
        return self._tool_names

    def get_openai_tool_schemas(self):
        """
        Convert MCP tool schemas to OpenAI function-calling format
        (compatible with llama-cpp-python's create_chat_completion).
        """
        schemas = []
        for tool in self._tools:
            schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema if tool.inputSchema else {
                        "type": "object",
                        "properties": {},
                    },
                },
            }
            schemas.append(schema)
        return schemas

    # ------------------------------------------------------------------
    # Tool invocation
    # ------------------------------------------------------------------

    def call_tool(self, name, arguments):
        """
        Invoke an MCP tool by name.

        Args:
            name: Tool name (e.g. "read_file", "create_directory").
            arguments: Dict of arguments to pass.

        Returns:
            Result text as a string.
        """
        return self.execute_tool(name, arguments).to_legacy()

    def execute_tool(self, name, arguments):
        started = time.monotonic()
        if not self._connected or not self._loop:
            return ToolResult.error('disconnected', 'MCP client is not connected')
        future = asyncio.run_coroutine_threadsafe(self._execute_tool_async(name, arguments), self._loop)
        try:
            result = future.result(timeout=30)
        except TimeoutError:
            future.cancel()
            result = ToolResult.error('tool_timeout', 'MCP tool timed out', retryable=True)
        except Exception as exc:
            result = ToolResult.error('mcp_error', exc)
        result.duration_seconds = time.monotonic() - started
        return result

    async def _call_tool_async(self, name, arguments):
        return (await self._execute_tool_async(name, arguments)).to_legacy()

    async def _execute_tool_async(self, name, arguments):
        result = await self._session.call_tool(name, arguments)
        parts = [block.text if hasattr(block, 'text') else str(block) for block in result.content]
        text = '\n'.join(parts) or '(empty result)'
        if getattr(result, 'isError', False):
            return ToolResult.error('mcp_tool_error', '\n'.join(parts) or 'unknown MCP error')
        return ToolResult(data=text)
