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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self):
        """Start the MCP server subprocess and discover tools."""
        try:
            # Create a new event loop in a background thread
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
            self._thread.start()

            # Run the async connect on the background loop
            future = asyncio.run_coroutine_threadsafe(self._connect_async(), self._loop)
            future.result(timeout=30)  # Wait up to 30s for server to start
        except Exception as e:
            print(f"[MCP] Failed to connect: {e}")
            self._connected = False

    async def _connect_async(self):
        server_params = StdioServerParameters(
            command="npx",
            args=[
                "-y",
                "@modelcontextprotocol/server-filesystem",
                self.allowed_dir,
            ],
        )

        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read_stream, write_stream = stdio_transport
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

        await self._session.initialize()

        # Discover tools
        tools_result = await self._session.list_tools()
        self._tools = tools_result.tools
        self._tool_names = {t.name for t in self._tools}
        self._connected = True
        print(f"[MCP] Discovered tools: {', '.join(sorted(self._tool_names))}")

    def close(self):
        """Shut down the MCP server subprocess."""
        if self._loop and self._exit_stack:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._exit_stack.__aexit__(None, None, None), self._loop
                )
                future.result(timeout=5)
            except Exception:
                pass

        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join(timeout=5)
            self._loop = None
            self._thread = None

        self._exit_stack = None
        self._session = None
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
        if not self._connected or not self._loop:
            return f"Error: MCP client is not connected"
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._call_tool_async(name, arguments), self._loop
            )
            return future.result(timeout=30)
        except Exception as e:
            return f"Error calling MCP tool '{name}': {e}"

    async def _call_tool_async(self, name, arguments):
        result = await self._session.call_tool(name, arguments)

        if result.isError:
            # Extract error message from content blocks
            error_parts = []
            for block in result.content:
                if hasattr(block, "text"):
                    error_parts.append(block.text)
            return f"Error: {' '.join(error_parts) or 'unknown MCP error'}"

        # MCP returns a list of content blocks; concatenate text parts
        parts = []
        for block in result.content:
            if hasattr(block, "text"):
                parts.append(block.text)
            else:
                parts.append(str(block))
        return "\n".join(parts) or "(empty result)"
