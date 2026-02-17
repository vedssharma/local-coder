import json
import re
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner

MAX_AGENT_ITERATIONS = 10


def _build_tool_schemas(mcp_client=None):
    """Return MCP tool schemas, or an empty list if MCP is not connected."""
    if not mcp_client or not mcp_client.is_connected:
        return []
    return mcp_client.get_openai_tool_schemas()


def run_agent_loop(llm, messages, console, max_tokens=512, mcp_client=None):
    """
    Run the agentic tool-calling loop.

    Calls the LLM with tool schemas. If the LLM returns tool calls,
    executes them via MCP, appends results, and loops. When the LLM
    returns a text response (no tool calls), returns that as the final answer.
    """
    schemas = _build_tool_schemas(mcp_client)

    tool_summary = f"{len(schemas)} MCP tools"
    if schemas:
        tool_summary += f" ({', '.join(s['function']['name'] for s in schemas)})"
    console.print(f"[dim]{tool_summary}[/dim]")

    for iteration in range(MAX_AGENT_ITERATIONS):
        call_kwargs = {"messages": messages, "max_tokens": max_tokens, "stream": False}
        if schemas:
            call_kwargs["tools"] = schemas

        with Live(Spinner("dots", text=f"Thinking... (step {iteration + 1})"),
                   console=console, refresh_per_second=4) as live:
            response = llm.create_chat_completion(**call_kwargs)

        choice = response["choices"][0]
        message = choice["message"]
        finish_reason = choice.get("finish_reason", "unknown")

        # Debug: show what the LLM returned
        has_tool_calls = bool(message.get("tool_calls"))
        content_preview = (message.get("content") or "")[:80]
        console.print(
            f"[dim]  finish_reason={finish_reason}, "
            f"tool_calls={has_tool_calls}, "
            f"content={'repr: ' + repr(content_preview) if content_preview else '(empty)'}[/dim]"
        )
        console.print(f"[dim]  raw message keys: {list(message.keys())}[/dim]")
        if message.get("content"):
            console.print(f"[dim]  raw content: {repr(message['content'][:200])}[/dim]")

        tool_calls = message.get("tool_calls")

        # Fallback: model emitted the tool call as a JSON block in content
        if not tool_calls:
            tool_calls = _parse_inline_tool_calls(message.get("content", ""))
            if tool_calls:
                console.print(f"[dim]  (parsed {len(tool_calls)} inline tool call(s) from content)[/dim]")
                message = {"role": "assistant", "content": None, "tool_calls": tool_calls}

        if tool_calls:
            messages.append(message)

            for tool_call in tool_calls:
                fn_name = tool_call["function"]["name"]
                fn_args_raw = tool_call["function"]["arguments"]

                if isinstance(fn_args_raw, str):
                    try:
                        fn_args = json.loads(fn_args_raw)
                    except json.JSONDecodeError:
                        fn_args = {}
                else:
                    fn_args = fn_args_raw

                console.print(f"  [dim]tool: {fn_name}({_format_args(fn_args)})[/dim]")

                if mcp_client and mcp_client.is_connected:
                    result = mcp_client.call_tool(fn_name, fn_args)
                    result_preview = str(result)[:120]
                    console.print(f"  [dim]result: {result_preview}{'...' if len(str(result)) > 120 else ''}[/dim]")
                else:
                    result = f"Error: MCP client not connected, cannot call tool '{fn_name}'"

                # Ensure result is a non-empty string
                if not result:
                    result = "(empty result)"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", f"call_{iteration}"),
                    "content": str(result)
                })

            continue

        final_text = message.get("content", "")
        if final_text:
            messages.append({"role": "assistant", "content": final_text})
            return final_text

        # LLM returned empty content with no tool calls — nudge it
        console.print("[dim]Empty response from LLM, nudging...[/dim]")
        messages.append(message)
        messages.append({
            "role": "user",
            "content": (
                "You must respond. If you need filesystem information, call the appropriate tool "
                "(e.g. list_directory, read_file). Otherwise provide your answer now."
            )
        })
        continue

    # Exhausted iterations — force a final text answer without tools
    console.print("[dim]Reached tool call limit, generating final answer...[/dim]")
    messages.append({
        "role": "user",
        "content": "Please provide your final answer now based on what you have learned."
    })
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        stream=False
    )
    final_text = response["choices"][0]["message"].get("content", "")
    messages.append({"role": "assistant", "content": final_text})
    return final_text


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
