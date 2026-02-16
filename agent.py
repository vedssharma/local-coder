import json
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner
from tools import TOOL_SCHEMAS, execute_tool

MAX_AGENT_ITERATIONS = 10


def run_agent_loop(llm, messages, console, max_tokens=512, require_write_confirmation=None):
    """
    Run the agentic tool-calling loop.

    Calls the LLM with tool schemas. If the LLM returns tool calls,
    executes them, appends results, and loops. When the LLM returns
    a text response (no tool calls), returns that as the final answer.
    """
    for iteration in range(MAX_AGENT_ITERATIONS):
        with Live(Spinner("dots", text=f"Thinking... (step {iteration + 1})"),
                   console=console, refresh_per_second=4) as live:
            response = llm.create_chat_completion(
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                max_tokens=max_tokens,
                stream=False
            )

        choice = response["choices"][0]
        message = choice["message"]

        tool_calls = message.get("tool_calls")
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

                result = execute_tool(fn_name, fn_args, require_write_confirmation)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", f"call_{iteration}"),
                    "content": result
                })

            continue

        final_text = message.get("content", "")
        if final_text:
            messages.append({"role": "assistant", "content": final_text})
            return final_text

        break

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


def _format_args(args):
    """Format tool arguments for display, truncating long values."""
    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > 60:
            s = s[:57] + "..."
        parts.append(f"{k}={s!r}")
    return ", ".join(parts)
