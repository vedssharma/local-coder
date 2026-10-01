from prompt_builder import build_messages, build_edit_system_message, build_user_message
from helpers import parse_file_references
from agent import run_agent_loop
from mcp_client import MCPClient
from workspace_tools import WorkspaceTools, MODES
from runtime import Runtime
from model_backend import create_model, EmbeddedModel
from agent import RunBudget
import config
import os
import glob
import typer
from llama_cpp import Llama
from rich.console import Console
from rich.markdown import Markdown

app = typer.Typer()

# Global variable to hold the model instance (lazy loaded)
llm = None

# Global MCP client instance (lazy loaded)
_mcp_client = None


def get_mcp_client():
    """Lazy-initialize and return the MCP filesystem client."""
    global _mcp_client
    if _mcp_client is None or not _mcp_client.is_connected:
        _mcp_client = MCPClient()
        typer.echo("Connecting to MCP filesystem server...")
        _mcp_client.connect()
        if _mcp_client.is_connected:
            typer.echo(f"MCP connected ({len(_mcp_client.tool_names)} tools available)")
        else:
            typer.echo("MCP unavailable; native tools remain available")
    return _mcp_client

def get_llm():
    """Lazy load the LLM model."""
    global llm
    if llm is None:
        llm = create_model(config.get_model_config())
    return llm



def handle_model_command():
    """Handle the /model slash command: show current model and optionally switch."""
    global llm
    current_config = config.get_model_config()
    model_path = current_config["model_path"]
    typer.echo(f"\nCurrent model: {os.path.basename(model_path)}")
    typer.echo(f"  Path: {model_path}")

    # Find available .gguf files in the current directory
    gguf_files = sorted(glob.glob("./*.gguf"))
    other_files = [f for f in gguf_files if os.path.abspath(f) != os.path.abspath(model_path)]

    if other_files:
        typer.echo(f"\nAvailable GGUF models in current directory:")
        for i, f in enumerate(other_files, 1):
            typer.echo(f"  {i}. {os.path.basename(f)}")

    typer.echo(f"\nEnter a path to a .gguf file to switch models, or press Enter to keep the current model:")
    new_path = input("> ").strip()

    if not new_path:
        typer.echo("Keeping current model.\n")
        return

    # Allow selecting by number from the list
    if new_path.isdigit() and other_files:
        idx = int(new_path) - 1
        if 0 <= idx < len(other_files):
            new_path = other_files[idx]
        else:
            typer.echo("Invalid selection.\n")
            return

    if not os.path.isfile(new_path):
        typer.echo(f"Error: File not found: {new_path}\n")
        return

    if not new_path.endswith(".gguf"):
        typer.echo(f"Error: Not a .gguf file: {new_path}\n")
        return

    typer.echo(f"Loading model: {new_path}...")
    try:
        abs_path = os.path.abspath(new_path)
        loaded = Llama(model_path=abs_path, n_ctx=current_config['n_ctx'], n_gpu_layers=current_config['n_gpu_layers'], verbose=False)
        config.set_model_path(abs_path)
        profile = config.get_model_config()
        profile['backend'] = 'embedded'
        config.save_config(profile)
        llm = EmbeddedModel(profile)
        llm._model = loaded
        typer.echo(f"Switched to: {os.path.basename(abs_path)}\n")
    except Exception as e:
        typer.echo(f"Error loading model: {e}\n")


def _gather_project_context():
    """Gather project information by reading key files from disk."""
    from pathlib import Path

    context_parts = []

    # List top-level directory
    cwd = Path(".")
    entries = sorted(cwd.iterdir())
    dir_listing = []
    skip = {".git", "__pycache__", "node_modules", "venv", ".venv", "llm"}
    for entry in entries:
        if entry.name in skip:
            continue
        suffix = "/" if entry.is_dir() else ""
        dir_listing.append(f"  {entry.name}{suffix}")
    context_parts.append("## Directory listing\n" + "\n".join(dir_listing))

    # Read key files (if they exist)
    key_files = [
        "README.md", "requirements.txt", "package.json", "setup.py",
        "pyproject.toml", "Dockerfile", "docker-compose.yml",
        "config.py", "main.py", "CLAUDE.md",
    ]
    max_file_chars = 3000
    for fname in key_files:
        try:
            p = WorkspaceTools().path(fname)
        except ValueError:
            continue
        if p.exists() and p.is_file():
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
                if len(content) > max_file_chars:
                    content = content[:max_file_chars] + "\n... [truncated]"
                context_parts.append(f"## Contents of {fname}\n```\n{content}\n```")
            except Exception:
                pass

    return "\n\n".join(context_parts)


def handle_md_command(console, max_tokens):
    """Handle the /md slash command: explore the project and generate CONTEXT.md."""
    typer.echo("\nGenerating CONTEXT.md by exploring the project...\n")

    md_max_tokens = max(max_tokens, 2048)

    # Step 1: Gather real project data from disk (no LLM needed)
    typer.echo("Reading project files...\n")
    project_data = _gather_project_context()

    # Step 2: Ask the LLM to synthesize into a CONTEXT.md
    generate_messages = [
        {
            "role": "system",
            "content": (
                "You are a technical writer. Generate a markdown document and nothing else. "
                "Do not use any tools. Just output the markdown content directly."
            )
        },
        {
            "role": "user",
            "content": (
                "Based on the following real project files, write the contents of a CONTEXT.md file. "
                "Include these sections:\n"
                "- Project name and one-line description\n"
                "- Tech stack and dependencies\n"
                "- Directory structure overview\n"
                "- Key files and what they do\n"
                "- How to run the project\n"
                "- Architecture notes\n\n"
                "Output ONLY the markdown content, no explanation. "
                "Base everything strictly on the file contents provided below.\n\n"
                f"{project_data}"
            )
        }
    ]

    typer.echo("Generating CONTEXT.md content...\n")
    md_content = run_agent_loop(
        llm=get_llm(),
        messages=generate_messages,
        console=console,
        max_tokens=md_max_tokens
    )

    if not md_content or not md_content.strip():
        typer.echo("Failed to generate CONTEXT.md content.\n")
        return

    # Step 3: Show and write with user confirmation
    console.print(Markdown(md_content))
    console.print()

    if typer.confirm("Write CONTEXT.md?"):
        try:
            with open("CONTEXT.md", "w", encoding="utf-8") as f:
                f.write(md_content)
            typer.echo("CONTEXT.md has been created. It will be auto-injected into future prompts.\n")
        except Exception as e:
            typer.echo(f"Error writing CONTEXT.md: {e}\n")
    else:
        typer.echo("Write cancelled.\n")


def make_runtime(mode, no_mcp, max_steps, max_seconds, token_budget, trace, console, task_kind="auto"):
    if mode not in MODES:
        raise typer.BadParameter('mode must be read-only, workspace-edit, or execute')
    streamed = False
    def emit(event):
        nonlocal streamed
        if event['type'] == 'assistant_delta':
            streamed = True
            console.print(event['text'], end='', markup=False, highlight=False)
        elif event['type'] == 'tool_started':
            console.print(f"Tool: {event['name']}", markup=False)
        elif event['type'] == 'assistant_text' and not streamed:
            console.print(Markdown(event['text']))
        elif event['type'] == 'run_finished':
            console.print()
            if event['status'] != 'completed':
                console.print(f"Run {event['status']}: {event['reason']}", markup=False)
            streamed = False
    return Runtime(get_llm(), os.getcwd(), config.CONFIG_DIR, mode=mode,
        mcp_client=None if no_mcp else get_mcp_client(), emit=emit,
        budget=RunBudget(max_steps, max_seconds, token_budget),
        context_window=config.get_model_config()['n_ctx'], trace=trace, task_kind=task_kind)


def execute_turn(runtime, prompt, max_tokens):
    original, files = parse_file_references(prompt, root=runtime.tools.root)
    result = runtime.turn(original, files, max_tokens)
    typer.echo(f'Session: {runtime.session_id}; outcome: {result.status}')
    if result.status != 'completed':
        typer.echo(result.text)
    return result


@app.command()
def ask(
    prompt: str = typer.Argument(...),
    max_tokens: int = typer.Option(512, '--max-tokens', '-n', min=1),
    no_mcp: bool = typer.Option(True, '--no-mcp/--mcp', help='Native tools are always available; MCP is optional'),
    mode: str = typer.Option('read-only', '--mode'),
    max_steps: int = typer.Option(30, min=1),
    max_seconds: float = typer.Option(300, min=1),
    token_budget: int = typer.Option(8192, min=1),
    trace: bool = typer.Option(False, '--trace'),
    task_kind: str = typer.Option('auto', '--task-kind', help='auto, answer, inspect, code, or all'),
):
    """Ask a question or run a bounded coding task."""
    with make_runtime(mode, no_mcp, max_steps, max_seconds, token_budget, trace, Console(), task_kind) as runtime:
        result = execute_turn(runtime, prompt, max_tokens)
    if result.status != 'completed':
        raise typer.Exit(1)


@app.command()
def chat(
    max_tokens: int = typer.Option(512, '--max-tokens', '-n', min=1),
    no_mcp: bool = typer.Option(True, '--no-mcp/--mcp', help='Native tools are always available; MCP is optional'),
    mode: str = typer.Option('read-only', '--mode'),
    resume: str = typer.Option(None, '--resume'),
    max_steps: int = typer.Option(30, min=1),
    max_seconds: float = typer.Option(300, min=1),
    token_budget: int = typer.Option(8192, min=1),
    trace: bool = typer.Option(False, '--trace'),
    task_kind: str = typer.Option('auto', '--task-kind', help='auto, answer, inspect, code, or all'),
):
    """Chat with persistent tool history; /resume ID, /sessions, /new, /undo, /exit."""
    console = Console()
    with make_runtime(mode, no_mcp, max_steps, max_seconds, token_budget, trace, console, task_kind) as runtime:
        if resume:
            runtime.resume(resume)
        while True:
            try:
                prompt = typer.prompt('You').strip()
                if prompt == '/exit':
                    break
                if prompt == '/sessions':
                    typer.echo('\n'.join(runtime.store.list()) or 'No saved sessions.')
                elif prompt.startswith('/resume '):
                    runtime.resume(prompt.split(maxsplit=1)[1])
                elif prompt == '/new':
                    runtime.new()
                elif prompt == '/undo':
                    typer.echo(runtime.tools.undo_last())
                elif prompt == '/model':
                    handle_model_command()
                    runtime.model = get_llm()
                    from model_backend import ModelAdapter
                    runtime.context.count_tokens = runtime.model.count_tokens if isinstance(runtime.model, ModelAdapter) else runtime.context.count_tokens
                elif prompt == '/md':
                    handle_md_command(console, max_tokens)
                elif prompt:
                    execute_turn(runtime, prompt, max_tokens)
            except (KeyboardInterrupt, EOFError):
                break
            except (OSError, ValueError) as exc:
                typer.echo(str(exc), err=True)


@app.command()
def edit(
    prompt: str = typer.Argument(...),
    max_tokens: int = typer.Option(2048, '--max-tokens', '-n', min=1),
    mode: str = typer.Option('workspace-edit', '--mode'),
    no_mcp: bool = typer.Option(True, '--no-mcp/--mcp', help='Native tools are always available; MCP is optional'),
    max_steps: int = typer.Option(30, min=1),
    max_seconds: float = typer.Option(300, min=1),
    token_budget: int = typer.Option(8192, min=1),
    trace: bool = typer.Option(False, '--trace'),
    task_kind: str = typer.Option('auto', '--task-kind', help='auto, answer, inspect, code, or all'),
):
    """Apply targeted edits; --mode execute also permits validation commands."""
    with make_runtime(mode, no_mcp, max_steps, max_seconds, token_budget, trace, Console(), task_kind) as runtime:
        result = execute_turn(runtime, prompt, max_tokens)
    if result.status != 'completed':
        raise typer.Exit(1)


@app.command()
def undo():
    """Undo the most recent harness patch if the file has not changed since."""
    try:
        typer.echo(WorkspaceTools(mode='workspace-edit').undo_last())
    except (OSError, ValueError) as exc:
        typer.echo(f'Cannot undo: {exc}', err=True)
        raise typer.Exit(1)


@app.command()
def models(
    set_model: str = typer.Option(None, "--set", "-s", help="Path to GGUF model file to use"),
    backend: str = typer.Option(None, '--backend', help='embedded or openai'),
    base_url: str = typer.Option(None, '--base-url'),
    model_name: str = typer.Option(None, '--model-name'),
    context_window: int = typer.Option(None, '--context-window', min=1024),
    chat_format: str = typer.Option(None, '--chat-format'),
    supports_tools: bool = typer.Option(None, '--tools/--no-tools'),
    streaming: bool = typer.Option(None, '--stream/--no-stream'),
    threads: int = typer.Option(None, '--threads', min=1),
    batch_threads: int = typer.Option(None, '--batch-threads', min=1),
    batch_size: int = typer.Option(None, '--batch-size', min=1),
    micro_batch_size: int = typer.Option(None, '--micro-batch-size', min=1),
    flash_attention: bool = typer.Option(None, '--flash-attention/--no-flash-attention'),
    key_cache_type: str = typer.Option(None, '--key-cache-type'),
    value_cache_type: str = typer.Option(None, '--value-cache-type'),
    prompt_cache_mb: int = typer.Option(None, '--prompt-cache-mb', min=0, max=4096),
    server_cache_prompt: bool = typer.Option(None, '--server-cache-prompt/--no-server-cache-prompt'),
):
    """Show current model or set a new model."""
    global llm
    updates = {'backend': backend, 'base_url': base_url, 'model': model_name,
               'n_ctx': context_window, 'chat_format': chat_format,
               'supports_tools': supports_tools, 'stream': streaming,
               'n_threads': threads, 'n_threads_batch': batch_threads, 'n_batch': batch_size,
               'n_ubatch': micro_batch_size, 'flash_attn': flash_attention,
               'type_k': key_cache_type, 'type_v': value_cache_type,
               'prompt_cache_mb': prompt_cache_mb, 'server_cache_prompt': server_cache_prompt}
    updates = {k: v for k, v in updates.items() if v is not None}
    if updates:
        profile = {**config.get_model_config(), **updates}
        try:
            create_model(profile)
        except ValueError as exc:
            raise typer.BadParameter(str(exc))
        config.save_config(profile)
        llm = None
        typer.echo('Model profile updated.')

    if set_model:
        # User wants to change the model
        if not os.path.exists(set_model):
            typer.echo(f"Error: Model file not found: {set_model}", err=True)
            raise typer.Exit(1)

        if not set_model.lower().endswith('.gguf'):
            typer.echo(f"Error: Model file must be a .gguf file", err=True)
            raise typer.Exit(1)

        # Get absolute path
        abs_path = os.path.abspath(set_model)

        # Update configuration
        if config.set_model_path(abs_path):
            profile = config.get_model_config()
            profile['backend'] = 'embedded'
            config.save_config(profile)
            llm = None
            typer.echo(f"✓ Model updated successfully!")
            typer.echo(f"  New model: {abs_path}")
            typer.echo(f"\nNote: Restart the application for the change to take effect.")
        else:
            typer.echo(f"Error: Failed to update model configuration", err=True)
            raise typer.Exit(1)
    else:
        # Show current model
        current_config = config.get_model_config()
        model_path = current_config["model_path"]

        typer.echo("Current Model Configuration:")
        typer.echo(f"  Backend: {current_config['backend']}")
        if current_config['backend'] == 'openai':
            typer.echo(f"  Server: {current_config.get('base_url', 'http://127.0.0.1:8080/v1')}")
            typer.echo(f"  Model: {current_config.get('model', 'local-model')}")
        typer.echo(f"  Model path: {model_path}")
        typer.echo(f"  Context size: {current_config['n_ctx']}")
        typer.echo(f"  GPU layers: {current_config['n_gpu_layers']}")

        # Check if model file exists
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024 * 1024)  # Convert to GB
            typer.echo(f"  File size: {file_size:.2f} GB")
            typer.echo(f"  Status: ✓ Available")
        else:
            typer.echo(f"  Status: ✗ Not found")


@app.command()
def benchmark(
    output: str = typer.Option(..., '--output'),
    prompt: str = typer.Option('Explain why binary search takes logarithmic time.'),
    repeats: int = typer.Option(3, min=1),
    warmups: int = typer.Option(1, min=0),
    max_tokens: int = typer.Option(128, min=1),
):
    """Measure real configured inference; missing models/errors never count as passes."""
    from performance import benchmark as measure
    import json
    from pathlib import Path
    try:
        report = measure(get_llm(), prompt, repeats, warmups, max_tokens)
        Path(output).write_text(json.dumps(report, indent=2) + '\n')
        typer.echo(json.dumps(report['median'], indent=2))
    except (OSError, ValueError) as exc:
        typer.echo(f'Benchmark failed: {exc}', err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
