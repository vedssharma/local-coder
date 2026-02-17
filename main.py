from prompt_builder import build_messages, build_edit_system_message, build_user_message
from helpers import parse_file_references
from agent import run_agent_loop
from mcp_client import MCPClient
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
    if _mcp_client is None:
        _mcp_client = MCPClient()
        typer.echo("Connecting to MCP filesystem server...")
        _mcp_client.connect()
        if _mcp_client.is_connected:
            typer.echo(f"MCP connected ({len(_mcp_client.tool_names)} tools available)")
        else:
            typer.echo("MCP unavailable — no tools will be available")
    return _mcp_client

def get_llm():
    """Lazy load the LLM model."""
    global llm
    if llm is None:
        model_config = config.get_model_config()
        llm = Llama(
            model_path=model_config["model_path"],
            n_ctx=model_config["n_ctx"],
            n_gpu_layers=model_config["n_gpu_layers"],
            verbose=False
        )
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
        llm = Llama(model_path=abs_path, n_ctx=8192, n_gpu_layers=-1, verbose=False)
        config.set_model_path(abs_path)
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
        p = Path(fname)
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


@app.command()
def ask(
    prompt: str = typer.Argument(..., help="Coding question"),
    max_tokens: int = typer.Option(512, "--max-tokens", "-n", help="Max number of new tokens to generate"),
    no_mcp: bool = typer.Option(False, "--no-mcp", help="Disable MCP filesystem server")
):
    """Ask a coding question with optional file references using @file syntax"""
    console = Console()
    original_prompt, file_contents = parse_file_references(prompt)
    messages = build_messages(original_prompt, file_contents)

    mcp = None if no_mcp else get_mcp_client()
    final_answer = run_agent_loop(
        llm=get_llm(),
        messages=messages,
        console=console,
        max_tokens=max_tokens,
        mcp_client=mcp
    )

    console.print()
    console.print(Markdown(final_answer))
    console.print()


@app.command()
def chat(
    max_tokens: int = typer.Option(512, "--max-tokens", "-n", help="Max number of new tokens to generate"),
    no_mcp: bool = typer.Option(False, "--no-mcp", help="Disable MCP filesystem server")
):
    """Start an interactive chat session. Type /exit to quit."""
    console = Console()
    typer.echo("Starting interactive chat session. Type /exit to quit.\n")

    mcp = None if no_mcp else get_mcp_client()
    history = []

    try:
        while True:
            try:
                prompt = typer.prompt("\nYou")

                if prompt.strip().lower() == "/exit":
                    typer.echo("Goodbye!")
                    break

                if prompt.strip().lower() == "/model":
                    handle_model_command()
                    continue

                if prompt.strip().lower() == "/md":
                    handle_md_command(console, max_tokens)
                    continue

                if not prompt.strip():
                    continue

                original_prompt, file_contents = parse_file_references(prompt)
                messages = build_messages(original_prompt, file_contents, history=history)

                typer.echo("\nAssistant:")
                final_answer = run_agent_loop(
                    llm=get_llm(),
                    messages=messages,
                    console=console,
                    max_tokens=max_tokens,
                    mcp_client=mcp
                )

                console.print(Markdown(final_answer))
                console.print()

                # Store only user/assistant messages for history (not tool call intermediates)
                history.append({"role": "user", "content": original_prompt})
                history.append({"role": "assistant", "content": final_answer})

                # Trim history to last 10 turns to manage context window
                if len(history) > 20:
                    history = history[-20:]

            except (KeyboardInterrupt, EOFError):
                typer.echo("\n\nGoodbye!")
                break
    finally:
        if mcp and mcp.is_connected:
            mcp.close()


@app.command()
def edit(
    prompt: str = typer.Argument(..., help="Edit request (use @file syntax to reference files)"),
    max_tokens: int = typer.Option(2048, "--max-tokens", "-n", help="Max number of tokens"),
):
    """Request code changes. Reference files with @file syntax."""
    console = Console()
    original_prompt, file_contents = parse_file_references(prompt)

    messages = [build_edit_system_message()]
    messages.append(build_user_message(original_prompt, file_contents))

    mcp = get_mcp_client()
    typer.echo("Generating changes...\n")
    final_answer = run_agent_loop(
        llm=get_llm(),
        messages=messages,
        console=console,
        max_tokens=max_tokens,
        mcp_client=mcp
    )

    if final_answer:
        console.print(Markdown(final_answer))
        console.print()

@app.command()
def models(
    set_model: str = typer.Option(None, "--set", "-s", help="Path to GGUF model file to use")
):
    """Show current model or set a new model."""
    console = Console()

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


if __name__ == "__main__":
    app()
