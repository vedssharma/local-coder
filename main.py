from prompt_builder import build_messages, build_edit_system_message, build_user_message
from helpers import parse_file_references
from agent import run_agent_loop
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


def _confirm_write(path, content):
    """Ask user before writing a file."""
    typer.echo(f"\nThe assistant wants to write to: {path}")
    typer.echo(f"  ({len(content)} characters)")
    return typer.confirm("Allow this write?")


def _confirm_write_or_dry_run(path, content, dry_run):
    """For edit command: show preview or ask confirmation."""
    if dry_run:
        typer.echo(f"\n{'='*60}")
        typer.echo(f"Would write: {path}")
        typer.echo(f"{'='*60}")
        typer.echo(content)
        return False
    typer.echo(f"\nThe assistant wants to write to: {path}")
    return typer.confirm("Apply this change?")


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


@app.command()
def ask(
    prompt: str = typer.Argument(..., help="Coding question"),
    max_tokens: int = typer.Option(512, "--max-tokens", "-n", help="Max number of new tokens to generate")
):
    """Ask a coding question with optional file references using @file syntax"""
    console = Console()
    original_prompt, file_contents = parse_file_references(prompt)
    messages = build_messages(original_prompt, file_contents)

    final_answer = run_agent_loop(
        llm=get_llm(),
        messages=messages,
        console=console,
        max_tokens=max_tokens,
        require_write_confirmation=_confirm_write
    )

    console.print()
    console.print(Markdown(final_answer))
    console.print()


@app.command()
def chat(
    max_tokens: int = typer.Option(512, "--max-tokens", "-n", help="Max number of new tokens to generate")
):
    """Start an interactive chat session. Type /exit to quit."""
    console = Console()
    typer.echo("Starting interactive chat session. Type /exit to quit.\n")

    history = []

    while True:
        try:
            prompt = typer.prompt("\nYou")

            if prompt.strip().lower() == "/exit":
                typer.echo("Goodbye!")
                break

            if prompt.strip().lower() == "/model":
                handle_model_command()
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
                require_write_confirmation=_confirm_write
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


@app.command()
def edit(
    prompt: str = typer.Argument(..., help="Edit request (use @file syntax to reference files)"),
    max_tokens: int = typer.Option(2048, "--max-tokens", "-n", help="Max number of tokens"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show changes without applying them"),
    apply_changes: bool = typer.Option(False, "--apply", "-y", help="Apply changes without confirmation")
):
    """Request code changes. Reference files with @file syntax."""
    console = Console()
    original_prompt, file_contents = parse_file_references(prompt)

    messages = [build_edit_system_message()]
    messages.append(build_user_message(original_prompt, file_contents))

    if apply_changes:
        confirm_fn = None
    else:
        confirm_fn = lambda p, c: _confirm_write_or_dry_run(p, c, dry_run)

    typer.echo("Generating changes...\n")
    final_answer = run_agent_loop(
        llm=get_llm(),
        messages=messages,
        console=console,
        max_tokens=max_tokens,
        require_write_confirmation=confirm_fn
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
