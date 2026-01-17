from prompt_builder import build_prompt_with_context, build_edit_prompt
from helpers import parse_edit_blocks, apply_edits, parse_file_references
import config
import typer
from llama_cpp import Llama
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner
import os

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
            n_gpu_layers=model_config["n_gpu_layers"]
        )
    return llm

@app.command()
def ask(
    prompt: str = typer.Argument(..., help="Coding question"),
    max_tokens: int = typer.Option(512, "--max-tokens", "-n", help="Max number of new tokens to generate")
):
    """Ask a coding question with optional file references using @file syntax"""
    console = Console()
    original_prompt, file_contents = parse_file_references(prompt)
    user_prompt = build_prompt_with_context(original_prompt, file_contents)

    response_text = ""
    stream = get_llm()(user_prompt, max_tokens=max_tokens, stream=True)

    # Show thinking spinner until first token arrives, then switch to markdown rendering
    first_token = True
    with Live(Spinner("dots", text="Thinking..."), console=console, refresh_per_second=10) as live:
        for output in stream:
            token = output["choices"][0]["text"]
            response_text += token

            # Switch from spinner to markdown after first token
            if first_token:
                first_token = False
                live.update(Markdown(response_text))
            else:
                live.update(Markdown(response_text))

    print()

@app.command()
def chat(
    max_tokens: int = typer.Option(512, "--max-tokens", "-n", help="Max number of new tokens to generate")
):
    """Start an interactive chat session. Type /exit to quit."""
    console = Console()
    typer.echo("Starting interactive chat session. Type /exit to quit.\n")

    while True:
        try:
            prompt = typer.prompt("\nYou")

            if prompt.strip().lower() == "/exit":
                typer.echo("Goodbye!")
                break

            if not prompt.strip():
                continue

            original_prompt, file_contents = parse_file_references(prompt)
            user_prompt = build_prompt_with_context(original_prompt, file_contents)

            typer.echo("\nAssistant:")
            response_text = ""
            stream = get_llm()(user_prompt, max_tokens=max_tokens, stream=True)

            # Show thinking spinner until first token arrives, then switch to markdown rendering
            first_token = True
            with Live(Spinner("dots", text="Thinking..."), console=console, refresh_per_second=10) as live:
                for output in stream:
                    token = output["choices"][0]["text"]
                    response_text += token

                    # Switch from spinner to markdown after first token
                    if first_token:
                        first_token = False
                        live.update(Markdown(response_text))
                    else:
                        live.update(Markdown(response_text))

            print()

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
    original_prompt, file_contents = parse_file_references(prompt)

    if not file_contents:
        typer.echo("Error: No files referenced. Use @filepath to reference files.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Analyzing {len(file_contents)} file(s)...")

    edit_prompt = build_edit_prompt(original_prompt, file_contents)

    typer.echo("Generating changes...\n")
    response = get_llm()(edit_prompt, max_tokens=max_tokens, stream=False)
    llm_output = response["choices"][0]["text"]

    typer.echo("=" * 60)
    typer.echo(llm_output)
    typer.echo("=" * 60)

    edits = parse_edit_blocks(llm_output)

    if not edits:
        typer.echo("\nNo valid edit blocks found in response.", err=True)
        raise typer.Exit(1)

    if dry_run:
        apply_edits(edits, dry_run=True)
        return

    if not apply_changes:
        typer.echo(f"\nFound {len(edits)} file(s) to edit:")
        for file_path in edits.keys():
            typer.echo(f"  - {file_path}")

        confirm = typer.confirm("\nApply these changes?")
        if not confirm:
            typer.echo("Changes not applied.")
            return

    apply_edits(edits, dry_run=False)

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
