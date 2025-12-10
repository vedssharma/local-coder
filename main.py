from prompt_builder import build_prompt_with_context, build_edit_prompt
from helpers import parse_edit_blocks, apply_edits, parse_file_references
import typer
from llama_cpp import Llama
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live

app = typer.Typer()

llm = Llama(
    model_path="./Qwen_Qwen2.5-Coder-7B-Instruct-GGUF_qwen2.5-coder-7b-instruct-q4_k_m.gguf",
    n_ctx=8192,
    n_gpu_layers=-1
)

@app.command()
def ask(
    prompt: str = typer.Argument(..., help="Coding question"),
    max_tokens: int = typer.Option(512, "--max-tokens", "-n", help="Max number of new tokens to generate")
):
    """Ask a coding question with optional file references using @file syntax"""
    original_prompt, file_contents = parse_file_references(prompt)
    user_prompt = build_prompt_with_context(original_prompt, file_contents)
    stream = llm(user_prompt, max_tokens=max_tokens, stream=True)
    for output in stream:
        print(output["choices"][0]["text"], end="", flush=True)
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
            stream = llm(user_prompt, max_tokens=max_tokens, stream=True)

            with Live(Markdown(""), console=console, refresh_per_second=10) as live:
                for output in stream:
                    response_text += output["choices"][0]["text"]
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
    response = llm(edit_prompt, max_tokens=max_tokens, stream=False)
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


if __name__ == "__main__":
    app()
