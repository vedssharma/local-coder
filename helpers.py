from pathlib import Path
import re
import typer


def parse_edit_blocks(llm_output: str) -> dict[str, str]:
    """Parse edit blocks from LLM output in the format:
    ```filepath
    content
    ```
    """
    edits = {}
    pattern = r'```([\w/.]+)\n(.*?)```'
    matches = re.findall(pattern, llm_output, re.DOTALL)

    for file_path, content in matches:
        edits[file_path] = content

    return edits


def apply_edits(edits: dict[str, str], dry_run: bool = False):
    """Apply the edits to files or just preview them."""
    for file_path, content in edits.items():
        if dry_run:
            typer.echo(f"\n{'='*60}")
            typer.echo(f"Would edit: {file_path}")
            typer.echo(f"{'='*60}")
            typer.echo(content)
        else:
            try:
                path = Path(file_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                typer.echo(f"✓ Updated: {file_path}")
            except Exception as e:
                typer.echo(f"✗ Failed to update {file_path}: {e}", err=True)


def parse_file_references(prompt: str) -> tuple[str, dict[str, str]]:
    pattern = r'@([^\s]+)'
    matches = re.findall(pattern, prompt)
    file_contents = {}
    for file_path in matches:
        path = Path(file_path)
        if not path.exists():
            typer.echo(f"Warning: File not found: {file_path}", err=True)
            continue
        if not path.is_file():
            typer.echo(f"Warning: Not a file: {file_path}", err=True)
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                file_contents[file_path] = f.read()
        except Exception as e:
            typer.echo(f"Warning: Could not read {file_path}: {e}", err=True)
    return prompt, file_contents
