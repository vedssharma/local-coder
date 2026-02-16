from pathlib import Path
import re
import typer


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
