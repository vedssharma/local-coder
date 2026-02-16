import os
import re
import json
from pathlib import Path

MAX_FILE_SIZE = 100_000
MAX_DIR_ENTRIES = 200
MAX_SEARCH_RESULTS = 50
SKIP_DIRS = {".git", "__pycache__", "node_modules", "venv", "llm", ".venv"}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the given path. Returns the file content as a string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories at the given path. Returns names with a trailing / for directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list. Defaults to current directory."
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for a text pattern in files under a directory. Returns matching lines with file paths and line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Text or regex pattern to search for"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in. Defaults to current directory."
                    },
                    "file_glob": {
                        "type": "string",
                        "description": "Optional glob to filter files, e.g. '*.py'"
                    }
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates the file if it does not exist. Overwrites existing content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    }
                },
                "required": ["path", "content"]
            }
        }
    }
]


def execute_tool(name, arguments, confirm_fn=None):
    """Dispatch to the appropriate tool function. Returns result as a string."""
    if name == "read_file":
        return _read_file(arguments.get("path", ""))
    elif name == "list_directory":
        return _list_directory(arguments.get("path", "."))
    elif name == "search_files":
        return _search_files(
            arguments.get("pattern", ""),
            arguments.get("path", "."),
            arguments.get("file_glob")
        )
    elif name == "write_file":
        return _write_file(
            arguments.get("path", ""),
            arguments.get("content", ""),
            confirm_fn
        )
    else:
        return f"Error: Unknown tool '{name}'"


def _read_file(path):
    try:
        p = Path(path)
        if not p.exists():
            return f"Error: File not found: {path}"
        if not p.is_file():
            return f"Error: Not a file: {path}"
        content = p.read_text(encoding="utf-8", errors="replace")
        if len(content) > MAX_FILE_SIZE:
            return content[:MAX_FILE_SIZE] + f"\n\n... [truncated, file is {len(content)} chars]"
        return content
    except Exception as e:
        return f"Error reading {path}: {e}"


def _list_directory(path):
    try:
        p = Path(path)
        if not p.exists():
            return f"Error: Directory not found: {path}"
        if not p.is_dir():
            return f"Error: Not a directory: {path}"
        entries = sorted(p.iterdir())
        lines = []
        for entry in entries[:MAX_DIR_ENTRIES]:
            if entry.name in SKIP_DIRS:
                continue
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"{entry.name}{suffix}")
        if len(entries) > MAX_DIR_ENTRIES:
            lines.append(f"... and {len(entries) - MAX_DIR_ENTRIES} more entries")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing {path}: {e}"


def _search_files(pattern, path=".", file_glob=None):
    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"

    results = []
    p = Path(path)

    for root, dirs, files in os.walk(p):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            if file_glob and not Path(fname).match(file_glob):
                continue
            fpath = Path(root) / fname
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f, 1):
                        if compiled.search(line):
                            results.append(f"{fpath}:{i}: {line.rstrip()}")
                            if len(results) >= MAX_SEARCH_RESULTS:
                                return "\n".join(results) + f"\n... [capped at {MAX_SEARCH_RESULTS} results]"
            except (OSError, UnicodeDecodeError):
                continue

    if not results:
        return "No matches found."
    return "\n".join(results)


def _write_file(path, content, confirm_fn=None):
    if confirm_fn and not confirm_fn(path, content):
        return "Write cancelled by user."
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to {path}"
    except Exception as e:
        return f"Error writing {path}: {e}"
