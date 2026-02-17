import os


def _load_context_md():
    """Load CONTEXT.md from the current directory if it exists."""
    path = os.path.join(os.getcwd(), "CONTEXT.md")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            content = f.read()
        # Truncate if very large to keep system message reasonable
        max_len = 4000
        if len(content) > max_len:
            content = content[:max_len] + "\n\n[... truncated ...]"
        return content
    except Exception:
        return None


def build_system_message():
    base = (
        "You are an expert coding assistant with access to filesystem tools.\n"
        "IMPORTANT: You MUST call tools to answer any question about files or directories. "
        "Never answer from memory or guess file contents. "
        "For listing files, call list_directory. For reading files, call read_file. "
        "For searching code, call search_files. "
        "Always call a tool first, then summarize the result in your answer."
    )

    context_md = _load_context_md()
    if context_md:
        base += (
            "\n\nHere is project context from CONTEXT.md:\n"
            f"<context>\n{context_md}\n</context>"
        )

    return {"role": "system", "content": base}


def build_edit_system_message():
    return {
        "role": "system",
        "content": (
            "You are an expert coding assistant that edits code files. "
            "IMPORTANT: Always call read_file first to read the target file, then call write_file to apply your changes. "
            "Never guess file contents — read them first. Use tools, then summarize what you did."
        )
    }


def build_user_message(prompt, file_contents):
    """Build the user message, optionally with pre-loaded file contents."""
    if not file_contents:
        return {"role": "user", "content": prompt}

    context_parts = []
    for file_path, content in file_contents.items():
        context_parts.append(f"<file path='{file_path}'>\n{content}\n</file>")
    context = "\n\n".join(context_parts)

    full_content = (
        f"The user has pre-loaded the following files for reference:\n\n"
        f"{context}\n\n"
        f"User request: {prompt}"
    )
    return {"role": "user", "content": full_content}


def build_messages(prompt, file_contents, history=None):
    """Build the full message list for the LLM."""
    messages = [build_system_message()]
    if history:
        messages.extend(history)
    messages.append(build_user_message(prompt, file_contents))
    return messages
