

def build_prompt_with_context(prompt: str, file_contents: dict[str, str]) -> str:
    if not file_contents:
        return f"You are an expert and helpful coding assistant. User: {prompt} Assistant:"
    context_parts = []
    for file_path, content in file_contents.items():
        context_parts.append(f"<file path='{file_path}'>\n{content}\n</file>")
    context = "\n\n".join(context_parts)
    full_prompt = f"You are an expert and helpful coding assistant. The user has provided the following files for context:\n\n{context}\n\nUser: {prompt}\n\nAssistant:"
    return full_prompt

def build_edit_prompt(prompt: str, file_contents: dict[str, str]) -> str:
    context_parts = []
    for file_path, content in file_contents.items():
        context_parts.append(f"<file path='{file_path}'>\n{content}\n</file>")
    context = "\n\n".join(context_parts)

    prompt = "You are an expert coding assistant that helps edit code files.\n\n"
    prompt += context + "\n\n"
    prompt += f"User request: {prompt}\n\n"
    prompt += "Instructions:\n"
    prompt += "1. Analyze the request and provided files\n"
    prompt += "2. Generate the complete modified content for each file that needs changes\n"
    prompt += "3. Output in this format:\n\n"
    prompt += "<edit file=\"path/to/file\">\n"
    prompt += "[Complete new content]\n"
    prompt += ""