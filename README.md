# Local Coder

A Claude Code-inspired AI coding assistant that runs entirely on your local machine. Powered by llama.cpp and local language models, Local Coder provides an interactive CLI for asking coding questions, getting explanations, and editing code files—all without sending your code to external servers.

## Features

- 🤖 **Local AI Model**: Runs completely offline using llama.cpp with GGUF models
- 💬 **Interactive Chat**: Multi-turn conversations with streaming markdown-rendered responses
- 🔧 **Agentic Tool Use**: The model can read files, list directories, and search code autonomously via MCP
- 📝 **Code Editing**: Request code changes—the model reads target files itself before applying edits
- 📁 **File Context**: Reference files in your prompts using `@filename` syntax
- 🎨 **Beautiful Terminal UI**: Syntax-highlighted markdown rendering with code blocks
- ⚡ **GPU Acceleration**: Leverages your GPU for fast inference
- 💾 **Persistent Config**: Model settings saved to `~/.local-coder/config.json`
- 📄 **CONTEXT.md Auto-Injection**: Generate a project context file the model reads automatically
- 🔌 **MCP Server Mode**: Expose local-coder as an MCP tool server for Claude Code, Cursor, and other agents

## Prerequisites

- Python 3.10 or higher
- Node.js (required for MCP filesystem integration)
- [Optional but recommended] CUDA-compatible GPU for faster inference
- ~5GB disk space for the default model

## Installation

You can run Local Coder either natively or using Docker.

### Option A: Native Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/local-coder.git
cd local-coder
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you have a CUDA-compatible GPU, install llama-cpp-python with GPU support:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

For Metal (macOS with Apple Silicon):

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

#### 3. Download a Model

Download a GGUF model file and put it in the root of the project. The example in this project uses the Qwen2.5-Coder 7B model

```bash
# Using huggingface-cli (recommended)
pip install huggingface-hub
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct-GGUF \
  qwen2.5-coder-7b-instruct-q4_k_m.gguf \
  --local-dir . \
  --local-dir-use-symlinks False
```

Or download manually from [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF).

**Alternative Models**: You can use any GGUF model. Popular options include:
- [Qwen2.5-Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF) (recommended for coding)
- [DeepSeek-Coder](https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF)
- [CodeLlama](https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF)

Whichever gguf model you choose to use, make sure to not commit it to the repo. It should be gitignored

#### 4. Set Your Model Path

```bash
python main.py models --set ./your-model-name.gguf
```

This saves the path to `~/.local-coder/config.json` so you don't need to configure it again.

### Option B: Docker Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/local-coder.git
cd local-coder
```

#### 2. Download a Model

Download your GGUF model file (see step 3 in Native Installation above).

#### 3. Build the Docker Image

```bash
docker build -t local-coder .
```

#### 4. Run with Docker

**Interactive chat:**
```bash
docker run -it --rm \
  -v $(pwd)/your-model.gguf:/app/models/model.gguf:ro \
  local-coder
```

**Ask a question:**
```bash
docker run -it --rm \
  -v $(pwd)/your-model.gguf:/app/models/model.gguf:ro \
  local-coder ask "your question"
```

**For GPU support and more Docker options, see [DOCKER.md](DOCKER.md)**

## Usage

### Interactive Chat Mode

Start an interactive session for multi-turn conversations:

```bash
python main.py chat
```

**Example session:**
```
Connecting to MCP filesystem server...
MCP connected (11 tools available)
Starting interactive chat session. Type /exit to quit.

You: How does the agentic loop in @agent.py work?
Assistant: [reads the file via MCP, then explains it]

You: /exit
Goodbye!
```

**Features in chat mode:**
- Reference files with `@filename` syntax (pre-loads file contents)
- Responses are rendered as formatted markdown
- The model can also call filesystem tools itself via MCP
- Conversation history is kept for up to 10 turns
- Use `--no-mcp` to disable MCP and run without filesystem tools

**Slash commands (type these at the `You:` prompt):**

| Command | Description |
|---------|-------------|
| `/exit` | Quit the chat session |
| `/model` | Show the current model and interactively switch to another GGUF file |
| `/md` | Explore the project and generate a `CONTEXT.md` file the model reads on startup |

#### `/model` — Switch models mid-session

```
You: /model

Current model: qwen2.5-coder-7b-instruct-q4_k_m.gguf
  Path: /Users/you/local-coder/qwen2.5-coder-7b-instruct-q4_k_m.gguf

Available GGUF models in current directory:
  1. codellama-7b.gguf

Enter a path to a .gguf file to switch models, or press Enter to keep the current model:
> 1
```

You can enter a number from the list, type a file path directly, or press Enter to keep the current model.

#### `/md` — Generate CONTEXT.md

```
You: /md

Generating CONTEXT.md by exploring the project...
Reading project files...
Generating CONTEXT.md content...

[preview of generated markdown]

Write CONTEXT.md? [y/N]: y
CONTEXT.md has been created. It will be auto-injected into future prompts.
```

Once `CONTEXT.md` exists in the project root, its contents are automatically included in the system prompt for every `ask`, `chat`, and `edit` call so the model has persistent project context.

### One-Shot Questions

Ask a single question without entering interactive mode:

```bash
python main.py ask "What's the difference between a list and a tuple?"
```

**With file context:**
```bash
python main.py ask "Explain what @main.py does"
```

**Disable MCP for a quick offline answer:**
```bash
python main.py ask "What is a decorator?" --no-mcp
```

The model will call filesystem tools as needed to answer questions about your codebase.

### Editing Code Files

Request changes to a code file. The model reads the file automatically before making changes:

```bash
python main.py edit "Add error handling to @helpers.py"
```

**Edit command options:**
- `--max-tokens N` or `-n N`: Set max response length (default: 2048)

**Example workflow:**
```bash
# The model reads helpers.py via MCP, applies changes, then summarizes what it did
python main.py edit "Add docstrings to all functions in @helpers.py"

# Larger changes may need more tokens
python main.py edit "Refactor @agent.py to use async/await throughout" -n 4096
```

### Managing Models

View the currently configured model and switch to a different one:

```bash
# Show current model configuration
python main.py models

# Set a different model (persisted to ~/.local-coder/config.json)
python main.py models --set ./codellama-7b.Q4_K_M.gguf
```

## Configuration

### Persistent Config File

Model settings are stored at `~/.local-coder/config.json`:

```json
{
  "model_path": "/path/to/your-model.gguf",
  "n_ctx": 8192,
  "n_gpu_layers": -1
}
```

Use `python main.py models --set <path>` to update the model path. Edit the JSON file directly to change `n_ctx` or `n_gpu_layers`.

### Adjusting Context Window

Edit `n_ctx` in `~/.local-coder/config.json`:

```json
{
  "n_ctx": 16384
}
```

Larger values allow longer conversations and bigger files, but use more VRAM.

### GPU Layers

Control how many model layers run on GPU by editing `n_gpu_layers`:

```json
{ "n_gpu_layers": -1 }   // All layers on GPU (recommended)
{ "n_gpu_layers": 0  }   // CPU only
{ "n_gpu_layers": 20 }   // First 20 layers on GPU, rest on CPU
```

### Response Length

Control response length per command:

```bash
python main.py chat --max-tokens 1024        # Longer responses in chat
python main.py ask "question" --max-tokens 256  # Shorter responses
```

## File Reference Syntax

Reference files in your prompts using `@`:

```bash
# Single file
python main.py ask "Explain @main.py"

# Multiple files
python main.py ask "How do @main.py and @helpers.py work together?"

# With paths
python main.py ask "Review @src/utils/parser.py for bugs"
```

The tool automatically:
- Reads the file contents
- Injects them into the prompt with proper formatting
- Handles missing files gracefully with warnings

In addition to `@filename` syntax, the model can also browse your filesystem autonomously using MCP tools—so you can ask "what files are in this project?" without pre-loading anything.

## MCP Filesystem Integration

Local Coder connects to `@modelcontextprotocol/server-filesystem` as a subprocess, giving the model a set of filesystem tools it can call during inference:

| Tool | What the model uses it for |
|------|---------------------------|
| `read_file` | Read any file before answering or editing |
| `list_directory` | Browse the project structure |
| `search_files` | Find code patterns across files |
| `write_file` | Apply edits to files |
| `create_directory`, `move_file`, etc. | Other filesystem operations |

The server is started automatically when you run `ask`, `chat`, or `edit`. Pass `--no-mcp` to skip it:

```bash
python main.py chat --no-mcp
python main.py ask "quick math question" --no-mcp
```

**Node.js is required** for MCP. If Node is not installed, the server will be unavailable and the model will fall back to answering without filesystem access.

## Skills (Use Local Coder from Claude Code or Cursor)

Local Coder ships with a skill package (`local-coder.skill`) that lets you use your local model as a tool from Claude Code, Cursor, or any other MCP-compatible agent.

### What the skill exposes

Once the MCP server is running, it provides five tools to the host agent:

| Tool | Description |
|------|-------------|
| `ask` | Single-turn Q&A with optional file context |
| `chat` | Stateful multi-turn conversation (use `session_id` to continue) |
| `edit` | Code editing—model reads files, applies changes, returns a summary |
| `get_model` | Inspect current model path, context size, GPU layers |
| `set_model` | Switch to a different GGUF model file |

### Starting the MCP server

```bash
/path/to/local-coder/llm/bin/python /path/to/local-coder/skills/local-coder/scripts/server.py
```

### Configuring in Claude Code

Add to `~/.claude/settings.json` (or a project-level `.claude/settings.json`):

```json
{
  "mcpServers": {
    "local-coder": {
      "command": "/absolute/path/to/local-coder/llm/bin/python",
      "args": ["/absolute/path/to/local-coder/skills/local-coder/scripts/server.py"],
      "cwd": "/absolute/path/to/local-coder"
    }
  }
}
```

Restart Claude Code—the `ask`, `chat`, `edit`, `get_model`, and `set_model` tools will appear in the available tools list.

### Configuring in Cursor

Add to `.cursor/mcp.json` in your workspace (or `~/.cursor/mcp.json` globally):

```json
{
  "mcpServers": {
    "local-coder": {
      "command": "/absolute/path/to/local-coder/llm/bin/python",
      "args": ["/absolute/path/to/local-coder/skills/local-coder/scripts/server.py"],
      "cwd": "/absolute/path/to/local-coder"
    }
  }
}
```

For a full setup walkthrough see `skills/local-coder/references/setup.md`.

## Troubleshooting

### Model Loading Issues

**Problem**: `FileNotFoundError: Model file not found`

**Solution**: Run `python main.py models` to check the configured path, then use `python main.py models --set ./your-model.gguf` to update it.

### MCP Not Connecting

**Problem**: `MCP unavailable — no tools will be available`

**Solution**: Install Node.js. The MCP filesystem server is an npm package that requires Node to run.

### GPU Not Being Used

**Problem**: Inference is slow despite having a GPU

**Solution**: Reinstall llama-cpp-python with GPU support:
```bash
# For NVIDIA/CUDA
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# For Apple Silicon
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Out of Memory Errors

**Problem**: `CUDA out of memory` or system crashes

**Solution**:
- Reduce `n_ctx` in `~/.local-coder/config.json`
- Use a smaller quantized model (e.g., Q4_K_M instead of Q6_K)
- Reduce `n_gpu_layers` to offload some layers to CPU

### Slow Response Times

**Solution**:
- Ensure GPU acceleration is enabled
- Use a smaller model
- Reduce `max_tokens` for shorter responses
- Lower quantization (Q4_K_M is faster than Q6_K)

## Architecture

### Project Structure

```
local-coder/
├── main.py              # CLI entry point (ask, chat, edit, models commands + slash commands)
├── agent.py             # Agentic tool-calling loop (up to 10 iterations)
├── helpers.py           # Utility functions (file parsing, @reference handling)
├── prompt_builder.py    # LLM prompt construction (injects CONTEXT.md automatically)
├── config.py            # Persistent config (~/.local-coder/config.json)
├── mcp_client.py        # MCP filesystem client (wraps @modelcontextprotocol/server-filesystem)
├── tools.py             # Built-in tool implementations (read_file, list_directory, etc.)
├── skills/
│   └── local-coder/
│       ├── SKILL.md             # Skill metadata and usage guide
│       ├── references/
│       │   └── setup.md         # Agent setup guide (Claude Code, Cursor, etc.)
│       └── scripts/
│           └── server.py        # MCP server entry point for use by other agents
├── local-coder.skill    # Packaged skill archive
├── requirements.txt     # Python dependencies
└── *.gguf               # Your downloaded model file
```

### How It Works

1. **User Input**: Commands are parsed by Typer in `main.py`
2. **File References**: `@filename` patterns are extracted and files are loaded by `helpers.py`
3. **Prompt Building**: Context, CONTEXT.md, and instructions are formatted by `prompt_builder.py`
4. **MCP Connection**: `mcp_client.py` starts the filesystem server as a subprocess and discovers tools
5. **Agentic Loop**: `agent.py` calls the LLM repeatedly until it produces a final text answer (no more tool calls), up to 10 iterations
6. **Tool Calls**: When the LLM calls a filesystem tool, `mcp_client.py` forwards it to the MCP server and returns the result
7. **Output Rendering**: Rich library renders the final markdown response

### Agentic Loop

The model operates in a loop rather than a single shot:

```
User message → LLM → tool call? → execute tool → append result → LLM → tool call? → ... → final text answer
```

This allows the model to read files, search for patterns, or list directories before answering—no need to pre-load everything with `@filename`.

## Contributing

Contributions are welcome! Areas for improvement:

- Support for more model formats
- Configuration file for settings
- Conversation history persistence
- Better error messages
- Additional commands (code search, refactoring, etc.)
- Unit tests

As you can tell from the CLAUDE.md file in the gitignore, I used AI to help me make this project. Feel free to use AI tools to help you contribute to the repo. Whatever you use, make sure to test properly on your local before you push changes.

## License

MIT License - feel free to use this project for any purpose.

## Acknowledgments

- Built with [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- Inspired by [Claude Code](https://claude.ai/code)
- Terminal UI powered by [Rich](https://github.com/Textualize/rich)
- CLI framework: [Typer](https://github.com/tiangolo/typer)
- Filesystem tools via [MCP](https://modelcontextprotocol.io)

## Privacy

Your code never leaves your machine. All inference happens locally using your hardware.
