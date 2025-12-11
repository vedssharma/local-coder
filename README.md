# Local Coder

A Claude Code-inspired AI coding assistant that runs entirely on your local machine. Powered by llama.cpp and local language models, Local Coder provides an interactive CLI for asking coding questions, getting explanations, and editing code files—all without sending your code to external servers.

## Features

- 🤖 **Local AI Model**: Runs completely offline using llama.cpp with GGUF models
- 💬 **Interactive Chat**: Multi-turn conversations with streaming markdown-rendered responses
- 📝 **Code Editing**: Request code changes with preview and confirmation workflow
- 📁 **File Context**: Reference files in your prompts using `@filename` syntax
- 🎨 **Beautiful Terminal UI**: Syntax-highlighted markdown rendering with code blocks
- ⚡ **GPU Acceleration**: Leverages your GPU for fast inference

## Prerequisites

- Python 3.10 or higher
- [Optional but recommended] CUDA-compatible GPU for faster inference
- ~5GB disk space for the default model

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/local-coder.git
cd local-coder
```

### 2. Install Dependencies

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

### 3. Download a Model

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

### 4. Update Model Path

Edit `main.py` and update the `model_path` to point to your downloaded model:

```python
llm = Llama(
    model_path="./your-model-name.gguf",  # Update this path
    n_ctx=8192,
    n_gpu_layers=-1
)
```

## Usage

### Interactive Chat Mode

Start an interactive session for multi-turn conversations:

```bash
python main.py chat
```

**Example session:**
```
Starting interactive chat session. Type /exit to quit.

You: How do I read a file in Python?
Assistant: You can read a file using the `open()` function...

[Markdown rendered output appears here]

You: /exit
Goodbye!
```

**Features in chat mode:**
- Type `/exit` to quit
- Reference files with `@filename` syntax
- Responses are rendered as formatted markdown
- Streaming output with real-time updates

### One-Shot Questions

Ask a single question without entering interactive mode:

```bash
python main.py ask "What's the difference between a list and a tuple?"
```

**With file context:**
```bash
python main.py ask "Explain what @main.py does"
```

The `@filename` syntax loads the file content and includes it in your prompt.

### Editing Code Files

Request changes to code files with an interactive workflow:

```bash
python main.py edit "Add error handling to @helpers.py" --dry-run
```

**Edit command options:**
- `--dry-run`: Preview changes without applying them
- `--apply` or `-y`: Apply changes without confirmation
- `--max-tokens N` or `-n N`: Set max response length (default: 2048)

**Example workflow:**
```bash
# Preview changes first
python main.py edit "Add docstrings to all functions in @helpers.py" --dry-run

# Apply changes with confirmation
python main.py edit "Add docstrings to all functions in @helpers.py"

# Auto-apply without confirmation
python main.py edit "Add docstrings to all functions in @helpers.py" --apply
```

## Configuration

### Adjusting Context Window

Edit the `n_ctx` parameter in `main.py` to change the context window size:

```python
llm = Llama(
    model_path="./your-model.gguf",
    n_ctx=8192,  # Increase for longer contexts (uses more VRAM)
    n_gpu_layers=-1
)
```

### GPU Layers

Control how many model layers run on GPU:

```python
n_gpu_layers=-1  # All layers on GPU (recommended)
n_gpu_layers=0   # CPU only
n_gpu_layers=20  # First 20 layers on GPU, rest on CPU
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

## Troubleshooting

### Model Loading Issues

**Problem**: `FileNotFoundError: Model file not found`

**Solution**: Verify the model path in `main.py` matches your downloaded model file name.

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
- Reduce `n_ctx` (context window) in `main.py`
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
├── main.py              # CLI entry point with Typer commands
├── helpers.py           # Utility functions (file parsing, edit handling)
├── prompt_builder.py    # LLM prompt construction
├── requirements.txt     # Python dependencies
└── *.gguf              # Your downloaded model file
```

### How It Works

1. **User Input**: Commands are parsed by Typer in `main.py`
2. **File References**: `@filename` patterns are extracted and files are loaded by `helpers.py`
3. **Prompt Building**: Context and instructions are formatted by `prompt_builder.py`
4. **LLM Inference**: llama.cpp processes the prompt and streams tokens
5. **Output Rendering**: Rich library renders markdown in real-time for chat mode

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

## Privacy

Your code never leaves your machine. All inference happens locally using your hardware.
