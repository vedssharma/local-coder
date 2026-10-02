from prompt_builder import build_messages, build_edit_system_message, build_user_message
from helpers import parse_file_references
from agent import run_agent_loop
from workspace_tools import WorkspaceTools, MODES
from runtime import Runtime
from model_backend import create_model, EmbeddedModel, import_llama
from agent import RunBudget
import config
import os
import glob
import typer
from rich.console import Console
from rich.markdown import Markdown

app = typer.Typer()

# Global variable to hold the model instance (lazy loaded)
llm = None

def get_llm():
    """Lazy load the LLM model."""
    global llm
    if llm is None:
        llm = create_model(config.get_model_config())
    return llm



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

    typer.echo(f"\nEnter a path to a .gguf file to switch models, 'api' to use a hosted model (OpenAI, Anthropic, Gemini, ...), or press Enter to keep the current model:")
    new_path = input("> ").strip()

    if new_path.lower() == 'api':
        import providers
        try:
            updates = providers.select_provider_interactively()
            if updates is None:
                typer.echo("Keeping current model.\n")
                return
            config.update_model_config({**config.get_model_config(), **updates})
            llm = None
            typer.echo(f"Switched to: {updates['model']} via {providers.PROVIDERS[updates['provider']]['label']}\n")
        except (ValueError, OSError) as e:
            typer.echo(f"Error: {e}\n")
        return

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
        import providers
        abs_path = os.path.abspath(new_path)
        profile = providers.local_profile({**current_config, 'model_path': abs_path})
        loaded = import_llama().Llama(model_path=abs_path, n_ctx=profile['n_ctx'], n_gpu_layers=profile['n_gpu_layers'], verbose=False)
        config.set_model_path(abs_path)
        profile = providers.local_profile(config.get_model_config())
        config.update_model_config(profile)
        llm = EmbeddedModel(profile)
        llm._model = loaded
        typer.echo(f"Switched to: {os.path.basename(abs_path)}\n")
    except Exception as e:
        typer.echo(f"Error loading model: {e}\n")


def _gather_project_context():
    """Gather project information by reading key files from disk."""
    from pathlib import Path

    context_parts = []

    # List top-level directory
    cwd = Path(".")
    entries = sorted(cwd.iterdir())
    dir_listing = []
    skip = {".git", "__pycache__", "node_modules", "venv", ".venv", "llm"}
    for entry in entries:
        if entry.name in skip:
            continue
        suffix = "/" if entry.is_dir() else ""
        dir_listing.append(f"  {entry.name}{suffix}")
    context_parts.append("## Directory listing\n" + "\n".join(dir_listing))

    # Read key files (if they exist)
    key_files = [
        "README.md", "requirements.txt", "package.json", "setup.py",
        "pyproject.toml", "Dockerfile", "docker-compose.yml",
        "config.py", "main.py", "CLAUDE.md",
    ]
    max_file_chars = 3000
    for fname in key_files:
        try:
            p = WorkspaceTools().path(fname)
        except ValueError:
            continue
        if p.exists() and p.is_file():
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
                if len(content) > max_file_chars:
                    content = content[:max_file_chars] + "\n... [truncated]"
                context_parts.append(f"## Contents of {fname}\n```\n{content}\n```")
            except Exception:
                pass

    return "\n\n".join(context_parts)


def handle_md_command(console, max_tokens):
    """Handle the /md slash command: explore the project and generate CONTEXT.md."""
    typer.echo("\nGenerating CONTEXT.md by exploring the project...\n")

    md_max_tokens = max(max_tokens or 0, 2048)

    # Step 1: Gather real project data from disk (no LLM needed)
    typer.echo("Reading project files...\n")
    project_data = _gather_project_context()

    # Step 2: Ask the LLM to synthesize into a CONTEXT.md
    generate_messages = [
        {
            "role": "system",
            "content": (
                "You are a technical writer. Generate a markdown document and nothing else. "
                "Do not use any tools. Just output the markdown content directly."
            )
        },
        {
            "role": "user",
            "content": (
                "Based on the following real project files, write the contents of a CONTEXT.md file. "
                "Include these sections:\n"
                "- Project name and one-line description\n"
                "- Tech stack and dependencies\n"
                "- Directory structure overview\n"
                "- Key files and what they do\n"
                "- How to run the project\n"
                "- Architecture notes\n\n"
                "Output ONLY the markdown content, no explanation. "
                "Base everything strictly on the file contents provided below.\n\n"
                f"{project_data}"
            )
        }
    ]

    typer.echo("Generating CONTEXT.md content...\n")
    md_content = run_agent_loop(
        llm=get_llm(),
        messages=generate_messages,
        console=console,
        max_tokens=md_max_tokens
    )

    if not md_content or not md_content.strip():
        typer.echo("Failed to generate CONTEXT.md content.\n")
        return

    # Step 3: Show and write with user confirmation
    console.print(Markdown(md_content))
    console.print()

    if typer.confirm("Write CONTEXT.md?"):
        try:
            with open("CONTEXT.md", "w", encoding="utf-8") as f:
                f.write(md_content)
            typer.echo("CONTEXT.md has been created. It will be auto-injected into future prompts.\n")
        except Exception as e:
            typer.echo(f"Error writing CONTEXT.md: {e}\n")
    else:
        typer.echo("Write cancelled.\n")


def make_runtime(mode, max_steps, max_seconds, token_budget, trace, console, task_kind="auto", persistent=False, profile_name=None, route=False, tool_workers=4):
    if mode not in MODES:
        raise typer.BadParameter('mode must be read-only, workspace-edit, or execute')
    streamed = False
    def emit(event):
        nonlocal streamed
        if event['type'] == 'assistant_delta':
            streamed = True
            console.print(event['text'], end='', markup=False, highlight=False)
        elif event['type'] == 'tool_started':
            console.print(f"Tool: {event['name']}", markup=False)
        elif event['type'] == 'assistant_text' and not streamed:
            console.print(Markdown(event['text']))
        elif event['type'] == 'verification_result':
            if event['changed_files'] or event['checks'] or event['outstanding_processes'] or event['verification_status']=='requires_review':
                console.print('Verification: '+event['verification_status'],markup=False)
        elif event['type'] == 'run_finished':
            console.print()
            if event['status'] != 'completed':
                console.print(f"Run {event['status']}: {event['reason']}", markup=False)
            streamed = False
    effective_task = task_kind if task_kind != 'auto' else ('inspect' if mode == 'read-only' else 'code')
    selected = config.get_model_config(profile_name, effective_task, route)
    model = create_model(selected) if profile_name or route else get_llm()
    if persistent:
        from inference_daemon import PersistentModel
        model = PersistentModel(selected, config.CONFIG_DIR)
    return Runtime(model, os.getcwd(), config.CONFIG_DIR, mode=mode,
        emit=emit,
        budget=RunBudget(max_steps, max_seconds, token_budget),
        context_window=selected['n_ctx'], trace=trace, task_kind=task_kind,
        tool_workers=tool_workers)


def execute_turn(runtime, prompt, max_tokens):
    original, files = parse_file_references(prompt, root=runtime.tools.root)
    result = runtime.turn(original, files, max_tokens)
    typer.echo(f'Session: {runtime.session_id}; outcome: {result.status}')
    if result.status != 'completed':
        typer.echo(result.text)
    return result


@app.command()
def ask(
    prompt: str = typer.Argument(...),
    max_tokens: int = typer.Option(None, '--max-tokens', '-n', min=1, help='Output tokens per model call (default: a quarter of the context window, 512 to 4096)'),
    mode: str = typer.Option('read-only', '--mode'),
    max_steps: int = typer.Option(30, min=1),
    max_seconds: float = typer.Option(300, min=1),
    token_budget: int = typer.Option(8192, min=1),
    trace: bool = typer.Option(False, '--trace'),
    task_kind: str = typer.Option('auto', '--task-kind', help='auto, answer, inspect, code, or all'),
    persistent: bool = typer.Option(False, '--persistent'),
    profile_name: str = typer.Option(None, '--profile'),
    route: bool = typer.Option(False, '--route', help='Opt in to configured task-kind routing'),
    tool_workers: int = typer.Option(4, '--tool-workers', min=1, max=8, help='Maximum concurrent independent read tools'),
):
    """Ask a question or run a bounded coding task."""
    with make_runtime(mode, max_steps, max_seconds, token_budget, trace, Console(), task_kind, persistent, profile_name, route, tool_workers) as runtime:
        result = execute_turn(runtime, prompt, max_tokens)
    if result.status != 'completed':
        raise typer.Exit(1)


@app.command()
def chat(
    max_tokens: int = typer.Option(None, '--max-tokens', '-n', min=1, help='Output tokens per model call (default: a quarter of the context window, 512 to 4096)'),
    mode: str = typer.Option('read-only', '--mode'),
    resume: str = typer.Option(None, '--resume'),
    max_steps: int = typer.Option(30, min=1),
    max_seconds: float = typer.Option(300, min=1),
    token_budget: int = typer.Option(8192, min=1),
    trace: bool = typer.Option(False, '--trace'),
    task_kind: str = typer.Option('auto', '--task-kind', help='auto, answer, inspect, code, or all'),
    persistent: bool = typer.Option(False, '--persistent'),
    profile_name: str = typer.Option(None, '--profile'),
    route: bool = typer.Option(False, '--route', help='Opt in to configured task-kind routing'),
    tool_workers: int = typer.Option(4, '--tool-workers', min=1, max=8, help='Maximum concurrent independent read tools'),
):
    """Chat with persistent tool history; /resume ID, /sessions, /new, /undo, /exit."""
    console = Console()
    with make_runtime(mode, max_steps, max_seconds, token_budget, trace, console, task_kind, persistent, profile_name, route, tool_workers) as runtime:
        if resume:
            runtime.resume(resume)
        while True:
            try:
                prompt = typer.prompt('You').strip()
                if prompt == '/exit':
                    break
                if prompt == '/sessions':
                    typer.echo('\n'.join(runtime.store.list()) or 'No saved sessions.')
                elif prompt.startswith('/resume '):
                    runtime.resume(prompt.split(maxsplit=1)[1])
                elif prompt == '/new':
                    runtime.new()
                elif prompt == '/undo':
                    typer.echo(runtime.tools.undo_last())
                elif prompt == '/acknowledge-interrupted':
                    runtime.acknowledge_interrupted()
                    typer.echo('Interrupted operations acknowledged; completed calls will not be replayed.')
                elif prompt == '/model':
                    handle_model_command()
                    runtime.model = get_llm()
                    from model_backend import ModelAdapter
                    runtime.context.count_tokens = runtime.model.count_tokens if isinstance(runtime.model, ModelAdapter) else runtime.context.count_tokens
                elif prompt == '/md':
                    handle_md_command(console, max_tokens)
                elif prompt:
                    execute_turn(runtime, prompt, max_tokens)
            except (KeyboardInterrupt, EOFError):
                break
            except (OSError, ValueError) as exc:
                typer.echo(str(exc), err=True)


@app.command()
def edit(
    prompt: str = typer.Argument(...),
    max_tokens: int = typer.Option(None, '--max-tokens', '-n', min=1, help='Output tokens per model call (default: a quarter of the context window, 512 to 4096)'),
    mode: str = typer.Option('workspace-edit', '--mode'),
    max_steps: int = typer.Option(30, min=1),
    max_seconds: float = typer.Option(300, min=1),
    token_budget: int = typer.Option(8192, min=1),
    trace: bool = typer.Option(False, '--trace'),
    task_kind: str = typer.Option('auto', '--task-kind', help='auto, answer, inspect, code, or all'),
    persistent: bool = typer.Option(False, '--persistent'),
    profile_name: str = typer.Option(None, '--profile'),
    route: bool = typer.Option(False, '--route', help='Opt in to configured task-kind routing'),
    tool_workers: int = typer.Option(4, '--tool-workers', min=1, max=8, help='Maximum concurrent independent read tools'),
):
    """Apply targeted edits; --mode execute also permits validation commands."""
    with make_runtime(mode, max_steps, max_seconds, token_budget, trace, Console(), task_kind, persistent, profile_name, route, tool_workers) as runtime:
        result = execute_turn(runtime, prompt, max_tokens)
    if result.status != 'completed':
        raise typer.Exit(1)


@app.command()
def undo():
    """Undo the most recent harness patch if the file has not changed since."""
    try:
        typer.echo(WorkspaceTools(mode='workspace-edit').undo_last())
    except (OSError, ValueError) as exc:
        typer.echo(f'Cannot undo: {exc}', err=True)
        raise typer.Exit(1)


@app.command()
def models(
    set_model: str = typer.Option(None, "--set", "-s", help="Path to GGUF model file to use"),
    backend: str = typer.Option(None, '--backend', help='embedded or openai'),
    base_url: str = typer.Option(None, '--base-url'),
    model_name: str = typer.Option(None, '--model-name'),
    context_window: int = typer.Option(None, '--context-window', min=1024),
    chat_format: str = typer.Option(None, '--chat-format'),
    supports_tools: bool = typer.Option(None, '--tools/--no-tools'),
    streaming: bool = typer.Option(None, '--stream/--no-stream'),
    threads: int = typer.Option(None, '--threads', min=1),
    batch_threads: int = typer.Option(None, '--batch-threads', min=1),
    batch_size: int = typer.Option(None, '--batch-size', min=1),
    micro_batch_size: int = typer.Option(None, '--micro-batch-size', min=1),
    flash_attention: bool = typer.Option(None, '--flash-attention/--no-flash-attention'),
    key_cache_type: str = typer.Option(None, '--key-cache-type'),
    value_cache_type: str = typer.Option(None, '--value-cache-type'),
    prompt_cache_mb: int = typer.Option(None, '--prompt-cache-mb', min=0, max=4096),
    server_cache_prompt: bool = typer.Option(None, '--server-cache-prompt/--no-server-cache-prompt'),
    speculative_mode: str = typer.Option(None, '--speculative-mode'),
    draft_model_path: str = typer.Option(None, '--draft-model'),
    draft_tokens: int = typer.Option(None, '--draft-tokens', min=1, max=32),
    draft_ngram_size: int = typer.Option(None, '--draft-ngram-size', min=1, max=8),
    draft_gpu_layers: int = typer.Option(None, '--draft-gpu-layers', min=-1),
    provider: str = typer.Option(None, '--provider', help='Hosted provider: ' + ', '.join(__import__('providers').PROVIDERS)),
    api_key: bool = typer.Option(False, '--api-key', help='Prompt for the provider API key (stored privately)'),
):
    """Show current model or set a new model."""
    global llm
    updates = {'backend': backend, 'base_url': base_url, 'model': model_name,
               'n_ctx': context_window, 'chat_format': chat_format,
               'supports_tools': supports_tools, 'stream': streaming,
               'n_threads': threads, 'n_threads_batch': batch_threads, 'n_batch': batch_size,
               'n_ubatch': micro_batch_size, 'flash_attn': flash_attention,
               'type_k': key_cache_type, 'type_v': value_cache_type,
               'prompt_cache_mb': prompt_cache_mb, 'server_cache_prompt': server_cache_prompt,
               'speculative_mode': speculative_mode, 'draft_model_path': draft_model_path,
               'draft_tokens': draft_tokens, 'draft_ngram_size': draft_ngram_size,
               'draft_n_gpu_layers': draft_gpu_layers}
    updates = {k: v for k, v in updates.items() if v is not None}
    if provider:
        import providers
        if provider not in providers.PROVIDERS:
            raise typer.BadParameter(f'Unknown provider; choose from {", ".join(providers.PROVIDERS)}')
        spec = providers.PROVIDERS[provider]
        model = updates.get('model') or spec['models'][0]
        updates = {**providers.provider_profile(provider, model), **updates}
        if api_key or not providers.resolve_key(updates):
            key = typer.prompt(f"{spec['label']} API key", hide_input=True, default='', show_default=False)
            if key:
                providers.save_key(provider, key)
            elif not providers.resolve_key(updates):
                raise typer.BadParameter('An API key is required (or set ' + spec['key_env'] + ')')
    elif api_key:
        raise typer.BadParameter('--api-key requires --provider')
    elif 'base_url' in updates or 'backend' in updates:
        updates.setdefault('provider', None)
    if updates:
        current = config.get_model_config()
        profile = {**current, **updates}
        if profile.get('backend') == 'embedded' and current.get('provider'):
            import providers
            profile = providers.local_profile(profile, keep_context='n_ctx' in updates)
        try:
            create_model(profile)
        except ValueError as exc:
            raise typer.BadParameter(str(exc))
        config.update_model_config(profile)
        llm = None
        typer.echo('Model profile updated.')

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
            import providers
            profile = providers.local_profile(config.get_model_config(), keep_context=context_window is not None)
            config.update_model_config(profile)
            llm = None
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
        typer.echo(f"  Backend: {current_config['backend']}")
        if current_config['backend'] == 'openai':
            typer.echo(f"  Server: {current_config.get('base_url', 'http://127.0.0.1:8080/v1')}")
            typer.echo(f"  Model: {current_config.get('model', 'local-model')}")
            if current_config.get('provider'):
                import providers
                typer.echo(f"  Provider: {current_config['provider']} (API key: "
                           f"{'set' if providers.resolve_key(current_config) else 'missing'})")
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


@app.command()
def benchmark(
    output: str = typer.Option(..., '--output'),
    prompt: str = typer.Option('Explain why binary search takes logarithmic time.'),
    repeats: int = typer.Option(3, min=1),
    warmups: int = typer.Option(1, min=0),
    max_tokens: int = typer.Option(128, min=1),
    profile_name: str = typer.Option(None, '--profile'),
    compare: list[str] = typer.Option(None, '--compare'),
    persistent: bool = typer.Option(False, '--persistent'),
):
    """Measure real configured inference; missing models/errors never count as passes."""
    from performance import benchmark as measure
    import json
    from pathlib import Path
    try:
        if compare and persistent:
            raise ValueError('Persistent inference holds one profile; compare without --persistent')
        names = compare or [profile_name]
        reports = []
        for name in names:
            profile = config.get_model_config(name)
            if persistent:
                from inference_daemon import PersistentModel
                model = PersistentModel(profile, config.CONFIG_DIR)
            else:
                model = create_model(profile)
            try:
                report = measure(model, prompt, repeats, warmups, max_tokens)
                safe_keys = ('backend', 'model', 'model_path', 'n_ctx', 'n_threads', 'n_threads_batch',
                             'n_batch', 'n_ubatch', 'type_k', 'type_v', 'flash_attn', 'prompt_cache_mb',
                             'speculative_mode', 'draft_model_path', 'draft_tokens', 'draft_ngram_size', 'draft_n_gpu_layers')
                report['profile'] = name or config.load_config().get('active_profile', 'default')
                report['settings'] = {key: profile[key] for key in safe_keys if key in profile}
                reports.append(report)
            finally:
                model.close()
        result = {'version': 1, 'comparisons': reports} if compare else reports[0]
        Path(output).write_text(json.dumps(result, indent=2) + '\n')
        typer.echo(json.dumps([r['median'] for r in reports], indent=2))
        if not all(r['all_completed'] for r in reports):
            typer.echo('Some responses were truncated; inspect the samples.', err=True)
    except (OSError, ValueError) as exc:
        typer.echo(f'Benchmark failed: {exc}', err=True)
        raise typer.Exit(1)


@app.command()
def profiles(action: str = typer.Argument('list'), name: str = typer.Argument(None),
             target: str = typer.Argument(None)):
    """Manage named configurations: list, save NAME, use NAME, route TASK NAME."""
    global llm
    import json
    try:
        if action == 'list':
            data = config.load_config()
            typer.echo(json.dumps({'active': data.get('active_profile', 'default'),
                                  'profiles': sorted(data.get('profiles', {})),
                                  'routes': data.get('routes', {})}, indent=2))
        elif action == 'save':
            config.save_profile(name)
        elif action == 'use':
            config.activate_profile(name)
            llm = None
        elif action == 'route':
            config.set_route(name, target)
        else:
            raise ValueError('Use list, save NAME, use NAME, or route TASK NAME')
    except ValueError as exc:
        raise typer.BadParameter(str(exc))


@app.command()
def inference_server(stop: bool = typer.Option(False, '--stop')):
    """Run the private inference daemon in the foreground, or stop it."""
    from inference_daemon import InferenceDaemon, PersistentModel
    if stop:
        client = PersistentModel({'backend': 'embedded'}, config.CONFIG_DIR, autostart=False)
        try:
            client.stop()
        except OSError as exc:
            raise typer.BadParameter(f'Daemon unavailable: {exc}')
        typer.echo('Daemon stopping.')
        return
    from inference_daemon import normalized_profile
    profile = config.get_model_config()
    if profile.get('backend', 'embedded') != 'embedded':
        raise typer.BadParameter('The private daemon is for embedded models')
    model = create_model(normalized_profile(profile))
    with InferenceDaemon(config.CONFIG_DIR / 'inference.sock', model) as server:
        typer.echo('Inference daemon running; Ctrl-C to stop.')
        try:
            server.serve_forever(poll_interval=0.05)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    app()
