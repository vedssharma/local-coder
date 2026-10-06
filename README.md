# Local Coder

A local-first coding harness with a terminal CLI and an MCP server. Run a GGUF model through embedded llama.cpp, or connect to an OpenAI-compatible model server. The harness can inspect code, edit files, run checks, retain working context, and resume sessions.

Embedded inference stays on your machine. The server backend sends prompts and file context to the endpoint you configure; use a local endpoint for local inference.

## Install

Python 3.10+ is required. Node.js is not required.

```bash
python -m venv llm
source llm/bin/activate
pip install -r requirements.txt
pip install pytest pytest-asyncio
python -m pytest -q
```

That is enough for hosted providers and OpenAI-compatible servers. To run GGUF models in-process, also install llama-cpp-python, which needs a C/C++ build toolchain:

```bash
pip install -r requirements-embedded.txt
```

For a CPU build when compiler environment variables point to unavailable tools:

```bash
CC=gcc CXX=g++ CMAKE_BUILD_PARALLEL_LEVEL=2 pip install -r requirements-embedded.txt
```

The test suite stubs llama-cpp-python, so it runs without the native build.

CUDA and Metal builds are optional; follow llama-cpp-python's installation instructions for your platform. Docker configuration is described in [DOCKER.md](DOCKER.md).

## Configure a model

Download a trusted GGUF model, verify its checksum against the publisher's metadata, and keep it out of Git. Qwen2.5-Coder is one example; model quality and tool-call support depend on the model, quantization, and chat template.

```bash
python main.py models --set /absolute/path/to/coder.gguf
python main.py models --context-window 8192
python main.py ask "What is 2 + 2?"
```

Configuration lives in `~/.local-coder/config.json`. Set `LOCAL_CODER_CONFIG_DIR` to select another directory. Existing configurations receive defaults for new fields.

For a local OpenAI-compatible server you started separately:

```bash
python main.py models --backend openai \
  --base-url http://127.0.0.1:8080/v1 --model-name local-coder \
  --context-window 8192 --tools --stream
```

To use a hosted frontier model instead, pick a provider (`openai`, `anthropic`, `google`, `xai`, `mistral`, `deepseek`). You are prompted for the API key with hidden input, or you can set the provider's environment variable (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `XAI_API_KEY`, `MISTRAL_API_KEY`, `DEEPSEEK_API_KEY`), which takes precedence. Keys are stored in `~/.local-coder/keys.json` (mode 0600), never in `config.json`. Inside `chat`, `/model` then `api` runs the same selection interactively. Prompts and file context are sent to the provider.

Selecting a provider sets the context window the harness budgets against to 128,000 tokens (64,000 for DeepSeek). This is below most hosted models' maximum on purpose, because each step resends the whole context; pass `--context-window` to change it. Hosted APIs don't expose a tokenizer, so prompt size is estimated: three UTF-8 bytes per token at first, then adjusted from the prompt token counts each response reports (with a 10% safety margin). A local OpenAI-compatible server is asked for exact counts through its `/tokenize` endpoint (llama-server and vLLM have one); if it doesn't answer, the same estimate is used. If you configured a provider before this default existed, run `models --provider ...` again or set `--context-window`. Switching back to a GGUF model restores the local default of 8,192 unless you pass `--context-window`.

```bash
python main.py models --provider anthropic --model-name claude-sonnet-5-5 --api-key
```

`--chat-format` selects an embedded llama.cpp chat format. `--no-tools` marks a model that cannot use native tool calls; the tools are then described in the system prompt, the model calls them with `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` blocks (or a fenced JSON block), and results come back as user messages. `--no-stream` disables streaming when a server does not support it. Optional API authentication uses `LOCAL_CODER_API_KEY`, or the environment variable named by `api_key_env` in configuration; never put key values in configuration or source. `request_timeout` controls the server's socket timeout (default 60 seconds). Requests to a server or provider that fail with 408, 429, 500, 502, 503, 504 or 529, or whose connection drops before any output, are retried up to twice. The retry honors `Retry-After` (capped at 60 seconds) or otherwise waits 1 and then 2 seconds, and it never waits past the run's time budget. Other errors, and failures after output has started streaming, end the run as before.

## Use the harness

```bash
# Read-only is the default for ask and chat.
python main.py ask "Explain @agent.py"
python main.py chat

# edit defaults to workspace-edit: targeted file edits, no arbitrary commands.
python main.py edit "Simplify @helpers.py"

# Explicitly permit commands so the agent can run tests after its edits.
python main.py edit "Fix the failing tests and verify the changes" --mode execute
python main.py chat --mode execute --max-steps 60 --max-seconds 600 --token-budget 16384
```

`--max-tokens` limits the output of each model call. It defaults to a quarter of the context window, between 512 and 4096 tokens, so a `write` call has room for a whole file without crowding out the prompt. `--token-budget` limits the total generated in a turn. If a tool call is cut off at the limit, it is not executed; the model is asked once to make the change in smaller steps, and a second cut-off ends the run as `budget_exhausted`.

The model has eight tools:

- `read`: a line range of a workspace file (default 100 lines).
- `list` (all modes): workspace file paths under a directory, optionally filtered by a glob such as `**/*.py` (default 200 entries, at most 1,000). In a Git repository it respects `.gitignore`; `.git`, `.local-coder`, `node_modules`, virtual environments, and caches are always skipped, as are symlinks that resolve outside the workspace.
- `search` (all modes): a regular-expression search of file contents, returning `path:line: text` matches (default 50, at most 200). Optional `path`, filename `glob`, and `case_sensitive`. It skips the same paths as `list`, binary files, and files over 1 MB.
- `write`: create a file or overwrite it entirely (`workspace-edit` mode or higher).
- `edit`: replace exactly one matching block of text in an existing file (`workspace-edit` mode or higher). If `old_text` appears more than once, the error lists the line where each match starts and an optional `start_line` picks one. If it isn't found exactly, a unique match that differs only in trailing whitespace or a uniform indentation shift is used (the same shift is applied to `new_text`) and the result says so; otherwise the error shows the closest block with line numbers.
- `bash` (`execute` mode): run a non-interactive Bash command, including pipes, redirects, and multiline scripts, and wait for it to finish. Use it for `git diff`, running checks, and anything `list` and `search` don't cover. It returns combined stdout/stderr and the exit status. The timeout is 1–300 seconds (default 60) and the optional `cwd` must resolve inside the workspace. Output retains at most 32,000 bytes and reports truncation. Bash must be installed on PATH; it runs without profile or rc files, with stdin closed. Each call starts a fresh shell; variables and working-directory changes do not persist between calls. `bash` runs with host privileges and the command body is not filesystem-sandboxed. The runtime terminates any remaining command processes when it closes.
- `web_search` and `web_fetch`: see below.

For example, the agent can call `bash` with `{"command": "python -m pytest -q", "cwd": ".", "timeout_seconds": 120}`. Repository instructions and tool output cannot elevate the selected mode.

The native `web_search` and `web_fetch` tools work in every permission mode, but they are **off by default for embedded and local-server models** and on for hosted providers. Any tool that reaches the network can carry workspace content out (a `web_fetch` URL can include text the model read), and a prompt injection in a file or page could ask for exactly that. Hosted providers already receive your context, so web access adds little there; for a fully local setup it would be the only path off the machine.

Turn them on or off per profile with `python main.py models --web` or `--no-web`, or for one run with `--web`/`--no-web` on `ask`, `chat`, and `edit`. The MCP server follows the profile unless `LOCAL_CODER_WEB=1` or `0` is set at launch. When they are off, the tools are not offered and the system prompt does not mention them. `--task-kind answer` still disables all tools. They need no extra Python dependencies.

- `web_search`: accepts `query`, optional `max_results` (1–10, default 5), and `timeout_seconds` (1–30, default 20). Returns source URLs, titles, snippets, and the provider name. It uses DuckDuckGo's HTML search by default. Set `BRAVE_SEARCH_API_KEY` in the harness process environment to use the Brave Search API instead; the key is not passed in model tool arguments or returned in results. Queries are sent to the selected search provider. Provider errors and bot challenges are reported as errors, not invented results.
- `web_fetch`: accepts `url`, optional `max_chars` (100–50,000, default 12,000), and `timeout_seconds` (1–30, default 20). Returns the final URL, HTTP status, content type, title, extracted text, and a truncation flag. HTML scripts and styles are removed; plain text, JSON, and XML are supported. It reads at most 1 MB per response. JavaScript rendering, authenticated browsing, PDFs, and binary downloads are not supported.

For example, ask `python main.py ask "Search the web for the latest Python asyncio documentation and cite the source URLs"`, or let the model call `web_fetch` with `{"url": "https://docs.python.org/3/library/asyncio.html"}`. Web output is marked as untrusted content, and the system prompt directs the model to use it as evidence rather than instructions.

Only public HTTP(S) URLs on standard ports are supported; embedded URL credentials and local/private IP literals are rejected. Direct requests validate destination DNS addresses, including redirects. When an HTTP proxy is configured, hostname resolution and destination access use that proxy's policy; these checks are not an OS network sandbox. The client preserves inherited proxies and TLS certificate verification. Restricted environments must allow the search provider (`html.duckduckgo.com`, or `api.search.brave.com`) and the sites being fetched. A blocked destination requires an environment policy change; the tools do not bypass it.

Permission modes:

| Mode | Read | Write/edit | Bash |
| --- | --- | --- | --- |
| `read-only` | Yes | No | No |
| `workspace-edit` | Yes | Yes | No |
| `execute` | Yes | Yes | Yes |

File tools and `@file` references reject paths outside the selected workspace, including symlink escapes. `write` and `edit` cannot modify `.git` or harness metadata. **Execute mode uses the host's privileges; it is not an OS sandbox.** Use a container or other isolation when running untrusted tasks.

Runs report `completed`, `blocked`, `cancelled`, or `budget_exhausted`. `completed` means the model ended its response; verification requires observing the actual check results. Exhaustion never forces an extra answer that claims completion. Cancellation and time limits are checked between operations; streaming adapters also check cancellation between chunks. A blocking embedded generation step cannot be forcibly interrupted by the engine.

## Sessions and context

Each CLI turn saves a session and prints its ID. Sessions include tool calls and observations, not only final answers. Resuming never restores a previous permission mode; select permissions explicitly for the new process.

```bash
python main.py chat --resume SESSION_ID
python main.py undo
```

Chat commands: `/sessions`, `/resume ID`, `/new`, `/undo`, `/model`, `/md`, and `/exit`. Undo restores only the latest recorded harness patch whose current contents still match the patch result. It preserves pre-existing user edits and refuses to overwrite subsequent changes. Arbitrary command edits are not automatically undoable.

Context budgeting counts tools, transcript, framing, and reserved generation space. Embedded inference uses its tokenizer; server inference conservatively estimates from UTF-8 bytes. Complete older turns are compacted into a bounded summary of requests, decisions, tool use, and observations. Summaries are lossy. The active turn is preserved; if it cannot fit, the run blocks instead of silently discarding it. Large tool results are shortened, with retained output artifacts under `.local-coder/artifacts` where applicable. Command output retains the latest 32 KB.

Root `AGENTS.md` applies to the workspace. Ranged reads include applicable ancestor and directory instructions for their target. `CONTEXT.md` provides optional project context. Neither can change executor permissions.

`--trace` writes structured events to `.local-coder/traces`. Transcripts, undo records, and traces can contain source code and tool output; keep them private. `.local-coder/` is ignored by Git and Docker.

## MCP server

Launch from the workspace you want it to operate on:

```bash
cd /path/to/your/project
/path/to/local-coder/llm/bin/python /path/to/local-coder/skills/local-coder/scripts/server.py
```

It exposes `ask`, `chat`, `edit`, `get_model`, and `set_model` over stdio. Responses from coding tools include `status`, `text`, budgets consumed, and `session_id`. The server defaults to read-only. Set `LOCAL_CODER_PERMISSION_MODE=workspace-edit` or `execute` at launch to enable those capabilities. MCP arguments cannot grant permissions. Each call gets the CLI's run budget (30 steps, 300 seconds, 8,192 generated tokens); set `LOCAL_CODER_MAX_STEPS`, `LOCAL_CODER_MAX_SECONDS` or `LOCAL_CODER_TOKEN_BUDGET` at launch to change it. See the [setup guide](skills/local-coder/references/setup.md).

## Evaluate changes

The ordinary test suite validates implementation behavior without loading an LLM:

```bash
python -m pytest -q
```

Six disposable-repository evaluations cover navigation, a bug fix, a multi-file change, recovery from a failing check, permissions, and continuing an unfinished task after restart:

```bash
# Scripted model decisions, real files/subprocesses/session storage.
python evaluations/run.py --scripted --output /tmp/harness-mechanics.json

# Actual configured model. Explicitly allow commands with host privileges.
python evaluations/run.py --allow-execution --output /tmp/model-evaluation.json

# Read-only evaluation without command permission.
python evaluations/run.py --case navigation --output /tmp/navigation.json
```

Reports record success checks, changed files, observed command exit codes, tool failures, generated tokens, and elapsed time. Tests must actually execute, test files must remain unchanged, and coding tasks must produce passing independent checks. A confident final answer alone does not pass. Scripted passes measure harness mechanics, not model coding ability. Use identical fixtures and record model/quantization/context settings when comparing models; these small tasks are a regression suite, not evidence of parity with commercial harnesses.

## Architecture

- `agent.py`: bounded orchestration, validated calls, explicit outcomes.
- `workspace_tools.py`: scoped tools, processes, permissions, and patch undo.
- `session.py`: persistent transcripts, context budgeting, scoped instructions.
- `runtime.py`: shared turn/session lifecycle and events.
- `model_backend.py`: embedded and OpenAI-compatible streaming adapters.
- `main.py`: CLI and terminal rendering.
- `skills/local-coder/scripts/server.py`: MCP interface to the same runtime.

Dependencies and inference behavior are covered by separate checks. The MCP server SDK is constrained to 1.x because this repository uses its 1.x result API.

## Inference measurements and tuning

```bash
python main.py models --threads 4 --batch-threads 4 --batch-size 512 --micro-batch-size 256
python main.py models --flash-attention --key-cache-type q8_0 --value-cache-type q8_0
python main.py benchmark --output /tmp/inference.json --warmups 1 --repeats 3
```

Thread, batch, attention, and KV-cache options apply to embedded inference. KV types are `f16`, `q8_0`, or `q4_0`; quantized value caches require Flash Attention. Backend/platform support still determines whether a setting works. Benchmark before adopting a setting; larger batches are not always faster.

Reports retain individual samples and medians for request latency, load time, first streamed output, and available native prompt/generation throughput. Unsupported rates remain null; server end-to-end latency is not mislabeled as decode speed. A warm-up excludes model loading from the sampled steady-state requests; use `--warmups 0` to include a cold first request. Run results also contain total, context preparation, model, and tool timings and call counts. Real benchmarks require a configured model.

`--task-kind auto` exposes inspection tools (`read`, `list`, `search`, and `web_search` and `web_fetch` when web access is on) in read-only mode and all permitted tools in edit/execute modes. Use `answer` for a tool-free question, `inspect` for reads and web lookups, `code` for permitted coding tools, or `all`. Selection narrows capabilities; it never grants permissions. Default reads return 100 lines, `@file` preloads are bounded to 8 KB, and unchanged preloads already retained in the transcript are not injected twice. Changed files and references whose old context was compacted away are injected again.

Duplicate reads within one model response share an execution. Reads across steps execute again to detect changes, but identical retained observations are referenced instead of appended in full. Errors and command results are never reused.

Prompt dictionaries and tool ordering are canonicalized without reordering conversation turns. Embedded llama.cpp keeps its live prefix state; `models --prompt-cache-mb 128` additionally enables a bounded in-memory llama.cpp state cache (`0` disables it). Cached model states may contain prompt data and are never written to disk. `--server-cache-prompt` is an explicit opt-in for servers that support the nonstandard `cache_prompt` request field; generic OpenAI servers receive no such field by default. API-reported cached prompt tokens are recorded when available. Stable prefixes improve the opportunity for reuse but do not guarantee cache hits or speedups.

### Persistent one-shot inference

On Unix platforms, `python main.py ask "question" --persistent` starts or reuses a private embedded-inference daemon. Subsequent CLI processes share its loaded model and prefix cache. File tools and permissions remain in each CLI process; the daemon accepts only inference and tokenization requests. Its socket and profile snapshot are private to your user. Startup is serialized and a mismatched profile is rejected instead of silently using another model.

Use `python main.py inference-server` for foreground operation, or `python main.py inference-server --stop` to stop it (including after changing configuration). Stop/restart after tuning or model changes. The daemon remains alive after a one-shot command; it occupies model memory until stopped. Logs and the socket live under `LOCAL_CODER_CONFIG_DIR`. Server backends already retain models and do not use this daemon.

Token budgeting uses a bounded in-memory cache of counts for serialized message and schema fragments. It reuses unchanged fragments, invalidates changed contents, and clears when the tokenizer changes. Framing/boundary margins remain conservative; this is an estimate rather than an exact model-specific chat-template count. Run metrics expose cache hits/misses. Cached and uncached budgeting use the same estimate, and the cache stores hashes/counts rather than prompt text.

### Named models and explicit routing

Configure/download each model first, then save a snapshot:

```bash
python main.py profiles save small
# Change model/tuning, then:
python main.py profiles save large
python main.py profiles use small
python main.py profiles route answer small
python main.py profiles route code large
python main.py ask "question" --task-kind answer --route
python main.py edit "fix the tests" --mode execute --profile large
python main.py benchmark --compare small --compare large --output /tmp/comparison.json
python evaluations/run.py --profile small --allow-execution --output /tmp/small-quality.json
```

`--profile` overrides routing. Routes are used only with `--route` and choose a model once at task startup; there is no hidden difficulty classifier or mid-turn model swap. Without routing, the active profile (or the base `default`) is used. `profiles use default` restores the base configuration; `profiles route code default` clears that route. Subsequent `models` changes update the active named profile while preserving other profiles. Named snapshots do not inherit later base tuning changes. Comparisons run sequentially, release each model, and record settings and raw samples. Compare coding success as well as latency before choosing a smaller/quantized model. A private daemon retains one profile at a time and requires stop/restart when switching.

### Speculative decoding

```bash
# Reuse matching text sequences from the prompt; no second model required.
python main.py models --speculative-mode prompt-lookup --draft-tokens 8 --draft-ngram-size 2

# Use a compatible smaller GGUF as a learned draft.
python main.py models --speculative-mode draft-model --draft-model /path/to/draft.gguf --draft-tokens 8

# Disable and compare against the baseline.
python main.py models --speculative-mode off
```

These modes use llama-cpp-python's target-verified draft callback. Prompt lookup is useful only when the prompt contains matching continuations. Learned drafts must match tokenizer metadata, vocabulary/token IDs, and tokenization checks; incompatible models are rejected and released. Both models must fit in memory; speculative decoding also enables additional target logits storage and can be slower than the baseline. The draft defaults to CPU; `--draft-gpu-layers` controls its offload independently. No draft is loaded unless explicitly enabled. External-server speculation must be configured using that server's supported startup options; this client does not send invented portable draft parameters.

Use named baseline/speculative profiles with `benchmark --compare`, then compare coding evaluations. Native throughput describes native target evaluation; `end_to_end_completion_tokens_per_second` includes full request time (including draft work, prefill, and loading where applicable). This distinction matters for speculative decoding. `benchmark --persistent` measures the private daemon; multi-profile comparisons require non-persistent inference or explicit daemon restarts. Real model measurements remain necessary before claiming a speedup.

### Structured tool outcomes

The agent consumes `ToolResult` objects through `execute_tool`; the existing `call_tool` string API remains a compatibility wrapper. Model tool messages are JSON envelopes containing `status`, `data`, `error_code`, `error_message`, `retryable`, `duration_seconds`, and `artifacts`. Large payloads are bounded without truncating the JSON envelope or its status, and the full payload is retained as an artifact when possible.

Statuses distinguish successful observations, nonzero command exits (`failed` / `command_failed`), launch or validation errors (`error`), timeouts, and cancellation. Tools report errors structurally; ordinary content beginning with `Error:` remains successful data. `tool_finished` events include both the model-facing output and the structured result. Retryability is consumed only by the bounded retry policy for explicitly safe reads.

### Tool registry

Each native tool has one `ToolSpec` registration in `WorkspaceTools._register_tools`: its schema, handler, minimum permission mode, side effects, concurrency policy, observation-reuse policy, task categories, and optional timeout defaults/limits. Schemas returned to callers are independent snapshots. JSON Schema validators are compiled once per registration and reused by direct calls and the agent loop. The registry controls tool exposure, dispatch, argument validation, and the loop's cache invalidation.

The scheduler uses this metadata to parallelize only explicitly eligible independent reads.

### Shared execution deadlines

A run now supplies one `ExecutionContext` to the model adapter and tools. Command timers and HTTP request timeouts are capped by the remaining run budget. Cancellation is checked before dispatch, during command capture, between streamed model chunks, and while waiting for active HTTP or daemon sockets. Active socket reads are interrupted by shutting down the connection; cancellation prevents subsequent mutations from starting.

Python file/DNS operations and connecting sockets retain platform interruption limits. Embedded llama.cpp calls cannot be forcibly interrupted inside a native generation step; persistent daemon clients can disconnect, but that does not forcibly terminate the server's native generation. Execute mode still uses host privileges.

### Recovery and progress detection

Explicitly retry-safe reads can retry transient failures at most twice, with short exponential backoff inside the shared run budget. Retry events and attempt counts are observable. Commands, edits, permission failures, invalid arguments, and stale edits are never blindly retried; a stale edit returns `stale_patch` so the model can reread and repair its arguments. HTTP status errors distinguish retryable throttling/server failures from access denials.

The loop detects repeated semantic observations and short cycles without progress and returns `blocked` with reason `no_progress`. 
### Concurrent read scheduling

Adjacent independent read/web calls marked parallel-safe execute with at most four workers by default; model tool results retain the original call order. Identical cacheable observations share one execution. Serial tools and mutations drain the preceding read group before executing, invalidate stale observation caches, and finish before subsequent reads begin. Unknown legacy tools remain serial.

Set `--tool-workers` on `ask`, `chat`, or `edit` (1–8), or `LOCAL_CODER_TOOL_WORKERS` at MCP launch. One worker preserves serial read execution. Output events may arrive in execution order, while transcript results remain in call order. Cancellation and validation apply to each scheduled operation.

### Active-turn context and artifacts

Long turns can compact older completed tool exchanges without splitting call/result pairs. The current request and newest exchange remain; compacted evidence retains concise failures, edit/check outcomes, decisions, and content-addressed archive references. If the newest output alone is oversized, its payload is shortened while status and exit metadata remain intact. Evidence summaries are untrusted user-role data, never new system instructions. Requests or schemas that cannot fit still fail explicitly.

Oversized tool output is archived under `.local-coder/artifacts/<sha256>.txt`; the truncated message names the file, and the model can page through it with `read` and a line range. Large model-facing tool envelopes retain structured command status and archive IDs rather than losing exit metadata to truncation.

### Checkpointed execution

Session snapshots now contain a durable call journal. The runtime saves the assistant batch before dispatch, validated arguments and side-effect metadata before executing each operation, and its outcome immediately afterward. Atomic snapshots are fsynced, and an exclusive session lease prevents two runtimes from concurrently continuing the same session. Version-one transcripts remain loadable.

Resume reconstructs missing tool results from completed journal entries and never automatically replays prior commands or edits. Pending calls receive `not_executed`; calls interrupted during execution or old live process handles receive `interrupted_operation`. Interrupted edits require a fresh read of the affected file before further mutations. Unknown command effects require explicit acknowledgement after inspection (`Runtime.acknowledge_interrupted()` or `/acknowledge-interrupted` in CLI chat). Acknowledgement does not mark an operation successful or grant additional tool permissions.

### Evidence-aware completion

`RunResult` and `turn_result`/`verification_result` events report changed files, observed check commands and exit codes, and a separate `verification_status`: `not_run`, `passed`, `failed`, `stale`, `in_progress`, or `requires_review`. Standard test commands (pytest, unittest, `npm test`, `cargo test`, ...) run through `bash` are recognized; use `verification=true` for other genuine checks. Preserve the check's exit status rather than masking it with a later successful shell command.

The latest rerun of the same command replaces its earlier result; independent failed checks remain failures. Any `write` or `edit` invalidates checks recorded before it. Final model prose is labeled as a model summary when evidence contradicts completion; the runtime returns `blocked` for failed/stale checks, outstanding jobs, and interrupted effects requiring inspection. Edits with no checks are explicitly reported as unverified.

Verification covers observed commands and changes, not test coverage or task correctness. Arbitrary shell mutations are not fully tracked, and an exit-zero command does not establish that a meaningful check ran. Real-model evaluations and appropriate project tests remain necessary.

### Loop regression evaluations

The expanded suite adds eight fault-injection cases: malformed arguments with recovery, transient read retries, cancellation with a pending mutation, overlapping reads with ordered results, long-turn compaction, inspection after an interrupted write, permission enforcement against untrusted web instructions, and a false claim that a failing check passed.

```bash
python evaluations/run.py --scripted --suite all --output /tmp/all-mechanics.json
python evaluations/run.py --scripted --suite all --baseline /tmp/all-mechanics.json --output /tmp/after.json
python evaluations/run.py --profile small --allow-execution --baseline /tmp/small-quality.json --output /tmp/small-after.json
```

Reports include model requests, tool calls, retries, generated tokens, elapsed time, verification state, and per-case success criteria. Baseline comparisons show metric deltas and success changes; modes must match. Keep environment, model settings, and workloads comparable, and repeat real-model runs before attributing timing differences to an optimization. Loop fault injection requires `--scripted`; its web case tests permission enforcement against deliberately hostile tool content, not a model's resistance to prompt injection. Actual abrupt-process crash recovery is additionally exercised by the subprocess regression tests.
