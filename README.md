# Local Coder

A local-first coding harness with a terminal CLI and an MCP server. Run a GGUF model through embedded llama.cpp, or connect to an OpenAI-compatible model server. The harness can inspect code, apply targeted patches, run checks, retain working context, and resume sessions.

Embedded inference stays on your machine. The server backend sends prompts and file context to the endpoint you configure; use a local endpoint for local inference.

## Install

Python 3.10+, a C/C++ build toolchain for llama-cpp-python, and [ripgrep](https://github.com/BurntSushi/ripgrep) are required. Node.js is needed only for optional external MCP filesystem tools.

```bash
python -m venv llm
source llm/bin/activate
pip install -r requirements.txt
pip install pytest pytest-asyncio
python -m pytest -q
```

For a CPU build when compiler environment variables point to unavailable tools:

```bash
CC=gcc CXX=g++ CMAKE_BUILD_PARALLEL_LEVEL=2 pip install -r requirements.txt
```

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

`--chat-format` selects an embedded llama.cpp chat format. `--no-tools` marks a model that cannot use native tool calls. `--no-stream` disables streaming when a server does not support it. Optional API authentication uses `LOCAL_CODER_API_KEY`, or the environment variable named by `api_key_env` in configuration; never put key values in configuration or source. `request_timeout` controls the server's socket timeout (default 60 seconds).

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

The native tools are `read_file` (line ranges), `list_directory`, `search_code` (content search via `rg`), `apply_patch` (exactly one matching block, or creation of a new file), `git_diff`, and—in execute mode—`run_command`, `poll_process`, and `cancel_process`. Commands take an argv array and return output, exit status, and a process handle. Their timeout is 1–300 seconds. The runtime terminates remaining command processes when it closes.

Native tools work without Node. `--mcp` adds known read-only tools from `@modelcontextprotocol/server-filesystem`; `--no-mcp` is the default. Arbitrary MCP tools and MCP mutation tools are not exposed. Repository instructions and tool output cannot elevate the selected mode.

Permission modes:

| Mode | Read/search/diff | Targeted patches | Arbitrary commands |
| --- | --- | --- | --- |
| `read-only` | Yes | No | No |
| `workspace-edit` | Yes | Yes | No |
| `execute` | Yes | Yes | Yes |

File tools and `@file` references reject paths outside the selected workspace, including symlink escapes. Patches cannot edit `.git` or harness metadata. **Execute mode uses the host's privileges; it is not an OS sandbox.** Use a container or other isolation when running untrusted tasks.

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

It exposes `ask`, `chat`, `edit`, `get_model`, and `set_model` over stdio. Responses from coding tools include `status`, `text`, budgets consumed, and `session_id`. The server defaults to read-only. Set `LOCAL_CODER_PERMISSION_MODE=workspace-edit` or `execute` at launch to enable those capabilities. MCP arguments cannot grant permissions. See the [setup guide](skills/local-coder/references/setup.md).

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

Dependencies and inference behavior are covered by separate checks. The native MCP SDK is constrained to 1.x because this repository uses its 1.x result API.

## Inference measurements and tuning

```bash
python main.py models --threads 4 --batch-threads 4 --batch-size 512 --micro-batch-size 256
python main.py models --flash-attention --key-cache-type q8_0 --value-cache-type q8_0
python main.py benchmark --output /tmp/inference.json --warmups 1 --repeats 3
```

Thread, batch, attention, and KV-cache options apply to embedded inference. KV types are `f16`, `q8_0`, or `q4_0`; quantized value caches require Flash Attention. Backend/platform support still determines whether a setting works. Benchmark before adopting a setting; larger batches are not always faster.

Reports retain individual samples and medians for request latency, load time, first streamed output, and available native prompt/generation throughput. Unsupported rates remain null; server end-to-end latency is not mislabeled as decode speed. A warm-up excludes model loading from the sampled steady-state requests; use `--warmups 0` to include a cold first request. Run results also contain total, context preparation, model, and tool timings and call counts. Real benchmarks require a configured model.

`--task-kind auto` exposes inspection tools in read-only mode and native coding tools in edit/execute modes. Use `answer` for a tool-free question, `inspect` for reads/searches, `code` for permitted native tools, or `all` to include optional MCP reads. Selection narrows capabilities; it never grants permissions. Default reads return 100 lines, `@file` preloads are bounded to 8 KB, and unchanged preloads already retained in the transcript are not injected twice. Changed files and references whose old context was compacted away are injected again.

`batch_read` and `batch_search` accept up to eight independent requests and return results in request order. Up to four reads/searches run concurrently; edits and commands remain ordered. Duplicate reads within one model response share an execution. Reads across steps execute again to detect changes, but identical retained observations are referenced instead of appended in full. Errors and command results are never reused.

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
