"""UI-independent sessions, model execution, events, and cleanup."""
from dataclasses import asdict
import json
from pathlib import Path
import threading
import time

from agent import run_agent, RunBudget
from model_backend import ModelAdapter
from prompt_builder import build_messages
from session import ContextManager, SessionStore
from workspace_tools import WorkspaceTools
from process_orchestration import ProcessOrchestrator, validate_wait_seconds


class Runtime:
    def __init__(self, model, workspace, state_dir, mode='read-only', mcp_client=None,
                 emit=None, budget=None, context_window=8192, trace=False, task_kind="auto",
                 process_wait_seconds=2, tool_workers=4):
        if type(tool_workers) is not int or not 1 <= tool_workers <= 8:
            raise ValueError('tool_workers must be between 1 and 8')
        self.tool_workers = tool_workers
        self._emit_lock = threading.RLock()
        self.process_wait_seconds = validate_wait_seconds(process_wait_seconds)
        self.model = model
        self.tools = WorkspaceTools(workspace, mcp_client=mcp_client, mode=mode)
        self.store = SessionStore(Path(state_dir) / 'sessions', workspace)
        self.context = ContextManager(context_window,
            count_tokens=model.count_tokens if isinstance(model, ModelAdapter) else None,
            artifact_dir=self.tools.path('.local-coder/artifacts'))
        self.messages = []
        self.session_id = None
        self.cancel_event = threading.Event()
        self.budget = budget or RunBudget()
        self.sink = emit or (lambda event: None)
        self.trace = trace
        self.task_kind = task_kind

    def emit(self, event):
        with self._emit_lock:
            self.sink(event)
            if self.trace:
                directory = self.tools.path('.local-coder/traces')
                directory.mkdir(parents=True, exist_ok=True, mode=0o700)
                with (directory / (self.session_id + '.jsonl')).open('a') as f:
                    f.write(json.dumps(event) + '\n')

    def resume(self, key):
        self.messages = self.store.load(key)
        self.session_id = key
        self.context.memory = []
        self.context.task = None

    def new(self):
        self.messages = []
        self.session_id = None
        self.context.memory = []
        self.context.task = None

    def turn(self, prompt, file_contents=None, max_tokens=512):
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError('max_tokens must be positive')
        self.cancel_event.clear()
        history = [m for m in self.messages if m.get('role') != 'system' or m.get('name') == 'working_memory']
        file_contents = file_contents or {}
        fresh, reused = {}, []
        for name, content in file_contents.items():
            block = f"<file path='{name}'>\n{content}\n</file>"
            if any(block in m.get('content', '') for m in history if isinstance(m.get('content'), str)):
                reused.append(name)
            else:
                fresh[name] = content
        if reused:
            prompt += '\nUnchanged file context retained earlier in this transcript: ' + ', '.join(reused)
        self.messages = build_messages(prompt, fresh, history=history, root=self.tools.root)
        self.session_id = self.store.save(self.messages, self.session_id)
        if isinstance(self.model, ModelAdapter):
            self.model.emit = self.emit
            self.model.cancel_event = self.cancel_event
        result = None
        orchestrator = ProcessOrchestrator(self.tools, self.process_wait_seconds,
            time.monotonic() + self.budget.max_seconds, self.cancel_event, self.emit)
        try:
            result = run_agent(self.model, self.messages, max_tokens, self.tools,
                               budget=self.budget, cancel_event=self.cancel_event,
                               emit=self.emit, context_manager=self.context,
                               tool_schemas=self.tools.selected_schemas(self.task_kind),
                               tool_executor=orchestrator, execution_context=orchestrator.context, tool_workers=self.tool_workers)
            self.emit({'type': 'turn_result', **asdict(result)})
            return result
        finally:
            if isinstance(self.model, ModelAdapter):
                self.model.execution_context = None
            if self.cancel_event.is_set() or (result and result.status in ('cancelled', 'budget_exhausted')):
                self.tools.cancel_all_processes()
            self.store.save(self.messages, self.session_id)

    def close(self):
        self.cancel_event.set()
        self.tools.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
