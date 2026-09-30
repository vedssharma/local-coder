"""UI-independent sessions, model execution, events, and cleanup."""
from dataclasses import asdict
import json
from pathlib import Path
import threading

from agent import run_agent, RunBudget
from model_backend import ModelAdapter
from prompt_builder import build_messages
from session import ContextManager, SessionStore
from workspace_tools import WorkspaceTools


class Runtime:
    def __init__(self, model, workspace, state_dir, mode='read-only', mcp_client=None,
                 emit=None, budget=None, context_window=8192, trace=False):
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

    def emit(self, event):
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

    def new(self):
        self.messages = []
        self.session_id = None
        self.context.memory = []

    def turn(self, prompt, file_contents=None, max_tokens=512):
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError('max_tokens must be positive')
        self.cancel_event.clear()
        history = [m for m in self.messages if m.get('role') != 'system' or m.get('name') == 'working_memory']
        self.messages = build_messages(prompt, file_contents or {}, history=history, root=self.tools.root)
        self.session_id = self.store.save(self.messages, self.session_id)
        if isinstance(self.model, ModelAdapter):
            self.model.emit = self.emit
            self.model.cancel_event = self.cancel_event
        try:
            result = run_agent(self.model, self.messages, max_tokens, self.tools,
                               budget=self.budget, cancel_event=self.cancel_event,
                               emit=self.emit, context_manager=self.context)
            self.emit({'type': 'turn_result', **asdict(result)})
            return result
        finally:
            self.store.save(self.messages, self.session_id)

    def close(self):
        self.cancel_event.set()
        self.tools.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
