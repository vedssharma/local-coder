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
from checkpoint_recovery import recover_checkpoint
from tool_result import ToolResult


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
        self.checkpoint = {}
        self._checkpoint_lock = threading.RLock()
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
        with self.store.lease(key):
            self.messages,self.checkpoint = self.store.load_state(key)
            self.messages = recover_checkpoint(self.messages,self.checkpoint,self.tools.processes)
            self.store.save(self.messages,key,self.checkpoint)
        self.session_id = key
        self.context.memory = []
        self.context.task = None
        self.context.active_notes = []

    def new(self):
        self.checkpoint = {}
        self.messages = []
        self.session_id = None
        self.context.memory = []
        self.context.task = None
        self.context.active_notes = []

    def turn(self, prompt, file_contents=None, max_tokens=512):
        if self.session_id is None:
            self.session_id = self.store.save(self.messages,checkpoint=self.checkpoint)
        with self.store.lease(self.session_id):
            self.messages,self.checkpoint = self.store.load_state(self.session_id)
            self.messages = recover_checkpoint(self.messages,self.checkpoint,self.tools.processes)
            return self._turn(prompt,file_contents,max_tokens)

    @property
    def interrupted_operations(self):
        return {key:record for key,record in self.checkpoint.get('calls',{}).items() if record.get('inspection_required')}

    def acknowledge_interrupted(self):
        if self.session_id is None:
            return
        with self.store.lease(self.session_id):
            for record in self.checkpoint.get('calls',{}).values():
                record['inspection_required']=False
            self.store.save(self.messages,self.session_id,self.checkpoint)

    def checkpoint_call(self, stage, call, arguments, result):
        with self._checkpoint_lock:
            calls=self.checkpoint.setdefault('calls',{})
            if stage=='batch':
                for entry in call['tool_calls']:
                    calls[entry['id']]={'state':'pending','name':entry['function']['name']}
            else:
                record=calls.setdefault(call['id'],{})
                record.update(name=call['function']['name'],arguments=arguments)
                spec=self.tools.registry.get(record['name'])
                record['side_effects']=spec.side_effects if spec else 'unknown'
                if stage=='executing':
                    record['state']='executing'
                else:
                    record.update(state='completed',model_output=result.to_model(self.context),result=json.loads(result.to_model(self.context)))
                    if result.error_code in ('io_error','execution_error') and record['side_effects'] in ('filesystem','process','unknown'):
                        record.update(state='interrupted',inspection_required=True)
            self.store.save(self.messages,self.session_id,self.checkpoint)

    def _turn(self, prompt, file_contents=None, max_tokens=512):
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
        self.session_id = self.store.save(self.messages, self.session_id,self.checkpoint)
        if isinstance(self.model, ModelAdapter):
            self.model.emit = self.emit
            self.model.cancel_event = self.cancel_event
        result = None
        orchestrator = ProcessOrchestrator(self.tools, self.process_wait_seconds,
            time.monotonic() + self.budget.max_seconds, self.cancel_event, self.emit)
        def execute(name,args):
            spec=self.tools.registry.get(name)
            if self.interrupted_operations and spec and spec.side_effects in ('filesystem','process','unknown'):
                return ToolResult.error('inspection_required','Inspect interrupted operations before making further changes; commands require explicit acknowledgement.')
            outcome=orchestrator(name,args)
            if name=='read_file' and (outcome.status=='success' or outcome.error_code=='not_found'):
                for record in self.interrupted_operations.values():
                    if record.get('side_effects')=='filesystem' and isinstance(record.get('arguments'),dict):
                        original=record['arguments'].get('path')
                        if original and self.tools.path(original)==self.tools.path(args['path']):
                            record['inspection_required']=False
            return outcome
        try:
            result = run_agent(self.model, self.messages, max_tokens, self.tools,
                               budget=self.budget, cancel_event=self.cancel_event,
                               emit=self.emit, context_manager=self.context,
                               tool_schemas=self.tools.selected_schemas(self.task_kind),
                               tool_executor=execute, execution_context=orchestrator.context, tool_workers=self.tool_workers,
                               checkpoint=self.checkpoint_call, reserved_call_ids=set(self.checkpoint.get('calls',{})))
            self.emit({'type': 'turn_result', **asdict(result)})
            return result
        finally:
            if isinstance(self.model, ModelAdapter):
                self.model.execution_context = None
            if self.cancel_event.is_set() or (result and result.status in ('cancelled', 'budget_exhausted')):
                self.tools.cancel_all_processes()
            self.store.save(self.messages, self.session_id,self.checkpoint)

    def close(self):
        self.cancel_event.set()
        self.tools.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
