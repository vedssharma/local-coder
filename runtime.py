"""UI-independent sessions, model execution, events, and cleanup."""
from dataclasses import asdict
import uuid
import json
from pathlib import Path
import threading
import time

from agent import run_agent, RunBudget, default_output_tokens
from model_backend import ModelAdapter
from prompt_builder import build_messages
from session import ContextManager, SessionStore
from workspace_tools import WorkspaceTools
from execution_context import ExecutionContext
from checkpoint_recovery import recover_checkpoint
from tool_result import ToolResult, invoke_tool
from verification import VerificationLedger


class Runtime:
    def __init__(self, model, workspace, state_dir, mode='read-only',
                 emit=None, budget=None, context_window=8192, trace=False, task_kind="auto",
                 tool_workers=4, web=True):
        if type(tool_workers) is not int or not 1 <= tool_workers <= 8:
            raise ValueError('tool_workers must be between 1 and 8')
        self.tool_workers = tool_workers
        self._emit_lock = threading.RLock()
        self.model = model
        self.tools = WorkspaceTools(workspace, mode=mode, web=web)
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
        self.tool_executor = None
        self.max_tokens = default_output_tokens(context_window)

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

    def turn(self, prompt, file_contents=None, max_tokens=None):
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
                    if stage=='completed':
                        VerificationLedger(self.checkpoint.setdefault('evidence',{})).observe(call['id'],record['name'],arguments,result)
                    record.update(state='completed',model_output=result.to_model(self.context),result=json.loads(result.to_model(self.context)))
                    if result.error_code in ('io_error','execution_error') and record['side_effects'] in ('filesystem','process','unknown'):
                        record.update(state='interrupted',inspection_required=True)
            self.store.save(self.messages,self.session_id,self.checkpoint)


    def _turn(self, prompt, file_contents=None, max_tokens=None):
        if max_tokens is None:
            max_tokens = default_output_tokens(self.context.window)
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError('max_tokens must be positive')
        self.max_tokens = max_tokens
        self.cancel_event.clear()
        self.tools.turn_id = uuid.uuid4().hex
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
        self.messages = build_messages(prompt, fresh, history=history, root=self.tools.root, tools=self.tools.tool_names)
        self.session_id = self.store.save(self.messages, self.session_id,self.checkpoint)
        if isinstance(self.model, ModelAdapter):
            self.model.emit = self.emit
            self.model.cancel_event = self.cancel_event
        result = None
        execution_context = ExecutionContext(time.monotonic() + self.budget.max_seconds, self.cancel_event)
        def execute(name,args):
            spec=self.tools.registry.get(name)
            if self.interrupted_operations and spec and spec.side_effects in ('filesystem','process','unknown'):
                return ToolResult.error('inspection_required','Inspect interrupted operations before making further changes; commands require explicit acknowledgement.')
            if self.cancel_event.is_set():
                return ToolResult(status='cancelled', error_code='interrupted')
            with execution_context.bind():
                outcome=invoke_tool(self.tools,name,args)
            if name=='read' and (outcome.status=='success' or outcome.error_code=='not_found'):
                for record in self.interrupted_operations.values():
                    if record.get('side_effects')=='filesystem' and isinstance(record.get('arguments'),dict):
                        original=record['arguments'].get('path')
                        if original and self.tools.path(original)==self.tools.path(args['path']):
                            record['inspection_required']=False
            return outcome
        self.tool_executor = execute
        try:
            result = run_agent(self.model, self.messages, max_tokens, self.tools,
                               budget=self.budget, cancel_event=self.cancel_event,
                               emit=self.emit, context_manager=self.context,
                               tool_schemas=self.tools.selected_schemas(self.task_kind),
                               tool_executor=execute, execution_context=execution_context, tool_workers=self.tool_workers,
                               checkpoint=self.checkpoint_call, reserved_call_ids=set(self.checkpoint.get('calls',{})))
            if result.status in ('cancelled','budget_exhausted'):
                self.tools.cancel_all_processes()
            evidence=VerificationLedger(self.checkpoint.setdefault('evidence',{})).summarize(self.tools,self.interrupted_operations)
            for field,value in evidence.items():
                setattr(result,field,value)
            if result.verification_status in ('failed','in_progress','stale','requires_review') and result.status=='completed':
                result.status='blocked'
                result.reason='verification_'+result.verification_status
                result.text='Verification '+result.verification_status+': the observed evidence does not establish successful completion.\n\nModel summary (not verification evidence):\n'+result.text
                self.messages.append({'role':'user','name':'verification_evidence','content':json.dumps(evidence)})
            elif result.changed_files and result.verification_status=='not_run':
                result.text='Changes are not verified: no validation command was observed.\n\nModel summary (not verification evidence):\n'+result.text
            self.emit({'type':'verification_result',**evidence})
            self.emit({'type': 'turn_result', **asdict(result)})
            return result
        finally:
            self.tool_executor = None
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
