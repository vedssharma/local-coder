import json
import threading
from unittest.mock import MagicMock

from agent import run_agent
from workspace_tools import WorkspaceTools, schema, STRING
from tool_registry import ToolSpec


def responses(names):
    return [{'choices': [{'message': {'content': None, 'tool_calls': [
        {'id': str(i), 'type': 'function', 'function': {'name': n, 'arguments': '{}'}} for i,n in enumerate(names)]},
        'finish_reason': 'tool_calls'}]}, {'choices': [{'message': {'content': 'done'}, 'finish_reason': 'stop'}]}]


def test_independent_reads_overlap_but_results_keep_call_order(tmp_path):
    tools = WorkspaceTools(tmp_path)
    barrier = threading.Barrier(2)
    def handler(value):
        def run(args):
            barrier.wait(timeout=2)
            return value
        return run
    for name in ('first', 'second'):
        tools.registry.register(ToolSpec(schema(name, name, {}), handler(name),
            side_effects='none', concurrency='parallel', task_kinds=('all',)))
    model = MagicMock()
    model.create_chat_completion.side_effect = responses(['first', 'second'])
    messages=[]
    assert run_agent(model, messages, mcp_client=tools, tool_workers=2).status == 'completed'
    assert [json.loads(m['content'])['data'] for m in messages if m['role']=='tool'] == ['first','second']


def test_mutation_is_a_barrier_and_duplicate_reads_share_one_execution(tmp_path):
    tools=WorkspaceTools(tmp_path, mode='workspace-edit')
    state={'value':'old'}
    read=MagicMock(side_effect=lambda args:state['value'])
    def edit(args):
        state['value']='new'
        return 'edited'
    tools.registry.register(ToolSpec(schema('read', 'Read', {}), read, side_effects='none',
        concurrency='parallel', cacheable=True))
    tools.registry.register(ToolSpec(schema('edit', 'Edit', {}), edit, minimum_mode='workspace-edit', side_effects='filesystem'))
    model=MagicMock();model.create_chat_completion.side_effect=responses(['read','read','edit','read'])
    messages=[]
    assert run_agent(model,messages,mcp_client=tools).status=='completed'
    assert read.call_count==2
    assert [json.loads(m['content'])['data'] for m in messages if m['role']=='tool']==['old','old','edited','new']


def test_worker_limit_and_serial_unknown_tools(tmp_path):
    tools=WorkspaceTools(tmp_path)
    active=0
    peak=0
    lock=threading.Lock()
    barrier=threading.Barrier(2)
    def read(args):
        nonlocal active,peak
        with lock:
            active+=1;peak=max(peak,active)
        barrier.wait(timeout=2)
        with lock:
            active-=1
        return 'read'
    for i in range(4):
        tools.registry.register(ToolSpec(schema(f'read{i}', 'Read', {}),read,side_effects='none',concurrency='parallel'))
    model=MagicMock();model.create_chat_completion.side_effect=responses([f'read{i}' for i in range(4)])
    assert run_agent(model,[],mcp_client=tools,tool_workers=2).status=='completed'
    assert peak==2
    ordered=[]
    for name in ('a','b'):
        tools.registry.register(ToolSpec(schema(name,name,{}),lambda args,name=name:ordered.append(name) or name))
    model.create_chat_completion.side_effect=responses(['a','b'])
    assert run_agent(model,[],mcp_client=tools).status=='completed'
    assert ordered==['a','b']
