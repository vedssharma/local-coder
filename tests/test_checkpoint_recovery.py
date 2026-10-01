import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock

import pytest
from runtime import Runtime
from session import SessionStore


def call_response(name,args):
    return {'choices':[{'message':{'content':None,'tool_calls':[{'id':'call','type':'function',
        'function':{'name':name,'arguments':json.dumps(args)}}]},'finish_reason':'tool_calls'}]}


def answer():
    return {'choices':[{'message':{'content':'done'},'finish_reason':'stop'}]}


@pytest.mark.parametrize('persist_completed',[False,True])
def test_crash_checkpoint_does_not_replay_command(tmp_path,persist_completed):
    repo=Path(__file__).resolve().parents[1]
    script=r'''
import json,os,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from runtime import Runtime
root=Path(sys.argv[2])
class Model:
    def create_chat_completion(self,**kwargs):
        return {'choices':[{'message':{'content':None,'tool_calls':[{'id':'effect','type':'function','function':{'name':'bash','arguments':json.dumps({'command':'printf x >> counter'})}}]},'finish_reason':'tool_calls'}]}
runtime=Runtime(Model(),root,root/'state',mode='execute')
original=runtime.checkpoint_call
def checkpoint(stage,*args):
    if stage=='completed':
        if sys.argv[3]=='True':
            original(stage,*args)
        os._exit(17)
    original(stage,*args)
runtime.checkpoint_call=checkpoint
runtime.turn('append once')
'''
    result=subprocess.run([sys.executable,'-c',script,str(repo),str(tmp_path),str(persist_completed)],timeout=10)
    assert result.returncode==17 and (tmp_path/'counter').read_text()=='x'
    store=SessionStore(tmp_path/'state/sessions',tmp_path)
    key=store.list()[0]
    model=MagicMock();model.create_chat_completion.return_value=answer()
    with Runtime(model,tmp_path,tmp_path/'state',mode='execute') as runtime:
        runtime.resume(key)
        observation=json.loads(next(m['content'] for m in runtime.messages if m['role']=='tool'))
        if persist_completed:
            assert observation['status']=='success' and not runtime.interrupted_operations
        else:
            assert observation['error_code']=='interrupted_operation' and runtime.interrupted_operations
            blocked=runtime.tools.registry.get('bash')
            model.create_chat_completion.side_effect=[call_response('bash',{'command':'printf x >> counter'}),answer()]
        assert runtime.turn('continue without replaying').status=='completed'
        assert (tmp_path/'counter').read_text()=='x'
        if not persist_completed:
            assert any('inspection_required' in m.get('content','') for m in runtime.messages if m['role']=='tool')
            runtime.acknowledge_interrupted()
            assert not runtime.interrupted_operations


def test_interrupted_patch_requires_reading_affected_file(tmp_path):
    (tmp_path/'a').write_text('applied')
    store=SessionStore(tmp_path/'state/sessions',tmp_path)
    message=call_response('apply_patch',{'path':'a','old_text':'old','new_text':'applied'})['choices'][0]['message']
    message['role']='assistant'
    key=store.save([{'role':'user','content':'edit'},message],checkpoint={'calls':{'call':{
        'state':'executing','name':'apply_patch','side_effects':'filesystem',
        'arguments':{'path':'a','old_text':'old','new_text':'applied'}}}})
    model=MagicMock();model.create_chat_completion.side_effect=[call_response('read_file',{'path':'a'}),
        call_response('apply_patch',{'path':'a','old_text':'applied','new_text':'reviewed'}),answer()]
    with Runtime(model,tmp_path,tmp_path/'state',mode='workspace-edit') as runtime:
        runtime.resume(key)
        assert runtime.interrupted_operations
        assert runtime.turn('inspect and continue').status=='completed'
        assert not runtime.interrupted_operations
    assert (tmp_path/'a').read_text()=='reviewed'


def test_pending_calls_receive_results_without_execution_and_lease_is_exclusive(tmp_path):
    store=SessionStore(tmp_path/'state/sessions',tmp_path)
    message=call_response('bash',{'command':'touch forbidden'})['choices'][0]['message'];message['role']='assistant'
    key=store.save([{'role':'user','content':'task'},message],checkpoint={'calls':{'call':{'state':'pending','name':'bash'}}})
    with store.lease(key):
        with pytest.raises(ValueError,match='already running'):
            with SessionStore(store.directory,tmp_path).lease(key):
                pass
    with Runtime(MagicMock(),tmp_path,tmp_path/'state',mode='execute') as runtime:
        runtime.resume(key)
        assert json.loads(runtime.messages[-1]['content'])['error_code']=='not_executed'
        assert not (tmp_path/'forbidden').exists()


def test_version_one_sessions_remain_loadable(tmp_path):
    store=SessionStore(tmp_path/'sessions',tmp_path)
    key=store.save([{'role':'user','content':'old'}])
    path=store.directory/(key+'.json')
    data=json.loads(path.read_text());data['version']=1;data.pop('checkpoint');path.write_text(json.dumps(data))
    assert store.load_state(key)==([{'role':'user','content':'old'}],{})
