import json
import pytest

from artifact_store import ArtifactStore
from session import ContextManager
from tool_result import ToolResult
from workspace_tools import WorkspaceTools


def exchange(i, name='read_file', status='success', text=None):
    data={'output':text or ('observation '+str(i)+' ')*180}
    if name=='run_command':
        data.update(exit_code=1 if status=='failed' else 0, running=False)
    return [{'role':'assistant','content':None,'tool_calls':[{'id':str(i),'type':'function',
        'function':{'name':name,'arguments':json.dumps({'path':'a.py'})}}]},
        {'role':'tool','tool_call_id':str(i),'content':json.dumps(ToolResult(status=status,data=data).to_dict())}]


def test_active_turn_compacts_complete_pairs_and_preserves_failure_evidence(tmp_path):
    context=ContextManager(5500,artifact_dir=tmp_path / '.local-coder/artifacts')
    messages=[{'role':'system','content':'rules'},{'role':'user','content':'Fix the failing check'}]
    messages += exchange(0,'run_command','failed')
    messages += exchange(1,'apply_patch',text='Patched a.py')
    for i in range(2,7):
        messages += exchange(i)
    context.fit(messages,[],300)
    assert context.size(messages,[])<=5200
    assert messages[-1]['tool_call_id']=='6'
    ids={c['id'] for m in messages for c in m.get('tool_calls',[])}
    assert ids=={m['tool_call_id'] for m in messages if m['role']=='tool'}
    memory=next(m for m in messages if m.get('name')=='active_tool_memory')
    assert memory['role']=='user' and 'failed' in memory['content'] and 'apply_patch' in memory['content']
    assert list((tmp_path / '.local-coder/artifacts').glob('*.txt'))


def test_last_large_failure_preserves_status_exit_code_and_retrievable_archive(tmp_path):
    context=ContextManager(2400,artifact_dir=tmp_path / '.local-coder/artifacts')
    messages=[{'role':'user','content':'Check'}]+exchange(0,'run_command','failed',text='failure trace\n'*1000)
    context.fit(messages,[],100)
    result=json.loads(messages[-1]['content'])
    assert result['status']=='failed' and result['data']['exit_code']==1
    tools=WorkspaceTools(tmp_path)
    archive=tools.execute_tool('read_artifact',{'artifact_id':result['artifacts'][0],'max_bytes':12000})
    assert archive.status=='success' and 'failure trace' in archive.data['text']
    assert not archive.data['eof']


def test_artifact_reader_rejects_traversal_and_symlink_escape(tmp_path):
    store=ArtifactStore(tmp_path / 'artifacts')
    key=store.put('data'*2000)
    first=store.read(key,max_bytes=100)
    assert first['next_offset']==100 and not first['eof']
    assert store.read(key,offset=7900,max_bytes=100)['next_offset']==8000
    with pytest.raises(ValueError):
        store.read('../private')
    (tmp_path / 'artifacts' / key).unlink()
    (tmp_path / 'private').write_text('private')
    (tmp_path / 'artifacts' / key).symlink_to(tmp_path / 'private')
    with pytest.raises(OSError):
        store.read(key)
    with pytest.raises(ValueError):
        store.put('data'*2000)
