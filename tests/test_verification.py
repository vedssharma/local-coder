from unittest.mock import MagicMock

from runtime import Runtime
from tool_result import ToolResult
from verification import VerificationLedger
from tests.helpers import response, answer


def test_failed_check_overrides_model_success_claim(tmp_path):
    model=MagicMock()
    model.create_chat_completion.side_effect=[response('bash',{
        'command':'exit 1','verification':True}),answer()]
    with Runtime(model,tmp_path,tmp_path/'state',mode='execute') as runtime:
        result=runtime.turn('verify the change')
        assert result.status=='blocked' and result.reason=='verification_failed'
        assert result.verification_status=='failed' and result.checks[0]['exit_code']==1
        assert result.text.startswith('Verification failed:')
        assert runtime.messages[-1]['name']=='verification_evidence'


def test_edit_without_check_is_explicitly_unverified(tmp_path):
    model=MagicMock()
    model.create_chat_completion.side_effect=[response('write',{
        'path':'new.py','content':'print(1)\n'}),answer()]
    with Runtime(model,tmp_path,tmp_path/'state',mode='workspace-edit') as runtime:
        result=runtime.turn('create file')
        assert result.changed_files==['new.py']
        assert result.verification_status=='not_run' and result.checks==[]
        assert 'not verified' in result.text


def check(ledger, call_id, code=0, identity='test'):
    ledger.observe(call_id,'bash',{'command':identity,'verification':True},
        ToolResult.process({'process_id':call_id,'running':False,'exit_code':code,'output':''}))


def test_latest_same_check_replaces_failure_but_other_failure_remains():
    tools=MagicMock(); tools.processes={}
    ledger=VerificationLedger()
    check(ledger,'failed',1)
    check(ledger,'rerun',0)
    assert ledger.summarize(tools)['verification_status']=='passed'
    check(ledger,'other_failed',2,'lint')
    assert ledger.summarize(tools)['verification_status']=='failed'
    assert len(ledger.summarize(tools)['checks'])==2


def test_check_before_edit_is_stale_until_rerun():
    tools=MagicMock();tools.processes={}
    ledger=VerificationLedger()
    check(ledger,'before')
    ledger.observe('patch','edit',{},ToolResult(data='patched',changed_files=['a.py']))
    assert ledger.summarize(tools)['verification_status']=='stale'
    check(ledger,'after')
    assert ledger.summarize(tools)['verification_status']=='passed'
    assert ledger.summarize(tools,{'unknown':{}})['verification_status']=='requires_review'

