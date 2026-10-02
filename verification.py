"""Observed command evidence, independent of the model's completion claims."""
import json
from pathlib import Path
import shlex


def is_check(name, arguments):
    if arguments.get('verification') is True:
        return True
    if name!='bash':
        return False
    try:
        argv=shlex.split(arguments.get('command',''))
    except ValueError:
        return False
    while argv and '=' in argv[0] and not argv[0].startswith(('/','.')):
        argv=argv[1:]  # leading VAR=value assignments
    if not argv:
        return False
    program=Path(argv[0]).name
    if program.startswith('pytest'):
        return True
    if program.startswith('python') and '-m' in argv:
        index=argv.index('-m')
        return index+1<len(argv) and argv[index+1] in ('pytest','unittest')
    return program in ('npm','pnpm','yarn','cargo','go','dotnet','mvn') and len(argv)>1 and (
        argv[1]=='test' or argv[1:3]==['run','test'])


class VerificationLedger:
    def __init__(self, state=None):
        self.state=state if state is not None else {}
        self.state.setdefault('revision',0)
        self.state.setdefault('changed_files',[])
        self.state.setdefault('commands',{})
        self.state.setdefault('observed_calls',[])

    def observe(self, call_id, name, arguments, result):
        if call_id in self.state['observed_calls']:
            return
        self.state['observed_calls'].append(call_id)
        if result.changed_files:
            if name in ('write','edit'):
                self.state['revision']+=1
            self.state['changed_files']=sorted(set(self.state['changed_files'])|set(result.changed_files))
        data=result.data
        if name!='bash' or not isinstance(data,dict) or not data.get('process_id'):
            return
        key=data['process_id']
        record=self.state['commands'].get(key,{})
        record={'name':name,'arguments':arguments,'verification':is_check(name,arguments),
                    'revision':self.state['revision'],'identity':json.dumps([name,arguments],sort_keys=True)}
        record.update(process_id=key,status=result.status,exit_code=data.get('exit_code'),
                      running=data.get('running',False),artifacts=result.artifacts,
                      output_preview=str(data.get('output',''))[-500:])
        self.state['commands'][key]=record

    def summarize(self, tools, interrupted=()):
        for key,record in self.state['commands'].items():
            if record.get('running') and key in tools.processes:
                data=tools._poll(key)
                if not data['running']:
                    from tool_result import ToolResult
                    result=ToolResult.process(data)
                    record.update(status=result.status,exit_code=data['exit_code'],running=False,
                                  output_preview=data['output'][-500:])
            elif record.get('running') and key not in tools.processes:
                record.update(status='interrupted',running=False,exit_code=None)
        checks={}
        for record in self.state['commands'].values():
            if record.get('verification'):
                checks[record.get('identity',record['process_id'])]=record
        checks=list(checks.values())
        outstanding=[r for r in self.state['commands'].values() if r.get('running')]
        if interrupted:
            status='requires_review'
        elif outstanding:
            status='in_progress'
        elif not checks:
            status='not_run'
        elif any(r['status']!='success' or r['exit_code']!=0 for r in checks):
            status='failed'
        elif any(r['revision']<self.state['revision'] for r in checks):
            status='stale'
        else:
            status='passed'
        return {'changed_files':self.state['changed_files'],'checks':checks,
                'outstanding_processes':outstanding,'verification_status':status,
                'verification_scope':'observed_commands_and_changes'}
