"""Repair interrupted transcripts without replaying side-effecting operations."""
import json
from tool_result import ToolResult


def recover_checkpoint(messages, checkpoint, live_processes=()):
    calls=checkpoint.setdefault('calls',{})
    restored=[]
    for message in messages:
        if message.get('role')=='tool':
            continue
        restored.append(message)
        if message.get('role')!='assistant' or not message.get('tool_calls'):
            continue
        if not isinstance(message['tool_calls'],list) or any(not isinstance(c,dict) or
            not isinstance(c.get('id'),str) or not isinstance(c.get('function'),dict) for c in message['tool_calls']):
            raise ValueError('Invalid checkpoint tool-call envelope')
        for call in message['tool_calls']:
            key=call['id']
            existing=next((m for m in messages if m.get('tool_call_id')==key),None)
            record=calls.get(key,{})
            if not isinstance(record,dict):
                raise ValueError('Invalid checkpoint call record')
            outcome=record.get('result',{})
            if existing and not outcome:
                try:
                    parsed=json.loads(existing['content'])
                    outcome=parsed if isinstance(parsed,dict) else {}
                except (ValueError,TypeError):
                    pass
            if not isinstance(outcome,dict):
                raise ValueError('Invalid checkpoint result')
            process=outcome.get('data') if isinstance(outcome.get('data'),dict) else {}
            was_running = outcome.get('status')=='running' and process.get('process_id') not in live_processes
            uncertain = record.get('state') in ('executing','interrupted') or was_running
            if uncertain:
                effects=record.get('side_effects','unknown')
                required=record.get('inspection_required',effects in ('filesystem','process','unknown'))
                result=ToolResult.error('interrupted_operation','Operation was interrupted; do not assume it completed or replay it.')
                result.data={'inspection_required':required,'side_effects_unknown':effects in ('filesystem','process','unknown'),
                             'tool':call['function']['name'],'arguments':record.get('arguments')}
                record.update(state='interrupted',inspection_required=required)
                calls[key]=record
                content=json.dumps(result.to_dict())
            elif record.get('state')=='completed' and record.get('model_output'):
                content=record['model_output']
            elif existing:
                content=existing['content']
            else:
                result=ToolResult.error('not_executed','Pending call was never executed; choose the next action explicitly.')
                content=json.dumps(result.to_dict())
                calls.setdefault(key,{}).update(state='interrupted',inspection_required=False)
            restored.append({'role':'tool','tool_call_id':key,'content':content})
    return restored
