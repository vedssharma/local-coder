import json


def response(name, args):
    return {'choices': [{'message': {'content': None, 'tool_calls': [
        {'id': 'process', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]},
        'finish_reason': 'tool_calls'}]}


def answer():
    return {'choices': [{'message': {'content': 'done'}, 'finish_reason': 'stop'}]}
