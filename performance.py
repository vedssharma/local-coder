"""Repeatable inference measurements; unavailable measurements remain null."""
import statistics


def benchmark(model, prompt, repeats=3, warmups=1, max_tokens=128):
    if repeats < 1 or warmups < 0 or max_tokens < 1:
        raise ValueError('Invalid benchmark counts')
    messages = [{'role': 'user', 'content': prompt}]
    samples = []
    for index in range(warmups + repeats):
        response = model.create_chat_completion(messages=messages, max_tokens=max_tokens,
                                                temperature=0, stream=False)
        choice = response['choices'][0]
        sample = dict(response.get('performance', {}))
        sample['finish_reason'] = choice.get('finish_reason')
        sample['completed'] = choice.get('finish_reason') == 'stop'
        if index >= warmups:
            samples.append(sample)
    def median(key):
        values = [s[key] for s in samples if s.get(key) is not None]
        return statistics.median(values) if values else None
    return {'version': 1, 'warmups': warmups, 'repeats': repeats, 'samples': samples,
            'median': {key: median(key) for key in ('elapsed_seconds', 'load_seconds',
                'time_to_first_output_seconds', 'prompt_tokens_per_second', 'generation_tokens_per_second',
                'end_to_end_completion_tokens_per_second')},
            'all_completed': all(s['completed'] for s in samples)}
