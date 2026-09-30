"""Model adapters for embedded llama.cpp and local OpenAI-compatible servers."""
import json
import os
import urllib.request
import threading
from urllib.parse import urlparse


class ModelAdapter:
    def __init__(self, profile):
        self.profile = profile
        self.emit = lambda event: None
        self.cancel_event = None

    def count_tokens(self, text):
        # Conservative fallback for servers without a tokenizer API.
        return len(text.encode('utf-8'))

    def collect(self, chunks):
        content, calls, reason, usage = [], {}, None, {}
        for chunk in chunks:
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise KeyboardInterrupt
            usage = chunk.get('usage') or usage
            for choice in chunk.get('choices', []):
                if choice.get('index', 0) != 0:
                    continue
                delta = choice.get('delta', {})
                if delta.get('content'):
                    content.append(delta['content'])
                    self.emit({'type': 'assistant_delta', 'text': delta['content']})
                for part in delta.get('tool_calls') or []:
                    index = part.get('index', 0)
                    entry = calls.setdefault(index, {'id': '', 'type': 'function',
                                                     'function': {'name': '', 'arguments': ''}})
                    if part.get('id'):
                        entry['id'] = part['id']
                    for key in ('name', 'arguments'):
                        entry['function'][key] += part.get('function', {}).get(key) or ''
                reason = choice.get('finish_reason') or reason
        if reason is None:
            raise ValueError('Model stream ended without a finish reason')
        return {'choices': [{'message': {'role': 'assistant', 'content': ''.join(content) or None,
                                        'tool_calls': [calls[k] for k in sorted(calls)] or None},
                             'finish_reason': reason}], 'usage': usage}


class EmbeddedModel(ModelAdapter):
    def __init__(self, profile):
        super().__init__(profile)
        self._model = None
        self._lock = threading.RLock()

    def load(self):
        if self._model is None:
            from llama_cpp import Llama
            kwargs = {key: self.profile[key] for key in ('model_path', 'n_ctx', 'n_gpu_layers')}
            if self.profile.get('chat_format'):
                kwargs['chat_format'] = self.profile['chat_format']
            self._model = Llama(**kwargs, verbose=False)
        return self._model

    def count_tokens(self, text):
        with self._lock:
            return len(self.load().tokenize(text.encode('utf-8')))

    def create_chat_completion(self, **kwargs):
        if not self.profile.get('supports_tools', True):
            kwargs.pop('tools', None)
        kwargs['stream'] = self.profile.get('stream', True)
        with self._lock:
            result = self.load().create_chat_completion(**kwargs)
            return self.collect(result) if kwargs['stream'] else result


class OpenAIModel(ModelAdapter):
    def __init__(self, profile):
        super().__init__(profile)
        base = profile.get('base_url', 'http://127.0.0.1:8080/v1').rstrip('/')
        parsed = urlparse(base)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('base_url must be an HTTP(S) URL without embedded credentials')
        self.url = base + '/chat/completions'

    def create_chat_completion(self, **kwargs):
        kwargs['model'] = self.profile.get('model', 'local-model')
        kwargs['stream'] = self.profile.get('stream', True)
        if not self.profile.get('supports_tools', True):
            kwargs.pop('tools', None)
        headers = {'Content-Type': 'application/json'}
        key_name = self.profile.get('api_key_env', 'LOCAL_CODER_API_KEY')
        if os.environ.get(key_name):
            headers['Authorization'] = 'Bearer ' + os.environ[key_name]
        request = urllib.request.Request(self.url, data=json.dumps(kwargs).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=self.profile.get('request_timeout', 60)) as response:
            if not kwargs['stream']:
                return json.load(response)
            def chunks():
                for line in response:
                    if not line.startswith(b'data:'):
                        continue
                    data = line[5:].strip()
                    if data == b'[DONE]':
                        return
                    yield json.loads(data)
            return self.collect(chunks())


def create_model(profile):
    backend = profile.get('backend', 'embedded')
    if backend == 'embedded':
        return EmbeddedModel(profile)
    if backend == 'openai':
        return OpenAIModel(profile)
    raise ValueError(f'Unknown model backend: {backend}')
