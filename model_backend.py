"""Model adapters for embedded llama.cpp and local OpenAI-compatible servers."""
import json
import os
import urllib.request
import threading
import time
from urllib.parse import urlparse


TUNING_INTS = ('n_threads', 'n_threads_batch', 'n_batch', 'n_ubatch')
KV_TYPES = ('f16', 'q8_0', 'q4_0')


def validate_tuning(profile):
    for key in TUNING_INTS:
        value = profile.get(key)
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f'{key} must be a positive integer')
    if profile.get('n_batch') and profile.get('n_ubatch') and profile['n_ubatch'] > profile['n_batch']:
        raise ValueError('n_ubatch cannot exceed n_batch')
    for key in ('type_k', 'type_v'):
        if profile.get(key) is not None and profile[key] not in KV_TYPES:
            raise ValueError(f'{key} must be one of {KV_TYPES}')
    if 'flash_attn' in profile and type(profile['flash_attn']) is not bool:
        raise ValueError('flash_attn must be a boolean')
    if profile.get('type_v', 'f16') != 'f16' and not profile.get('flash_attn'):
        raise ValueError('Quantized value cache requires flash_attn')


class ModelAdapter:
    def __init__(self, profile):
        validate_tuning(profile)
        self.profile = dict(profile)
        self.emit = lambda event: None
        self.cancel_event = None
        self._first_output = None
        self.load_seconds = 0.0

    def create_chat_completion(self, **kwargs):
        started = time.perf_counter()
        old_load = self.load_seconds
        self._first_output = None
        response = self._complete(**kwargs)
        elapsed = time.perf_counter() - started
        usage = response.get('usage') or {}
        metrics = {'elapsed_seconds': elapsed, 'load_seconds': self.load_seconds - old_load,
                   'time_to_first_output_seconds': self._first_output - started if self._first_output else None,
                   'prompt_tokens': usage.get('prompt_tokens'), 'completion_tokens': usage.get('completion_tokens'),
                   'prompt_tokens_per_second': None, 'generation_tokens_per_second': None}
        metrics.update(self.backend_metrics())
        response['performance'] = metrics
        self.emit({'type': 'inference_metrics', **metrics})
        return response

    def backend_metrics(self):
        return {}

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
                if (delta.get('content') or delta.get('tool_calls')) and self._first_output is None:
                    self._first_output = time.perf_counter()
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
            kwargs.update({k: self.profile[k] for k in TUNING_INTS + ('flash_attn',) if k in self.profile})
            if 'n_batch' in kwargs and 'n_ubatch' not in kwargs:
                kwargs['n_ubatch'] = min(kwargs['n_batch'], 512)
            from llama_cpp import llama_cpp
            for key in ('type_k', 'type_v'):
                if key in self.profile:
                    kwargs[key] = getattr(llama_cpp, 'GGML_TYPE_' + self.profile[key].upper())
            kwargs['no_perf'] = False
            started = time.perf_counter()
            self._model = Llama(**kwargs, verbose=False)
            self.load_seconds += time.perf_counter() - started
            self.emit({'type': 'model_loaded', 'load_seconds': self.load_seconds})
        return self._model

    def count_tokens(self, text):
        with self._lock:
            return len(self.load().tokenize(text.encode('utf-8')))

    def _complete(self, **kwargs):
        if not self.profile.get('supports_tools', True):
            kwargs.pop('tools', None)
        kwargs['stream'] = self.profile.get('stream', True)
        with self._lock:
            model = self.load()
            from llama_cpp import llama_cpp
            if hasattr(model, '_ctx'):
                llama_cpp.llama_perf_context_reset(model._ctx.ctx)
            result = model.create_chat_completion(**kwargs)
            return self.collect(result) if kwargs['stream'] else result

    def backend_metrics(self):
        try:
            from llama_cpp import llama_cpp
            stats = llama_cpp.llama_perf_context(self._model._ctx.ctx)
            return {'prompt_tokens_per_second': stats.n_p_eval / (stats.t_p_eval_ms / 1000) if stats.t_p_eval_ms > 0 else None,
                    'generation_tokens_per_second': stats.n_eval / (stats.t_eval_ms / 1000) if stats.t_eval_ms > 0 else None,
                    'evaluated_prompt_tokens': stats.n_p_eval, 'evaluated_generation_tokens': stats.n_eval}
        except (AttributeError, TypeError):
            return {}


class OpenAIModel(ModelAdapter):
    def __init__(self, profile):
        super().__init__(profile)
        base = profile.get('base_url', 'http://127.0.0.1:8080/v1').rstrip('/')
        parsed = urlparse(base)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('base_url must be an HTTP(S) URL without embedded credentials')
        self.url = base + '/chat/completions'

    def _complete(self, **kwargs):
        kwargs['model'] = self.profile.get('model', 'local-model')
        kwargs['stream'] = self.profile.get('stream', True)
        if not self.profile.get('supports_tools', True):
            kwargs.pop('tools', None)
        if kwargs['stream']:
            kwargs['stream_options'] = {'include_usage': True}
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
