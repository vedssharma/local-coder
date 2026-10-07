import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import pytest
from model_backend import ModelAdapter, OpenAIModel


def test_stream_assembles_fragmented_calls_and_rejects_incomplete_stream():
    adapter = ModelAdapter({})
    chunks = [
        {'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'c', 'function': {'name': 'read_file', 'arguments': '{"pa'}}]}}]},
        {'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': 'th":"a"}'}}]}, 'finish_reason': 'tool_calls'}]}]
    result = adapter.collect(chunks)
    call = result['choices'][0]['message']['tool_calls'][0]
    assert call['function'] == {'name': 'read_file', 'arguments': '{"path":"a"}'}
    with pytest.raises(ValueError, match='finish reason'):
        adapter.collect(chunks[:1])


def test_openai_transport_streams_and_sends_model_and_tools():
    received = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            for chunk in [
                {'choices': [{'delta': {'content': 'Hello '}}]},
                {'choices': [{'delta': {'content': 'world'}, 'finish_reason': 'stop'}]},
            ]:
                self.wfile.write(b'data: ' + json.dumps(chunk).encode() + b'\n\n')
            self.wfile.write(b'data: [DONE]\n\n')
        def log_message(self, *_):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        model = OpenAIModel({'base_url': f'http://127.0.0.1:{server.server_port}/v1', 'model': 'test'})
        events = []
        model.emit = events.append
        result = model.create_chat_completion(messages=[], tools=[{'type': 'function'}], max_tokens=32)
        assert result['choices'][0]['message']['content'] == 'Hello world'
        assert received[0]['model'] == 'test' and received[0]['stream']
        assert 'tools' in received[0]
        assert ''.join(e['text'] for e in events if e['type'] == 'assistant_delta') == 'Hello world'
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_tuning_reaches_native_constructor_and_invalid_values_are_rejected(monkeypatch, tmp_path):
    (tmp_path / 'm.gguf').touch()
    monkeypatch.chdir(tmp_path)
    from unittest.mock import MagicMock
    import llama_cpp
    from model_backend import EmbeddedModel
    fake = MagicMock()
    seen = {}
    def constructor(**kwargs):
        seen.update(kwargs)
        return fake
    monkeypatch.setattr(llama_cpp, 'Llama', constructor)
    model = EmbeddedModel({'model_path': 'm.gguf', 'n_ctx': 1024, 'n_gpu_layers': 0,
                           'n_threads': 4, 'n_batch': 256, 'type_k': 'q8_0'})
    model.load()
    assert seen['n_threads'] == 4 and seen['n_ubatch'] == 256
    assert seen['type_k'] == llama_cpp.llama_cpp.GGML_TYPE_Q8_0
    with pytest.raises(ValueError):
        EmbeddedModel({'n_threads': 0})
    with pytest.raises(ValueError):
        EmbeddedModel({'type_v': 'q4_0'})


def test_measurements_do_not_invent_token_rates():
    from performance import benchmark
    class Fake(ModelAdapter):
        def _complete(self, **kwargs):
            return self.collect([{'choices': [{'delta': {'content': 'ok'}, 'finish_reason': 'stop'}]}])
    report = benchmark(Fake({}), 'question', repeats=2, warmups=1)
    assert len(report['samples']) == 2
    assert report['median']['time_to_first_output_seconds'] is not None
    assert report['median']['generation_tokens_per_second'] is None
    assert report['all_completed']


def test_cache_configuration_and_stable_tool_prefix(monkeypatch, tmp_path):
    (tmp_path / 'm.gguf').touch()
    monkeypatch.chdir(tmp_path)
    from unittest.mock import MagicMock
    import llama_cpp
    from model_backend import EmbeddedModel
    fake = MagicMock()
    cache = MagicMock()
    monkeypatch.setattr(llama_cpp, 'Llama', lambda **kwargs: fake)
    monkeypatch.setattr(llama_cpp, 'LlamaRAMCache', cache)
    model = EmbeddedModel({'model_path': 'm.gguf', 'n_ctx': 1024, 'n_gpu_layers': 0, 'prompt_cache_mb': 64})
    model.load()
    cache.assert_called_once_with(capacity_bytes=64 * 1024 * 1024)
    fake.set_cache.assert_called_once()
    class Capture(ModelAdapter):
        def _complete(self, **kwargs):
            self.request = json.dumps(kwargs)
            return {'choices': [{'message': {'content': 'ok'}, 'finish_reason': 'stop'}]}
    capture = Capture({})
    tools = [{'function': {'name': 'z', 'parameters': {'b': 1, 'a': 2}}}, {'function': {'name': 'a'}}]
    capture.create_chat_completion(messages=[{'content': 'question', 'role': 'user'}], tools=tools)
    first = capture.request
    capture.create_chat_completion(messages=[{'role': 'user', 'content': 'question'}], tools=list(reversed(tools)))
    assert capture.request == first
    capture.create_chat_completion(messages=[{'role': 'user', 'content': 'changed'}], tools=tools)
    assert capture.request != first


def test_embedded_inference_reports_a_missing_model_or_llama_cpp(monkeypatch, tmp_path):
    import sys
    from model_backend import EmbeddedModel, EMBEDDED_INSTALL_HINT
    with pytest.raises(ValueError, match='does not exist'):
        EmbeddedModel({'model_path': str(tmp_path / 'missing.gguf'), 'n_ctx': 1024, 'n_gpu_layers': 0}).load()
    (tmp_path / 'm.gguf').touch()
    monkeypatch.setitem(sys.modules, 'llama_cpp', None)  # Simulates llama-cpp-python not being installed.
    with pytest.raises(RuntimeError, match='requirements-embedded.txt') as raised:
        EmbeddedModel({'model_path': str(tmp_path / 'm.gguf'), 'n_ctx': 1024, 'n_gpu_layers': 0}).load()
    assert str(raised.value) == EMBEDDED_INSTALL_HINT


def test_cli_starts_without_llama_cpp(tmp_path):
    import os, subprocess, sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    # A sitecustomize hook makes importing llama_cpp fail in the child process.
    (tmp_path / 'sitecustomize.py').write_text("import sys\nsys.modules['llama_cpp'] = None\n")
    env = {**os.environ, 'PYTHONPATH': str(tmp_path), 'LOCAL_CODER_CONFIG_DIR': str(tmp_path / 'config')}
    done = subprocess.run([sys.executable, str(root / 'main.py'), '--help'], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert 'ask' in done.stdout


def _serve(statuses, retry_after=None):
    """A server that fails with each status in turn, then succeeds without streaming."""
    statuses = list(statuses)
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            status = statuses.pop(0) if statuses else 200
            body = json.dumps({'choices': [{'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'stop'}]}
                              if status == 200 else {'error': 'busy'}).encode()
            self.send_response(status)
            if retry_after is not None and status != 200:
                self.send_header('Retry-After', retry_after)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *_):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    model = OpenAIModel({'base_url': f'http://127.0.0.1:{server.server_port}/v1', 'stream': False})
    return server, model, requests


@pytest.mark.parametrize('status', [429, 503, 529])
def test_transient_model_errors_are_retried_with_retry_after(status):
    server, model, requests = _serve([status, status], retry_after='0')
    events = []
    model.emit = events.append
    try:
        result = model.create_chat_completion(messages=[], max_tokens=8)
        assert result['choices'][0]['message']['content'] == 'ok'
        assert len(requests) == 3
        assert [e['attempt'] for e in events if e['type'] == 'model_retry'] == [2, 3]
    finally:
        server.shutdown()
        server.server_close()


def test_model_retries_are_bounded_and_skip_request_errors():
    import urllib.error
    server, model, requests = _serve([500, 500, 500, 500], retry_after='0')
    try:
        with pytest.raises(urllib.error.HTTPError, match='500'):
            model.create_chat_completion(messages=[], max_tokens=8)
        assert len(requests) == 3
    finally:
        server.shutdown()
        server.server_close()
    server, model, requests = _serve([400])
    try:
        with pytest.raises(urllib.error.HTTPError, match='400'):
            model.create_chat_completion(messages=[], max_tokens=8)
        assert len(requests) == 1
    finally:
        server.shutdown()
        server.server_close()


def test_model_retry_never_waits_past_the_run_deadline():
    import time, urllib.error
    from execution_context import ExecutionContext
    server, model, requests = _serve([429], retry_after='30')
    model.execution_context = ExecutionContext(time.monotonic() + 5, threading.Event())
    try:
        started = time.monotonic()
        with pytest.raises(urllib.error.HTTPError, match='429'):
            model.create_chat_completion(messages=[], max_tokens=8)
        assert len(requests) == 1 and time.monotonic() - started < 2
    finally:
        server.shutdown()
        server.server_close()


def test_retry_delay_policy():
    import urllib.error
    from email.message import Message
    from model_backend import retry_delay
    headers = Message()
    headers['Retry-After'] = '600'
    assert retry_delay(urllib.error.HTTPError('u', 429, 'x', headers, None), 0) == 60
    assert retry_delay(urllib.error.HTTPError('u', 503, 'x', Message(), None), 1) == 2
    assert retry_delay(urllib.error.HTTPError('u', 401, 'x', Message(), None), 0) is None
    assert retry_delay(urllib.error.URLError(ConnectionResetError()), 0) == 1
    assert retry_delay(urllib.error.URLError(ConnectionRefusedError()), 0) is None
    assert retry_delay(ValueError('bad json'), 0) is None


# ---------------------------------------------------------------------------
# Profile validation, embedded inference and server details
# ---------------------------------------------------------------------------

from types import SimpleNamespace
from unittest.mock import MagicMock


@pytest.mark.parametrize('profile, message', [
    ({'speculative_mode': 'sometimes'}, 'speculative_mode must be'),
    ({'draft_tokens': 33}, 'draft_tokens must be between 1 and 32'),
    ({'draft_ngram_size': 0}, 'draft_ngram_size must be between 1 and 8'),
    ({'draft_n_gpu_layers': -2}, 'draft_n_gpu_layers'),
    ({'prompt_cache_mb': 5000}, 'prompt_cache_mb'),
    ({'server_cache_prompt': 'yes'}, 'server_cache_prompt must be a boolean'),
    ({'n_batch': 128, 'n_ubatch': 256}, 'n_ubatch cannot exceed n_batch'),
    ({'type_k': 'q2_k'}, 'type_k must be one of'),
    ({'flash_attn': 1}, 'flash_attn must be a boolean'),
])
def test_invalid_tuning_is_rejected(profile, message):
    from model_backend import validate_tuning
    with pytest.raises(ValueError, match=message):
        validate_tuning(profile)


def test_create_model_rejects_unknown_backends():
    from model_backend import create_model
    with pytest.raises(ValueError, match='Unknown model backend: cloud'):
        create_model({'backend': 'cloud'})


def test_base_url_must_be_http_without_credentials():
    for url in ('ftp://example.com/v1', 'http://user:secret@example.com/v1', 'http:///v1'):
        with pytest.raises(ValueError, match='base_url'):
            OpenAIModel({'base_url': url})


def test_adapter_fallback_counts_bytes_and_collect_skips_other_choices():
    adapter = ModelAdapter({})
    assert adapter.count_tokens('héllo') == 6
    result = adapter.collect([{'choices': [{'index': 1, 'delta': {'content': 'ignored'}}]},
                              {'choices': [{'index': 0, 'delta': {'content': 'kept'}, 'finish_reason': 'stop'}]}])
    assert result['choices'][0]['message']['content'] == 'kept'


def test_collect_checks_the_execution_context_and_cancellation():
    import time
    from execution_context import ExecutionContext, ExecutionCancelled
    chunk = {'choices': [{'delta': {'content': 'x'}, 'finish_reason': 'stop'}]}
    adapter = ModelAdapter({})
    cancelled = threading.Event()
    adapter.execution_context = ExecutionContext(time.monotonic() + 30, cancelled)
    assert adapter.collect([chunk])['choices'][0]['message']['content'] == 'x'
    cancelled.set()
    with pytest.raises(ExecutionCancelled):
        adapter.collect([chunk])
    adapter.execution_context = None
    adapter.cancel_event = cancelled
    with pytest.raises(KeyboardInterrupt):
        adapter.collect([chunk])


def _embedded(monkeypatch, tmp_path, **profile):
    import llama_cpp
    from model_backend import EmbeddedModel
    (tmp_path / 'm.gguf').touch()
    native = MagicMock()
    native.tokenize.side_effect = lambda data: list(data)
    seen = {}
    monkeypatch.setattr(llama_cpp, 'Llama', lambda **kwargs: seen.update(kwargs) or native)
    model = EmbeddedModel({'model_path': str(tmp_path / 'm.gguf'), 'n_ctx': 1024, 'n_gpu_layers': 0, **profile})
    return model, native, seen


def test_embedded_completion_streams_drops_tools_and_closes(monkeypatch, tmp_path):
    model, native, seen = _embedded(monkeypatch, tmp_path, chat_format='chatml', supports_tools=False)
    native.create_chat_completion.return_value = iter([
        {'choices': [{'delta': {'content': 'hel'}}]},
        {'choices': [{'delta': {'content': 'lo'}, 'finish_reason': 'stop'}]}])
    events = []
    model.emit = events.append
    result = model.create_chat_completion(messages=[{'role': 'user', 'content': 'hi'}],
                                          tools=[{'function': {'name': 'read'}}], max_tokens=8)
    assert seen['chat_format'] == 'chatml'
    sent = native.create_chat_completion.call_args.kwargs
    assert 'tools' not in sent and sent['stream'] is True
    assert result['choices'][0]['message']['content'] == 'hello'
    assert [e['type'] for e in events][:2] == ['model_loaded', 'assistant_delta']
    assert model.count_tokens('abc') == 3
    assert model.backend_metrics() == {}  # The stub has no perf counters.
    model.close()
    native.close.assert_called_once()
    assert model._model is None
    model.close()  # Closing twice is harmless.


def test_embedded_completion_without_streaming_returns_the_response(monkeypatch, tmp_path):
    model, native, _ = _embedded(monkeypatch, tmp_path, stream=False)
    response = {'choices': [{'message': {'content': 'whole'}, 'finish_reason': 'stop'}]}
    native.create_chat_completion.return_value = response
    native._ctx = SimpleNamespace(ctx=object())
    result = model.create_chat_completion(messages=[], tools=[{'function': {'name': 'read'}}])
    assert result['choices'][0]['message']['content'] == 'whole'
    assert 'tools' in native.create_chat_completion.call_args.kwargs


def test_embedded_backend_metrics_from_perf_counters(monkeypatch, tmp_path):
    from llama_cpp import llama_cpp
    model, native, _ = _embedded(monkeypatch, tmp_path)
    model.load()
    native._ctx = SimpleNamespace(ctx=object())
    stats = SimpleNamespace(n_p_eval=100, t_p_eval_ms=500.0, n_eval=20, t_eval_ms=0.0)
    monkeypatch.setattr(llama_cpp, 'llama_perf_context', lambda ctx: stats)
    assert model.backend_metrics() == {'prompt_tokens_per_second': 200.0, 'generation_tokens_per_second': None,
                                       'evaluated_prompt_tokens': 100, 'evaluated_generation_tokens': 20}


def test_embedded_close_releases_the_draft_model(monkeypatch, tmp_path):
    model, native, _ = _embedded(monkeypatch, tmp_path)
    model.load()
    draft = MagicMock()
    model._draft = draft
    model.close()
    draft.close.assert_called_once()
    assert model._draft is None


def _json_server(handler_body):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            body = handler_body(self)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *_):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_server_token_counts_accept_count_fields_and_ignore_unknown_shapes():
    replies = iter([b'{"count": 7}', b'{"unexpected": true}'])
    server = _json_server(lambda handler: next(replies))
    try:
        model = OpenAIModel({'base_url': f'http://127.0.0.1:{server.server_port}/v1'})
        assert model.count_tokens('anything') == 7
        assert model._server_count('anything') is None
    finally:
        server.shutdown()


def test_server_requests_ask_for_prompt_caching_and_drop_unsupported_tools():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            body = json.dumps({'choices': [{'message': {'content': 'ok'}, 'finish_reason': 'stop'}]}).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *_):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        model = OpenAIModel({'base_url': f'http://127.0.0.1:{server.server_port}/v1', 'stream': False,
                             'server_cache_prompt': True, 'supports_tools': False})
        model.create_chat_completion(messages=[], tools=[{'function': {'name': 'read'}}])
        assert seen[0]['cache_prompt'] is True and 'tools' not in seen[0]
    finally:
        server.shutdown()


def test_retries_wait_on_the_execution_context(monkeypatch):
    import time
    import urllib.error
    from execution_context import ExecutionContext
    model = OpenAIModel({'base_url': 'http://127.0.0.1:9/v1', 'stream': False})
    attempts = []

    def request(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise urllib.error.URLError(ConnectionResetError('reset'))
        return {'choices': [{'message': {'content': 'ok'}, 'finish_reason': 'stop'}]}
    monkeypatch.setattr(model, '_request', request)
    context = MagicMock(wraps=ExecutionContext(time.monotonic() + 60, threading.Event()))
    context.deadline = time.monotonic() + 60
    model.execution_context = context
    monkeypatch.setattr('model_backend.time.sleep', lambda _: pytest.fail('slept outside the context'))
    model._complete(messages=[])
    context.wait.assert_called_once_with(1.0)
    assert len(attempts) == 2
