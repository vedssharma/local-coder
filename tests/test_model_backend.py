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


def test_tuning_reaches_native_constructor_and_invalid_values_are_rejected(monkeypatch):
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


def test_cache_configuration_and_stable_tool_prefix(monkeypatch):
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
