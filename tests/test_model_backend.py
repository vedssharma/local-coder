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
