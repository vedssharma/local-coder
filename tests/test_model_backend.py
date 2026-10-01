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
