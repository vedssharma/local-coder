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
        assert ''.join(e['text'] for e in events) == 'Hello world'
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
