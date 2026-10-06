"""Prompt-size counting for OpenAI-compatible servers: /tokenize when offered, else a calibrated estimate."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from model_backend import OpenAIModel
from session import ContextManager


@pytest.fixture
def tokenize_server():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        mode = 'tokens'

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append((self.path, body))
            if self.path != '/tokenize' or Handler.mode == 'missing':
                self.send_response(404); self.end_headers(); return
            words = body['content'].split()
            payload = {'tokens': list(range(len(words)))} if Handler.mode == 'tokens' else {'count': len(words)}
            data = json.dumps(payload).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server, Handler, requests
    server.shutdown()


def local(server):
    return OpenAIModel({'backend': 'openai', 'base_url': f'http://127.0.0.1:{server.server_port}/v1', 'model': 'm'})


@pytest.mark.parametrize('mode', ['tokens', 'count'])
def test_self_hosted_server_tokenizer_gives_exact_counts(tokenize_server, mode):
    server, handler, requests = tokenize_server
    handler.mode = mode
    model = local(server)
    assert model.count_tokens('one two three') == 3
    assert requests[-1][0] == '/tokenize' and requests[-1][1]['content'] == 'one two three'


def test_server_without_tokenize_falls_back_once_to_the_estimate(tokenize_server):
    server, handler, requests = tokenize_server
    handler.mode = 'missing'
    model = local(server)
    assert model.count_tokens('a' * 300) == 100
    assert model.count_tokens('a' * 300) == 100
    assert len(requests) == 1


def test_estimate_follows_reported_prompt_tokens_with_a_margin():
    model = OpenAIModel({'backend': 'openai', 'provider': 'openai', 'base_url': 'https://api.openai.com/v1'})
    request = {'messages': [{'role': 'user', 'content': 'x' * 4000}]}
    size = len(json.dumps(request['messages'], separators=(',', ':')).encode()) + len(b'[]')  # messages + no tools
    model.observe_prompt(request, size // 5)
    assert model.bytes_per_token == pytest.approx(min(6.0, size / (size // 5) * 0.9))
    assert model.count_version == 1
    before = model.bytes_per_token
    model.observe_prompt(request, int(size / (before / 0.9)))  # within 5%: no change
    assert model.count_version == 1 and model.bytes_per_token == before
    model.observe_prompt(request, size * 10)
    assert model.bytes_per_token == 1.5
    model.observe_prompt(request, None)
    assert model.bytes_per_token == 1.5


def test_context_manager_drops_cached_counts_when_the_estimate_changes():
    model = OpenAIModel({'backend': 'openai', 'provider': 'openai', 'base_url': 'https://api.openai.com/v1'})
    context = ContextManager(10000, count_tokens=model.count_tokens)
    assert context.count_tokens('a' * 600) == 200
    model.bytes_per_token, model.count_version = 6.0, 1
    assert context.count_tokens('a' * 600) == 100
