import os
import subprocess
import sys
import threading
import pytest
from inference_daemon import InferenceDaemon, PersistentModel
from model_backend import ModelAdapter


def test_two_cli_processes_reuse_one_daemon_and_profile_mismatch_is_rejected(tmp_path):
    class Fake(ModelAdapter):
        def __init__(self):
            super().__init__({'backend': 'embedded'})
            self.calls = 0
        def _complete(self, **kwargs):
            self.calls += 1
            self.emit({'type': 'assistant_delta', 'text': 'hello'})
            return {'choices': [{'message': {'content': 'hello'}, 'finish_reason': 'stop'}]}
    model = Fake()
    server = InferenceDaemon(tmp_path / 'inference.sock', model)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert os.stat(server.path).st_mode & 0o777 == 0o600
        code = 'from inference_daemon import PersistentModel; import sys; m=PersistentModel({"backend":"embedded"},sys.argv[1],autostart=False); print(m.create_chat_completion(messages=[],max_tokens=4)["choices"][0]["message"]["content"])'
        for _ in range(2):
            result = subprocess.run([sys.executable, '-c', code, str(tmp_path)], capture_output=True, text=True, timeout=10)
            assert result.returncode == 0, result.stderr
            assert 'hello' in result.stdout
        assert model.calls == 2
        client = PersistentModel({'backend': 'embedded', 'n_threads': 2}, tmp_path, autostart=False)
        with pytest.raises(ValueError, match='profile differs'):
            client.count_tokens('hello')
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_autostart_reports_missing_model_and_can_be_stopped():
    import tempfile
    import time
    with tempfile.TemporaryDirectory(prefix='lc-daemon-') as state:
        client = PersistentModel({'backend': 'embedded', 'model_path': '/nonexistent-draft-test-model.gguf',
                                  'n_ctx': 1024, 'n_gpu_layers': 0}, state)
        try:
            with pytest.raises(ValueError, match='does not exist'):
                client.count_tokens('hello')
        finally:
            if client.socket_path.exists():
                client.stop()
                deadline = time.monotonic() + 5
                while client.socket_path.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                assert not client.socket_path.exists()


# ---------------------------------------------------------------------------
# Daemon protocol and client edge cases (no subprocesses)
# ---------------------------------------------------------------------------

import contextlib
import json
import socket
import time
from unittest.mock import MagicMock

import inference_daemon
from execution_context import ExecutionContext
from inference_daemon import fingerprint, normalized_profile


class Echo(ModelAdapter):
    def __init__(self, profile=None):
        super().__init__(profile or {'backend': 'embedded'})

    def count_tokens(self, text):
        return len(text.split())

    def _complete(self, **kwargs):
        self.emit({'type': 'inference_metrics', 'elapsed_seconds': 0})
        self.emit({'type': 'assistant_delta', 'text': 'hi'})
        return {'choices': [{'message': {'content': 'hi'}, 'finish_reason': 'stop'}]}


@contextlib.contextmanager
def serving(directory, model=None):
    server = InferenceDaemon(directory / 'inference.sock', model or Echo())
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def raw(server, payload):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(5)
        connection.connect(str(server.path))
        connection.sendall(payload)
        connection.shutdown(socket.SHUT_WR)
        with connection.makefile('rb') as stream:
            return [json.loads(line) for line in stream]


def request(server, operation, arguments=None, profile_hash=None):
    body = {'operation': operation, 'arguments': arguments or {},
            'profile_hash': server.profile_hash if profile_hash is None else profile_hash}
    return raw(server, json.dumps(body).encode() + b'\n')


def test_normalized_profile_resolves_model_and_draft_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    profile = normalized_profile({'backend': 'embedded', 'model_path': 'm.gguf', 'draft_model_path': 'd.gguf'})
    assert profile['model_path'] == str(tmp_path / 'm.gguf')
    assert profile['draft_model_path'] == str(tmp_path / 'd.gguf')
    assert normalized_profile({'backend': 'openai', 'model_path': 'm.gguf'})['model_path'] == 'm.gguf'
    assert fingerprint({'model_path': 'm.gguf'}) == fingerprint({'model_path': str(tmp_path / 'm.gguf')})


def test_daemon_refuses_an_existing_socket_path(tmp_path):
    (tmp_path / 'inference.sock').write_text('')
    with pytest.raises(ValueError, match='Socket already exists'):
        InferenceDaemon(tmp_path / 'inference.sock', Echo())


def test_daemon_handles_each_operation_and_rejects_bad_requests(tmp_path):
    with serving(tmp_path) as server:
        assert request(server, 'ping', profile_hash='anything') == [{'response': {'profile_hash': server.profile_hash}}]
        assert request(server, 'tokenize', {'text': 'three short words'}) == [{'response': 3}]
        lines = request(server, 'complete', {'messages': [], 'max_tokens': 4, 'stream': False})
        assert lines[0] == {'event': {'type': 'inference_metrics', 'elapsed_seconds': 0}}
        assert lines[-1]['response']['choices'][0]['message']['content'] == 'hi'
        assert 'profile differs' in request(server, 'tokenize', {'text': 'x'}, profile_hash='other')[0]['error']
        assert request(server, 'complete', {'messages': [], 'model_path': '/etc/passwd'}) == [
            {'error': 'Unsupported inference arguments'}]
        assert request(server, 'load', {}) == [{'error': 'Unknown operation'}]
        assert raw(server, b'{"operation": "ping"}') == [{'error': 'Request too large or incomplete'}]
        assert 'error' in raw(server, b'not json\n')[0]


def test_daemon_shutdown_request_stops_serving(tmp_path):
    server = InferenceDaemon(tmp_path / 'inference.sock', Echo())
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01}, daemon=True)
    thread.start()
    try:
        assert request(server, 'shutdown') == [{'response': {'stopped': True}}]
        thread.join(timeout=5)
        assert not thread.is_alive()
    finally:
        server.server_close()
    assert not server.path.exists()


def test_client_round_trips_events_metrics_and_stop(tmp_path):
    with serving(tmp_path) as server:
        client = PersistentModel({'backend': 'embedded'}, tmp_path, autostart=False)
        events = []
        client.emit = events.append
        response = client.create_chat_completion(messages=[], max_tokens=4, stream=False)
        assert response['choices'][0]['message']['content'] == 'hi'
        assert [e['type'] for e in events] == ['assistant_delta', 'inference_metrics']
        assert response['performance']['time_to_first_output_seconds'] is not None
        assert client.count_tokens('two words') == 2
        client._server_performance = {'load_seconds': 1.5, 'elapsed_seconds': 2.0, 'cached_prompt_tokens': 3}
        assert client.backend_metrics() == {'prompt_tokens_per_second': None, 'generation_tokens_per_second': None,
                                            'cached_prompt_tokens': 3, 'load_seconds': 1.5,
                                            'server_load_seconds': 1.5, 'server_elapsed_seconds': 2.0}
        other = PersistentModel({'backend': 'embedded', 'n_threads': 3}, tmp_path, autostart=False)
        assert other.stop() == {'stopped': True}
        deadline = time.monotonic() + 5
        while server.path.exists() and time.monotonic() < deadline:
            time.sleep(0.02)


def test_client_honors_the_execution_context_and_cancellation(tmp_path):
    with serving(tmp_path):
        client = PersistentModel({'backend': 'embedded'}, tmp_path, autostart=False)
        client.execution_context = ExecutionContext(time.monotonic() + 30, threading.Event())
        assert client.count_tokens('a b c') == 3
        client.execution_context = None
        client.cancel_event = threading.Event()
        client.cancel_event.set()
        with pytest.raises(KeyboardInterrupt):
            client.count_tokens('a')


def test_client_rejects_oversized_requests_and_truncated_responses(tmp_path, monkeypatch):
    client = PersistentModel({'backend': 'embedded'}, tmp_path, autostart=False)
    monkeypatch.setattr(inference_daemon, 'MAX_REQUEST', 120)
    with pytest.raises(ValueError, match='exceeds daemon limit'):
        client._rpc('tokenize', {'text': 'x' * 200}, ensure=False)
    monkeypatch.undo()
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(client.socket_path))
    listener.listen(1)

    def hang_up():
        connection, _ = listener.accept()
        connection.recv(65536)
        connection.close()
    thread = threading.Thread(target=hang_up, daemon=True)
    thread.start()
    try:
        with pytest.raises(ValueError, match='incomplete or oversized'):
            client._rpc('tokenize', {'text': 'x'}, ensure=False)
    finally:
        thread.join(timeout=5)
        listener.close()


def test_client_rejects_non_embedded_profiles_and_long_socket_paths(tmp_path):
    with pytest.raises(ValueError, match='Persistent mode is for embedded'):
        PersistentModel({'backend': 'openai'}, tmp_path)
    client = PersistentModel({'backend': 'embedded'}, tmp_path / ('d' * 120))
    with pytest.raises(ValueError, match='socket path is too long'):
        client.count_tokens('x')


def test_client_without_autostart_reports_a_missing_daemon(tmp_path):
    client = PersistentModel({'backend': 'embedded'}, tmp_path, autostart=False)
    with pytest.raises(ValueError, match='not running'):
        client.count_tokens('x')


def test_autostart_refuses_to_replace_a_regular_file(tmp_path, monkeypatch):
    monkeypatch.setattr(inference_daemon.subprocess, 'Popen', MagicMock(side_effect=AssertionError('spawned')))
    client = PersistentModel({'backend': 'embedded'}, tmp_path)
    client.socket_path.write_text('not a socket')
    # A regular file refuses connections like a dead socket would.
    monkeypatch.setattr(client, '_rpc', MagicMock(side_effect=ConnectionRefusedError))
    with pytest.raises(ValueError, match='non-socket file'):
        client._ensure()


def _stale_socket(path):
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    listener.close()


def test_autostart_removes_a_stale_socket_and_reports_a_crashed_daemon(tmp_path, monkeypatch):
    client = PersistentModel({'backend': 'embedded', 'model_path': 'm.gguf'}, tmp_path)
    _stale_socket(client.socket_path)
    child = MagicMock()
    child.poll.return_value = 1
    popen = MagicMock(return_value=child)
    monkeypatch.setattr(inference_daemon.subprocess, 'Popen', popen)
    with pytest.raises(ValueError, match='Daemon startup failed'):
        client.count_tokens('x')
    assert not client.socket_path.exists()
    assert '--profile-file' in popen.call_args.args[0]
    written = json.loads((tmp_path / 'inference-profile.json').read_text())
    assert written['model_path'] == client.profile['model_path']
    child.terminate.assert_not_called()


def test_autostart_terminates_a_daemon_that_never_listens(tmp_path, monkeypatch):
    client = PersistentModel({'backend': 'embedded'}, tmp_path)
    child = MagicMock()
    child.poll.return_value = None
    monkeypatch.setattr(inference_daemon.subprocess, 'Popen', MagicMock(return_value=child))
    clock = iter([0.0, 0.0, 10.0])
    monkeypatch.setattr(inference_daemon.time, 'monotonic', lambda: next(clock, 10.0))
    monkeypatch.setattr(inference_daemon.time, 'sleep', lambda _: None)
    with pytest.raises(ValueError, match='Daemon startup failed'):
        client.count_tokens('x')
    child.terminate.assert_called_once()
    child.wait.assert_called_once_with(timeout=5)


def test_daemon_main_serves_the_profile_file(tmp_path, monkeypatch):
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps({'backend': 'embedded', 'model_path': 'm.gguf'}))
    created = MagicMock()
    server = MagicMock()
    server.__enter__.return_value = server
    daemon = MagicMock(return_value=server)
    monkeypatch.setattr(inference_daemon, 'create_model', created)
    monkeypatch.setattr(inference_daemon, 'InferenceDaemon', daemon)
    monkeypatch.setattr(sys, 'argv', ['inference_daemon.py', '--socket', str(tmp_path / 's.sock'),
                                      '--profile-file', str(profile)])
    inference_daemon.main()
    assert created.call_args.args[0]['model_path'] == 'm.gguf'
    assert daemon.call_args.args[0] == str(tmp_path / 's.sock')
    server.serve_forever.assert_called_once()
