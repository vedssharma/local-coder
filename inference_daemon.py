"""Private same-user inference daemon. It exposes inference, never workspace tools."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import socketserver
import stat
import subprocess
import sys
import threading
import time

from model_backend import ModelAdapter, create_model

MAX_REQUEST = 4 * 1024 * 1024


def normalized_profile(profile):
    result = dict(profile)
    if result.get('backend', 'embedded') == 'embedded' and 'model_path' in result:
        result['model_path'] = str(Path(result['model_path']).resolve())
        if result.get('draft_model_path'):
            result['draft_model_path'] = str(Path(result['draft_model_path']).resolve())
    return result


def fingerprint(profile):
    return hashlib.sha256(json.dumps(normalized_profile(profile), sort_keys=True).encode()).hexdigest()


class InferenceDaemon(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, path, model):
        self.path = Path(path)
        self.model = model
        self.profile_hash = fingerprint(model.profile)
        self.model_lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.path.exists():
            raise ValueError('Socket already exists; stop the running daemon or remove a confirmed stale socket')
        super().__init__(str(self.path), Handler)
        os.chmod(self.path, 0o600)

    def server_close(self):
        super().server_close()
        self.path.unlink(missing_ok=True)


class Handler(socketserver.StreamRequestHandler):
    def send(self, value):
        self.wfile.write(json.dumps(value).encode() + b'\n')
        self.wfile.flush()

    def handle(self):
        try:
            self.request.settimeout(300)
            data = self.rfile.readline(MAX_REQUEST + 1)
            if len(data) > MAX_REQUEST or not data.endswith(b'\n'):
                raise ValueError('Request too large or incomplete')
            request = json.loads(data)
            operation = request['operation']
            if operation == 'ping':
                self.send({'response': {'profile_hash': self.server.profile_hash}})
                return
            if request.get('profile_hash') != self.server.profile_hash:
                raise ValueError('Daemon model profile differs; stop/restart it before using this profile')
            if operation == 'shutdown':
                self.send({'response': {'stopped': True}})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            with self.server.model_lock:
                model = self.server.model
                model.emit = lambda event: self.send({'event': event})
                model.cancel_event = None
                if operation == 'tokenize':
                    result = model.count_tokens(request['arguments']['text'])
                elif operation == 'complete':
                    kwargs = request['arguments']
                    allowed = {'messages', 'tools', 'max_tokens', 'temperature', 'stream', 'top_p', 'seed'}
                    if set(kwargs) - allowed:
                        raise ValueError('Unsupported inference arguments')
                    result = model.create_chat_completion(**kwargs)
                else:
                    raise ValueError('Unknown operation')
                self.send({'response': result})
        except (Exception, KeyboardInterrupt) as exc:
            try:
                self.send({'error': str(exc)})
            except OSError:
                pass


class PersistentModel(ModelAdapter):
    def __init__(self, profile, state_dir, autostart=True):
        if profile.get('backend', 'embedded') != 'embedded':
            raise ValueError('Persistent mode is for embedded inference; server backends already persist models')
        super().__init__(normalized_profile(profile))
        self.state_dir = Path(state_dir).resolve()
        self.socket_path = self.state_dir / 'inference.sock'
        self.profile_hash = fingerprint(self.profile)
        self.autostart = autostart

    def _ensure(self):
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if len(os.fsencode(str(self.socket_path))) > 100:
            raise ValueError('Daemon socket path is too long; choose a shorter LOCAL_CODER_CONFIG_DIR')
        lock_path = self.state_dir / 'inference.lock'
        with os.fdopen(os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600), 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                response = self._rpc('ping', {}, ensure=False)
            except (ConnectionRefusedError, FileNotFoundError):
                if not self.autostart:
                    raise ValueError('Inference daemon is not running')
                if self.socket_path.exists():
                    if not stat.S_ISSOCK(self.socket_path.lstat().st_mode) or self.socket_path.is_symlink():
                        raise ValueError('Refusing to replace a non-socket file')
                    self.socket_path.unlink()
                path = self.state_dir / 'inference-profile.json'
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
                with os.fdopen(descriptor, 'w') as profile_file:
                    json.dump(self.profile, profile_file)
                with os.fdopen(os.open(self.state_dir / 'inference.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600), 'ab') as log:
                    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                        '--socket', str(self.socket_path), '--profile-file', str(path)],
                        stdout=log, stderr=log, start_new_session=True)
                deadline = time.monotonic() + 5
                while True:
                    try:
                        response = self._rpc('ping', {}, ensure=False)
                        break
                    except (ConnectionRefusedError, FileNotFoundError):
                        if child.poll() is not None or time.monotonic() >= deadline:
                            if child.poll() is None:
                                child.terminate()
                                child.wait(timeout=5)
                            raise ValueError('Daemon startup failed; inspect inference.log')
                        time.sleep(0.05)
            if response['profile_hash'] != self.profile_hash:
                raise ValueError('Daemon model profile differs; stop/restart it before using this profile')

    def _rpc(self, operation, arguments, ensure=True):
        if ensure:
            self._ensure()
        payload = json.dumps({'operation': operation, 'arguments': arguments,
                              'profile_hash': self.profile_hash}).encode() + b'\n'
        if len(payload) > MAX_REQUEST:
            raise ValueError('Inference request exceeds daemon limit')
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(2 if operation == 'ping' else self.profile.get('request_timeout', 300))
            connection.connect(str(self.socket_path))
            connection.sendall(payload)
            with connection.makefile('rb') as stream:
                while True:
                    if self.cancel_event is not None and self.cancel_event.is_set():
                        raise KeyboardInterrupt
                    line = stream.readline(MAX_REQUEST + 1)
                    if not line or len(line) > MAX_REQUEST:
                        raise ValueError('Daemon returned an incomplete or oversized response')
                    value = json.loads(line)
                    if 'error' in value:
                        raise ValueError(value['error'])
                    if 'event' in value:
                        event = value['event']
                        if event['type'] == 'assistant_delta' and self._first_output is None:
                            self._first_output = time.perf_counter()
                        if event['type'] != 'inference_metrics':
                            self.emit(event)
                    if 'response' in value:
                        return value['response']

    def count_tokens(self, text):
        return self._rpc('tokenize', {'text': text})

    def _complete(self, **kwargs):
        result = self._rpc('complete', kwargs)
        self._server_performance = result.get('performance', {})
        return result

    def backend_metrics(self):
        metrics = getattr(self, '_server_performance', {})
        return {**{key: metrics.get(key) for key in ('prompt_tokens_per_second', 'generation_tokens_per_second', 'cached_prompt_tokens')},
                'load_seconds': metrics.get('load_seconds', 0),
                'server_load_seconds': metrics.get('load_seconds'), 'server_elapsed_seconds': metrics.get('elapsed_seconds')}

    def stop(self):
        # Stopping one's own private daemon must work after configuration changes.
        observed = self._rpc('ping', {}, ensure=False)
        self.profile_hash = observed['profile_hash']
        return self._rpc('shutdown', {}, ensure=False)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--socket', required=True)
    parser.add_argument('--profile-file', required=True)
    args = parser.parse_args()
    profile = json.loads(Path(args.profile_file).read_text())
    with InferenceDaemon(args.socket, create_model(profile)) as server:
        server.serve_forever(poll_interval=0.05)


if __name__ == '__main__':
    main()
