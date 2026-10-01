import json
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
