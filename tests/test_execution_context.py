import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from execution_context import ExecutionContext, ExecutionCancelled, DeadlineExceeded
from model_backend import OpenAIModel
from workspace_tools import WorkspaceTools


def test_shared_context_caps_command_deadline_even_without_wait(tmp_path):
    tools = WorkspaceTools(tmp_path, mode='execute')
    context = ExecutionContext(time.monotonic() + .15, threading.Event())
    try:
        result = tools.execute_tool('bash', {'command': 'sleep 30'}, context=context)
        assert result.status == 'timed_out'
    finally:
        tools.close()


def test_expired_or_cancelled_context_never_starts_mutation(tmp_path):
    tools = WorkspaceTools(tmp_path, mode='workspace-edit')
    event = threading.Event()
    context = ExecutionContext(time.monotonic() - 1, event)
    args = {'path': 'forbidden', 'content': 'bad'}
    assert tools.execute_tool('write', args, context=context).error_code == 'run_deadline'
    event.set()
    assert tools.execute_tool('write', args, context=context).status == 'cancelled'
    assert not (tmp_path / 'forbidden').exists()


@pytest.mark.parametrize('cancel', [False, True])
def test_active_http_read_observes_deadline_and_cancellation(cancel):
    ready, release = threading.Event(), threading.Event()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200)
            self.send_header('Content-Length', '10000')
            self.end_headers()
            self.wfile.flush()
            ready.set()
            release.wait(2)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    event = threading.Event()
    model = OpenAIModel({'base_url': f'http://127.0.0.1:{server.server_port}/v1', 'stream': False})
    model.execution_context = ExecutionContext(time.monotonic() + (.3 if not cancel else 2), event)
    def trigger():
        if ready.wait(1):
            time.sleep(.05)
            event.set()
    if cancel:
        threading.Thread(target=trigger, daemon=True).start()
    started = time.monotonic()
    try:
        with pytest.raises(ExecutionCancelled if cancel else DeadlineExceeded):
            model.create_chat_completion(messages=[], max_tokens=10)
        assert time.monotonic() - started < 1
    finally:
        release.set()
        server.shutdown()
        server.server_close()


def test_socket_guard_shuts_down_the_socket_when_cancelled():
    class Socket:
        def __init__(self, fail):
            self.fail, self.calls = fail, 0

        def shutdown(self, how):
            self.calls += 1
            if self.fail:
                raise OSError('already closed')
    for fail in (False, True):
        cancel = threading.Event()
        context = ExecutionContext(time.monotonic() + 30, cancel)
        sock = Socket(fail)
        with pytest.raises(ExecutionCancelled):
            with context.socket_guard(sock):
                cancel.set()
                deadline = time.monotonic() + 2
                while not sock.calls and time.monotonic() < deadline:
                    time.sleep(0.01)
        assert sock.calls == 1


def test_socket_guard_reraises_errors_while_the_run_is_live():
    context = ExecutionContext(time.monotonic() + 30, threading.Event())
    with pytest.raises(OSError, match='connection reset'):
        with context.socket_guard(None):
            raise OSError('connection reset')
