"""A shared run deadline and cooperative cancellation across operation boundaries."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import socket
import threading
import time


class ExecutionCancelled(Exception):
    pass


class DeadlineExceeded(TimeoutError):
    pass


CURRENT_CONTEXT = ContextVar('local_coder_execution_context', default=None)


@dataclass(frozen=True)
class ExecutionContext:
    deadline: float
    cancel_event: threading.Event

    def check(self):
        if self.cancel_event.is_set():
            raise ExecutionCancelled('Run cancelled')
        if time.monotonic() >= self.deadline:
            raise DeadlineExceeded('Run deadline exceeded')

    def timeout(self, requested):
        self.check()
        return max(0.001, min(requested, self.deadline - time.monotonic()))

    def wait(self, seconds):
        self.cancel_event.wait(self.timeout(seconds))
        self.check()

    @contextmanager
    def bind(self):
        self.check()
        token = CURRENT_CONTEXT.set(self)
        try:
            yield
        finally:
            CURRENT_CONTEXT.reset(token)

    @contextmanager
    def response_guard(self, response):
        """Interrupt active HTTP sockets; connecting/DNS retain platform limits."""
        sock = getattr(getattr(getattr(response, 'fp', None), 'raw', None), '_sock', None)
        with self.socket_guard(sock):
            yield

    @contextmanager
    def socket_guard(self, sock):
        done = threading.Event()
        def watch():
            while not done.wait(0.05):
                if self.cancel_event.is_set() or time.monotonic() >= self.deadline:
                    if sock is not None:
                        try:
                            sock.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                    return
        worker = threading.Thread(target=watch, daemon=True)
        worker.start()
        try:
            self.check()
            yield
            self.check()
        except Exception:
            self.check()
            raise
        finally:
            done.set()
            worker.join(timeout=0.1)
