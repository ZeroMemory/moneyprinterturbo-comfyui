"""Cooperative cancellation shared by WebUI workers and direct LLM requests."""

from contextlib import contextmanager
from functools import wraps
import socket
import threading


class GenerationStopped(BaseException):
    """A user stop must pass through provider retry and fallback handlers."""


_lock = threading.RLock()
_tokens = set()
_local = threading.local()


class Token:
    def __init__(self, task_id=None):
        self.task_id = task_id
        self.event = threading.Event()
        self.stop_error = None
        self.result_committed = False
        self.preserve_previous_result = False
        self._callbacks = set()
        self._lock = threading.RLock()

    def check(self):
        if self.event.is_set() and not self.result_committed:
            raise GenerationStopped("Stopped by user.")

    def add_cancel_callback(self, callback):
        with self._lock:
            if not self.event.is_set():
                self._callbacks.add(callback)
                return lambda: self._remove_callback(callback)
        callback()
        self.check()

    def _remove_callback(self, callback):
        with self._lock:
            self._callbacks.discard(callback)

    def cancel(self):
        with self._lock:
            self.event.set()
            callbacks = tuple(self._callbacks)
            self._callbacks.clear()
        for callback in callbacks:
            try:
                callback()
            except Exception:
                # A closed resource must not prevent stopping the remaining jobs.
                pass


def new_token(task_id=None):
    token = Token(task_id)
    with _lock:
        _tokens.add(token)
    return token


def release(token):
    with _lock:
        _tokens.discard(token)


def current_token():
    return getattr(_local, "token", None)


def checkpoint():
    token = current_token()
    if token is not None:
        token.check()


@contextmanager
def operation(task_id=None, *, token=None):
    previous = current_token()
    if previous is not None and token is None:
        previous.check()
        yield previous
        previous.check()
        return
    token = token or new_token(task_id)
    _local.token = token
    try:
        token.check()
        yield token
        token.check()
    finally:
        _local.token = previous
        release(token)


def cancellable(function):
    @wraps(function)
    def run(*args, **kwargs):
        with operation():
            return function(*args, **kwargs)
    return run


def stop_all():
    with _lock:
        tokens = tuple(_tokens)
        # Signal the entire snapshot before any resource teardown can block.
        for token in tokens:
            token.event.set()
    for token in tokens:
        token.cancel()
    return tokens


def abort_http_client(client):
    """Release a blocking HTTP read before closing this request's own client."""
    http_client = getattr(client, "_client", client)
    # HTTPX has no public abort-in-flight API; shutdown releases recv on Windows.
    transports = {getattr(http_client, "_transport", None)}
    transports.update(getattr(http_client, "_mounts", {}).values())
    for transport in transports:
        pool = getattr(transport, "_pool", None)
        for connection in getattr(pool, "connections", ()):
            while getattr(connection, "_connection", None) is not None:
                connection = connection._connection
            stream = getattr(connection, "_network_stream", None)
            if stream is not None:
                sock = stream.get_extra_info("socket")
                if sock is not None:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
    http_client.close()
