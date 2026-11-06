"""ComfyUI API client with owned-prompt cancellation."""

from dataclasses import dataclass
import json
from http.client import HTTPConnection, HTTPSConnection, HTTPException, IncompleteRead
import os
from queue import Empty, SimpleQueue
import socket
from socket import SHUT_RDWR
from threading import Event, Thread, Timer
import time
from typing import Callable, Iterable
from urllib.parse import quote, urlencode, urlsplit

from .profiles import load_profile_snapshot, local_endpoint_error, remote_endpoint_error


MAX_REMOTE_VIDEO_BYTES = 1024 * 1024 * 1024
WebSocketEvents = Callable[[str, float], Iterable[dict]]


class ComfyApiError(RuntimeError):
    pass


class ComfyPromptNotSubmitted(ComfyApiError):
    """A transport failure before any prompt request could be sent."""


class ComfyHttpError(ComfyApiError):
    def __init__(self, status, detail, *, prompt_rejected=False):
        self.status = status
        self.prompt_rejected = prompt_rejected
        super().__init__(f"ComfyUI HTTP {status}: {detail}")


class PromptOwnershipError(PermissionError):
    pass


@dataclass(frozen=True)
class CompletionResult:
    prompt_id: str
    status: str
    history: object
    event: dict | None = None


@dataclass(frozen=True)
class CancellationResult:
    prompt_id: str
    status: str
    queue: object
    history: object


MAX_RESPONSE_BYTES = 32 * 1024 * 1024


def _prompt_outputs_rejected(raw):
    try:
        payload = json.loads(raw)
    except (UnicodeError, ValueError):
        return False
    if not isinstance(payload, dict) or set(payload) != {"error", "node_errors"}:
        return False
    error = payload["error"]
    return (isinstance(error, dict) and error.get("type") == "prompt_outputs_failed_validation"
            and isinstance(error.get("message"), str) and isinstance(error.get("details"), str)
            and isinstance(payload["node_errors"], dict))


def remaining_timeout(deadline):
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("ComfyUI request deadline expired")
    return remaining


def queue_counts(queue):
    if not isinstance(queue, dict):
        return None
    return {name: len(queue[name]) for name in ("queue_running", "queue_pending")
            if isinstance(queue.get(name), list)}


def _connect_socket(address, deadline, source_address):
    # The OS resolver can outlive a socket timeout. Only resolution runs off-thread.
    resolved = SimpleQueue()

    def resolve():
        try:
            resolved.put(socket.getaddrinfo(*address, 0, socket.SOCK_STREAM))
        except Exception as exc:
            resolved.put(exc)

    Thread(target=resolve, daemon=True).start()
    try:
        addresses = resolved.get(timeout=remaining_timeout(deadline))
    except Empty:
        raise TimeoutError("ComfyUI request deadline expired during DNS resolution") from None
    if isinstance(addresses, Exception):
        raise addresses
    last_error = OSError("No address found for ComfyUI endpoint")
    for family, kind, protocol, _, sockaddr in addresses:
        budget = remaining_timeout(deadline)
        try:
            conn = socket.socket(family, kind, protocol)
            try:
                conn.settimeout(budget)
                if source_address:
                    conn.bind(source_address)
                conn.connect(sockaddr)
                conn.settimeout(remaining_timeout(deadline))
            except BaseException:
                conn.close()
                raise
            return conn
        except OSError as exc:
            last_error = exc
    raise last_error


def _connection_for(endpoint, deadline):
    parsed = urlsplit(endpoint)
    connection_class = HTTPSConnection if parsed.scheme == "https" else HTTPConnection
    connection = connection_class(parsed.hostname, parsed.port, timeout=remaining_timeout(deadline))
    # Keep HTTPConnection's TLS hostname and share the deadline across all addresses and the handshake.
    connection._create_connection = lambda address, _, source: _connect_socket(address, deadline, source)
    return connection


class UrlTransport:
    def __init__(self, endpoint: str, timeout: float = 10, *, remote=False, token=None):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.remote = remote
        self.token = token

    def request(self, method: str, path: str, json_body=None, *, timeout=None, output=None):
        limit = 120 if output is not None else self.timeout
        budget = limit if timeout is None else min(limit, timeout)
        if budget <= 0:
            raise TimeoutError("ComfyUI request deadline expired")
        deadline = time.monotonic() + budget
        data = None
        headers = {}
        if self.token is not None:
            headers["Authorization"] = "Bearer " + self.token
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        connection = None
        request_started = False
        expired = Event()
        watchdog = None
        try:
            connection = _connection_for(self.endpoint, deadline)
            connection.connect()
            socket = connection.sock

            def expire_request():
                expired.set()
                try:
                    socket.shutdown(SHUT_RDWR)
                except OSError:
                    pass

            # Buffered header reads can outlive the inactivity timeout if bytes arrive slowly.
            watchdog = Timer(max(0, deadline - time.monotonic()), expire_request)
            watchdog.daemon = True
            watchdog.start()

            def set_remaining_timeout():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("ComfyUI request deadline expired")
                socket.settimeout(remaining)

            set_remaining_timeout()
            request_started = True
            connection.request(method, path, body=data, headers=headers)
            set_remaining_timeout()
            with connection.getresponse() as response:
                if output is not None:
                    if response.status != 200:
                        raise ComfyHttpError(response.status, "video download failed")
                    length = response.length
                    if length is not None and not 0 < length <= MAX_REMOTE_VIDEO_BYTES:
                        raise ComfyApiError("ComfyUI video size is outside the download limit")
                    received = 0
                    while not response.isclosed():
                        set_remaining_timeout()
                        chunk = response.read1(min(65536, MAX_REMOTE_VIDEO_BYTES + 1 - received))
                        if not chunk:
                            break
                        received += len(chunk)
                        if received > MAX_REMOTE_VIDEO_BYTES:
                            raise ComfyApiError("ComfyUI video exceeds the download limit")
                        output.write(chunk)
                    if length is not None and received != length:
                        raise IncompleteRead(b"", length - received)
                    if not received:
                        raise ComfyApiError("ComfyUI returned an empty video")
                    return
                raw = bytearray()
                while not response.isclosed():
                    set_remaining_timeout()
                    chunk = response.read1(min(65536, MAX_RESPONSE_BYTES + 1 - len(raw)))
                    raw.extend(chunk)
                    if len(raw) > MAX_RESPONSE_BYTES:
                        raise ComfyApiError("ComfyUI response exceeds 32 MiB limit")
                    if not chunk:
                        break
                if response.length not in (None, 0):
                    raise IncompleteRead(b"", response.length)
                if not 200 <= response.status < 300:
                    detail = raw[:2048].decode("utf-8", errors="replace")
                    rejected = (method == "POST" and path == "/prompt" and response.status == 400
                                and _prompt_outputs_rejected(raw))
                    if self.remote:
                        detail = "request failed"
                    raise ComfyHttpError(response.status, detail, prompt_rejected=rejected)
            result = json.loads(raw.decode("utf-8")) if raw else {}
            if time.monotonic() >= deadline:
                raise TimeoutError("ComfyUI request deadline expired")
            return result
        except Exception as error:
            if (method == "POST" and path == "/prompt" and not request_started
                    and isinstance(error, (OSError, HTTPException))):
                raise ComfyPromptNotSubmitted(f"ComfyUI connection failed before prompt submission: {error}") from error
            if expired.is_set():
                raise TimeoutError("ComfyUI request deadline expired") from error
            raise
        finally:
            if watchdog is not None:
                watchdog.cancel()
                watchdog.join()
            if connection is not None:
                connection.close()


class ComfyClient:
    def __init__(self, endpoint: str, *, transport=None, owned_prompt_ids=(),
                 remote_server=False, allow_insecure_http=False, token_env=None):
        token = None
        if remote_server:
            if error := remote_endpoint_error(endpoint, allow_insecure_http, token_env):
                raise ValueError(error)
            if token_env is not None:
                token = os.environ.get(token_env)
                if not token or any(not 33 <= ord(char) <= 126 for char in token):
                    raise ValueError("Remote ComfyUI bearer token is missing or invalid")
        elif error := local_endpoint_error(endpoint):
            raise ValueError(error)
        self.endpoint = endpoint.rstrip("/")
        self.remote_server = remote_server
        self.transport = transport or UrlTransport(self.endpoint, remote=remote_server, token=token)
        self._owned_prompt_ids = {_prompt_id(value) for value in owned_prompt_ids}

    def system_stats(self, *, timeout=None):
        return self._get("/system_stats", timeout=timeout)

    def object_info(self, *, timeout=None):
        return self._get("/object_info", timeout=timeout)

    def queue(self, *, timeout=None):
        return self._get("/queue", timeout=timeout)

    def history(self, prompt_id: str | None = None, *, timeout=None):
        path = "/history"
        if prompt_id is not None:
            path += "/" + quote(_prompt_id(prompt_id), safe="")
        return self._get(path, timeout=timeout)

    def download_output(self, prompt_id, filename, subfolder, output, *, timeout=None):
        if _prompt_id(prompt_id) not in self._owned_prompt_ids:
            raise PromptOwnershipError("refusing to fetch an output for a prompt not owned by client")
        query = urlencode({"filename": filename, "subfolder": subfolder, "type": "output"})
        self.transport.request("GET", "/view?" + query, timeout=timeout, output=output)

    def submit(self, prompt: dict, *, client_id: str, timeout=None):
        if not isinstance(prompt, dict):
            raise TypeError("prompt must be a dictionary")
        if not isinstance(client_id, str) or not client_id:
            raise ValueError("client_id must be a non-empty string")
        result = self.transport.request(
            "POST", "/prompt", {"prompt": prompt, "client_id": client_id},
            **({"timeout": timeout} if timeout is not None else {}),
        )
        prompt_id = result.get("prompt_id") if isinstance(result, dict) else None
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ComfyApiError("submit response did not contain prompt_id")
        self._owned_prompt_ids.add(prompt_id)
        return result

    def delete_pending(self, prompt_id: str, *, timeout=None) -> None:
        prompt_id = _prompt_id(prompt_id)
        if prompt_id not in self._owned_prompt_ids:
            raise PromptOwnershipError("refusing to cancel a prompt not owned by client")
        self.transport.request("POST", "/queue", {"delete": [prompt_id]},
                               **({"timeout": timeout} if timeout is not None else {}))

    def cancel(self, prompt_id: str, *, timeout=None) -> bool:
        prompt_id = _prompt_id(prompt_id)
        if prompt_id not in self._owned_prompt_ids:
            raise PromptOwnershipError("refusing to cancel a prompt not owned by client")
        deadline = time.monotonic() + timeout if timeout is not None else None
        try:
            budget = remaining_timeout(deadline)
            response = self.transport.request("POST", "/api/jobs/" + quote(prompt_id, safe="") + "/cancel",
                                              **({"timeout": budget} if budget is not None else {}))
            return isinstance(response, dict) and response.get("cancelled") is True
        except ComfyHttpError as exc:
            if exc.status not in (404, 405):
                raise
            # Older /interrupt handlers can stop a different user's running job.
            self.delete_pending(prompt_id, timeout=remaining_timeout(deadline))
            running = self.queue(timeout=remaining_timeout(deadline))["queue_running"]
            if any(len(row) > 1 and row[1] == prompt_id for row in running):
                raise ComfyApiError("This server does not support targeted cancellation of running jobs.") from exc
            return False

    def cancel_and_confirm(
        self, prompt_id: str, *, timeout: float = 10, poll_interval: float = 0.1
    ) -> CancellationResult:
        prompt_id = _prompt_id(prompt_id)
        if timeout <= 0 or poll_interval < 0:
            raise ValueError("timeout must be positive and poll_interval non-negative")
        deadline = time.monotonic() + timeout
        queue = {}
        history = {}
        try:
            acknowledged = self.cancel(prompt_id, timeout=remaining_timeout(deadline))
        except TimeoutError:
            return CancellationResult(prompt_id, "uncertain", queue, history)
        while time.monotonic() < deadline:
            try:
                # A job finishing after this queue snapshot must show up in the following history read.
                snapshot = self.queue(timeout=remaining_timeout(deadline)) if acknowledged else None
                history = self.history(prompt_id, timeout=remaining_timeout(deadline))
            except TimeoutError:
                break
            if isinstance(history, dict) and prompt_id in history:
                status = history_status(history[prompt_id], prompt_id)
                if status == "cancelled":
                    return CancellationResult(
                        prompt_id, "confirmed_cancelled", queue, history
                    )
                if status in {"completed", "failed"}:
                    return CancellationResult(prompt_id, status, queue, history)
            try:
                if not acknowledged:
                    snapshot = self.queue(timeout=remaining_timeout(deadline))
            except TimeoutError:
                break
            queue = queue_counts(snapshot)
            if acknowledged and history == {} and queue and len(queue) == 2:
                if all(isinstance(row, (list, tuple)) and len(row) > 1 and isinstance(row[1], str)
                       and row[1] != prompt_id for name in queue for row in snapshot[name]):
                    return CancellationResult(prompt_id, "confirmed_cancelled", queue, history)
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(poll_interval, remaining))
        return CancellationResult(prompt_id, "uncertain", queue, history)

    def wait_for_completion(
        self,
        prompt_id: str,
        client_id: str,
        *,
        timeout: float = 300,
        poll_interval: float = 1,
        websocket_events: WebSocketEvents | None = None,
    ) -> CompletionResult:
        prompt_id = _prompt_id(prompt_id)
        if prompt_id not in self._owned_prompt_ids:
            raise PromptOwnershipError("refusing to wait for a prompt not owned by client")
        if timeout <= 0 or poll_interval < 0:
            raise ValueError("timeout must be positive and poll_interval non-negative")
        deadline = time.monotonic() + timeout
        history = {}
        terminal_event = None

        def check_history(budget):
            nonlocal history
            if budget > 0:
                try:
                    history = self.history(prompt_id, timeout=budget)
                except (TimeoutError, ConnectionError, HTTPException):
                    return None
                except ComfyHttpError as exc:
                    if exc.status in (404, 429) or exc.status >= 500:
                        return None
                    raise
                if isinstance(history, dict) and prompt_id in history:
                    status = history_status(history[prompt_id], prompt_id)
                    if status in {"completed", "failed", "cancelled"}:
                        return CompletionResult(prompt_id, status, history, terminal_event)

        # A job may already have finished before this client starts listening.
        result = check_history(min(1, timeout / 4))
        if result:
            return result
        if self.remote_server and websocket_events is None:
            websocket_events = lambda *_: ()
        receiver = websocket_events or (
            lambda current_client_id, current_timeout: _default_websocket_events(
                self.endpoint, current_client_id, current_timeout
            )
        )
        terminal_status = None
        try:
            remaining = max(0, deadline - time.monotonic())
            # Leave a final HTTP read even if a handshake or fragmented frame stalls.
            socket_budget = remaining - min(1, remaining / 2)
            if websocket_events is not None:
                socket_budget = min(1, socket_budget)
            next_poll = time.monotonic() + min(1, poll_interval)
            for event in receiver(client_id, socket_budget):
                if time.monotonic() >= deadline:
                    break
                result = _terminal_event(event, prompt_id)
                if result:
                    terminal_status, terminal_event = result, event
                    break
                if time.monotonic() >= next_poll:
                    result = check_history(min(1, deadline - time.monotonic()))
                    if result:
                        return result
                    next_poll = time.monotonic() + min(1, poll_interval)
        except Exception:
            pass
        if terminal_status in {"failed", "cancelled"}:
            try:
                history = self.history(prompt_id, timeout=max(0, deadline - time.monotonic()))
            except (OSError, HTTPException, ComfyApiError, ValueError):
                history = {}
            return CompletionResult(prompt_id, terminal_status, history, terminal_event)
        # ComfyUI can signal success before task_done publishes the output files in history.
        while time.monotonic() < deadline:
            result = check_history(deadline - time.monotonic())
            if result:
                return result
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(poll_interval, remaining))
        return CompletionResult(prompt_id, "timeout", history, terminal_event)

    def _get(self, path: str, *, timeout=None):
        if timeout is not None:
            return self.transport.request("GET", path, timeout=timeout)
        return self.transport.request("GET", path)


def remote_client_options(profile):
    if not profile.remote_server:
        return {}
    return {"remote_server": True, "allow_insecure_http": profile.allow_insecure_http,
            "token_env": profile.token_env}


def client_for_snapshot(snapshot, client_factory):
    if snapshot.get("remote_server") is True:
        profile = load_profile_snapshot(snapshot)
        return client_factory(profile.endpoint, **remote_client_options(profile))
    return client_factory(snapshot["endpoint"])


def _prompt_id(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("prompt_id must be a non-empty string")
    return value


def _terminal_event(event: object, prompt_id: str) -> str | None:
    if not isinstance(event, dict):
        return None
    data = event.get("data")
    if not isinstance(data, dict) or data.get("prompt_id") != prompt_id:
        return None
    event_type = event.get("type")
    if event_type == "execution_success":
        return "completed"
    if event_type == "execution_error":
        return "failed"
    if event_type == "execution_interrupted":
        return "cancelled"
    if event_type == "executing" and data.get("node") is None:
        return "completed"
    return None


def history_status(entry: object, prompt_id: str | None = None) -> str:
    if not isinstance(entry, dict):
        return "unknown"
    status = entry.get("status")
    if not isinstance(status, dict):
        return "unknown"
    completed = status.get("completed")
    if type(completed) is not bool or (
        "status_str" in status and status["status_str"] != ("success" if completed else "error")
    ):
        return "unknown"
    messages = status.get("messages", [])
    if not isinstance(messages, list):
        return "unknown"
    event_types = set()
    for item in messages:
        if not (isinstance(item, list) and len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], dict)):
            return "unknown"
        if item[0] in {"execution_success", "execution_error", "execution_interrupted"}:
            if prompt_id is not None and item[1].get("prompt_id") != prompt_id:
                return "unknown"
            event_types.add(item[0])
    if len(event_types) > 1 or ("execution_success" in event_types and not completed):
        return "unknown"
    if event_types & {"execution_error", "execution_interrupted"}:
        if completed:
            return "unknown"
        return "failed" if "execution_error" in event_types else "cancelled"
    return "completed" if completed else "running"


def execution_error(result: CompletionResult) -> str | None:
    """Return a bounded owned failure summary without execution inputs or traceback."""
    if result.status != "failed" or not isinstance(result.prompt_id, str) or not result.prompt_id:
        return None
    candidates = []
    entry = result.history.get(result.prompt_id) if isinstance(result.history, dict) else None
    if history_status(entry, result.prompt_id) == "failed":
        candidates.extend(data for kind, data in entry["status"]["messages"] if kind == "execution_error")
    if _terminal_event(result.event, result.prompt_id) == "failed":
        candidates.append(result.event["data"])
    for data in candidates:
        message = data.get("exception_message")
        if data.get("prompt_id") != result.prompt_id or not isinstance(message, str) or not message.strip():
            continue
        message = " ".join("".join(char for char in message.strip().splitlines()[0] if char.isprintable()).split())[:500]
        if not message:
            continue
        details = {}
        for field in ("node_id", "node_type", "exception_type"):
            value = data.get(field)
            if isinstance(value, str) and value.strip() and not any(char in value for char in "\r\n\0"):
                details[field] = value.strip()[:120]
        location = "ComfyUI"
        if "node_id" in details:
            location += " node " + details["node_id"]
        if "node_type" in details:
            location += " (" + details["node_type"] + ")"
        kind = details.get("exception_type")
        return f"{location}: {kind + ': ' if kind else ''}{message}"
    return None


def _default_websocket_events(endpoint: str, client_id: str, timeout: float):
    """Yield ComfyUI WebSocket events when optional websocket-client is installed."""
    import websocket

    parsed = urlsplit(endpoint)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    url = f"{scheme}://{parsed.netloc}/ws?{urlencode({'clientId': client_id})}"
    deadline = time.monotonic() + timeout
    connection = websocket.WebSocket()
    transport = _connection_for(endpoint, deadline)
    expired = Event()

    def expire_connection():
        expired.set()
        owned_socket = connection.sock or transport.sock
        if owned_socket is not None:
            try:
                owned_socket.shutdown(SHUT_RDWR)
            except OSError:
                pass

    watchdog = Timer(timeout, expire_connection)
    watchdog.daemon = True
    watchdog.start()
    try:
        transport.connect()
        transport.sock.settimeout(remaining_timeout(deadline))
        connection.connect(url, timeout=remaining_timeout(deadline), socket=transport.sock, redirect_limit=0)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            connection.settimeout(min(1, remaining))
            try:
                message = connection.recv()
            except websocket.WebSocketTimeoutException:
                message = None
            # Also wake the history poller on idle sockets and binary previews.
            yield json.loads(message) if isinstance(message, str) else {}
    except Exception as error:
        if expired.is_set():
            raise TimeoutError("ComfyUI WebSocket deadline expired") from error
        raise
    finally:
        watchdog.cancel()
        watchdog.join()
        # A close handshake can itself block beyond the wait budget.
        connection.shutdown()
        transport.close()
