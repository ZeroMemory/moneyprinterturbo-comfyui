"""Ownership-aware ComfyUI server process lifecycle."""

from dataclasses import dataclass
import subprocess
from typing import Callable

from .profiles import LaunchProfile, validate_profile


@dataclass(frozen=True)
class ServerOwnership:
    started_by_operator: bool
    pid: int | None = None
    process: object | None = None
    allow_operator_stop_owned: bool = False


def may_stop_server(ownership: ServerOwnership) -> bool:
    return (
        ownership.started_by_operator
        and ownership.pid is not None
        and ownership.allow_operator_stop_owned
    )


def start_owned_server(
    profile: LaunchProfile,
    *,
    stdout_log: object,
    stderr_log: object,
    process_factory: Callable[..., object] = subprocess.Popen,
) -> ServerOwnership:
    """Start a hidden owned server process using the profile root as cwd."""
    if not profile.allow_operator_start:
        raise PermissionError("launch profile does not allow operator start")
    errors = validate_profile(profile)
    if errors:
        raise ValueError("invalid launch profile: " + "; ".join(errors))
    command = [str(profile.python_executable), *profile.launch_args]
    process = process_factory(
        command,
        cwd=str(profile.comfyui_root),
        stdout=stdout_log,
        stderr=stderr_log,
        stdin=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    pid = getattr(process, "pid", None)
    if not isinstance(pid, int):
        raise RuntimeError("started process did not provide a pid")
    return ServerOwnership(
        started_by_operator=True,
        pid=pid,
        process=process,
        allow_operator_stop_owned=profile.allow_operator_stop_owned,
    )


def stop_owned_server(ownership: ServerOwnership, *, timeout: float = 10) -> None:
    """Terminate an owned process, escalating to kill only after its timeout."""
    if not may_stop_server(ownership):
        raise PermissionError("refusing to stop a server not started by the operator")
    process = ownership.process
    if process is None or getattr(process, "pid", None) != ownership.pid:
        raise PermissionError("refusing to stop a process without matching ownership")

    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout)
