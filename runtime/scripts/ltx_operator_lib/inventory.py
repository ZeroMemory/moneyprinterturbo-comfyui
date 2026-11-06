"""Read-only local and HTTP inventory for a ComfyUI launch profile."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

from .api import ComfyClient
from .profiles import LaunchProfile, validate_profile


HttpGetter = Callable[[str], object]


@dataclass(frozen=True)
class InventoryReport:
    root: Path
    executable: Path
    endpoint: str
    root_available: bool
    executable_available: bool
    server_reachable: bool
    models_available: bool
    custom_nodes_available: bool
    system_stats_available: bool
    object_info_available: bool
    models: tuple[str, ...]
    custom_nodes: tuple[str, ...]
    system_stats: object | None
    object_info: object | None
    errors: tuple[str, ...]


def inspect_inventory(
    profile: LaunchProfile,
    *,
    http_get: HttpGetter | None = None,
) -> InventoryReport:
    """Inspect filenames and read-only server endpoints without hashing models."""
    if profile.remote_server:
        raise ValueError("Use preflight --template-request to verify a remote ComfyUI profile")
    errors = list(validate_profile(profile))
    comfyui_directory = profile.application_directory
    models_directory = comfyui_directory / "models"
    custom_nodes_directory = comfyui_directory / "custom_nodes"

    models = _relative_files(models_directory, errors, "models")
    custom_nodes = _custom_node_names(custom_nodes_directory, errors)

    system_stats = None
    object_info = None
    getter = http_get or _get_json
    if not any(error.startswith("unsafe endpoint") for error in errors):
        system_stats = _get_server_value(
            getter, f"{profile.endpoint.rstrip('/')}/system_stats", errors
        )
        object_info = _get_server_value(
            getter, f"{profile.endpoint.rstrip('/')}/object_info", errors
        )

    return InventoryReport(
        root=profile.comfyui_root,
        executable=profile.python_executable,
        endpoint=profile.endpoint,
        root_available=profile.comfyui_root.is_dir(),
        executable_available=profile.python_executable.is_file(),
        server_reachable=system_stats is not None or object_info is not None,
        models_available=models_directory.is_dir(),
        custom_nodes_available=custom_nodes_directory.is_dir(),
        system_stats_available=system_stats is not None,
        object_info_available=object_info is not None,
        models=models,
        custom_nodes=custom_nodes,
        system_stats=system_stats,
        object_info=object_info,
        errors=tuple(errors),
    )


def _relative_files(directory: Path, errors: list[str], label: str) -> tuple[str, ...]:
    if not directory.is_dir():
        errors.append(f"{label} directory is unavailable: {directory}")
        return ()
    try:
        return tuple(
            sorted(
                path.relative_to(directory).as_posix()
                for path in directory.rglob("*")
                if path.is_file()
            )
        )
    except OSError as error:
        errors.append(f"could not enumerate {label}: {error}")
        return ()


def _custom_node_names(directory: Path, errors: list[str]) -> tuple[str, ...]:
    if not directory.is_dir():
        errors.append(f"custom nodes directory is unavailable: {directory}")
        return ()
    try:
        return tuple(
            sorted(
                path.name
                for path in directory.iterdir()
                if path.name != "__pycache__"
                and not path.name.endswith(".disabled")
                and (path.is_dir() or (path.is_file() and path.suffix == ".py"))
            )
        )
    except OSError as error:
        errors.append(f"could not enumerate custom nodes: {error}")
        return ()


def _get_server_value(getter: HttpGetter, url: str, errors: list[str]) -> object | None:
    try:
        return getter(url)
    except Exception as error:
        errors.append(f"{url.rsplit('/', 1)[-1]} unavailable: {error}")
        return None


def _get_json(url: str) -> object:
    parsed = urlsplit(url)
    client = ComfyClient(f"{parsed.scheme}://{parsed.netloc}")
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    return client.transport.request("GET", path, timeout=3)
