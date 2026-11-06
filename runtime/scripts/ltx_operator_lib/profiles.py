"""Immutable ComfyUI launch profiles and fail-closed safety validation."""

from dataclasses import dataclass, fields
from ipaddress import ip_address, ip_network
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Callable
from urllib.parse import urlsplit


LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
PRIVATE_NETWORKS = tuple(map(ip_network, ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7")))
DANGEROUS_OPTIONS = (
    "--listen",
    "--port",
    "--tls-keyfile",
    "--tls-certfile",
    "--enable-cors-header",
)
DIRECTORY_CHANGING_OPTIONS = (
    "--output-directory",
    "--input-directory",
    "--temp-directory",
    "--base-directory",
    "--user-directory",
    "--extra-model-paths-config",
)
SAFE_BOOLEAN_OPTIONS = frozenset(
    {
        "--windows-standalone-build",
        "--enable-manager",
        "--use-sage-attention",
    }
)
DEFAULT_OPERATOR_ENDPOINT = "http://127.0.0.1:8188"


@dataclass(frozen=True)
class LaunchProfile:
    name: str
    comfyui_root: Path | None
    python_executable: Path | None
    endpoint: str
    output_directory: Path
    launch_args: tuple[str, ...]
    optional_modules: tuple[str, ...]
    startup_timeout_seconds: float
    expected_gpu_name: str | None
    expected_vram_mb: int | None
    attention_backend: str
    allow_operator_start: bool
    allow_operator_stop_owned: bool
    model_roots: tuple[Path, ...] = ()
    external_server: bool = False
    remote_server: bool = False
    allow_insecure_http: bool = False
    token_env: str | None = None

    @property
    def application_directory(self) -> Path:
        """Resolve both the standalone source tree and the portable distribution."""
        target = _launch_target(self)
        if target is None:
            raise ValueError("unsupported ComfyUI launch target")
        return target.parent


def load_profile(path: str | Path) -> LaunchProfile:
    """Load a JSON profile and resolve all paths relative to its file."""
    profile_path = Path(path).expanduser().resolve()
    return profile_from_dict(json.loads(profile_path.read_text(encoding="utf-8")), profile_path.parent)


def profile_from_dict(data: dict, base: Path) -> LaunchProfile:
    if not isinstance(data, dict):
        raise ValueError("launch profile JSON must be an object")

    if "remote_server" in data and _required_bool(data, "remote_server"):
        if set(data) - {"name", "endpoint", "remote_server", "allow_insecure_http", "token_env"}:
            raise ValueError("Remote ComfyUI profiles cannot specify local launch or file settings")
        return LaunchProfile(
            name=_required_string(data, "name"), comfyui_root=None, python_executable=None,
            endpoint=_required_string(data, "endpoint"), output_directory=base / ".remote-output",
            launch_args=(), optional_modules=(), startup_timeout_seconds=120,
            expected_gpu_name=None, expected_vram_mb=None, attention_backend="default",
            allow_operator_start=False, allow_operator_stop_owned=False, remote_server=True,
            allow_insecure_http=_required_bool(data, "allow_insecure_http") if "allow_insecure_http" in data else False,
            token_env=_required_string(data, "token_env") if "token_env" in data else None,
        )

    return LaunchProfile(
        name=_required_string(data, "name"),
        comfyui_root=_resolved_path(base, _required_string(data, "comfyui_root")),
        python_executable=_interpreter_path(base, _required_string(data, "python_executable")),
        endpoint=_required_string(data, "endpoint"),
        output_directory=_resolved_path(
            base, _required_string(data, "output_directory")
        ),
        model_roots=tuple(_resolved_path(base, root) for root in _string_tuple(data, "model_roots")) if "model_roots" in data else (),
        launch_args=_string_tuple(data, "launch_args"),
        optional_modules=_string_tuple(data, "optional_modules"),
        startup_timeout_seconds=_positive_number(data, "startup_timeout_seconds"),
        expected_gpu_name=_required_string(data, "expected_gpu_name") if "expected_gpu_name" in data else None,
        expected_vram_mb=_positive_integer(data, "expected_vram_mb") if "expected_vram_mb" in data else None,
        attention_backend=_required_string(data, "attention_backend"),
        allow_operator_start=_required_bool(data, "allow_operator_start"),
        allow_operator_stop_owned=_required_bool(data, "allow_operator_stop_owned"),
        external_server=_required_bool(data, "external_server") if "external_server" in data else False,
    )


def load_profile_snapshot(snapshot: dict) -> LaunchProfile:
    """Require displayed profile fields to match the selected source profile."""
    source = Path(snapshot["source_path"]).expanduser().resolve()
    profile = load_profile(source)
    if "endpoint" not in snapshot:
        raise ValueError("profile snapshot must include endpoint")
    if profile.remote_server and snapshot.get("remote_server") is not True:
        raise ValueError("Remote ComfyUI profile snapshot must declare its mode")
    for field in fields(profile):
        if field.name not in snapshot:
            continue
        expected = getattr(profile, field.name)
        actual = snapshot[field.name]
        if field.name == "model_roots":
            actual = tuple(_resolved_path(source.parent, root) for root in _string_tuple(snapshot, "model_roots"))
        elif field.name == "python_executable" and isinstance(actual, str):
            actual = _interpreter_path(source.parent, actual)
        elif isinstance(expected, Path) and isinstance(actual, str):
            actual = _resolved_path(source.parent, actual)
        elif isinstance(expected, tuple) and isinstance(actual, list):
            actual = tuple(actual)
        if actual != expected or isinstance(actual, bool) != isinstance(expected, bool):
            raise ValueError(f"profile snapshot mismatch: {field.name}")
    return profile


def validate_profile(
    profile: LaunchProfile,
    module_exists: Callable[[str], bool] | None = None,
) -> tuple[str, ...]:
    """Return concrete validation errors without changing the installation."""
    if profile.remote_server:
        error = remote_endpoint_error(profile.endpoint, profile.allow_insecure_http, profile.token_env)
        return (error,) if error else ()
    errors: list[str] = []
    if not profile.comfyui_root.is_dir():
        errors.append(f"ComfyUI root does not exist: {profile.comfyui_root}")
    if not profile.external_server and not profile.python_executable.is_file():
        errors.append(f"Python executable does not exist: {profile.python_executable}")
    if profile.external_server and (profile.allow_operator_start or profile.allow_operator_stop_owned):
        errors.append("An external server cannot allow operator start or stop")
    if profile.external_server and profile.optional_modules:
        errors.append("An external server cannot check local optional Python modules")
    if not profile.output_directory.is_dir():
        errors.append(f"ComfyUI output directory does not exist: {profile.output_directory}")
    launch_target = _launch_target(profile)
    if launch_target is None:
        errors.append("configured launch target must be main.py or ComfyUI/main.py")
    else:
        if not launch_target.is_file():
            errors.append(f"configured launch target does not exist: {launch_target}")
        expected = (launch_target.parent / "output").resolve()
        if profile.output_directory.resolve() != expected:
            errors.append(f"ComfyUI output directory must match default output directory: {expected}")

    endpoint_error = local_endpoint_error(profile.endpoint)
    if endpoint_error:
        errors.append(endpoint_error)
    if profile.allow_operator_start and profile.endpoint != DEFAULT_OPERATOR_ENDPOINT:
        errors.append(
            f"unsafe operator-start endpoint: expected {DEFAULT_OPERATOR_ENDPOINT}"
        )
    launch_argv_error = _launch_argv_error(profile.launch_args)
    if launch_argv_error:
        errors.append(launch_argv_error)
    errors.extend(_launch_arg_errors(profile.launch_args))

    for module in profile.optional_modules:
        if profile.external_server:
            continue
        try:
            available = (
                module_exists(module)
                if module_exists is not None
                else _module_exists(
                    profile.python_executable,
                    profile.comfyui_root,
                    _interpreter_flags(profile.launch_args),
                    module,
                )
            )
        except Exception as error:
            errors.append(f"optional Python module check failed for {module}: {error}")
        else:
            if not available:
                errors.append(f"optional Python module is unavailable: {module}")

    if not _is_positive_finite_number(profile.startup_timeout_seconds):
        errors.append("startup_timeout_seconds must be a finite positive number")
    if profile.expected_gpu_name is not None and (
        not isinstance(profile.expected_gpu_name, str) or not profile.expected_gpu_name
    ):
        errors.append("expected_gpu_name must be a non-empty string")
    if profile.expected_vram_mb is not None and (
        isinstance(profile.expected_vram_mb, bool)
        or not isinstance(profile.expected_vram_mb, int)
        or profile.expected_vram_mb <= 0
    ):
        errors.append("expected_vram_mb must be a positive integer")
    if not isinstance(profile.attention_backend, str) or not profile.attention_backend:
        errors.append("attention_backend must be a non-empty string")
    if not isinstance(profile.allow_operator_start, bool):
        errors.append("allow_operator_start must be a boolean")
    if not isinstance(profile.allow_operator_stop_owned, bool):
        errors.append("allow_operator_stop_owned must be a boolean")
    if not isinstance(profile.external_server, bool):
        errors.append("external_server must be a boolean")
    return tuple(errors)


def _required_string(data: dict, name: str) -> str:
    value = data.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _string_tuple(data: dict, name: str) -> tuple[str, ...]:
    value = data.get(name)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{name} must be a list of non-empty strings")
    return tuple(value)


def _positive_number(data: dict, name: str) -> float:
    value = data.get(name)
    if not _is_positive_finite_number(value):
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def _positive_integer(data: dict, name: str) -> int:
    value = data.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _required_bool(data: dict, name: str) -> bool:
    value = data.get(name)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _resolved_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _interpreter_path(base: Path, value: str) -> Path:
    # Resolving a venv's bin/python symlink selects the system environment on POSIX.
    path = base / Path(value).expanduser()
    return path.parent.resolve() / path.name


def _launch_target(profile: LaunchProfile) -> Path | None:
    if profile.comfyui_root is None:
        return None
    target_index = 1 if profile.launch_args[:1] == ("-s",) else 0
    if len(profile.launch_args) <= target_index:
        return None
    argument = profile.launch_args[target_index]
    if argument not in ("main.py", "ComfyUI/main.py"):
        return None
    return (profile.comfyui_root / Path(argument)).resolve()


def _interpreter_flags(launch_args: tuple[str, ...]) -> tuple[str, ...]:
    return ("-s",) if launch_args[:1] == ("-s",) else ()


def local_endpoint_error(endpoint: str) -> str | None:
    if not isinstance(endpoint, str) or any(char.isspace() or ord(char) < 32 for char in endpoint):
        return "unsafe endpoint: expected a loopback HTTP origin"
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except ValueError as error:
        return f"unsafe endpoint: {error}"
    if (
        parsed.scheme != "http"
        or not parsed.hostname
        or port is None
        or not 1 <= port <= 65535
    ):
        return "unsafe endpoint: only loopback http endpoints with an explicit port are allowed"
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or "?" in endpoint
        or "#" in endpoint
        or "\\" in endpoint
    ):
        return "unsafe endpoint: path, credentials, query strings, and fragments are not allowed"
    if parsed.hostname.lower() not in LOOPBACK_HOSTS:
        return "unsafe endpoint: only loopback http endpoints are allowed"
    return None


def remote_endpoint_error(endpoint: str, allow_insecure_http: bool, token_env: str | None = None) -> str | None:
    if not isinstance(endpoint, str) or any(char.isspace() or ord(char) < 32 for char in endpoint):
        return "Remote ComfyUI requires an explicit HTTP(S) origin"
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except ValueError:
        return "Remote ComfyUI requires an explicit HTTP(S) origin"
    if (not parsed.hostname or parsed.username is not None or parsed.password is not None
            or parsed.path not in {"", "/"} or "?" in endpoint or "#" in endpoint or "\\" in endpoint
            or port == 0 or parsed.netloc.endswith(":")):
        return "Remote ComfyUI origin cannot include credentials, a path, a query or a fragment"
    if token_env is not None and (not isinstance(token_env, str)
                                  or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token_env)):
        return "Remote ComfyUI token must name an environment variable"
    if parsed.scheme == "https":
        return None
    if parsed.scheme != "http" or allow_insecure_http is not True or token_env is not None:
        return "Remote ComfyUI requires HTTPS, or explicit token-free HTTP on a private network"
    if parsed.hostname in LOOPBACK_HOSTS:
        return None
    try:
        host = ip_address(parsed.hostname)
    except ValueError:
        return "Remote ComfyUI over HTTP requires a private LAN IP address"
    if not any(host in network for network in PRIVATE_NETWORKS):
        return "Remote ComfyUI over HTTP requires a private LAN IP address"
    return None


def _launch_arg_errors(launch_args: tuple[str, ...]) -> tuple[str, ...]:
    errors: list[str] = []
    for index, argument in enumerate(launch_args):
        if not argument.startswith("--"):
            continue
        option = argument.partition("=")[0]
        matches = [
            dangerous
            for dangerous in DANGEROUS_OPTIONS
            if option == dangerous or dangerous.startswith(option)
        ]
        directory_matches = [
            dangerous
            for dangerous in DIRECTORY_CHANGING_OPTIONS
            if option == dangerous or dangerous.startswith(option)
        ]
        if matches:
            errors.append(f"unsafe endpoint-changing launch option: {argument}")
        if directory_matches:
            errors.append(f"unsafe directory-changing launch option: {argument}")
        if not matches:
            continue
        if option == "--listen" and argument == "--listen":
            if index + 1 >= len(launch_args) or launch_args[index + 1].startswith("-"):
                errors.append("unsafe launch args: bare --listen is not allowed")
                continue
            host = launch_args[index + 1]
        elif option == "--listen" and argument.startswith("--listen="):
            host = argument.partition("=")[2]
            if not host:
                errors.append("unsafe launch args: bare --listen is not allowed")
                continue
        else:
            continue
        if host.strip("[]").lower() not in LOOPBACK_HOSTS:
            errors.append(f"unsafe launch args: non-loopback --listen host {host}")
    return tuple(errors)


def _launch_argv_error(launch_args: tuple[str, ...]) -> str | None:
    index = 0
    if launch_args[:1] == ("-s",):
        index = 1
    if len(launch_args) <= index or launch_args[index] not in ("main.py", "ComfyUI/main.py"):
        return "unsafe launch argv: expected optional -s followed by main.py or ComfyUI/main.py"

    index += 1
    seen: set[str] = set()
    while index < len(launch_args):
        option = launch_args[index]
        if option in SAFE_BOOLEAN_OPTIONS and option not in seen:
            seen.add(option)
            index += 1
            continue
        if option == "--reserve-vram" and option not in seen:
            if index + 1 >= len(launch_args):
                return "unsafe launch argv: --reserve-vram requires a numeric value"
            try:
                reserve_vram = float(launch_args[index + 1])
            except ValueError:
                return "unsafe launch argv: --reserve-vram requires a numeric value"
            if not math.isfinite(reserve_vram) or reserve_vram < 0:
                return "unsafe launch argv: --reserve-vram requires a finite non-negative value"
            seen.add(option)
            index += 2
            continue
        return f"unsafe launch argv: unsupported or misplaced argument {option}"
    return None


def _module_exists(
    python_executable: Path,
    comfyui_root: Path,
    interpreter_flags: tuple[str, ...],
    name: str,
) -> bool:
    script = (
        "import importlib.util,sys;"
        "print('true' if importlib.util.find_spec(sys.argv[1]) is not None else 'false')"
    )
    try:
        result = subprocess.run(
            [str(python_executable), *interpreter_flags, "-c", script, name],
            cwd=str(comfyui_root),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and result.stdout.strip().lower() == "true"


def _is_positive_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value > 0
    )
