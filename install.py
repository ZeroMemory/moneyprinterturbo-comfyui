"""Install the ComfyUI add-on in MoneyPrinterTurbo source, portable or pip installations."""

import argparse
import contextlib
import base64
import csv
from email.parser import Parser
import errno
import hashlib
import json
import io
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import tomllib
from uuid import NAMESPACE_URL, uuid5


ADDON = Path(__file__).resolve().parent
sys.path.insert(0, str(ADDON))  # Embedded Python may omit the script directory from sys.path.
VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
MAIN_COMMIT = "ace6c8f3a8dd9508776dbbf6f0bdc9b2d973254a"
STATE = ".ltx-operator"
PROTECTED = {"config.toml", ".git", ".github", STATE}
MPT_FILES = (
    "app/services/execution_control.py",
    "app/services/ltx.py",
    "app/services/media_timing.py",
    "app/services/nim_routes.py",
    "resource/ltx/ltx_2_3.md",
    "resource/ltx/ltx_2_5.md",
    "resource/ltx/minimax_h3.md",
    "webui/ltx_ui.py",
)
EDITOR_FILES = ("__init__.py", "js/editor.js")


class InstallError(Exception):
    pass


def digest(data):
    return hashlib.sha256(data).hexdigest() if data is not None else None


def checked(root, name):
    """Reject aliases to files outside the tree, including Windows junctions."""
    parts = PurePosixPath(name).parts
    if not parts or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", p) or p.endswith(".")
                        or re.fullmatch(r"CON|PRN|AUX|NUL|(?:COM|LPT)[0-9]", p.split(".")[0], re.I)
                        for p in parts) or "/".join(parts) != name:
        raise InstallError(f"Unsafe path: {name!r}")
    location = root
    for part in parts:
        location = location / part
        try:
            info = location.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise InstallError(f"Link in managed path: {name}")
        if location != root / name and not stat.S_ISDIR(info.st_mode):
            raise InstallError(f"Invalid parent of {name}")
    return location


def source_file(root, name):
    parts = name.casefold().split("/")
    if (parts[0] in PROTECTED or parts[:2] in (["resource", "fonts"], ["resource", "songs"])
            or any(p.startswith(".") for p in parts)):
        raise InstallError(f"Protected path: {name}")
    return checked(root, name)


def contents(path):
    if not path.exists():
        return None
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink > 1:
        raise InstallError(f"Expected a regular, unshared file: {path.name}")
    return path.read_bytes()


def packaged_sources():
    sources = {}
    for name in MPT_FILES:
        data = contents(checked(ADDON, "mpt/" + name))
        if data is None:
            raise InstallError(f"Installation package is missing mpt/{name}.")
        sources[name] = data
    return sources


def packaged_editor_sources():
    sources = {}
    for name in EDITOR_FILES:
        path = checked(ADDON, "runtime/comfyui/mpt_scene_editor/" + name)
        data = contents(path)
        if data is None:
            raise InstallError(f"Installation package is missing the ComfyUI scene editor: {name}.")
        sources[name] = data
    return sources


def package_info(target):
    """Return wheel metadata only for a package installation, never a source tree."""
    if checked(target, "pyproject.toml").is_file():
        return None
    entries = list(target.glob("mpt-*.dist-info"))
    if not entries:
        return None
    if len(entries) != 1:
        raise InstallError("Expected one installed mpt distribution in this Python environment.")
    name = entries[0].name
    metadata = Parser().parsestr(checked(target, name + "/METADATA").read_text("utf-8"))
    if metadata.get("Name", "").lower() != "mpt":
        raise InstallError("The selected package metadata does not identify mpt.")
    return name, metadata.get("Version")


def resolve_target(value=None, mpt_python=None):
    target = Path(value).expanduser().resolve(strict=True) if value else None
    if target and not (target / "pyvenv.cfg").is_file():
        return target
    if mpt_python is None and target:
        mpt_python = next((target / name for name in ("Scripts/python.exe", "bin/python")
                           if (target / name).is_file()), None)
        if mpt_python is None:
            raise InstallError("The selected Python environment has no interpreter.")
    interpreter = str(Path(mpt_python or sys.executable).expanduser().absolute())
    probe = """import importlib.metadata as m, json, sys
if sys.version_info < (3, 11): raise SystemExit('Python 3.11 or newer is required')
d = m.distribution('mpt')
print(json.dumps(str(d.locate_file(''))))
"""
    try:
        result = subprocess.run([interpreter, "-I", "-B", "-c", probe], capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired as exc:
        raise InstallError("The selected mpt Python did not answer within 30 seconds.") from exc
    if result.returncode:
        raise InstallError("No installed mpt package was found in the selected Python. "
                           "Select its source/portable folder or use --mpt-python with the interpreter from an MPT virtual environment.")
    target = Path(json.loads(result.stdout)).resolve(strict=True)
    if not package_info(target):
        raise InstallError("Select the source folder for an editable mpt installation.")
    return target


def package_path(name, packaged):
    if packaged:
        if name == "cli.py":
            return "app/cli.py"
        if name == "config.example.toml" or name.startswith("resource/"):
            return "app/_assets/" + name
    return name


def package_record(original, outputs):
    """Keep pip's file inventory accurate, including newly supplied modules/assets."""
    rows = list(csv.reader(io.StringIO(original.decode("utf-8"))))
    remaining = dict(outputs)
    for row in rows:
        if len(row) != 3:
            raise InstallError("Invalid mpt RECORD file.")
        if row[0] in remaining:
            data = remaining.pop(row[0])
            row[1:] = ["sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode(),
                       str(len(data))]
    for name, data in remaining.items():
        rows.append([name, "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode(),
                     str(len(data))])
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerows(rows)
    return stream.getvalue().encode("utf-8")


def check_source_streams(path):
    if os.name != "nt" or not path.exists():
        return
    import ctypes

    class StreamData(ctypes.Structure):
        _fields_ = [("size", ctypes.c_longlong), ("name", ctypes.c_wchar * 296)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.FindFirstStreamW.restype = ctypes.c_void_p
    stream = StreamData()
    handle = kernel.FindFirstStreamW(str(path), 0, ctypes.byref(stream), 0)
    if handle == ctypes.c_void_p(-1).value:
        error = ctypes.get_last_error()
        if error in (38, 87):  # No streams, or the filesystem does not support streams.
            return
        raise ctypes.WinError(error)
    handle = ctypes.c_void_p(handle)
    try:
        if stream.name != "::$DATA" or kernel.FindNextStreamW(handle, ctypes.byref(stream)):
            raise InstallError(f"{path} has a named data stream. Preserve it before replacing or removing this path.")
        error = ctypes.get_last_error()
        if error != 38:  # ERROR_HANDLE_EOF
            raise ctypes.WinError(error)
    finally:
        kernel.FindClose(handle)


def check_tree_streams(root):
    if os.name != "nt":
        return
    check_source_streams(root)
    for path in root.rglob("*"):
        checked(root, path.relative_to(root).as_posix())
        check_source_streams(path)


def delete_readonly(path):
    import _winapi
    import ctypes

    # FileDispositionInfoEx deletes this name without clearing READONLY on other links.
    handle = _winapi.CreateFile(str(path), 0x10000, 7, 0, 3, 0, 0)  # DELETE, share all, OPEN_EXISTING
    try:
        flags = ctypes.c_ulong(0x11)  # FILE_DISPOSITION_DELETE | FILE_DISPOSITION_IGNORE_READONLY_ATTRIBUTE
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        if not kernel.SetFileInformationByHandle(ctypes.c_void_p(handle), 21, ctypes.byref(flags), ctypes.sizeof(flags)):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        _winapi.CloseHandle(handle)


def check_replacement(target, path):
    check_source_streams(path)
    if os.name == "nt" and path.is_file() and path.stat().st_file_attributes & stat.FILE_ATTRIBUTE_READONLY:
        # Windows deletes this disposable probe on close, including on process exit.
        with tempfile.NamedTemporaryFile(dir=target) as probe:
            sample = Path(probe.name)
            sample.chmod(stat.S_IREAD)
            try:
                delete_readonly(sample)
            except OSError as exc:
                raise InstallError(f"This volume cannot safely replace the read-only file {path}.") from exc


def write_file(path, data, mode=0o644):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
    path.chmod(mode)


def write_atomic(path, data, mode=0o644):
    # An interrupted write to live state must leave the journal itself readable.
    check_source_streams(path)
    with installer_scratch(path.parent.parent, "write") as tmp:
        pending = tmp / "pending"
        write_file(pending, data, mode)
        pending.replace(path)


def discard_scratch(target, path):
    checked(target, path.name)
    check_tree_streams(path)
    marker = checked(path, ".gitignore")
    stamp = contents(marker)
    expected = scratch_marker(path)
    children = [child for child in path.iterdir() if child != marker]
    if stamp != expected:
        # A crash can precede marker publication or follow its removal. Leave uncertain ownership alone.
        if not children and (stamp is None or expected.startswith(stamp)):
            return
        raise InstallError(f"Unrecognized installer scratch directory: {path}. Move it aside before retrying.")
    allowed = {"pending"} if path.name.startswith(".ltx-write-") else {"source", STATE, "profile.json", "workflow.json"}
    for child in children:
        if child.name not in allowed or (child.name == "pending" and not child.is_file()):
            raise InstallError(f"Unexpected file in installer scratch: {child}. Move it aside before retrying.")
        checked(path, child.name)
    for child in children:
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    marker.unlink()
    path.rmdir()


def scratch_marker(path):
    return f"*\n# LTX Operator temporary directory {path.name}\n".encode("ascii")


@contextlib.contextmanager
def installer_scratch(target, purpose):
    path = Path(tempfile.mkdtemp(dir=target, prefix=f".ltx-{purpose}-"))
    (path / ".gitignore").write_bytes(scratch_marker(path))
    try:
        yield path
    finally:
        discard_scratch(target, path)


@contextlib.contextmanager
def install_lock(target):
    # The OS releases both types of lock if the installer is interrupted.
    descriptor = None
    try:
        if os.name == "nt":
            descriptor = os.open(target / ".ltx-operator.lock",
                                 os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_TEMPORARY, 0o600)
        else:
            import fcntl
            descriptor = os.open(target, os.O_RDONLY)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (FileExistsError, BlockingIOError) as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise InstallError("Another installer is using this project.") from exc
    try:
        yield
    finally:
        os.close(descriptor)


def git(directory, *args, staged=False):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    options = ["-c", "core.fsmonitor=false"]
    if staged:
        env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
        options += ["-c", "core.autocrlf=false"]
    result = subprocess.run(["git", *options, *args], cwd=directory, env=env, capture_output=True)
    if result.returncode:
        raise InstallError(result.stderr.decode("utf-8", "replace").strip() or "Git failed.")
    return result.stdout


def release_catalog():
    path = checked(ADDON, "supported-releases.json")
    if not path.is_file():
        raise InstallError("Installation package is missing supported-releases.json.")
    releases = json.loads(path.read_text("utf-8"))
    if not isinstance(releases, dict) or not releases:
        raise InstallError("Expected a nonempty object in supported-releases.json.")
    for release, filename in releases.items():
        if (not VERSION.fullmatch(release) or not isinstance(filename, str)
                or not re.fullmatch(r"[a-zA-Z0-9_-]+(?:\.[a-zA-Z0-9_-]+)*\.patch", filename)
                or filename.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL"}
                or re.fullmatch(r"(?:COM|LPT)\d", filename.split(".")[0], re.I)):
            raise InstallError("Invalid release version or patch filename in supported-releases.json.")
    return releases


def is_pinned_main(target):
    try:
        source = git(target, "rev-parse", "--show-toplevel", "HEAD").splitlines()
    except InstallError:
        return False  # A source ZIP or portable installation can lack Git metadata.
    return (len(source) == 2 and source[1] == MAIN_COMMIT.encode("ascii")
            and Path(os.fsdecode(source[0])).resolve() == target)


def project_patch(target, trial, installed=False):
    package = package_info(target)
    project = ({"name": "mpt", "version": package[1]} if package else
               tomllib.loads(checked(target, "pyproject.toml").read_text("utf-8-sig"))["project"])
    if project.get("name") != "mpt" or not checked(target, "webui/Main.py").is_file():
        raise InstallError("Select the MoneyPrinterTurbo installation containing webui/Main.py.")
    releases = release_catalog()
    version = project.get("version")
    filename = releases.get(version, "mpt.patch") if isinstance(version, str) else "mpt.patch"
    if trial is not None:
        if package:
            raise InstallError("A compatibility trial requires an exact tagged source checkout.")
        if not VERSION.fullmatch(trial):
            raise InstallError("A trial requires an exact stable X.Y.Z version.")
        if version != trial:
            raise InstallError(f"Expected MoneyPrinterTurbo {trial}, found {version}.")
        top = Path(os.fsdecode(git(target, "rev-parse", "--show-toplevel").strip())).resolve()
        if (top != target or git(target, "rev-parse", "HEAD") !=
                git(target, "rev-parse", "--verify", f"refs/tags/v{trial}^{{commit}}")
                or (not installed and git(target, "status", "--porcelain", "--", ".", ":(exclude).ltx-operator.lock"))):
            raise InstallError("A compatibility trial requires a clean checkout of that exact tag.")
    elif not package and is_pinned_main(target):
        if not installed and git(target, "status", "--porcelain", "--", ".", ":(exclude).ltx-operator.lock"):
            raise InstallError("Use a clean checkout of the supported MoneyPrinterTurbo main commit.")
        filename = "mpt-main.patch"
    elif not isinstance(version, str) or version not in releases:
        raise InstallError(f"Unsupported MoneyPrinterTurbo {version}. Supported: {', '.join(releases)}.")
    patch = checked(ADDON, "payload/" + filename)
    if not patch.is_file():
        raise InstallError(f"Installation package is missing payload/{filename}.")
    return patch


def prepare_model(state, comfyui, model_version, comfy_python, model_roots, endpoint, external_server=False, *,
                  workflow_api, remote_server=False, allow_insecure_http=False, token_env=None):
    from configure_ltx import build_template
    from ltx_operator_lib.api import ComfyClient, client_for_snapshot
    from ltx_operator_lib.planning import preflight

    with installer_scratch(state.parent, "check") as tmp:
        probe = Path(tmp) / "profile.json"
        profile, request = build_template(comfyui, probe, python=comfy_python, model_roots=model_roots,
                                          endpoint=endpoint, model_version=model_version,
                                          external_server=external_server, workflow_api=workflow_api,
                                          remote_server=remote_server, allow_insecure_http=allow_insecure_http,
                                          token_env=token_env)
        profile_bytes = (json.dumps(profile, indent=2) + "\n").encode()
        probe.write_bytes(profile_bytes)
        snapshot = tmp / "workflow.json"
        graph_bytes = (json.dumps(request["api_prompt"], indent=2) + "\n").encode()
        snapshot.write_bytes(graph_bytes)
        request["variant"]["source_path"] = str(snapshot)
        report = preflight(request, client_for_snapshot(request["profile"], ComfyClient), allow_empty_prompt=True)
        if report["errors"]:
            raise InstallError(f"Check ComfyUI at {endpoint}. Start the server and resolve the errors below.\n"
                               + "\n".join(report["errors"]))
        if not all(isinstance(report["checks"][k], dict) for k in ("system_stats", "object_info", "queue")):
            raise InstallError(f"Expected ComfyUI API responses from {endpoint}.")
    # Immutable files keep earlier scene plans usable when another model is registered.
    profile_name = f"profile-{digest(profile_bytes)}.json"
    graph_name = f"workflow-{digest(graph_bytes)}.json"
    request["profile"]["source_path"] = str(state / profile_name)
    request["variant"]["source_path"] = str(state / graph_name)
    request["client_id"] = str(uuid5(NAMESPACE_URL, f"{state}/{profile_name}/{model_version}"))
    request_bytes = (json.dumps(request, indent=2) + "\n").encode()
    request_name = f"request-{digest(request_bytes)}.json"
    return (model_version, state / request_name), {
        profile_name: profile_bytes, graph_name: graph_bytes, request_name: request_bytes,
    }


def generated_file(state, name, allow_missing=False):
    match = re.fullmatch(r"(?:request|profile|workflow)-([0-9a-f]{64})\.json", name)
    if not match:
        raise InstallError("Invalid model file in the installer receipt.")
    path = checked(state, name)
    data = contents(path)
    if not (allow_missing and data is None) and digest(data) != match[1]:
        raise InstallError(f"Generated model file changed: {path}. Preserve your edits before continuing.")
    return path


def operator_settings(python, template, saved=None, builtin=None):
    # A venv's bin/python is commonly a symlink. Resolving it loses that venv.
    values = {"python_executable": Path(python or sys.executable).expanduser().absolute(),
              "operator_script": ADDON / "runtime/scripts/ltx_operator.py"}
    saved = saved or {}
    registered = dict(saved.get("request_templates", {}))
    previous = saved.get("request_template")
    paths = template if isinstance(template, list) else ([template] if template else [])
    # A previous single-template installation has no model registry yet.
    legacy = [previous] if previous and not registered and Path(previous).is_file() else []
    if previous:
        values["request_template"] = Path(previous)
    for candidate in [*legacy, *paths]:
        path = Path(candidate).expanduser().resolve()
        if not path.is_file():
            raise InstallError(f"request_template must point to an existing file: {path}")
        request = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(request, dict) or not isinstance(request.get("variant", {}), dict):
            raise InstallError(f"Invalid request template: {path}")
        if candidate in paths:
            from runtime.scripts.ltx_operator_lib.planning import validate_request_template

            try:
                validate_request_template(request)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise InstallError(f"Invalid request template {path}: {exc}") from exc
        version = request.get("variant", {}).get("model_version")
        key = version if version in ("2.3", "2.5", "h3") else "custom"
        registered[key] = str(path)
        values["request_template"] = path
    for key, path in values.items():
        if key == "request_template" and builtin:
            continue  # The selected built-in template will replace this default.
        if not path.is_file():
            raise InstallError(f"{key} must point to an existing file.")
    if builtin:
        version, path = builtin
        registered[version] = str(path)
        values["request_template"] = path
    lines = ["[ltx]", "enabled = true"]
    for key, path in values.items():
        lines.append(f"{key} = {json.dumps(str(path), ensure_ascii=False)}")
    if registered:
        lines += ["", "[ltx.request_templates]"]
        lines += [f"{json.dumps(key)} = {json.dumps(path, ensure_ascii=False)}"
                  for key, path in sorted(registered.items())]
    return ("\n".join(lines) + "\n").encode()


def save_receipt(state, receipt, staged=False):
    writer = write_file if staged else write_atomic
    writer(checked(state, "receipt.json"), json.dumps(receipt, indent=2).encode())


def publish_exclusive(source, destination):
    if os.name == "nt":
        source.rename(destination)  # Windows rename never replaces an existing file.
        return
    try:
        os.link(source, destination)
    except OSError as exc:
        if exc.errno not in (errno.EPERM, errno.ENOTSUP, errno.ENOSYS):
            raise
        # Filesystems without hard links can still support an exclusive rename.
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        paths = os.fsencode(source), os.fsencode(destination)
        if sys.platform == "darwin" and hasattr(libc, "renamex_np"):
            result = libc.renamex_np(*paths, 4)  # RENAME_EXCL
        elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
            result = libc.renameat2(-100, paths[0], -100, paths[1], 1)  # AT_FDCWD, RENAME_NOREPLACE
        else:
            raise InstallError(f"The filesystem cannot safely publish {destination} without overwriting it.") from exc
        if result:
            code = ctypes.get_errno()
            if code in (errno.EINVAL, errno.ENOTSUP, errno.ENOSYS):
                if destination.exists() or destination.is_symlink():
                    raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(destination))
                raise InstallError(f"The filesystem cannot safely publish {destination} without overwriting it.")
            raise OSError(code, os.strerror(code), str(destination))
    else:
        source.unlink()


def unlink_exchange(path):
    try:
        path.unlink(missing_ok=True)
    except PermissionError:
        info = path.lstat()
        if (os.name != "nt" or info.st_nlink != 1
                or not info.st_file_attributes & stat.FILE_ATTRIBUTE_READONLY):
            raise
        delete_readonly(path)


def finish_exchange(target, state, receipt, applied=False):
    box = checked(state, "exchange")
    if not box.exists():
        return
    check_tree_streams(box)
    if any(p.name not in {"name", "old", "new"} for p in box.iterdir()):
        raise InstallError(f"Unexpected files in {box}. Preserve them before retrying.")
    namefile = checked(box, "name")
    old, new = checked(box, "old"), checked(box, "new")
    if old.exists() or new.exists():
        name = namefile.read_text(encoding="utf-8")
        if name not in receipt["files"]:
            raise InstallError(f"Unknown file in {box}. Preserve it before retrying.")
        dest = source_file(target, name)
        if old.exists():
            record = receipt["files"][name]
            if not applied and not dest.exists():
                publish_exclusive(old, dest)
            elif dest.exists() and old.samefile(dest):
                old.unlink()
            elif digest(contents(old)) in (record["before"], record["after"]):
                check_source_streams(old)
                unlink_exchange(old)
            else:
                raise InstallError(f"A concurrent save was preserved in {old} for {dest}. Keep it before retrying.")
        unlink_exchange(new)
    namefile.unlink(missing_ok=True)
    box.rmdir()


def change_source(target, state, receipt, name, expected, data):
    """Claim the previous file before checking its hash; only publish into a vacant name."""
    dest = source_file(target, name)
    box = checked(state, "exchange")
    box.mkdir()
    applied = False
    try:
        (box / "name").write_text(name, encoding="utf-8")
        if data is not None:
            new = box / "new"
            new.write_bytes(data)
            new.chmod(receipt["files"][name]["mode"])
            dest.parent.mkdir(parents=True, exist_ok=True)
        if expected is not None:
            old = box / "old"
            dest.rename(old)
            if digest(contents(old)) != expected:
                raise InstallError(f"{name} changed during installation or removal. Preserve your edits before retrying.")
            check_source_streams(old)
        if data is not None:
            publish_exclusive(new, dest)
        elif dest.exists():
            raise InstallError(f"{name} was recreated during removal. Preserve your edits before retrying.")
        applied = True
    finally:
        finish_exchange(target, state, receipt, applied)


def install_scene_editor(comfyui):
    root = Path(comfyui).expanduser().resolve(strict=True)
    app = root if checked(root, "main.py").is_file() else checked(root, "ComfyUI")
    if not checked(app, "main.py").is_file():
        raise InstallError("Choose the directory containing main.py or ComfyUI/main.py.")
    sources = packaged_editor_sources()
    nodes = checked(app, "custom_nodes")
    check_source_streams(nodes)
    if nodes.exists() and not nodes.is_dir():
        raise InstallError(f"Preserve the existing path before installing the scene editor: {nodes}")
    nodes.mkdir(exist_ok=True)
    with install_lock(nodes):
        pack = checked(nodes, "mpt_scene_editor")
        if pack.exists() and not pack.is_dir():
            raise InstallError(f"Preserve the existing path before installing the scene editor: {pack}")
        check_tree_streams(pack)
        state = checked(pack, STATE)
        if state.exists():
            raise InstallError(f"Preserve the interrupted scene editor transaction in {state} before retrying.")
        manifest = checked(pack, "mpt-editor.json")
        previous = contents(manifest)
        originals = {"mpt-editor.json": previous}
        if pack.exists():
            try:
                saved = json.loads(previous) if previous is not None else None
            except (ValueError, UnicodeError) as exc:
                raise InstallError(f"Unrecognized scene editor manifest: {manifest}. Preserve this folder first.") from exc
            if (not isinstance(saved, dict) or set(saved) != {"format", "files"} or saved["format"] != 1
                    or not isinstance(saved["files"], dict) or set(saved["files"]) != set(EDITOR_FILES)
                    or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                           for value in saved["files"].values())):
                raise InstallError(f"Unrecognized scene editor manifest: {manifest}. Preserve this folder first.")
            for name, expected in saved["files"].items():
                originals[name] = contents(checked(pack, name))
                if digest(originals[name]) != expected:
                    raise InstallError(f"Scene editor file changed: {pack / name}. Preserve your edits before updating.")
        outputs = {**sources, "mpt-editor.json": (json.dumps({"format": 1, "files": {
            name: digest(data) for name, data in sources.items()}}, indent=2) + "\n").encode()}
        originals = {name: originals.get(name) for name in outputs}
        changes = [name for name, data in outputs.items() if originals[name] != data]
        if not changes:
            return False
        files = {}
        for name in changes:
            path = checked(pack, name)
            check_replacement(nodes, path)
            files[name] = {"before": digest(originals[name]), "after": digest(outputs[name]),
                           "mode": stat.S_IMODE(path.stat().st_mode) if originals[name] is not None else 0o644}
        created = not pack.exists()
        state.mkdir(parents=True)
        applied = []
        try:
            for name in changes:
                change_source(pack, state, {"files": files}, name, files[name]["before"], outputs[name])
                applied.append(name)
        except BaseException:
            for name in reversed(applied):
                change_source(pack, state, {"files": files}, name, files[name]["after"], originals[name])
            raise
        finally:
            # Never recursively delete an uncertain exchange or any unrelated user data.
            with contextlib.suppress(OSError):
                state.rmdir()
            if created:
                with contextlib.suppress(OSError):
                    checked(pack, "js").rmdir()
                with contextlib.suppress(OSError):
                    pack.rmdir()
        return True


def scene_editor_status(comfyui):
    if comfyui is None:
        return ""
    try:
        changed = install_scene_editor(comfyui)
    except (OSError, ValueError, InstallError) as exc:
        return f"\nManual scene editing is unavailable: {exc}\nThe MoneyPrinterTurbo installation is valid."
    return ("\nManual scene editor installed. Restart ComfyUI to load it."
            if changed else "\nManual scene editor is installed. Restart ComfyUI if it has not loaded the editor yet.")


def managed_state(target):
    state = checked(target, STATE)
    if not state.exists():
        return state, None
    check_tree_streams(state)
    path = checked(state, "receipt.json")
    if contents(path) is None:
        raise InstallError(f"{STATE} exists without an installer receipt. Move it aside first.")
    receipt = json.loads(path.read_text("utf-8"))
    if receipt.get("format") != 1 or receipt.get("phase") not in ("installing", "installed", "removing", "configuring"):
        raise InstallError("Unrecognized installer receipt.")
    if receipt.get("private_ignore") and contents(checked(state, ".gitignore")) != b"*\n":
        raise InstallError(f"{STATE}/.gitignore changed. Restore it before continuing.")
    if receipt["phase"] == "installed" and not receipt.get("private_ignore"):
        if contents(checked(state, ".gitignore")) is not None:
            raise InstallError(f"Preserve the existing {STATE}/.gitignore before upgrading this install.")
        receipt.update(phase="configuring", next_settings=receipt["settings"], adding_ignore=True)
        save_receipt(state, receipt)
    if receipt["phase"] == "configuring":
        actual = digest(contents(checked(state, "settings.toml")))
        if actual not in (receipt["settings"], receipt["next_settings"]):
            raise InstallError("Settings changed during configuration. Preserve your edits before retrying.")
        for name in receipt.pop("adding", []):
            if actual == receipt["next_settings"]:
                generated_file(state, name)
                receipt.setdefault("generated", []).append(name)
            else:
                generated_file(state, name, allow_missing=True).unlink(missing_ok=True)
        if receipt.pop("adding_ignore", False):
            ignore = checked(state, ".gitignore")
            if contents(ignore) not in (None, b"*\n"):
                raise InstallError(f"{ignore} changed during setup. Preserve your edits before continuing.")
            if actual == receipt["next_settings"]:
                if not ignore.exists():
                    with ignore.open("xb") as stream:
                        stream.write(b"*\n")
                receipt["private_ignore"] = True
            else:
                ignore.unlink(missing_ok=True)
        receipt.update(phase="installed", settings=actual)
        del receipt["next_settings"]
        save_receipt(state, receipt)
    return state, receipt


def restore(target, state, receipt):
    # Precheck all files before rolling any back. A crashed transaction may be half applied.
    check_tree_streams(state)
    for name in receipt["directories"]:
        check_source_streams(source_file(target, name))
    finish_exchange(target, state, receipt)
    originals = {}
    fully_applied = True
    for name, record in receipt["files"].items():
        dest = source_file(target, name)
        current = digest(contents(dest))
        allowed = {record["after"]} if receipt["phase"] == "installed" else {record["after"], record["before"]}
        if current not in allowed:
            raise InstallError(f"{name} changed after installation. Preserve your edits before uninstalling.")
        if current != record["after"]:
            fully_applied = False
        if current != record["before"]:
            check_replacement(target, dest)
        backup = contents(checked(state, "backup/" + name))
        if digest(backup) != record["before"]:
            raise InstallError(f"Missing or changed backup: {name}")
        originals[name] = backup
    if digest(contents(checked(state, "settings.toml"))) != receipt["settings"]:
        raise InstallError(f"{STATE}/settings.toml changed. Preserve your edits before uninstalling.")
    # Backups and settings are the only installer-owned data. Keep extra files.
    known = {"settings.toml", "receipt.json"} | {"backup/" + n for n, v in originals.items() if v is not None}
    for name in receipt.get("generated", []):
        generated_file(state, name)
        known.add(name)
    if receipt.get("private_ignore"):
        if contents(checked(state, ".gitignore")) != b"*\n":
            raise InstallError(f"{STATE}/.gitignore changed. Preserve your edits before uninstalling.")
        known.add(".gitignore")
    # Older installers could leave the complete final receipt beside the live journal.
    legacy_temp = None
    if receipt["phase"] == "installing" and fully_applied:
        pending = [p for p in state.iterdir() if re.fullmatch(r"\.ltx-write-[a-z0-9_]{8}", p.name)]
        final = json.dumps({**receipt, "phase": "installed"}, indent=2).encode()
        if len(pending) == 1 and contents(checked(state, pending[0].name)) == final:
            check_source_streams(pending[0])
            legacy_temp = pending[0]
            known.add(legacy_temp.name)
    for entry in state.rglob("*"):
        rel = entry.relative_to(state).as_posix()
        checked(state, rel)
        if rel not in known and not (entry.is_dir() and any(k.startswith(rel + "/") for k in known)):
            raise InstallError(f"Move the extra file(s) in {STATE} aside, then retry.")
    if legacy_temp:
        legacy_temp.unlink()
    receipt["phase"] = "removing"
    save_receipt(state, receipt)
    for name, original in originals.items():
        dest = source_file(target, name)
        record = receipt["files"][name]
        if digest(contents(dest)) != record["before"]:
            change_source(target, state, receipt, name, record["after"], original)
    for name in reversed(receipt["directories"]):
        with contextlib.suppress(OSError):
            source_file(target, name).rmdir()
    # Rename the completed journal before cleanup so an interruption cannot half-delete a live receipt.
    with installer_scratch(target, "removed") as temporary:
        state.rename(Path(temporary) / STATE)


def install(target, state, settings, patch, generated=None):
    sources = packaged_sources()
    package = package_info(target)
    exclusions = ["--exclude=pyproject.toml", "--exclude=docker-compose.yml"] if package else []
    with installer_scratch(target, "install") as tmp:
        stage = Path(tmp) / "source"
        stage.mkdir()
        git(stage, "init", "--quiet", "--template=", staged=True)
        stats = git(stage, "apply", "--numstat", "-z", *exclusions, str(patch), staged=True).split(b"\0")
        virtual_names = [line.split(b"\t", 2)[2].decode("utf-8") for line in stats if line]
        if not virtual_names:
            raise InstallError("Empty patch.")
        virtual_names.extend(sources)
        record_name = package[0] + "/RECORD" if package else None
        if record_name:
            virtual_names.append(record_name)
        names = [package_path(name, bool(package)) for name in virtual_names]
        mapping = dict(zip(virtual_names, names))
        path_keys = {name.casefold() for name in names}
        if len(names) != len(path_keys) or any(
            parent.as_posix() in path_keys for name in path_keys for parent in PurePosixPath(name).parents
        ):
            raise InstallError("Conflicting paths in the installation package.")
        journal = Path(tmp) / STATE
        journal.mkdir(mode=0o700)
        write_file(journal / ".gitignore", b"*\n", 0o600)
        files, new_dirs, windows_eol = {}, set(), set()
        for virtual, name in mapping.items():
            dest = source_file(target, name)
            before = contents(dest)
            check_replacement(target, dest)
            if virtual in sources and before is not None:
                raise InstallError(f"Cannot install mpt/{name}: {name} already exists.")
            for parent in dest.relative_to(target).parents:
                if parent != Path(".") and not (target / parent).exists():
                    new_dirs.add(parent.as_posix())
            files[name] = {"before": digest(before), "mode": stat.S_IMODE(dest.stat().st_mode) if before is not None else 0o644}
            if before is not None:
                write_file(checked(journal, "backup/" + name), before)
                if b"\r\n" in before and b"\n" not in before.replace(b"\r\n", b""):
                    windows_eol.add(name)
                    before = before.replace(b"\r\n", b"\n")
                write_file(checked(stage, virtual), before)
        try:
            git(stage, "apply", "--whitespace=nowarn", *exclusions, str(patch), staged=True)
        except InstallError as exc:
            raise InstallError("This installation differs from the expected MoneyPrinterTurbo source. "
                               "Use a supported release, the pinned main checkout, or the exact trial tag; "
                               "your files have not been changed.\n" + str(exc)) from exc
        outputs = {}
        for virtual, name in mapping.items():
            data = sources[virtual] if virtual in sources else contents(checked(stage, virtual))
            if data is not None and name in windows_eol:
                data = data.replace(b"\n", b"\r\n")
            outputs[name] = data
            if digest(contents(source_file(target, name))) != files[name]["before"]:
                raise InstallError(f"{name} changed while preparing the installation.")
        if record_name:
            original = contents(checked(target, record_name))
            if original is None:
                raise InstallError("The installed mpt distribution is missing its RECORD file.")
            outputs[record_name] = package_record(original, {name: data for name, data in outputs.items()
                                                           if name != record_name})
        for name, data in outputs.items():
            files[name]["after"] = digest(data)
        receipt = {"format": 1, "phase": "installing", "files": files, "patch": digest(patch.read_bytes()),
                   "settings": digest(settings), "directories": sorted(new_dirs, key=lambda p: (p.count("/"), p))}
        receipt["private_ignore"] = True
        if generated:
            for name, data in generated.items():
                write_file(journal / name, data, 0o600)
            receipt["generated"] = list(generated)
        write_file(journal / "settings.toml", settings)
        save_receipt(journal, receipt, staged=True)
        journal.rename(state)
        try:
            for name, data in outputs.items():
                change_source(target, state, receipt, name, files[name]["before"], data)
            receipt["phase"] = "installed"
            save_receipt(state, receipt)
        except BaseException:
            receipt["phase"] = "installing"
            restore(target, state, receipt)
            raise


def run(action, target, python=None, template=None, trial=None, *, comfyui=None, model_version="2.5",
        comfy_python=None, model_roots=(), endpoint="http://127.0.0.1:8188", mpt_python=None, external_server=False,
        workflow_api=None, remote_comfyui=None, allow_insecure_http=False, token_env=None):
    connects_comfyui = comfyui is not None or remote_comfyui is not None
    if connects_comfyui != (workflow_api is not None):
        raise InstallError("Supply --comfyui or --remote-comfyui with --workflow-api FILE exported from ComfyUI.")
    target = resolve_target(target, mpt_python)
    if not target.is_dir():
        raise InstallError("The project path must be a directory.")
    with install_lock(target):
        scratch = [checked(target, path.name) for path in target.iterdir()
                   if re.fullmatch(r"\.ltx-(?:install|removed|check|write)-[a-z0-9_]{8}", path.name)]
        # Reject user streams before cleanup or recovery changes any existing journal.
        state = checked(target, STATE)
        for path in [state, *scratch]:
            check_tree_streams(path)
        previous = contents(checked(state, "receipt.json"))
        if previous is not None:
            receipt = json.loads(previous)
            if action == "uninstall" or receipt.get("phase") in ("installing", "removing"):
                for name in receipt.get("directories", []):
                    check_source_streams(source_file(target, name))
        for path in scratch:
            discard_scratch(target, path)
        state, receipt = managed_state(target)
        if receipt and receipt["phase"] != "installed":
            restore(target, state, receipt)
            receipt = None
        if action == "uninstall":
            if receipt:
                restore(target, state, receipt)
            return "LTX Operator integration is uninstalled."
        if action == "configure" and not receipt:
            raise InstallError("Install the integration before configuring it.")
        if action == "install":
            patch = project_patch(target, trial, installed=bool(receipt))
            if receipt:
                sources = {package_path(name, bool(package_info(target))): data
                           for name, data in packaged_sources().items()}
                if ((receipt.get("patch") and receipt["patch"] != digest(patch.read_bytes()))
                        or any(digest(data) != receipt["files"].get(name, {}).get("after")
                               for name, data in sources.items())):
                    raise InstallError("The integration package has changed. Uninstall the previous integration, then install again.")
        builtin, generated = None, {}
        if connects_comfyui:
            if template:
                raise InstallError("Choose one ComfyUI setup or --request-template.")
            builtin, generated = prepare_model(state, comfyui, model_version, comfy_python, model_roots,
                                               remote_comfyui if remote_comfyui is not None else endpoint,
                                               external_server, workflow_api=workflow_api,
                                               remote_server=remote_comfyui is not None,
                                               allow_insecure_http=allow_insecure_http, token_env=token_env)
        if receipt:
            for name, record in receipt["files"].items():
                if digest(contents(source_file(target, name))) != record["after"]:
                    raise InstallError(f"Installed file has changed: {name}")
            settings_file = checked(state, "settings.toml")
            previous = contents(settings_file)
            if digest(previous) != receipt["settings"]:
                raise InstallError("Installed settings were modified. Preserve your edits before reconfiguring.")
            for name in receipt.get("generated", []):
                generated_file(state, name)
            if action == "configure" or python or template or builtin:
                saved = tomllib.loads(previous.decode("utf-8"))["ltx"]
                new = operator_settings(python or saved["python_executable"], template, saved, builtin)
                if new != previous:
                    if action != "configure" and not builtin:
                        raise InstallError("Already installed. Use the configure command to update the LTX settings.")
                    adding = [name for name in generated if name not in receipt.get("generated", [])]
                    for name in adding:
                        if checked(state, name).exists():
                            raise InstallError(f"Model file already exists outside this installer's receipt: {name}")
                    if adding:
                        receipt["adding"] = adding
                    receipt.update(phase="configuring", next_settings=digest(new))
                    save_receipt(state, receipt)
                    try:
                        for name in adding:
                            write_atomic(state / name, generated[name], 0o600)
                        write_atomic(settings_file, new)
                    except BaseException:
                        managed_state(target)
                        raise
                    managed_state(target)
                return "LTX Operator settings are configured. Restart MoneyPrinterTurbo." + scene_editor_status(comfyui)
            return "LTX Operator integration is already installed." + scene_editor_status(comfyui)
        install(target, state, operator_settings(python, template, builtin=builtin), patch, generated)
        return ("Installed. Restart MoneyPrinterTurbo and select Local ComfyUI."
                + ("" if template or builtin else " To connect a model, rerun install with --comfyui and --workflow-api.")
                + scene_editor_status(comfyui))


def wizard_heading(text):
    color = sys.stdout.isatty() and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"
    if color and os.name == "nt":
        # Classic Windows consoles may print ANSI escapes literally.
        import ctypes
        import msvcrt
        mode = ctypes.c_ulong()
        try:
            handle = ctypes.c_void_p(msvcrt.get_osfhandle(sys.stdout.fileno()))
            color = ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)) and mode.value & 0x0004
        except (OSError, ValueError):
            color = False
    print(f"\n\033[1;36m{text}\033[0m" if color else f"\n{text}")


def wizard_choice(options):
    choices = {}
    for number, (value, label) in enumerate(options, start=1):
        print(f"  {number}. {label}" + ("  (default)" if number == 1 else ""))
        choices[str(number)] = choices[value] = value
    while True:
        answer = input("Choice [1]: ").strip().lower() or "1"
        if answer in choices:
            return choices[answer]
        print(f"Enter a number from 1 to {len(options)}.")


def run_wizard():
    wizard_heading("MoneyPrinterTurbo + ComfyUI")
    print("Local video setup\n" + "-" * 32)
    action = wizard_choice((("install", "Install or add a model"), ("uninstall", "Uninstall the add-on")))
    wizard_heading("Project folder" if action == "uninstall" else "Project folders")
    print("  Stop MoneyPrinterTurbo." if action == "uninstall" else
          "  Stop MoneyPrinterTurbo and leave ComfyUI running.")
    mpt = (os.environ.get("MPT_ADDON_BOOTSTRAP_TARGET") or
           input("MoneyPrinterTurbo source/portable folder or Python environment folder: ")).strip().strip('"')
    if not mpt:
        raise InstallError("No MoneyPrinterTurbo folder selected.")
    if action == "uninstall":
        run("uninstall", mpt)
        wizard_heading("Removal complete")
        print("The add-on is no longer installed in this MoneyPrinterTurbo folder.")
        return 0

    comfyui = input("ComfyUI folder: ").strip().strip('"')
    if not comfyui:
        raise InstallError("No ComfyUI folder selected.")
    wizard_heading("Video model")
    models = (("2.5", "LTX-2.5"), ("2.3", "LTX-2.3"), ("h3", "MiniMax H3"))
    version = wizard_choice(models)
    label = dict(models)[version]
    workflow_api = input(f"{label} JSON saved with ComfyUI Export (API): ").strip().strip('"')
    if not workflow_api:
        raise InstallError("No API workflow selected.")
    from configure_ltx import MissingComfyPython, MissingWeights
    python, roots = None, []
    while True:
        print(f"\nConnecting {label} to MoneyPrinterTurbo...", flush=True)
        try:
            result = run("install", mpt, comfyui=comfyui, model_version=version, comfy_python=python,
                         model_roots=roots, workflow_api=workflow_api)
            break
        except (MissingComfyPython, MissingWeights) as exc:
            print(exc, file=sys.stderr)
            needs_python = isinstance(exc, MissingComfyPython)
            question = "ComfyUI Python executable" if needs_python else "Additional model root (folder containing the model categories)"
            answer = input(f"{question}, or Enter to cancel: ").strip().strip('"')
            if not answer:
                print("Setup cancelled.")
                return 1
            path = Path(answer).expanduser()
            if needs_python:
                if not path.is_file():
                    raise InstallError(f"Python executable does not exist: {path}")
                python = path
            else:
                if not path.is_dir():
                    raise InstallError(f"Model root does not exist: {path}")
                if path.resolve() in exc.searched_roots:
                    raise InstallError(f"Already searched this model root: {path}")
                roots.append(path)
    wizard_heading("Setup complete")
    print(f"{result}\nChoose {label} as the local video model under Local ComfyUI.")
    return 0


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action")
    parser.set_defaults(comfyui=None, model_version=None, comfy_python=None, model_root=None, endpoint=None)
    for action in ("install", "uninstall", "configure"):
        cmd = commands.add_parser(action)
        cmd.add_argument("mpt", type=Path, nargs="?", help="MoneyPrinterTurbo source, portable or Python environment folder")
        cmd.add_argument("--mpt-python", type=Path, help="Interpreter from a virtual environment with mpt installed by pip")
        if action != "uninstall":
            source = cmd.add_mutually_exclusive_group()
            source.add_argument("--comfyui", type=Path, help="Connect a running local ComfyUI installation")
            source.add_argument("--remote-comfyui", metavar="URL", help="Connect a trusted ComfyUI server over HTTP(S)")
            source.add_argument("--request-template", action="append", metavar="FILE",
                                help="Register existing request JSON instead (repeatable)")
            cmd.add_argument("--workflow-api", type=Path, help="Local JSON saved with ComfyUI Export (API)")
            cmd.add_argument("--model-version", choices=("2.5", "2.3", "h3"), help="Workflow model (default: 2.5)")
            cmd.add_argument("--comfy-python", type=Path, help="ComfyUI interpreter outside its usual venv")
            cmd.add_argument("--model-root", action="append", type=Path, help="Additional ComfyUI model root (repeatable)")
            cmd.add_argument("--endpoint", help="Local ComfyUI URL (default: http://127.0.0.1:8188)")
            cmd.add_argument("--operator-python", help="Python for the standalone operator")
            cmd.add_argument("--external-server", action="store_true",
                             help="ComfyUI runs outside this environment; use mounted files and its loopback API")
            cmd.add_argument("--allow-insecure-comfyui-http", action="store_true",
                             help="Allow token-free HTTP for an explicit private IP or loopback")
            cmd.add_argument("--comfyui-token-env", metavar="NAME",
                             help="Read the HTTPS server's bearer token from this environment variable")
        if action == "install":
            cmd.add_argument("--trial-version", metavar="X.Y.Z", help="test a clean checkout of this exact upstream tag")
    argv = sys.argv[1:]
    if argv and argv[0] not in ("install", "uninstall", "configure") and not argv[0].startswith("-"):
        argv.insert(0, "install")
    args = parser.parse_args(argv)
    try:
        if not args.action:
            if not sys.stdin.isatty():
                parser.error("specify install or uninstall and a MoneyPrinterTurbo folder")
            try:
                return run_wizard()
            except (EOFError, KeyboardInterrupt):
                print("\nSetup interrupted.", file=sys.stderr)
                return 130
        remote = getattr(args, "remote_comfyui", None)
        allow_http = getattr(args, "allow_insecure_comfyui_http", False)
        token_env = getattr(args, "comfyui_token_env", None)
        if not remote and (allow_http or token_env):
            parser.error("Remote ComfyUI options require --remote-comfyui")
        if remote and (args.comfy_python or args.model_root or args.endpoint or args.external_server):
            parser.error("Remote ComfyUI cannot use local installation options")
        if not args.comfyui and not remote and any((args.model_version, args.comfy_python, args.model_root, args.endpoint,
                                    getattr(args, "external_server", False), getattr(args, "workflow_api", None))):
            parser.error("model options require --comfyui or --remote-comfyui")
        print(run(args.action, args.mpt, getattr(args, "operator_python", None),
                  getattr(args, "request_template", None), getattr(args, "trial_version", None),
                  comfyui=args.comfyui, model_version=args.model_version or "2.5", comfy_python=args.comfy_python,
                  model_roots=args.model_root or [], endpoint=args.endpoint or "http://127.0.0.1:8188",
                  mpt_python=args.mpt_python, external_server=getattr(args, "external_server", False),
                  workflow_api=getattr(args, "workflow_api", None), remote_comfyui=remote,
                  allow_insecure_http=allow_http, token_env=token_env))
    except (OSError, ValueError, KeyError, EOFError, InstallError) as exc:
        print(f"Installer: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
