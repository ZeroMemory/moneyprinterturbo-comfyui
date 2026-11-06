"""Atomic JSON artifacts and file hashing."""

import hashlib
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys

from .canonical import canonical_json


def ignore_private_directory(directory: Path) -> None:
    """Exclude a newly created operator directory before writing private data."""
    with (directory / ".gitignore").open("xb") as stream:
        stream.write(b"*\n")


def ensure_output_directory(directory: Path) -> None:
    """Protect directories the operator creates, leaving existing user folders alone."""
    try:
        directory.mkdir(parents=True)
    except FileExistsError:
        if not directory.is_dir():
            raise
    else:
        ignore_private_directory(directory)


def protect_private_file(path: Path) -> None:
    """Keep a private artifact out of ordinary Git adds without hiding its neighbors."""
    reserved = {".git", ".gitignore", ".gitattributes", ".gitmodules", ".mailmap"}
    if any(part.lower() in reserved for part in path.parts):
        raise ValueError("choose a private output outside Git control files")
    ensure_output_directory(path.parent)
    root = next((parent for parent in path.parents if (parent / ".git").exists()), None)
    if root is None:
        return

    git_env = {key: value for key, value in os.environ.items()
               if key not in {"GIT_DIR", "GIT_COMMON_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"}}

    def git(*args, **kwargs):
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, env=git_env, **kwargs)

    relative = path.relative_to(root).as_posix()
    name = "./" + relative
    match = git("check-ignore", "-v", "-z", "--stdin", input=os.fsencode(name) + b"\0")
    if match.returncode == 0:
        if match.stdout.split(b"\0")[2].startswith(b"!"):
            raise ValueError(f"Git explicitly includes {path}; choose a private location")
        return
    if match.returncode != 1:
        match.check_returncode()
    if git("--literal-pathspecs", "ls-files", "-z", "--", name, check=True).stdout:
        raise ValueError(f"Git already tracks {path}; choose a private location")

    exclude = root / os.fsdecode(git("rev-parse", "--git-path", "info/exclude", check=True).stdout.rstrip(b"\r\n"))
    if "\n" in relative or "\r" in relative:
        raise ValueError("Git cannot exclude a filename containing a newline")
    # Leading slash anchors the rule. Escape wildcard characters and trailing spaces.
    rule = re.sub(r"([\\*?\[ ])", r"\\\1", relative)
    if exclude.parent.is_symlink() or exclude.is_symlink() or (
        exclude.exists() and (not exclude.is_file() or exclude.stat().st_nlink != 1)
    ):
        raise ValueError("Git's local exclude file must be a regular, unlinked file")
    exclude.parent.mkdir(exist_ok=True)
    with exclude.open("ab") as stream:
        stream.write(os.fsencode("\n/" + rule + "\n"))
    if git("check-ignore", "-q", "--", name).returncode != 0:
        raise ValueError(f"Git could not exclude {path}; choose a private location")


def _ensure_runtime_output(destination: Path) -> None:
    files = [__file__, sys.executable, sys.argv[0]] + [vars(module).get('__file__') for module in tuple(sys.modules.values()) if module is not None]
    try:
        output_stat = destination.stat()
    except FileNotFoundError:
        output_stat = None
    for value in files:
        if not isinstance(value, (str, Path)) or not value:
            continue
        source = Path(value)
        if destination == source.absolute():
            raise ValueError('JSON output would overwrite runtime source')
        if output_stat is not None:
            try:
                source_stat = source.stat()
            except OSError:
                continue
            if (output_stat.st_dev, output_stat.st_ino) == (source_stat.st_dev, source_stat.st_ino):
                raise ValueError('JSON output would overwrite runtime source')


def create_json_exclusive(path: str | Path, value: object, *, private: bool = False) -> Path:
    """Create one canonical JSON artifact without overwriting an existing path."""
    destination = Path(path).expanduser()
    if private and os.path.lexists(destination):
        raise FileExistsError(f"output already exists: {destination}")
    destination = destination.resolve()
    if private:
        protect_private_file(destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical_json(value) + "\n")
    return destination


def replace_json_atomic(path: str | Path, value: object) -> Path:
    """Atomically replace a JSON artifact using a temporary sibling file."""
    destination = Path(path).expanduser().resolve()
    _ensure_runtime_output(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (
        f".{destination.name}.{secrets.token_hex(8)}.tmp"
    )
    try:
        create_json_exclusive(temporary, value)
        _ensure_runtime_output(destination)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def sha256_file(path: str | Path) -> str:
    """Hash one existing file without modifying it."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"file is unavailable: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
