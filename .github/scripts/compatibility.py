"""Exercise an add-on install against a disposable upstream checkout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import stat
import subprocess
import sys
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
BASE_TAG = "v1.3.9"
BASE_SHA = "6992380c5e82f07f3e1fc46b90927b6f1c655407"
API = "https://api.github.com/repos/harry0703/MoneyPrinterTurbo"
TAG = re.compile(r"v[0-9]+\.[0-9]+\.[0-9]+")
SHA = re.compile(r"[0-9a-f]{40}")
STAGES = ("install", "repeat", "media", "features", "render", "uninstall", "restore")


def supported(tag):
    return tag[1:] in json.loads((ROOT / "supported-releases.json").read_text(encoding="utf-8"))


def api(path):
    request = urllib.request.Request(API + path, headers={"User-Agent": "mpt-compatibility"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def output(name, value):
    print(f"{name}={value}", flush=True)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
            print(f"{name}={value}", file=stream)


def resolve():
    requested = os.environ.get("REQUESTED_TAG") or "baseline"
    if requested == "baseline":
        tag, sha = BASE_TAG, BASE_SHA
    elif requested == "main":
        tag = requested
        sha = runpy.run_path(str(ROOT / "install.py"))["MAIN_COMMIT"]
        if not SHA.fullmatch(sha):
            raise ValueError("Invalid pinned upstream main commit")
    else:
        if requested != "latest" and not TAG.fullmatch(requested):
            raise ValueError("Expected main, latest, or a release tag such as v1.3.9")
        release = api("/releases/latest" if requested == "latest" else "/releases/tags/" + requested)
        tag = release["tag_name"]
        if (not TAG.fullmatch(tag) or release["draft"] or release["prerelease"]
                or requested not in {"latest", tag}):
            raise ValueError("Expected a published stable upstream release")
        sha = api("/commits/" + tag)["sha"]
        if not SHA.fullmatch(sha) or (tag == BASE_TAG and sha != BASE_SHA):
            raise ValueError("Unexpected upstream commit")
    output("tag", tag)
    output("sha", sha)
    output("trial", str(tag != "main" and not supported(tag)).lower())


def git(directory, *args):
    return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()


def checkout_files(target):
    """Hash actual checkout bytes, including ignored files and symbolic link targets."""
    def raise_walk_error(error):
        raise error

    files = {}
    for root, dirs, names in os.walk(target, onerror=raise_walk_error):
        folder = Path(root)
        if folder == target:
            dirs[:] = [name for name in dirs if name != ".git"]
            names = [name for name in names if name != ".git"]
        for name in dirs.copy():
            path = folder / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                dirs.remove(name)
                names.append(name)
            else:
                files[path.relative_to(target).as_posix()] = ["dir"]
        for name in names:
            path = folder / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                identity = ["link", os.readlink(path)]
            elif stat.S_ISREG(info.st_mode):
                with path.open("rb") as stream:
                    identity = ["file", hashlib.file_digest(stream, "sha256").hexdigest()]
            else:
                raise ValueError(f"Unexpected file type in checkout: {path}")
            files[path.relative_to(target).as_posix()] = identity
    return files


def snapshot_path(target):
    return target.parent / (target.name + ".initial-files.json")


def prepare(snapshot, tag, sha, target):
    if (tag != "main" and not TAG.fullmatch(tag)) or not SHA.fullmatch(sha):
        raise ValueError("Invalid version or commit")
    ref = "refs/heads/main" if tag == "main" else f"refs/tags/{tag}"
    if git(snapshot, "rev-parse", f"{ref}^{{commit}}") != sha:
        raise ValueError("Upstream ref differs from resolved commit")
    subprocess.run(["git", "clone", "-q", "--no-local", "--depth=1", "--branch", tag,
                    str(snapshot), str(target)], check=True)
    if git(target, "rev-parse", "HEAD") != sha:
        raise ValueError("Candidate checkout does not match requested commit")
    config = target / "config.toml"
    if (not config.exists() and not config.is_symlink()
            and subprocess.run(["git", "-C", str(target), "check-ignore", "-q", "--", "config.toml"]).returncode == 0):
        config.write_bytes(b"# Local MPT settings\n")
    snapshot_path(target).write_text(json.dumps(checkout_files(target)), encoding="utf-8")


def test_environment():
    return {k: v for k, v in os.environ.items() if not k.upper().startswith(("GITHUB_", "ACTIONS_", "RUNNER_"))}


def stage(name, target, tag, interpreter, normal):
    scratch = target.parent
    env = test_environment()
    env.update(MPT_HOME=str(scratch / "mpt-home"), PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(target))
    install = [interpreter, str(ROOT / "install.py"), "install", str(target)]
    if tag != "main" and not normal and not supported(tag):
        install.extend(["--trial-version", tag[1:]])
    commands = {
        "install": install, "repeat": install,
        "features": [interpreter, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                     str(ROOT / "tests/integration"), str(target / "test/services/test_webui_i18n.py"),
                     str(target / "test/services/test_webui_grouped_select.py")],
        "render": [interpreter, str(ROOT / "tests/smoke_mpt.py"), "--mpt", str(target),
                   "--work-dir", str(scratch / "render")],
        "uninstall": [interpreter, str(ROOT / "install.py"), "uninstall", str(target)],
    }
    print(f"Checking {name} on {tag}", flush=True)
    if name in ("media", "restore"):
        paths = ["--", "resource/fonts", "resource/songs"] if name == "media" else []
        status = git(target, "status", "--porcelain", "--untracked-files=all", *paths)
        if status:
            print(status, file=sys.stderr)
        passed = not status
        if name == "restore":
            initial = snapshot_path(target)
            if not initial.is_file():
                print("Missing initial checkout snapshot.", file=sys.stderr)
                passed = False
            else:
                before = json.loads(initial.read_text(encoding="utf-8"))
                after = checkout_files(target)
                changed = sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))
                if changed:
                    print("Checkout files differ:", *changed, sep="\n", file=sys.stderr)
                    passed = False
    elif name == "features":
        patch = str(ROOT / ".github/tests/grouped-select.patch")
        apply = ["git", "-C", str(target), "apply"]
        passed = subprocess.run([*apply, patch]).returncode == 0
        if passed:
            try:
                passed = subprocess.run(commands[name], cwd=ROOT, env=env).returncode == 0
            finally:
                if subprocess.run([*apply, "--reverse", patch]).returncode:
                    passed = False
    else:
        passed = subprocess.run(commands[name], cwd=ROOT, env=env).returncode == 0
    # A short, stable exit code per scenario. Git, setup and infrastructure errors return 1.
    return 0 if passed else 20 + STAGES.index(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("resolve")
    for mode in ("prepare", "stage", "check"):
        p = sub.add_parser(mode)
        p.add_argument("--tag", required=True)
        if mode != "stage":
            p.add_argument("--upstream", required=True, type=Path)
            p.add_argument("--sha", required=True)
        if mode != "prepare":
            p.add_argument("--normal-install", action="store_true")
            p.add_argument("--python", default=sys.executable)
        if mode == "check":
            p.add_argument("--work-dir", type=Path)
            p.add_argument("--hosted-result", action="store_true", help="Return a named functional failure to the hosted workflow")
        else:
            p.add_argument("--target", required=True, type=Path)
        if mode == "stage":
            p.add_argument("name", choices=STAGES)
    args = parser.parse_args()
    if args.command == "resolve":
        resolve()
        return 0
    if args.tag != "main" and not TAG.fullmatch(args.tag):
        raise ValueError("Expected main or an upstream release tag")
    if args.command == "prepare":
        prepare(args.upstream.resolve(), args.tag, args.sha, args.target.resolve())
        return 0
    args.python = str(Path(args.python).absolute())
    if args.command == "stage":
        return stage(args.name, args.target.resolve(), args.tag, args.python, args.normal_install)
    with tempfile.TemporaryDirectory(prefix="ltx-ci-", dir=args.work_dir) as tmp:
        target = Path(tmp) / "MoneyPrinterTurbo"
        prepare(args.upstream.resolve(), args.tag, args.sha, target)
        for name in STAGES:
            code = stage(name, target, args.tag, args.python, args.normal_install)
            if code:
                if args.hosted_result and code == 20 + STAGES.index(name):
                    output("failed_stage", name)
                    return 0
                return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
