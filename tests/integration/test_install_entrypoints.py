"""Native launcher selection and argument forwarding, without installing models."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest


ADDON = Path(__file__).resolve().parents[2]
WINDOWS = os.name == "nt"


@pytest.fixture
def launcher(tmp_path):
    folder = tmp_path / "add-on space пример"
    folder.mkdir()
    for name in ("install.cmd", "install.sh", "install.command"):
        (folder / name).write_bytes((ADDON / name).read_bytes())
    (folder / "install.py").write_text(
        "import json,os,sys\n"
        "print(json.dumps({'argv':sys.argv[1:],'prefix':sys.prefix,'executable':sys.executable,"
        "'bootstrap':os.getenv('MPT_ADDON_BOOTSTRAP_TARGET')}))\n"
        "raise SystemExit(7)\n", encoding="utf-8")
    return folder


def runtime(folder):
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(folder)], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return folder / ("Scripts/python.exe" if WINDOWS else "bin/python")


def no_global_python(tmp_path):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    env.pop("MPT_ADDON_BOOTSTRAP_TARGET", None)
    env.pop("PYTHONPATH", None)
    if WINDOWS:
        env["PATH"] = str(Path(os.environ["SystemRoot"]) / "System32")
    else:
        commands = tmp_path / "commands"
        commands.mkdir(exist_ok=True)
        for name in ("dirname", "sh"):
            (commands / name).symlink_to(shutil.which(name))
        env["PATH"] = str(commands)
    return env


def invoke(launcher, args, env, *, script=None, input=None):
    if WINDOWS:
        script = launcher / "install.cmd"
        command = f'"{os.environ["SystemRoot"]}\\System32\\cmd.exe" /d /s /c ""{script}" '
        command += " ".join(f'"{arg}"' for arg in args) + '"'
    else:
        command = ["/bin/sh", str(launcher / (script or "install.sh")), *map(str, args)]
    return subprocess.run(command, cwd=launcher, env=env, input=input, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=20)


def received(result):
    assert result.returncode == 7, (result.stdout, result.stderr)
    return json.loads(re.search(r'\{"argv".*\}', result.stdout).group())


def test_operator_entrypoint_runs_with_an_isolated_embedded_style_path():
    script = ADDON / "runtime/scripts/ltx_operator.py"
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c",
         "import runpy,sys;sys.argv=[sys.argv[1],'--help'];runpy.run_path(sys.argv[0],run_name='__main__')",
         str(script)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize("action", [None, "install", "configure", "uninstall"])
@pytest.mark.parametrize("layout", ["venv", "bundled", "environment"])
def test_project_python_is_used_without_global_python(tmp_path, launcher, action, layout):
    target = tmp_path / "MPT space пример"
    env_root = target if layout == "environment" else target / (".venv" if layout == "venv" else "lib/python")
    python = runtime(env_root)
    if WINDOWS and layout == "bundled":
        shutil.copy2(python, env_root / "python.exe")
        python = env_root / "python.exe"
    args = ([] if action is None else [action]) + [str(target)]
    result = invoke(launcher, args, no_global_python(tmp_path))
    record = received(result)
    assert record["argv"] == args
    assert Path(record["executable"]) == python


@pytest.mark.parametrize("form", ["separate", "equals"])
def test_explicit_pip_python_needs_no_source_target(tmp_path, launcher, form):
    env_root = tmp_path / "pip environment пример"
    python = runtime(env_root)
    option = ["--mpt-python", str(python)] if form == "separate" else [f"--mpt-python={python}"]
    args = ["install", *option]
    record = received(invoke(launcher, args, no_global_python(tmp_path)))
    assert record["argv"] == args and Path(record["prefix"]) == env_root


def test_option_values_are_not_mistaken_for_the_mpt_target(tmp_path, launcher):
    target = tmp_path / "MPT=source !&() пример"
    env_root = target / ".venv"
    runtime(env_root)
    args = ["install", "--endpoint=http://127.0.0.1:8188", "--external-server", "--model-version", "2.5",
            "--allow-insecure-comfyui-http", str(target)]
    record = received(invoke(launcher, args, no_global_python(tmp_path)))
    assert record["argv"] == args and Path(record["prefix"]) == env_root


def test_missing_python_reports_requirement_without_running_installer(tmp_path, launcher):
    result = invoke(launcher, ["install", str(tmp_path / "no-python")], no_global_python(tmp_path))
    assert result.returncode == 1 and "Python 3.11 or later was not found" in result.stderr
    assert '"argv"' not in result.stdout


def test_old_python_version_is_rejected_before_running_installer(tmp_path, launcher):
    env_root = tmp_path / "old-runtime-control"
    python = runtime(env_root)
    site = env_root / ("Lib/site-packages" if WINDOWS else f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
    (site / "sitecustomize.py").write_text("import sys\nsys.version_info=(3,10,0,'final',0)\n")
    env = no_global_python(tmp_path)
    env["PYTHONPATH"] = str(site)
    result = invoke(launcher, ["install", "--mpt-python", str(python)], env)
    assert result.returncode == 1 and "must run Python 3.11 or later" in result.stderr
    assert '"argv"' not in result.stdout


def test_no_args_bootstraps_from_prompted_project_python(tmp_path, launcher):
    target = tmp_path / "portable MPT пример"
    env_root = target / ("lib/python" if WINDOWS else ".venv")
    python = runtime(env_root)
    if WINDOWS:
        shutil.copy2(python, env_root / "python.exe")
        python = env_root / "python.exe"
        result = invoke(launcher, [], no_global_python(tmp_path), input=str(target) + "\n\n")
        record = received(result)
    else:
        import pty
        import select
        import time
        master, slave = pty.openpty()
        proc = subprocess.Popen(["/bin/sh", str(launcher / "install.command")], cwd=launcher,
                                env=no_global_python(tmp_path), stdin=slave, stdout=slave, stderr=slave)
        os.close(slave)
        os.write(master, (str(target) + "\n\n").encode())
        output = b""
        deadline = time.monotonic() + 20
        try:
            while time.monotonic() < deadline:
                if select.select([master], [], [], 0.1)[0]:
                    try:
                        chunk = os.read(master, 65536)
                    except OSError:
                        break
                    if not chunk:
                        break
                    output += chunk
                if proc.poll() is not None:
                    break
            assert proc.wait(timeout=2) == 7, output.decode(errors="replace")
        finally:
            if proc.poll() is None:
                proc.kill()
            os.close(master)
        record = json.loads(re.search(rb'\{"argv".*\}', output).group())
    assert record["argv"] == [] and record["bootstrap"] == str(target)
    assert Path(record["executable"]) == python
