"""Check the installed PyPI layout and restore its original files afterward."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ADDON = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ADDON))
import install


def check(python, work):
    python = Path(python).absolute()
    work.mkdir(parents=True, exist_ok=True)
    target = install.resolve_target(None, python)
    metadata, _ = install.package_info(target)

    def snapshot():
        return {path.relative_to(target).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for name in ("app", "webui", metadata) for path in (target / name).rglob("*")
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}

    before = snapshot()
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "MPT_HOME": str(work / "home")}
    command = [sys.executable, "-B", str(ADDON / "install.py")]
    try:
        for _ in range(2):
            subprocess.run([*command, "install", "--mpt-python", str(python)], env=env, check=True)
        code = """
from app import paths
from app.config import config
from app.services import llm, ltx
from app.models.schema import VideoParams
assert config.ltx['enabled'] is True
assert not paths.is_source_checkout()
for version in ('2.3', '2.5', 'h3'):
    assert llm._ltx_scene_director_brief(version)
    assert VideoParams(video_subject='Paper boats', video_source='ltx', ltx_model_version=version).ltx_model_version == version
from streamlit.testing.v1 import AppTest
from app.services import version_checker
version_checker.get_available_update = lambda: None
config.ui['language'] = 'en'
app = AppTest.from_file(str(paths.SOURCE_ROOT / 'webui/Main.py'), default_timeout=30).run()
assert not app.exception, app.exception
assert any(item.label == 'Stop all' for item in app.button)
"""
        subprocess.run([str(python), "-I", "-B", "-c", code], cwd=work, env=env, check=True)
        subprocess.run([str(python), "-I", "-B", "-m", "app.launcher", "--help"],
                       cwd=work, env=env, check=True, stdout=subprocess.DEVNULL)
    finally:
        if (target / install.STATE).exists():
            subprocess.run([*command, "uninstall", "--mpt-python", str(python)], env=env, check=True)
    assert snapshot() == before, "Package files were not restored exactly"
    assert not (target / install.STATE).exists()
    print(json.dumps({"package_layout": "verified", "install_repeat_webui_cli_uninstall": True,
                      "restored_files": len(before)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    check(args.python, args.work_dir.resolve())
