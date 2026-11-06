"""Source and installed-package layouts share a reversible installation."""
import base64
import csv
import hashlib
import io
from pathlib import Path
import sys

import pytest

import install as installer

PROJECT_PATCH = installer.project_patch


@pytest.fixture
def wheel_layout(tmp_path, monkeypatch):
    target = tmp_path / "Python environment ü" / "site-packages"
    original = b"value = 1\r\n"
    installer.write_file(target / "app/cli.py", original)
    installer.write_file(target / "webui/Main.py", b"# webui\n")
    installer.write_file(target / "mpt-1.3.9.dist-info/METADATA", b"Name: mpt\nVersion: 1.3.9\n")
    installer.write_file(target / "mpt-1.3.9.dist-info/RECORD",
                         b"app/cli.py,sha256=old,11\n../Scripts/mpt.exe,,\nmpt-1.3.9.dist-info/RECORD,,\n")
    patch = tmp_path / "mpt.patch"
    patch.write_bytes(b"diff --git a/cli.py b/cli.py\n--- a/cli.py\n+++ b/cli.py\n"
                      b"@@ -1 +1 @@\n-value = 1\n+value = 2\n"
                      b"diff --git a/pyproject.toml b/pyproject.toml\n"
                      b"--- a/pyproject.toml\n+++ b/pyproject.toml\n@@ -1 +1 @@\n-name = 1\n+name = 2\n")
    monkeypatch.setattr(installer, "project_patch", lambda *a, **kw: patch)
    monkeypatch.setattr(installer, "packaged_sources", lambda: {"resource/ltx/guide.md": b"Model guide\n"})
    return target


def test_package_install_repeat_and_exact_restore(wheel_layout):
    target = wheel_layout
    before = {p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()}
    installer.run("install", target)
    assert (target / "app/cli.py").read_bytes() == b"value = 2\r\n"
    guide = target / "app/_assets/resource/ltx/guide.md"
    assert guide.read_bytes() == b"Model guide\n"
    rows = {row[0]: row[1:] for row in csv.reader(io.StringIO(
        (target / "mpt-1.3.9.dist-info/RECORD").read_text()))}
    for name in ("app/cli.py", "app/_assets/resource/ltx/guide.md"):
        data = (target / name).read_bytes()
        expected = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        assert rows[name] == ["sha256=" + expected, str(len(data))]
    assert rows["../Scripts/mpt.exe"] == ["", ""]
    installer.run("install", target)
    installer.run("uninstall", target)
    assert not guide.exists()
    after = {p.relative_to(target): p.read_bytes() for p in target.rglob("*")
             if p.is_file() and p.name != ".ltx-operator.lock"}
    assert after == before


def test_package_metadata_is_required(wheel_layout):
    metadata = wheel_layout / "mpt-1.3.9.dist-info/METADATA"
    metadata.write_text("Name: another-package\nVersion: 1.3.9\n")
    with pytest.raises(installer.InstallError, match="mpt"):
        installer.package_info(wheel_layout)


def test_package_cannot_use_source_tag_trial(wheel_layout):
    with pytest.raises(installer.InstallError, match="source checkout"):
        PROJECT_PATCH(wheel_layout, "1.3.9")


def test_explicit_source_does_not_select_another_python_package(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname="mpt"\nversion="1.3.9"\n')
    assert installer.resolve_target(tmp_path, Path(sys.executable)) == tmp_path.resolve()


def test_source_commit_selects_patch_independently_of_package_version(tmp_path, monkeypatch):
    addon = tmp_path / "addon"
    installer.write_file(addon / "supported-releases.json", b'{"1.3.9": "mpt.patch"}')
    for filename in ("mpt.patch", "mpt-main.patch"):
        installer.write_file(addon / "payload" / filename, b"fixture")
    monkeypatch.setattr(installer, "ADDON", addon)
    target = (tmp_path / "MoneyPrinterTurbo").resolve()
    installer.write_file(target / "pyproject.toml", b'[project]\nname="mpt"\nversion="1.3.9"\n')
    installer.write_file(target / "webui/Main.py", b"# original\n")
    assert PROJECT_PATCH(target, None).name == "mpt.patch"  # Portable source without Git metadata.
    git = installer.git
    git(target, "init", "--quiet", staged=True)
    git(target, "add", ".", staged=True)
    identity = ("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "-c", "commit.gpgsign=false")
    git(target, *identity, "commit", "--quiet", "-m", "release", staged=True)
    git(target, "tag", "v1.3.9", staged=True)
    release = git(target, "rev-parse", "HEAD").strip().decode("ascii")
    git(target, *identity, "commit", "--quiet", "--allow-empty", "-m", "main", staged=True)
    monkeypatch.setattr(installer, "MAIN_COMMIT", git(target, "rev-parse", "HEAD").strip().decode("ascii"))

    assert PROJECT_PATCH(target, None).name == "mpt-main.patch"
    with pytest.raises(installer.InstallError, match="exact tag"):
        PROJECT_PATCH(target, "1.3.9")
    (target / "webui/Main.py").write_text("# local change\n", encoding="utf-8")
    with pytest.raises(installer.InstallError, match="clean checkout"):
        PROJECT_PATCH(target, None)
    assert PROJECT_PATCH(target, None, installed=True).name == "mpt-main.patch"

    git(target, "checkout", "--quiet", "--detach", "--force", release, staged=True)
    assert PROJECT_PATCH(target, None).name == PROJECT_PATCH(target, "1.3.9").name == "mpt.patch"


def test_patch_and_copied_source_cannot_own_the_same_path(wheel_layout):
    patch = wheel_layout.parents[1] / "mpt.patch"
    with patch.open("ab") as stream:
        stream.write(b"diff --git a/resource/ltx/guide.md b/resource/ltx/guide.md\n"
                     b"new file mode 100644\n--- /dev/null\n+++ b/resource/ltx/guide.md\n"
                     b"@@ -0,0 +1 @@\n+Different guide\n")
    with pytest.raises(installer.InstallError, match="Conflicting paths"):
        installer.run("install", wheel_layout)
    assert not (wheel_layout / installer.STATE).exists()
