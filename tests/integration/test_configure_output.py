"""Configuration output and installer state preserve user data."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

ADDON = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ADDON), str(ADDON / "tests")]
from configure_ltx import configure, MODEL_INPUTS
from ltx_operator_lib import api
from ltx_operator_lib.adapters import adapter_for
from ltx_operator_lib.editor import _rebuild_request
from ltx_operator_lib.planning import preflight
from ltx_operator_lib.preparation import prepare_request
from ltx_operator_lib.runs import _collect_outputs
from ltx_operator_lib.workflow import load_workflow
from smoke_mpt import fixture_object_info
from workflow_fixtures import sample_graph, write_workflow
import install as installer


def comfy_with_weights(tmp_path, graph):
    app = tmp_path / "ComfyUI"
    (app / "output").mkdir(parents=True)
    (app / "main.py").touch()
    weights = {}
    for node in graph.values():
        for field, category in MODEL_INPUTS.get(node["class_type"], {}).items():
            filename = node["inputs"][field]
            weight = app / "models" / category / filename
            weight.parent.mkdir(parents=True, exist_ok=True)
            weight.write_bytes(b"unloaded test dependency")
            weights[filename] = weight
    return app, weights


def test_ltx23_configuration_preserves_imported_two_stage_workflow(tmp_path):
    original = sample_graph("2.3")
    source = write_workflow(tmp_path / "external.api.json", original)
    original_bytes = source.read_bytes()
    app, weights = comfy_with_weights(tmp_path, original)
    request = json.loads(configure(app, tmp_path / "settings", sys.executable,
                                   model_version="2.3", workflow_api=source).read_bytes())
    graph, requirements = request["api_prompt"], request["requirements"]
    assert request["variant"]["model_version"] == "2.3"
    assert request["variant"]["supports_negative_prompt"] is True
    lora = original["lora"]["inputs"]["lora_name"]
    assert requirements["required_loras"] == [lora]
    assert requirements["resolved_loras"] == {lora: str(weights.pop(lora))}
    assert set(requirements["required_models"]) == set(weights)
    assert requirements["resolved_models"] == {name: str(path) for name, path in weights.items()}
    for key in ("model", "lora", "guidance", "sampler_choice", "schedule", "upscale", "first_sample", "second_sample"):
        assert graph[key] == original[key]
    mapping = adapter_for(request["variant"]["adapter_family"]).discover(load_workflow(graph))
    assert mapping.negative_prompt.value == request["negative_prompt"] == graph["negative"]["inputs"]["text"]
    assert source.read_bytes() == original_bytes


def test_h3_import_resolves_official_style_linked_scene_controls(tmp_path):
    graph = sample_graph("h3")
    graph["display_size"] = {"class_type": "ResolutionSelector", "inputs": {
        "aspect_ratio": "1:1 (Square)", "megapixels": 0.25, "multiple": 32}}
    graph["frame_control"] = {"class_type": "ComfyMathExpression", "inputs": {
        "expression": "max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17", "values.a": 2}}
    graph["latent"]["inputs"].update(width=["display_size", 0], height=["display_size", 1], length=["frame_control", 1])
    source = write_workflow(tmp_path / "external.api.json", graph)
    app, _ = comfy_with_weights(tmp_path, graph)
    request = json.loads(configure(app, tmp_path / "settings", sys.executable,
                                   model_version="h3", workflow_api=source).read_bytes())
    assert {key: request["parameters"][key] for key in ("width", "height", "frame_count")} == {
        "width": 512, "height": 512, "frame_count": 56}
    assert {"display_size", "frame_control"}.isdisjoint(request["api_prompt"])
    assert json.loads(source.read_bytes()) == graph
    source.rename(tmp_path / "moved-export.json")
    snapshot = Path(request["variant"]["source_path"])
    assert snapshot.parent == tmp_path / "settings" and json.loads(snapshot.read_bytes()) == request["api_prompt"]
    edited = prepare_request(request, {"prompt": "A paper boat", "width": 640, "frame_count": 73})
    assert edited["api_prompt"]["latent"]["inputs"] == {
        **request["api_prompt"]["latent"]["inputs"], "prompt": "A paper boat", "width": 640, "length": 73}


def test_configure_rejects_ui_workflow_without_saving_a_profile(tmp_path):
    app = tmp_path / "ComfyUI"
    (app / "output").mkdir(parents=True)
    (app / "main.py").touch()
    source = write_workflow(tmp_path / "canvas.json", {"nodes": [], "links": []})
    output = tmp_path / "settings"
    with pytest.raises(ValueError, match="API"):
        configure(app, output, sys.executable, workflow_api=source)
    assert not output.exists()


def test_api_export_can_omit_optional_metadata(tmp_path):
    graph = sample_graph()
    graph["save"]["_meta"] = None
    source = write_workflow(tmp_path / "workflow.api.json", graph)
    app, _ = comfy_with_weights(tmp_path, graph)
    request = json.loads(configure(app, tmp_path / "settings", sys.executable, workflow_api=source).read_bytes())
    assert request["api_prompt"]["save"] == graph["save"]


@pytest.mark.parametrize("key,field,alias", [("model", "unet_name", "unet"), ("clip", "clip_name", "clip")])
def test_legacy_comfy_model_directory_cannot_shadow_an_approved_weight(tmp_path, key, field, alias):
    graph = sample_graph()
    source = write_workflow(tmp_path / "workflow.api.json", graph)
    app, weights = comfy_with_weights(tmp_path, graph)
    object_info = fixture_object_info(app, graph)
    filename = graph[key]["inputs"][field]
    canonical = weights[filename]
    legacy = app / "models" / alias / filename
    legacy.parent.mkdir()
    canonical.rename(legacy)

    request = json.loads(configure(app, tmp_path / "settings", sys.executable, workflow_api=source).read_bytes())
    prepared = prepare_request(request, {"prompt": "A paper boat"})
    edited = _rebuild_request(prepared, prepared["api_prompt"], object_info)
    assert edited["requirements"]["resolved_models"][filename] == str(legacy)

    canonical.write_bytes(b"Different model with the same selector")
    with pytest.raises(ValueError, match="Multiple files match"):
        configure(app, tmp_path / "ambiguous", sys.executable, workflow_api=source)
    with pytest.raises(ValueError, match="not approval-covered"):
        prepare_request(request, {"prompt": "A paper boat"})
    with pytest.raises(ValueError, match="Edited weight is unavailable, ambiguous"):
        _rebuild_request(prepared, prepared["api_prompt"], object_info)
    assert not (tmp_path / "ambiguous").exists()


def test_api_export_follows_an_exact_nested_model_link(tmp_path):
    graph = sample_graph()
    app, weights = comfy_with_weights(tmp_path, graph)
    filename = graph["model"]["inputs"]["unet_name"]
    shared = tmp_path / "shared"
    shared.mkdir()
    target = weights[filename].rename(shared / filename)
    link = app / "models/diffusion_models/linked"
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(str(shared), str(link))
    else:
        link.symlink_to(shared, target_is_directory=True)
    selector = str(Path("linked") / filename)
    graph["model"]["inputs"]["unet_name"] = selector
    source = write_workflow(tmp_path / "workflow.api.json", graph)
    request = json.loads(configure(app, tmp_path / "settings", sys.executable, workflow_api=source).read_bytes())
    prepared = prepare_request(request, {"prompt": "A paper boat"})
    assert prepared["requirements"]["resolved_models"][selector] == str(target)
    edited = _rebuild_request(prepared, prepared["api_prompt"], {})
    assert edited["requirements"]["resolved_models"][selector] == str(target)


@pytest.mark.parametrize("container,codec,suffix", [("mkv", "auto", "mkv"), ("webm", "av1", "webm"),
                                                     ("auto", "av1", "webm")])
def test_imported_output_matches_save_format_without_a_prefix_override(tmp_path, container, codec, suffix):
    graph = sample_graph()
    graph["save"]["inputs"].update(format=container, codec=codec)
    source = write_workflow(tmp_path / "workflow.api.json", graph)
    app, _ = comfy_with_weights(tmp_path, graph)
    request = json.loads(configure(app, tmp_path / "settings", sys.executable, workflow_api=source).read_bytes())
    prepared = prepare_request(request, {"prompt": "A paper boat"})
    prefix = graph["save"]["inputs"]["filename_prefix"]
    assert prepared["outputs"] == [f"{prefix}*.{suffix}"]
    clip = app / "output" / f"{prefix}_00001_.{suffix}"
    clip.touch()
    report = {"outputs": {"save": {"video": [{"filename": clip.name, "type": "output"}]}}}
    records = _collect_outputs(report, app / "output", graph=prepared["api_prompt"], prefixes={"save": prefix},
                               expected=prepared["outputs"], primary_node="save")
    assert [(Path(item["path"]), item["primary"]) for item in records] == [(clip, True)]


def test_import_preserves_one_ltx_rate_control_for_video_and_audio(tmp_path):
    graph = sample_graph()
    graph["rate"] = {"class_type": "ComfyMathExpression", "inputs": {"expression": "a", "values.a": 24}}
    graph["video"]["inputs"]["fps"] = ["rate", 0]
    graph["conditioning"]["inputs"]["frame_rate"] = ["rate", 0]
    graph["audio_latent"]["inputs"]["frame_rate"] = ["rate", 1]
    source = write_workflow(tmp_path / "external.api.json", graph)
    app, _ = comfy_with_weights(tmp_path, graph)
    request = json.loads(configure(app, tmp_path / "settings", sys.executable, workflow_api=source).read_bytes())
    imported = request["api_prompt"]
    rate = imported["video"]["inputs"]["fps"]
    assert imported[rate[0]]["inputs"]["value"] == request["parameters"]["fps"] == 24
    assert imported["rate"]["inputs"]["values.a"] == rate
    assert imported["conditioning"]["inputs"]["frame_rate"] == ["rate", 0]
    assert imported["audio_latent"]["inputs"]["frame_rate"] == ["rate", 1]


def test_configure_rejects_known_upscaler_from_another_release(tmp_path):
    graph = sample_graph("2.5")
    source = write_workflow(tmp_path / "external.api.json", graph)
    app, _ = comfy_with_weights(tmp_path, graph)
    output = tmp_path / "settings"
    with pytest.raises(ValueError, match="Select model version 2.5"):
        configure(app, output, sys.executable, model_version="2.3", workflow_api=source)
    assert not output.exists()


@pytest.mark.parametrize("encoding", ["cp1251", "ascii", "utf-8"])
def test_unicode_configure_result_and_repeat_are_encoding_safe(tmp_path, encoding):
    graph = sample_graph()
    app, _ = comfy_with_weights(tmp_path, graph)
    output = tmp_path / "settings ü"
    env = {**os.environ, "PYTHONUTF8": "0", "PYTHONIOENCODING": encoding + ":strict"}
    source = write_workflow(tmp_path / "official-export.api.json", graph)
    command = [sys.executable, "-B", str(ADDON / "configure_ltx.py"), "--comfyui", str(app), "--workflow-api", str(source),
               "--output", str(output), "--python", sys.executable]
    result = subprocess.run(command, env=env, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr.decode(encoding, errors="replace")
    request = output / "request.json"
    assert result.stdout.decode(encoding).strip() == str(request).encode(encoding, "backslashreplace").decode(encoding)
    saved = {file.name: file.read_bytes() for file in output.iterdir()}
    assert json.loads(saved["request.json"])["profile"]["source_path"] == str(output / "profile.json")
    repeated = subprocess.run(command, env=env, capture_output=True, timeout=20)
    assert repeated.returncode != 0 and b"UnicodeEncodeError" not in repeated.stderr
    assert saved == {file.name: file.read_bytes() for file in output.iterdir()}


def test_installer_unicode_error_does_not_crash_its_error_handler(tmp_path):
    env = {**os.environ, "PYTHONUTF8": "0", "PYTHONIOENCODING": "ascii:strict"}
    result = subprocess.run([sys.executable, "-B", str(ADDON / "install.py"), "install",
                             str(tmp_path / "missing ü")], env=env, capture_output=True, timeout=20)
    assert result.returncode == 1 and b"Installer:" in result.stderr
    assert b"UnicodeEncodeError" not in result.stderr and b"Traceback" not in result.stderr


def test_installer_unicode_optional_editor_warning_does_not_fail(tmp_path):
    env = {**os.environ, "PYTHONUTF8": "0", "PYTHONIOENCODING": "ascii:strict"}
    script = ("import install, sys; "
              "install.run = lambda *args, **kwargs: 'Installed. Manual editor unavailable: ' + sys.argv[-1]; "
              "sys.argv = ['install.py', 'install', sys.argv[-1]]; "
              "raise SystemExit(install.main())")
    result = subprocess.run([sys.executable, "-B", "-c", script, str(tmp_path / "comfy ü")],
                            cwd=ADDON, env=env, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert b"Installed. Manual editor unavailable:" in result.stdout
    assert b"\\xfc" in result.stdout
    assert b"UnicodeEncodeError" not in result.stderr


@pytest.mark.skipif(os.name != "nt", reason="NTFS named data streams")
@pytest.mark.parametrize("action", ["configure", "uninstall", "recover"])
@pytest.mark.parametrize("location", ["settings.toml", "receipt.json", ".gitignore", "profile", "request",
                                      "backup/app/config/config.py", ".", "backup", "backup/app", "backup/app/config",
                                      "source-directory"])
def test_installer_preserves_state_with_named_streams(tmp_path, action, location):
    target = tmp_path / "project"
    state = target / installer.STATE
    original = b"original source\n"
    installer.write_file(target / "app/config/config.py", original)
    installer.write_file(state / "backup/app/config/config.py", original)
    installer.write_file(state / ".gitignore", b"*\n")
    source_directory = target / "generated/child"
    source_directory.mkdir(parents=True)
    settings = installer.operator_settings(sys.executable, None)
    installer.write_file(state / "settings.toml", settings)
    generated = {}
    for kind in ("profile", "request"):
        data = json.dumps({"kind": kind}).encode()
        name = f"{kind}-{installer.digest(data)}.json"
        installer.write_file(state / name, data)
        generated[kind] = name
    receipt = {"format": 1, "phase": "installed", "settings": installer.digest(settings),
               "directories": ["generated", "generated/child"],
               "files": {"app/config/config.py": {"before": installer.digest(original),
                                                 "after": installer.digest(original), "mode": 0o644}},
               "private_ignore": True, "generated": list(generated.values())}
    if action == "recover":
        receipt.update(phase="configuring", next_settings=receipt["settings"])
    installer.save_receipt(state, receipt, staged=True)
    path = source_directory if location == "source-directory" else state / generated.get(location, location)
    stream = Path(str(path) + ":user-note")
    stream.write_bytes(b"preserve named user data")
    assert stream.read_bytes() == b"preserve named user data"
    before = {p.relative_to(target): p.read_bytes() if p.is_file() else None for p in target.rglob("*")}
    template = tmp_path / "custom.json"
    template.write_text('{"variant":{"model_version":"custom"}}', encoding="utf-8")

    if location == "source-directory" and action != "uninstall":
        graph = sample_graph()
        app, _ = comfy_with_weights(tmp_path, graph)
        source = write_workflow(tmp_path / "workflow.api.json", graph)
        template = configure(app, tmp_path / "settings", sys.executable, workflow_api=source)
        installer.run("configure", target, template=template)
        assert stream.read_bytes() == b"preserve named user data"
        assert installer.managed_state(target)[1]["phase"] == "installed"
        return

    with pytest.raises(installer.InstallError, match="named data stream"):
        installer.run("configure" if action == "recover" else action, target, template=template)

    assert stream.read_bytes() == b"preserve named user data"
    assert before == {p.relative_to(target): p.read_bytes() if p.is_file() else None for p in target.rglob("*")}


@pytest.fixture
def editor_package(monkeypatch):
    files = {"__init__.py": b'NODE_CLASS_MAPPINGS = {}\nWEB_DIRECTORY = "./js"\n',
             "js/editor.js": b'export const name = "MPT scene editor";\n'}
    monkeypatch.setattr(installer, "packaged_editor_sources", lambda: dict(files), raising=False)
    return files


@pytest.mark.parametrize("portable", [False, True])
def test_scene_editor_installs_in_actual_comfyui_and_repeats(tmp_path, portable, editor_package):
    root = tmp_path / "comfy"
    app = root / "ComfyUI" if portable else root
    app.mkdir(parents=True)
    (app / "main.py").touch()
    assert installer.install_scene_editor(root) is True
    pack = app / "custom_nodes/mpt_scene_editor"
    for name, data in editor_package.items():
        assert (pack / name).read_bytes() == data
    manifest = json.loads((pack / "mpt-editor.json").read_bytes())
    assert manifest == {"format": 1, "files": {name: installer.digest(data)
                                               for name, data in editor_package.items()}}
    before = {p.relative_to(pack): (p.read_bytes(), p.stat().st_mtime_ns)
              for p in pack.rglob("*") if p.is_file()}
    assert installer.install_scene_editor(root) is False
    assert before == {p.relative_to(pack): (p.read_bytes(), p.stat().st_mtime_ns)
                      for p in pack.rglob("*") if p.is_file()}


def test_scene_editor_upgrades_only_managed_files(tmp_path, editor_package):
    (tmp_path / "main.py").touch()
    installer.install_scene_editor(tmp_path)
    pack = tmp_path / "custom_nodes/mpt_scene_editor"
    note = pack / "user-note.txt"
    note.write_bytes(b"keep extra user files")
    editor_package.update({name: data + b"\n# new version\n" for name, data in editor_package.items()})
    assert installer.install_scene_editor(tmp_path) is True
    assert note.read_bytes() == b"keep extra user files"
    for name, data in editor_package.items():
        assert (pack / name).read_bytes() == data
    assert json.loads((pack / "mpt-editor.json").read_bytes())["files"] == {
        name: installer.digest(data) for name, data in editor_package.items()}


@pytest.mark.parametrize("name", ["__init__.py", "js/editor.js", "mpt-editor.json"])
def test_scene_editor_preserves_user_edits(tmp_path, editor_package, name):
    (tmp_path / "main.py").touch()
    installer.install_scene_editor(tmp_path)
    pack = tmp_path / "custom_nodes/mpt_scene_editor"
    (pack / name).write_bytes(b"user edits")
    before = {p.relative_to(pack): p.read_bytes() for p in pack.rglob("*") if p.is_file()}
    with pytest.raises(installer.InstallError, match="Preserve|Unrecognized"):
        installer.install_scene_editor(tmp_path)
    assert before == {p.relative_to(pack): p.read_bytes() for p in pack.rglob("*") if p.is_file()}


@pytest.mark.parametrize("collision", ["custom_nodes", "custom_nodes/mpt_scene_editor",
                                      "custom_nodes/mpt_scene_editor/js"])
def test_scene_editor_refuses_foreign_paths(tmp_path, editor_package, collision):
    (tmp_path / "main.py").touch()
    path = tmp_path / collision
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"foreign data")
    with pytest.raises(installer.InstallError):
        installer.install_scene_editor(tmp_path)
    assert path.read_bytes() == b"foreign data"


@pytest.mark.parametrize("failure", ["js/editor.js", "mpt-editor.json"])
@pytest.mark.parametrize("upgrade", [False, True])
def test_scene_editor_rolls_back_all_files(tmp_path, monkeypatch, editor_package, failure, upgrade):
    (tmp_path / "main.py").touch()
    pack = tmp_path / "custom_nodes/mpt_scene_editor"
    if upgrade:
        installer.install_scene_editor(tmp_path)
    before = {p.relative_to(pack): p.read_bytes() for p in pack.rglob("*") if p.is_file()}
    editor_package.update({name: data + b"\n# new version\n" for name, data in editor_package.items()})
    publish = installer.publish_exclusive
    failed = False

    def fail_once(source, destination):
        nonlocal failed
        if destination == pack / failure and not failed:
            failed = True
            raise OSError("injected editor publication failure")
        return publish(source, destination)

    monkeypatch.setattr(installer, "publish_exclusive", fail_once)
    with pytest.raises(OSError, match="injected editor"):
        installer.install_scene_editor(tmp_path)
    assert failed
    assert before == {p.relative_to(pack): p.read_bytes() for p in pack.rglob("*") if p.is_file()}
    assert not list((tmp_path / "custom_nodes").glob(".ltx-*"))


@pytest.mark.skipif(os.name != "nt", reason="NTFS named data streams")
@pytest.mark.parametrize("name", [".", "js", "__init__.py", "js/editor.js", "mpt-editor.json"])
def test_scene_editor_preserves_named_streams(tmp_path, editor_package, name):
    (tmp_path / "main.py").touch()
    installer.install_scene_editor(tmp_path)
    pack = tmp_path / "custom_nodes/mpt_scene_editor"
    stream = Path(str(pack / name) + ":user-note")
    stream.write_bytes(b"keep named stream")
    with pytest.raises(installer.InstallError, match="named data stream"):
        installer.install_scene_editor(tmp_path)
    assert stream.read_bytes() == b"keep named stream"


@pytest.mark.parametrize("name", ["__init__.py", "js/editor.js", "mpt-editor.json"])
def test_scene_editor_preserves_shared_files(tmp_path, editor_package, name):
    (tmp_path / "main.py").touch()
    installer.install_scene_editor(tmp_path)
    path = tmp_path / "custom_nodes/mpt_scene_editor" / name
    alias = tmp_path / "user-link"
    os.link(path, alias)
    before = alias.read_bytes()
    with pytest.raises(installer.InstallError, match="regular, unshared file"):
        installer.install_scene_editor(tmp_path)
    assert alias.read_bytes() == path.read_bytes() == before


def test_scene_editor_refuses_manifest_paths_outside_fixed_files(tmp_path, editor_package):
    (tmp_path / "main.py").touch()
    installer.install_scene_editor(tmp_path)
    pack = tmp_path / "custom_nodes/mpt_scene_editor"
    manifest = pack / "mpt-editor.json"
    saved = json.loads(manifest.read_bytes())
    saved["files"]["../../user.txt"] = installer.digest(b"user data")
    manifest.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(installer.InstallError, match="Unrecognized"):
        installer.install_scene_editor(tmp_path)
    assert json.loads(manifest.read_bytes()) == saved


def test_scene_editor_preserves_a_concurrent_save(tmp_path, monkeypatch, editor_package):
    (tmp_path / "main.py").touch()
    installer.install_scene_editor(tmp_path)
    pack = tmp_path / "custom_nodes/mpt_scene_editor"
    previous_js = (pack / "js/editor.js").read_bytes()
    previous_manifest = (pack / "mpt-editor.json").read_bytes()
    editor_package["__init__.py"] += b"\n# new version\n"
    change = installer.change_source

    def concurrent_save(target, state, receipt, name, expected, data):
        if name == "__init__.py":
            (pack / name).write_bytes(b"a concurrent user save")
        return change(target, state, receipt, name, expected, data)

    monkeypatch.setattr(installer, "change_source", concurrent_save)
    with pytest.raises(installer.InstallError, match="changed during"):
        installer.install_scene_editor(tmp_path)
    assert (pack / "__init__.py").read_bytes() == b"a concurrent user save"
    assert (pack / "js/editor.js").read_bytes() == previous_js
    assert (pack / "mpt-editor.json").read_bytes() == previous_manifest


@pytest.mark.skipif(os.name != "nt", reason="Windows installer lock")
def test_scene_editor_refuses_another_installer(tmp_path, editor_package):
    (tmp_path / "main.py").touch()
    nodes = tmp_path / "custom_nodes"
    nodes.mkdir()
    with installer.install_lock(nodes):
        with pytest.raises(installer.InstallError, match="Another installer"):
            installer.install_scene_editor(tmp_path)
    assert not (nodes / "mpt_scene_editor").exists()


@pytest.fixture
def minimal_mpt(tmp_path, monkeypatch):
    target = tmp_path / "mpt"
    installer.write_file(target / "app/config/config.py", b"value = 1\n")
    patch = tmp_path / "mpt.patch"
    patch.write_bytes(b"diff --git a/app/config/config.py b/app/config/config.py\n"
                      b"--- a/app/config/config.py\n+++ b/app/config/config.py\n"
                      b"@@ -1 +1 @@\n-value = 1\n+value = 2\n")
    monkeypatch.setattr(installer, "project_patch", lambda *args, **kwargs: patch)
    monkeypatch.setattr(installer, "packaged_sources", lambda: {})
    return target


@pytest.fixture
def fake_model(monkeypatch):
    monkeypatch.setattr(installer, "prepare_model", lambda state, *args, **kwargs: (
        ("2.5", state / "test-request.json"), {}))


def test_reimport_keeps_older_managed_workflow_without_original_export(tmp_path, monkeypatch, minimal_mpt):
    graph = sample_graph()
    app, _ = comfy_with_weights(tmp_path, graph)
    source = write_workflow(tmp_path / "external.api.json", graph)

    class Client:
        def system_stats(self):
            return {"system": {}}

        def object_info(self):
            return fixture_object_info(app, graph)

        def queue(self):
            return {"queue_running": [], "queue_pending": []}

    client = Client()
    monkeypatch.setattr(api, "ComfyClient", lambda endpoint: client)
    state = minimal_mpt / installer.STATE

    def registered_request():
        settings = tomllib.loads((state / "settings.toml").read_text(encoding="utf-8"))
        return json.loads(Path(settings["ltx"]["request_templates"]["2.5"]).read_bytes())

    installer.run("install", minimal_mpt, comfyui=app, comfy_python=sys.executable, workflow_api=source)
    previous = registered_request()
    previous_snapshot = Path(previous["variant"]["source_path"])
    assert previous_snapshot.parent == state
    previous_bytes = previous_snapshot.read_bytes()
    source.rename(tmp_path / "first-export.api.json")
    assert preflight(prepare_request(previous, {"prompt": "A paper boat"}), client)["ok"]

    graph["schedule"]["inputs"]["sigmas"] = "1.0, 0.5, 0.0"
    write_workflow(source, graph)
    installer.run("configure", minimal_mpt, comfyui=app, comfy_python=sys.executable, workflow_api=source)
    current = registered_request()
    snapshot = Path(current["variant"]["source_path"])
    assert snapshot != previous_snapshot and previous_snapshot.read_bytes() == previous_bytes
    assert json.loads(snapshot.read_bytes()) == current["api_prompt"]
    assert current["api_prompt"]["schedule"] == graph["schedule"]
    assert preflight(prepare_request(previous, {"prompt": "A paper boat"}), client)["ok"]
    assert preflight(prepare_request(current, {"prompt": "A new paper boat"}), client)["ok"]
    installer.run("uninstall", minimal_mpt)
    assert not state.exists() and source.is_file()


@pytest.mark.parametrize("action", ["install", "configure"])
@pytest.mark.parametrize("blocked", [False, True])
def test_mpt_comfyui_setup_installs_editor_after_valid_mpt(tmp_path, editor_package, minimal_mpt, fake_model, action, blocked):
    comfy = tmp_path / "ComfyUI"
    comfy.mkdir()
    (comfy / "main.py").touch()
    pack = comfy / "custom_nodes/mpt_scene_editor"
    if action == "configure":
        installer.run("install", minimal_mpt)
    if blocked:
        installer.write_file(pack / "__init__.py", b"foreign plugin")
    source = write_workflow(tmp_path / "external.api.json", sample_graph())
    message = installer.run(action, minimal_mpt, comfyui=comfy, workflow_api=source)
    assert installer.managed_state(minimal_mpt)[1]["phase"] == "installed"
    assert (minimal_mpt / "app/config/config.py").read_bytes() == b"value = 2\n"
    if blocked:
        assert "Manual scene editing is unavailable" in message
        assert "installation is valid" in message
        assert (pack / "__init__.py").read_bytes() == b"foreign plugin"
    else:
        assert message.count("Restart ComfyUI") == 1
        assert (pack / "js/editor.js").read_bytes() == editor_package["js/editor.js"]
        installer.run("uninstall", minimal_mpt)
        assert (pack / "js/editor.js").read_bytes() == editor_package["js/editor.js"]


def test_mpt_validation_failure_does_not_install_editor(tmp_path, editor_package, minimal_mpt, fake_model):
    comfy = tmp_path / "ComfyUI"
    comfy.mkdir()
    (comfy / "main.py").touch()
    python = tmp_path / "missing-python"
    source = write_workflow(tmp_path / "external.api.json", sample_graph())
    with pytest.raises(installer.InstallError, match="python_executable"):
        installer.run("install", minimal_mpt, python=python, comfyui=comfy, workflow_api=source)
    assert not (minimal_mpt / installer.STATE).exists()
    assert not (comfy / "custom_nodes").exists()
    assert (minimal_mpt / "app/config/config.py").read_bytes() == b"value = 1\n"


def test_mpt_external_template_does_not_install_editor(tmp_path, monkeypatch, minimal_mpt):
    graph = sample_graph()
    app, _ = comfy_with_weights(tmp_path, graph)
    source = write_workflow(tmp_path / "workflow.api.json", graph)
    template = configure(app, tmp_path / "settings", sys.executable, workflow_api=source)
    request = json.loads(template.read_bytes())
    del request["variant"]["model_version"]
    template.write_text(json.dumps(request), encoding="utf-8")

    def unexpected_editor_setup(*args):
        pytest.fail("A template-only installation must not select a ComfyUI folder")

    monkeypatch.setattr(installer, "install_scene_editor", unexpected_editor_setup)
    installer.run("install", minimal_mpt, template=template)
    installer.run("configure", minimal_mpt, template=template)
    assert installer.managed_state(minimal_mpt)[1]["phase"] == "installed"
    settings = tomllib.loads((minimal_mpt / installer.STATE / "settings.toml").read_text("utf-8"))
    assert Path(settings["ltx"]["request_templates"]["custom"]) == template


@pytest.mark.parametrize(("case", "error"), [
    ("incomplete", "missing request fields"),
    ("output_directory", "output directory must match default output directory"),
    ("h3_two_seeds", "MiniMax H3 requires 24 FPS and one seed"),
    ("h3_as_ltx", "MiniMax H3 requests must use the H3 adapter and model version together"),
    ("ltx_mismatch", "Select model version 2.3"),
])
def test_incomplete_request_does_not_replace_registered_model(tmp_path, minimal_mpt, case, error):
    graph = sample_graph()
    app, _ = comfy_with_weights(tmp_path, graph)
    source = write_workflow(tmp_path / "workflow.api.json", graph)
    template = configure(app, tmp_path / "settings", sys.executable, workflow_api=source)
    installer.run("install", minimal_mpt, template=template)
    state = minimal_mpt / installer.STATE
    before = {path.name: path.read_bytes() for path in state.iterdir() if path.is_file()}
    request = json.loads(template.read_bytes())
    if case == "incomplete":
        request = {"variant": {"model_version": "2.5"}}
    elif case == "output_directory":
        profile_path = Path(request["profile"]["source_path"])
        profile = json.loads(profile_path.read_bytes())
        wrong_output = tmp_path / "wrong-output"
        wrong_output.mkdir()
        profile["output_directory"] = str(wrong_output)
        wrong_profile = tmp_path / "wrong-profile.json"
        wrong_profile.write_text(json.dumps(profile), encoding="utf-8")
        request["profile"]["source_path"] = str(wrong_profile)
    elif case == "h3_two_seeds":
        request["variant"]["model_version"] = "h3"
    else:
        version = "h3" if case == "h3_as_ltx" else "2.3"
        other_graph = sample_graph(version)
        if version == "2.3":
            for node in other_graph.values():
                if node["class_type"] == "LatentUpscaleModelLoader":
                    node["inputs"]["model_name"] = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
        other_app, _ = comfy_with_weights(tmp_path / "other", other_graph)
        other_workflow = write_workflow(tmp_path / "other.api.json", other_graph)
        other = configure(other_app, tmp_path / "other-settings", sys.executable,
                          workflow_api=other_workflow, model_version=version)
        request = json.loads(other.read_bytes())
        request["variant"]["model_version"] = "2.5"
    bad = tmp_path / "bad-request.json"
    bad.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(installer.InstallError, match=f"Invalid request template.*{error}"):
        installer.run("configure", minimal_mpt, template=bad)

    assert {path.name: path.read_bytes() for path in state.iterdir() if path.is_file()} == before


def test_packaged_scene_editor_contains_only_fixed_files():
    sources = installer.packaged_editor_sources()
    assert tuple(sources) == installer.EDITOR_FILES == ("__init__.py", "js/editor.js")
    assert b"WEB_DIRECTORY" in sources["__init__.py"]
    assert b"registerExtension" in sources["js/editor.js"]



def test_configure_rejects_empty_credentials_in_local_origin(tmp_path):
    graph = sample_graph()
    app, _ = comfy_with_weights(tmp_path, graph)
    source = write_workflow(tmp_path / "workflow.api.json", graph)
    destination = tmp_path / "settings"
    with pytest.raises(ValueError, match="credentials"):
        configure(app, destination, sys.executable, endpoint="http://@127.0.0.1:8188", workflow_api=source)
    assert not destination.exists()


def test_editor_rejects_video_rate_conflicting_with_ltx_audio_and_conditioning(tmp_path):
    graph = sample_graph()
    app, _ = comfy_with_weights(tmp_path, graph)
    source = write_workflow(tmp_path / "workflow.api.json", graph)
    request = json.loads(configure(app, tmp_path / "settings", sys.executable, workflow_api=source).read_bytes())
    request = prepare_request(request, {"prompt": "A paper boat"})
    edited = json.loads(json.dumps(request["api_prompt"]))
    edited["video"]["inputs"]["fps"] = 12
    with pytest.raises(ValueError, match="frame_rate"):
        _rebuild_request(request, edited, fixture_object_info(app, graph))
