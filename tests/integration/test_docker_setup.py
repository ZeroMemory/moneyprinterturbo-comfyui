"""Container setup preserves native validation and host-owned output handoffs."""
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import sys
import tomllib

import pytest

ADDON = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ADDON), str(ADDON / "tests")]
import configure_ltx
from ltx_operator_lib.api import ComfyApiError
from ltx_operator_lib.profiles import profile_from_dict, validate_profile
from ltx_operator_lib.runs import _collect_outputs
from smoke_mpt import comfy_fixture
from workflow_fixtures import sample_graph, write_workflow


def mounted_comfy(tmp_path, monkeypatch, version="2.5", separator="\\"):
    app = tmp_path / "ComfyUI"
    (app / "output").mkdir(parents=True)
    (app / "main.py").touch()
    info = {}
    graph = sample_graph(version)
    workflow_api = write_workflow(tmp_path / f"{version}.api.json", graph)
    for node in graph.values():
        for field, category in configure_ltx.MODEL_INPUTS.get(node["class_type"], {}).items():
            filename = node["inputs"][field]
            path = app / "models" / category / "nested" / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"Inert mounted weight; never loaded")
            choices = info.setdefault(node["class_type"], {"input": {"required": {}}})["input"]["required"]
            choices.setdefault(field, [[]])[0].append("nested" + separator + filename)
    for node in info.values():
        for spec in node["input"]["required"].values():
            spec[0] = sorted(set(spec[0]))
    class Client:
        def object_info(self):
            return info
    monkeypatch.setattr(configure_ltx, "ComfyClient", lambda endpoint: Client())
    return app, info, workflow_api


@pytest.mark.parametrize("version", ["2.3", "2.5", "h3"])
@pytest.mark.parametrize("separator", ["\\", "/"])
def test_external_server_uses_host_weight_names_and_mounted_files(tmp_path, monkeypatch, version, separator):
    app, _, workflow_api = mounted_comfy(tmp_path, monkeypatch, version, separator)
    profile, request = configure_ltx.build_template(app, tmp_path / "profile.json", model_version=version,
                                                   python=tmp_path / "host-python.exe", external_server=True,
                                                   workflow_api=workflow_api)
    assert profile["external_server"] is True
    assert not profile["allow_operator_start"] and not profile["allow_operator_stop_owned"]
    assert not validate_profile(profile_from_dict(profile, tmp_path))
    for group in ("models", "loras"):
        for key, path in request["requirements"]["resolved_" + group].items():
            assert key.startswith("nested" + separator)
            assert Path(path).is_file()
    for node in request["api_prompt"].values():
        for field in configure_ltx.MODEL_INPUTS.get(node["class_type"], {}):
            assert node["inputs"][field].startswith("nested" + separator)


def test_external_server_does_not_relax_native_or_network_validation(tmp_path, monkeypatch):
    app, _, workflow_api = mounted_comfy(tmp_path, monkeypatch)
    profile, _ = configure_ltx.build_template(app, tmp_path / "profile.json", external_server=True,
                                             workflow_api=workflow_api)
    launch = profile_from_dict(profile, tmp_path)
    missing = tmp_path / "not-mounted-python"
    assert not validate_profile(replace(launch, python_executable=missing))
    assert any("Python executable" in item for item in validate_profile(replace(launch, external_server=False, python_executable=missing)))
    for field in ("allow_operator_start", "allow_operator_stop_owned"):
        assert any("external server" in item for item in validate_profile(replace(launch, **{field: True})))
    def do_not_execute(module):
        pytest.fail("External server setup must not execute host Python")
    assert any("optional" in item for item in validate_profile(replace(launch, optional_modules=("torch",)), do_not_execute))
    assert any("loopback" in item for item in validate_profile(replace(launch, endpoint="http://192.0.2.10:8188")))
    assert any("output directory" in item for item in validate_profile(replace(launch, output_directory=tmp_path / "elsewhere")))
    with pytest.raises(configure_ltx.MissingComfyPython):
        configure_ltx.build_template(app, tmp_path / "profile.json", workflow_api=workflow_api)
    native, _ = configure_ltx.build_template(app, tmp_path / "profile.json", python=sys.executable,
                                            workflow_api=workflow_api)
    assert "external_server" not in native


def test_external_server_rejects_missing_or_ambiguous_advertised_weight(tmp_path, monkeypatch):
    app, info, workflow_api = mounted_comfy(tmp_path, monkeypatch)
    choices = next(iter(next(iter(info.values()))["input"]["required"].values()))[0]
    selected = choices[0]
    choices.clear()
    with pytest.raises(ValueError, match="does not advertise one matching weight"):
        configure_ltx.build_template(app, tmp_path / "profile.json", external_server=True, workflow_api=workflow_api)
    choices.extend([selected, selected.replace("\\", "/")])
    with pytest.raises(ValueError, match="does not advertise one matching weight"):
        configure_ltx.build_template(app, tmp_path / "profile.json", external_server=True, workflow_api=workflow_api)
    assert not (tmp_path / "profile.json").exists()


def test_external_server_network_failure_is_a_configuration_error(tmp_path, monkeypatch):
    app, _, workflow_api = mounted_comfy(tmp_path, monkeypatch)
    class Client:
        def object_info(self):
            raise ComfyApiError("server unavailable")
    monkeypatch.setattr(configure_ltx, "ComfyClient", lambda endpoint: Client())
    with pytest.raises(ValueError, match="Cannot read ComfyUI's weight list"):
        configure_ltx.build_template(app, tmp_path / "profile.json", external_server=True, workflow_api=workflow_api)


def test_host_reported_backslash_output_is_collected_from_the_mount(tmp_path):
    folder = tmp_path / "neutral" / "video"
    folder.mkdir(parents=True)
    clip = folder / "scene_00001_.mp4"
    clip.write_bytes(b"Inert video fixture; never decoded")
    graph = {"45": {"class_type": "SaveVideo", "inputs": {"format": "mp4", "codec": "h264"}}}
    entry = {"outputs": {"45": {"images": [{"filename": clip.name, "subfolder": "neutral\\video", "type": "output"}]}}}
    records = _collect_outputs(entry, tmp_path, graph=graph, prefixes={"45": "neutral/video/scene"},
                               expected=["neutral/video/scene*.mp4"], primary_node="45")
    assert len(records) == 1 and records[0]["path"] == str(clip) and records[0]["primary"] is True


@pytest.mark.parametrize("field,value", [("subfolder", "../outside"), ("subfolder", "..\\outside"),
                                          ("subfolder", "C:\\outside"), ("subfolder", "/outside"),
                                          ("filename", "..\\outside.mp4"), ("filename", "a/outside.mp4")])
def test_host_reported_output_cannot_escape_the_mount(tmp_path, field, value):
    file = {"filename": "scene.mp4", "subfolder": "neutral", "type": "output", field: value}
    graph = {"45": {"class_type": "SaveVideo", "inputs": {"format": "mp4", "codec": "h264"}}}
    with pytest.raises(ValueError, match="relative ComfyUI output subfolder"):
        _collect_outputs({"outputs": {"45": {"images": [file]}}}, tmp_path, graph=graph,
                         prefixes={"45": "neutral/scene"}, expected=["neutral/scene*.mp4"], primary_node="45")


@pytest.fixture
def container_entrypoint(monkeypatch):
    monkeypatch.delenv("MPT_LTX_SETTINGS_PATH", raising=False)
    monkeypatch.delenv("MPT_LISTEN_HOST", raising=False)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    spec = importlib.util.spec_from_file_location("container_entrypoint", ADDON / "docker/entrypoint.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    return entry


def test_container_entrypoint_registers_each_model_then_starts_loopback_mpt(tmp_path, monkeypatch, container_entrypoint):
    entry = container_entrypoint
    app = tmp_path / "ComfyUI"
    app.mkdir()
    (app / "main.py").touch()
    monkeypatch.setattr(entry, "COMFYUI", app)
    monkeypatch.setenv("MPT_MODELS", "2.5,2.3,h3,2.5")
    monkeypatch.setenv("MPT_WEBUI_PORT", "8508")
    monkeypatch.setenv("COMFYUI_ENDPOINT", "http://127.0.0.1:8197")
    monkeypatch.setenv("MPT_HOME", str(tmp_path))
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    for name in ("2.5", "2.3", "h3"):
        (workflows / f"{name}.api.json").touch()
    calls = []
    monkeypatch.setattr(entry, "register_model", lambda *args: calls.append(args))
    started = []
    monkeypatch.setattr(entry.os, "execv", lambda executable, args: started.append((executable, args)))
    entry.main([])
    assert len(calls) == 3
    assert [args[1] for args in calls] == ["2.5", "2.3", "h3"]
    assert [args[2] for args in calls] == [
        workflows / "2.5.api.json", workflows / "2.3.api.json", workflows / "h3.api.json"
    ]
    assert all(args[3] == "http://127.0.0.1:8197" for args in calls)
    assert len(started) == 1
    assert "--server.address=127.0.0.1" in started[0][1] and "--server.port=8508" in started[0][1]
    monkeypatch.setenv("MPT_MODELS", "unknown")
    with pytest.raises(ValueError, match="MPT_MODELS"):
        entry.main([])
    assert len(calls) == 3 and len(started) == 1


@pytest.mark.parametrize("argv,script", [(["api"], "main.py"),
                                         (["cli", "--params", "/data/video.json"], "cli.py")])
def test_container_entrypoint_preserves_api_and_cli_routes(tmp_path, monkeypatch, argv, script, container_entrypoint):
    entry = container_entrypoint
    app = tmp_path / "ComfyUI"
    app.mkdir()
    (app / "main.py").touch()
    monkeypatch.setattr(entry, "COMFYUI", app)
    monkeypatch.setenv("MPT_MODELS", "2.5")
    monkeypatch.setenv("MPT_HOME", str(tmp_path))
    monkeypatch.delenv("MPT_LISTEN_HOST", raising=False)
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    (workflows / "2.5.api.json").touch()
    calls, started = [], []
    monkeypatch.setattr(entry, "register_model", lambda *args: calls.append(args))
    monkeypatch.setattr(entry.os, "execv", lambda executable, args: started.append(args))
    entry.main(argv)
    assert len(calls) == 1
    assert started == [[sys.executable, "-B", str(entry.MPT / script), *argv[1:]]]
    if argv == ["api"]:
        assert entry.os.environ["MPT_LISTEN_HOST"] == "127.0.0.1"


@pytest.mark.parametrize("argv", [[], ["api"], ["cli", "--params", "/data/video.json"]])
def test_unavailable_comfy_model_does_not_block_other_mpt_routes(tmp_path, monkeypatch, capsys, argv, container_entrypoint):
    entry = container_entrypoint
    app = tmp_path / "ComfyUI"
    app.mkdir()
    (app / "main.py").touch()
    monkeypatch.setattr(entry, "COMFYUI", app)
    monkeypatch.setenv("MPT_MODELS", "2.5,2.3,h3")
    monkeypatch.setenv("MPT_HOME", str(tmp_path))
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    (workflows / "2.3.api.json").touch()
    (workflows / "h3.api.json").touch()
    attempted, started = [], []
    def configure(state, model, workflow, endpoint):
        attempted.append(model)
        if model == "2.3":
            raise ValueError("server unavailable")
    monkeypatch.setattr(entry, "register_model", configure)
    monkeypatch.setattr(entry.os, "execv", lambda executable, args: started.append(args))
    entry.main(argv)
    assert attempted == ["2.3", "h3"] and len(started) == 1
    warnings = capsys.readouterr().err
    assert "Local model 2.5 was not registered" in warnings and str(workflows / "2.5.api.json") in warnings
    assert "Local model 2.3 was not registered" in warnings


def test_registered_models_are_available_in_a_recreated_container(tmp_path, monkeypatch, container_entrypoint):
    entry = container_entrypoint
    app, _, exported = mounted_comfy(tmp_path, monkeypatch, separator=os.sep)
    data = tmp_path / "data"
    workflows = data / "workflows"
    workflows.mkdir(parents=True)
    export = exported.rename(workflows / "2.5.api.json")
    first, replacement = tmp_path / "first-image", tmp_path / "replacement-image"
    first.mkdir()
    replacement.mkdir()
    monkeypatch.setattr(entry, "MPT", first)
    monkeypatch.setattr(entry, "COMFYUI", app)
    monkeypatch.setenv("MPT_HOME", str(data))
    monkeypatch.setenv("MPT_MODELS", "2.5")
    monkeypatch.delenv("MPT_LTX_SETTINGS_PATH", raising=False)
    active_settings = []
    monkeypatch.setattr(entry.os, "execv", lambda *args: active_settings.append(Path(os.environ["MPT_LTX_SETTINGS_PATH"])))

    with comfy_fixture(app, None, sample_graph()) as (endpoint, submissions):
        monkeypatch.setenv("COMFYUI_ENDPOINT", endpoint)
        entry.main([])
        assert not submissions
    settings = active_settings[-1]
    before = settings.read_bytes()
    registered = tomllib.loads(before.decode("utf-8"))["ltx"]["request_templates"]
    request = json.loads(Path(registered["2.5"]).read_bytes())
    assert Path(request["variant"]["source_path"]).is_file()
    assert Path(request["profile"]["source_path"]).is_file()
    assert settings.is_relative_to(data)
    assert not list(first.iterdir())

    monkeypatch.setattr(entry, "MPT", replacement)
    monkeypatch.delenv("MPT_LTX_SETTINGS_PATH")
    entry.main(["api"])  # The endpoint is down and registration fails.
    export.unlink()
    entry.main([])
    assert active_settings == [settings, settings, settings]
    assert settings.read_bytes() == before
    assert not list(replacement.iterdir())


@pytest.mark.parametrize("invalid", [b"[ltx\n", b"[ltx]\nrequest_templates = 1\n"])
def test_corrupt_registry_keeps_other_sources_available(tmp_path, monkeypatch, capsys, container_entrypoint, invalid):
    entry = container_entrypoint
    app = tmp_path / "ComfyUI"
    app.mkdir()
    (app / "main.py").touch()
    data = tmp_path / "data"
    registry = data / ".ltx-models"
    registry.mkdir(parents=True)
    settings = registry / "settings.toml"
    settings.write_bytes(invalid)
    monkeypatch.setattr(entry, "COMFYUI", app)
    monkeypatch.setenv("MPT_HOME", str(data))
    started = []
    monkeypatch.setattr(entry.os, "execv", lambda *args: started.append(os.environ.get("MPT_LTX_SETTINGS_PATH")))

    entry.main([])

    assert started == [None]
    assert settings.read_bytes() == invalid
    assert "Local model registry is unavailable" in capsys.readouterr().err
