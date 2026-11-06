"""Exercise the installed add-on against MoneyPrinterTurbo on PYTHONPATH."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import httpx
import pytest
import toml
from openai import OpenAI

from app import paths
from app.config import config
from app.controllers.base import validate_api_access
from app.services import llm
from app.services import material_cache
from app.utils import file_security


def test_script_worker_survives_page_interaction_and_commits_once(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from app.services import execution_control, version_checker

    started, release = threading.Event(), threading.Event()
    calls = []
    generated_script = "A display reads Error: file not found, then the operator restores it."

    def script(*, on_progress, **kwargs):
        calls.append(kwargs["video_subject"])
        on_progress("A paper boat")
        started.set()
        while not release.wait(0.05):
            execution_control.checkpoint()
        on_progress(generated_script)
        return generated_script

    monkeypatch.setattr(llm, "generate_script", script)
    monkeypatch.setattr(llm, "generate_terms", lambda *args, **kwargs: ["paper boat", "pond"])
    monkeypatch.setattr(config, "save_config", lambda: None)
    monkeypatch.setattr(version_checker, "get_available_update", lambda: None)
    monkeypatch.setattr(config, "ui", {**config.ui, "language": "en"})
    st.cache_data.clear()
    try:
        app = AppTest.from_file(str(paths.SOURCE_ROOT / "webui/Main.py"), default_timeout=15).run()
        app.text_area(key="video_subject").set_value("Paper boats")
        app.text_area(key="video_script").set_value("Previously saved narration")
        app.text_area(key="video_terms").set_value("previous, keywords").run()
        app.button(key="auto_generate_script").click().run(timeout=10)
        assert not app.exception and started.wait(2)
        run = app.session_state["local_script_generation_run"]
        assert not run["done"]
        setting = app.checkbox(key="match_materials_to_script")
        setting.set_value(not setting.value).run(timeout=10)
        assert not app.exception and app.session_state["local_script_generation_run"] is run
        assert calls == ["Paper boats"] and not run["token"].event.is_set()
        assert app.session_state["video_script"] == "Previously saved narration"
        release.set()
        for _ in range(100):
            if run["done"]:
                break
            threading.Event().wait(0.02)
        assert run["done"]
        app.run()
        assert not app.exception and app.text_area(key="video_script").value == generated_script
        assert app.text_area(key="video_terms").value == "paper boat, pond"
        app.text_area(key="video_script").set_value("Edited final narration").run()
        assert app.text_area(key="video_script").value == "Edited final narration"
        assert calls == ["Paper boats"]
    finally:
        release.set()
        st.cache_data.clear()


def test_installer_profile_remains_separate_from_user_settings(tmp_path):
    source = tmp_path / "source"
    shutil.copytree(paths.SOURCE_ROOT / "app", source / "app")
    sidecar = source / ".ltx-operator" / "settings.toml"
    sidecar.parent.mkdir()
    sidecar.write_text('[ltx]\nenabled = true\noperator_script = "installed"\n', encoding="utf-8")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    cfg = runtime / "config.toml"
    cfg.write_text('[ltx]\nenabled = false\n[app]\napi_key = "user-key"\n', encoding="utf-8")
    command = """
from app.config import config
assert not config.ltx['enabled']
assert config.ltx['operator_script'] == 'installed'
config.app['llm_provider'] = 'nvidia'
config.save_config()
"""
    subprocess.run(
        [sys.executable, "-c", command], check=True, cwd=source,
        env={**os.environ, "MPT_HOME": str(runtime), "PYTHONPATH": str(source)},
    )
    saved = toml.load(str(cfg))
    assert saved["ltx"] == {"enabled": False}
    assert saved["app"]["api_key"] == "user-key"
    assert saved["app"]["llm_provider"] == "nvidia"


def test_task_manager_reopens_an_existing_scene_plan_without_regeneration(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from unittest.mock import Mock
    from app.models import const
    from app.models.schema import VideoParams
    from app.services import ltx, state, webui_task

    memory = state.MemoryState()
    memory.update_task("resume-existing", phase="awaiting_approval", progress=40,
                       state=const.TASK_STATE_PROCESSING, video_subject="Paper boats",
                       params=VideoParams(video_subject="Paper boats", video_source="ltx",
                                          ltx_model_version="2.5").model_dump(),
                       ltx_scene_plan_response="Existing scene response", ltx_scenes=[])
    monkeypatch.setattr(state, "state", memory)
    monkeypatch.setattr(ltx, "load_batch", lambda _: {"phase": "awaiting_approval", "scenes": []})
    submit = Mock(side_effect=AssertionError("Opening a task must not generate anything"))
    monkeypatch.setattr(webui_task, "submit_generation", submit)
    monkeypatch.setattr(webui_task, "submit_ltx_render", submit)
    monkeypatch.setattr(config, "save_config", lambda: None)
    monkeypatch.setattr(config, "ui", {**config.ui, "language": "en"})
    st.cache_data.clear()
    try:
        app = AppTest.from_file(str(paths.SOURCE_ROOT / "webui/Main.py"), default_timeout=15).run()
        assert app.session_state["current_generation_task_id"] == ""
        app.text_area(key="video_script").set_value("Unsaved draft in the new browser").run()
        assert app.text_area(key="video_script").value == "Unsaved draft in the new browser"
        app.button(key="view_task_all_resume-existing").click().run()
        assert not app.exception
        assert app.session_state["current_generation_task_id"] == "resume-existing"
        assert app.text_area(key="video_script").value == "Unsaved draft in the new browser"
        assert any(code.value == "Existing scene response" for code in app.code)
        assert any(button.label == "Send to ComfyUI" for button in app.button)
        submit.assert_not_called()
    finally:
        st.cache_data.clear()


@pytest.mark.parametrize("mode", ["completion", "length", "probe-unsupported"])
@pytest.mark.parametrize("custom_key", ["", "custom-key"])
def test_nim_custom_endpoint_uses_only_its_own_key(monkeypatch, mode, custom_key):
    calls = []

    class Endpoint(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            prompt = request["messages"][-1]["content"]
            probe = prompt == "Reply with OK."
            calls.append((
                self.path, self.headers["Authorization"], request["model"], probe,
                self.headers["OpenAI-Organization"], self.headers["OpenAI-Project"],
            ))
            result = {"choices": [{"message": {"role": "assistant", "content": "OK" if probe else "A paper boat floats."}}]}
            status = 200
            if probe and mode == "length":
                result["choices"] = [{"message": {"role": "assistant", "content": ""}, "finish_reason": "length"}]
            if probe and mode == "probe-unsupported":
                status = 400
                result = {"error": {"message": "Unsupported parameter 'max_tokens'", "param": "max_tokens"}}
            body = json.dumps(result).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    with HTTPServer(("127.0.0.1", 0), Endpoint) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-environment-key")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-environment-key")
        monkeypatch.setenv("OPENAI_ORG_ID", "openai-environment-organization")
        monkeypatch.setenv("OPENAI_PROJECT_ID", "openai-environment-project")
        monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        key_field = f"nvidia_custom_{hashlib.sha256(base_url.encode()).hexdigest()}_api_key"
        for key, value in {
            "llm_provider": "nvidia", "nvidia_api_key": "hosted-only" if custom_key else "",
            "nvidia_model_name": "custom/model", "nvidia_base_url": base_url + "/", key_field: custom_key,
        }.items():
            monkeypatch.setitem(config.app, key, value)
        try:
            result = llm.probe_llm_model(force=True)
            assert result["ok"]
            if mode == "probe-unsupported":
                assert result["status"] == "unsupported"
            else:
                assert llm.probe_llm_model()["cached"]
            assert llm.generate_script("A paper boat") == "A paper boat floats."
            assert llm.test_connection()[0]
        finally:
            server.shutdown()
            worker.join()
    authorization = f"Bearer {custom_key}" if custom_key else None
    assert calls == [("/v1/chat/completions", authorization, "custom/model", probe, None, None)
                     for probe in (True, False, False)]


@pytest.mark.parametrize(("endpoint", "send_model"), [
    ("https://integrate.api.nvidia.com/v1/chat/completions", True),
    ("https://ai.api.nvidia.com/v1/vlm/test/direct", False),
    ("https://integrate.api.nvidia.com/v1/responses", True),
])
def test_nim_hosted_request_uses_the_selected_operation(monkeypatch, endpoint, send_model):
    settings = {"llm_provider": "nvidia", "nvidia_model_name": "test/model"}
    monkeypatch.setattr(config, "app", settings)
    monkeypatch.setenv("NVIDIA_API_KEY", "probe-key")
    monkeypatch.setattr(llm, "nvidia_chat_route", lambda model, base: (endpoint, send_model))
    requests = []
    response_status = 200
    responses_api = endpoint.endswith("/responses")

    def serve(request):
        body = json.loads(request.content)
        assert str(request.url) == endpoint and (body.get("model") == "test/model") is send_model
        requests.append((body, request.headers["authorization"]))
        if response_status != 200:
            return httpx.Response(response_status, headers={"Location": "https://other.test/redirect"},
                                  json={"error": {"message": "do not expose prompt or key"}})
        if responses_api:
            message, = body["input"]
            assert message["type"] == "message" and message["role"] == "user"
            assert message["content"][0]["type"] == "input_text" and "messages" not in body
            probe = "max_output_tokens" in body
            output = [{"id": "r", "type": "reasoning", "summary": []}, {
                "id": "m", "type": "message", "role": "assistant",
                "content": [{"type": "text", "text": "Open"}, {"type": "output_text", "text": " water"}],
            }]
            data = {"id": "r1", "object": "response", "created_at": 1, "model": "test/model",
                    "status": "incomplete" if probe else "completed", "output": [] if probe else output,
                    "incomplete_details": {"reason": "max_output_tokens"} if probe else None}
        else:
            assert body["messages"][0]["role"] == "user" and "input" not in body
            data = {"choices": [{"message": {"role": "assistant", "content": "Open water"}, "finish_reason": "stop"}]}
        return httpx.Response(200, json=data)

    monkeypatch.setattr(llm, "OpenAI", lambda **kw: OpenAI(**kw, http_client=httpx.Client(
        transport=httpx.MockTransport(serve), follow_redirects=True)))
    assert llm.probe_llm_model(force=True)["status"] == "available"
    assert requests[-1][0]["max_output_tokens" if responses_api else "max_tokens"] == 1
    assert requests[-1][1] == "Bearer probe-key"
    assert llm._generate_response("private prompt", {**settings, "nvidia_api_key": "snapshot-key"}) == "Open water"
    assert requests[-1][1] == "Bearer snapshot-key"
    assert llm.test_connection()[0]
    for response_status in (307, 404):
        failed = llm._generate_response("private prompt", settings)
        assert str(response_status) in failed and "do not expose" not in failed
    assert len(requests) == 5


def test_remote_listen_requires_explicit_key(monkeypatch):
    monkeypatch.delenv("MPT_API_KEY", raising=False)
    monkeypatch.setitem(config.app, "api_key", "")
    monkeypatch.setattr(config, "listen_host", "127.0.0.1")
    validate_api_access()
    monkeypatch.setattr(config, "listen_host", "0.0.0.0")
    with pytest.raises(ValueError, match="requires"):
        validate_api_access()
    monkeypatch.setenv("MPT_API_KEY", "local-test")
    validate_api_access()


def test_output_alias_cannot_overwrite_input(tmp_path):
    original = tmp_path / "narration.wav"
    original.write_bytes(b"input")
    alias = tmp_path / "result.mp4"
    os.link(original, alias)
    with pytest.raises(ValueError, match="overwrite an input"):
        file_security.ensure_output_path(alias, [original])
    assert original.read_bytes() == b"input"


def test_newly_written_material_cache_tolerates_clock_roundoff(tmp_path, monkeypatch):
    monkeypatch.setattr(material_cache.utils, "storage_dir", lambda *args, **kwargs: str(tmp_path))
    key = ("pixabay", "boat", 3, "9:16")
    file = material_cache._cache_path(*key)
    file.write_text(json.dumps({"version": 2, "items": [{"provider": "pixabay", "url": "https://example.org/boat.mp4",
                        "duration": 4, "source_info": {"provider": "pixabay"}}]}), encoding="utf-8")
    now = file.stat().st_mtime - 0.000001
    assert material_cache.cleanup_expired_material_search_cache(now=now, force=True) == 0
    assert material_cache.load_material_search_cache(*key, now=now)[0].duration == 4


@pytest.mark.parametrize("phase", ["stopped", "stop_unconfirmed"])
def test_stopped_local_task_keeps_scene_review(monkeypatch, phase):
    import ast
    from types import SimpleNamespace
    from unittest.mock import Mock
    from app.models import const

    source = paths.SOURCE_ROOT / "webui/Main.py"
    names = {"_normalize_task_state", "_render_generation_task_snapshot"}
    nodes = [node for node in ast.parse(source.read_text(encoding="utf-8")).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    review = Mock()
    namespace = dict(const=const, tr=lambda text: text,
                     st=SimpleNamespace(error=Mock(), info=Mock(), progress=Mock()),
                     _render_ltx_scene_response=Mock(), _render_ltx_review=review,
                     _render_generation_logs=Mock())
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    snapshot = {"state": const.TASK_STATE_FAILED, "phase": phase,
                "params": {"video_source": "ltx"}, "ltx_scenes": [{"index": 1, "status": "complete"}]}
    namespace["_render_generation_task_snapshot"]("saved-scenes", snapshot)
    review.assert_called_once_with("saved-scenes", snapshot)


def test_completed_local_task_can_open_scene_review(monkeypatch, tmp_path):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from app.models import const
    from app.models.schema import VideoParams
    from app.services import state, version_checker

    # Only the task action is exercised; the player is not opened or decoded.
    saved_video = tmp_path / "final.mp4"
    saved_video.write_bytes(b"saved video")
    memory = state.MemoryState()
    params = VideoParams(video_subject="A boat", video_source="ltx", ltx_model_version="2.5")
    memory.update_task("completed-local", params=params.model_dump(),
                       state=const.TASK_STATE_COMPLETE, phase="complete", progress=100,
                       videos=[str(saved_video)], ltx_scenes=[{"index": 1, "status": "complete"}])
    monkeypatch.setattr(state, "state", memory)
    monkeypatch.setattr(config, "save_config", lambda: None)
    monkeypatch.setattr(version_checker, "get_available_update", lambda: None)
    monkeypatch.setattr(config, "ui", {**config.ui, "language": "en"})
    st.cache_data.clear()
    try:
        app = AppTest.from_file(str(paths.SOURCE_ROOT / "webui/Main.py"), default_timeout=20).run()
        assert not app.exception
        assert "view_task_all_completed-local" in [button.key for button in app.button]
    finally:
        st.cache_data.clear()
