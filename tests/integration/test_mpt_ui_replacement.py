"""Exercise input replacement and scene review through the installed WebUI."""

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app import paths
from app.config import config
from app.models import const
from app.models.schema import VideoParams
from app.services import llm, ltx, state, version_checker
from app.services.loomloom import LoomLoomScriptCandidate
from app.utils import utils


@pytest.fixture
def ui(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "save_config", lambda: None)
    monkeypatch.setitem(config.app, "script_generation_backend", "local")
    monkeypatch.setattr(config, "ui", {**config.ui, "language": "en"})
    monkeypatch.setattr(version_checker, "get_available_update", lambda: None)
    monkeypatch.setattr(state, "state", state.MemoryState())

    def task_dir(sub_dir=""):
        target = tmp_path / "tasks" / sub_dir
        target.mkdir(parents=True, exist_ok=True)
        return str(target)

    monkeypatch.setattr(utils, "task_dir", task_dir)
    st.cache_data.clear()
    yield
    st.cache_data.clear()


@pytest.mark.parametrize("replacement", ["history", "preset", "candidate", "invalid-preset"])
def test_pending_script_respects_input_replacement(ui, monkeypatch, replacement):
    started, release = threading.Event(), threading.Event()

    def script(*, on_progress, **kwargs):
        on_progress("Original generation in progress")
        started.set()
        # Allow an old provider to return after its document has been replaced.
        assert release.wait(10)
        return "Generated for the original boats."

    monkeypatch.setattr(llm, "generate_script", script)
    monkeypatch.setattr(llm, "generate_terms", lambda *args, **kwargs: ["old boats keyword"])
    restored = VideoParams(video_subject="Restored forest", video_script="Keep the forest screenplay.",
                           video_terms=["forest keyword"])
    task_dir = Path(utils.task_dir("saved-forest"))
    (task_dir / "script.json").write_text(json.dumps({
        "script": restored.video_script, "search_terms": restored.video_terms,
        "params": restored.model_dump(mode="json"),
    }), encoding="utf-8")
    state.state.update_task("saved-forest", params=restored.model_dump(mode="json"),
                            state=const.TASK_STATE_COMPLETE, phase="complete", progress=100,
                            video_subject=restored.video_subject)
    import_preset = False
    original_uploader = st.file_uploader

    def uploader(*args, **kwargs):
        if kwargs.get("key") == "settings_preset_uploader" and import_preset:
            params = restored.model_dump(mode="json")
            if replacement == "invalid-preset":
                params["paragraph_number"] = 0
            payload = {"schema": "moneyprinterturbo.settings-preset", "version": 1, "params": params}
            return SimpleNamespace(file_id="forest-preset", getvalue=lambda: json.dumps(payload).encode())
        return original_uploader(*args, **kwargs)

    monkeypatch.setattr(st, "file_uploader", uploader)
    try:
        app = AppTest.from_file(str(paths.SOURCE_ROOT / "webui/Main.py"), default_timeout=20).run()
        app.text_area(key="video_subject").set_value("Original boats")
        app.text_area(key="video_script").set_value("Previous boat draft")
        app.text_area(key="video_terms").set_value("previous keyword").run()
        app.button(key="auto_generate_script").click().run()
        assert not app.exception and started.wait(2)
        old_run = app.session_state["local_script_generation_run"]
        if replacement == "history":
            app.button(key="restore_task_all_saved-forest").click().run()
            app.button(key="confirm_task_restore").click().run()
        elif replacement in {"preset", "invalid-preset"}:
            import_preset = True
            app.run()
        else:
            app.session_state["loomloom_script_candidates"] = (
                LoomLoomScriptCandidate(0, restored.video_script, ("forest keyword",)),
            )
            app.selectbox(key="script_generation_backend_select_en").set_value("loomloom").run()
            assert app.session_state["local_script_generation_run"] is old_run
            assert not old_run["token"].event.is_set()
            app.button(key="loomloom_apply_candidate").click().run()
        assert not app.exception
        if replacement == "invalid-preset":
            assert app.session_state["local_script_generation_run"] is old_run
            assert not old_run["token"].event.is_set()
            assert app.session_state["video_script"] == "Previous boat draft"
            assert app.session_state["video_terms"] == "previous keyword"
        release.set()
        for _ in range(200):
            if old_run["done"]:
                break
            threading.Event().wait(0.01)
        assert old_run["done"]
        app.run()
        assert not app.exception
        if replacement == "invalid-preset":
            assert app.session_state["video_script"] == "Generated for the original boats."
            assert app.session_state["video_terms"] == "old boats keyword"
        else:
            assert old_run["token"].event.is_set()
            assert app.session_state["video_script"] == "Keep the forest screenplay."
            assert app.session_state["video_terms"] == "forest keyword"
        assert app.session_state["video_subject"] == (
            "Original boats" if replacement in {"candidate", "invalid-preset"} else "Restored forest"
        )
        assert "local_script_generation_run" not in app.session_state
    finally:
        release.set()


@pytest.mark.parametrize("phase,task_phase,statuses,disabled", [
    ("awaiting_approval", "awaiting_approval", ["planned"], False),
    ("awaiting_approval", "awaiting_approval", ["failed"], True),
    ("stop_unconfirmed", "stop_unconfirmed", ["approved"], True),
    ("approved", "approved", ["approved", "complete"], False),
    ("complete", "complete", ["complete"], False),
    ("approved", "approved", ["approved", "failed"], True),
    ("approved", "stop_unconfirmed", ["approved"], True),
    ("complete", "stop_unconfirmed", ["complete"], True),
    ("approved", "failed", ["approved"], False),
    ("failed", "failed", ["approved"], True),
])
def test_scene_review_render_respects_batch_phase(ui, monkeypatch, tmp_path, phase, task_phase, statuses, disabled):
    task_id = "review-scenes"
    params = VideoParams(video_subject="A forest", video_source="ltx")
    scenes = []
    for index, status in enumerate(statuses, 1):
        plan = tmp_path / f"scene-{index}.json"
        plan.write_text(json.dumps({"approval_hash": "a" * 64, "prompt": "A forest"}), encoding="utf-8")
        scenes.append({"index": index, "status": status, "plan_file": str(plan)})
    batch = {"task_id": task_id, "phase": phase, "scenes": scenes}
    if phase == "failed":
        batch["retry_error"] = "Create a new task and review its scene plans before rendering again."
    monkeypatch.setattr(ltx, "load_batch", lambda _: batch)
    stop_error = "Remote stop unconfirmed. Check the owned job in ComfyUI."
    state.state.update_task(task_id, params=params.model_dump(mode="json"),
                            state=const.TASK_STATE_FAILED, phase=task_phase,
                            error=stop_error if task_phase == "stop_unconfirmed" else None,
                            video_subject="A forest", ltx_scenes=scenes)
    app = AppTest.from_file(str(paths.SOURCE_ROOT / "webui/Main.py"), default_timeout=20).run()
    app.button(key=f"view_task_all_{task_id}").click().run()
    assert not app.exception
    assert next(button for button in app.button if button.key and button.key.startswith(f"ltx-render-{task_id}-")).disabled is disabled
    if phase == "failed":
        assert batch["retry_error"] in [error.value for error in app.error]
    if task_phase == "stop_unconfirmed":
        assert stop_error in [error.value for error in app.error]


@pytest.mark.parametrize("phase,disabled", [
    ("awaiting_approval", False), ("approved", False), ("editing", False),
    ("planning", True), ("queued", True),
])
def test_task_manager_deletes_idle_ltx_reviews_only(ui, monkeypatch, tmp_path, phase, disabled):
    task_id = "discard-local-scenes"
    monkeypatch.setattr(ltx.utils, "storage_dir", lambda name: str(tmp_path / name))
    params = VideoParams(video_subject="A forest", video_source="ltx")
    ltx._create_json(ltx._batch_path(task_id), {"task_id": task_id, "phase": phase, "scenes": []})
    state.state.update_task(task_id, params=params.model_dump(mode="json"), phase=phase)

    app = AppTest.from_file(str(paths.SOURCE_ROOT / "webui/Main.py"), default_timeout=20).run()
    app.button(key=f"view_task_all_{task_id}").click().run()
    delete = app.button(key=f"delete_task_all_{task_id}")
    assert delete.disabled is disabled
    if not disabled:
        delete.click().run()
        assert not app.exception
        assert state.state.get_task(task_id) is None
        assert not (tmp_path / "tasks" / task_id).exists()
        assert app.session_state["current_generation_task_id"] == ""
        assert not any(button.key == f"delete_task_all_{task_id}" for button in app.button)


@pytest.fixture
def verify_fixture_plan(monkeypatch):
    def verify(args, timeout):
        command, plan_flag, filename, hash_flag, expected = args
        assert (command, plan_flag, hash_flag) == ("verify-plan", "--plan", "--approval-hash")
        if json.loads(Path(filename).read_bytes())["approval_hash"] != expected:
            raise ltx.LtxError("Scene plan changed")
        return {"ok": True, "approval_hash": expected, "submissions": 0}

    monkeypatch.setattr(ltx, "run_cli", verify)


@pytest.mark.parametrize("changed_since_review", [False, True])
def test_scene_review_sends_once_and_binds_the_displayed_plan(ui, monkeypatch, changed_since_review, verify_fixture_plan):
    from unittest.mock import Mock
    from app.services import webui_task

    task_id = "send-reviewed-scenes"
    folder = Path(utils.task_dir(task_id)) / "ltx"
    folder.mkdir()
    plan_file = folder / "plan.json"
    plan = {"approval_hash": "a" * 64, "prompt": "A paper boat on a pond."}
    plan_file.write_text(json.dumps(plan), encoding="utf-8")
    batch = {"task_id": task_id, "phase": "awaiting_approval", "scenes": [
        {"index": 1, "status": "planned", "plan_file": str(plan_file), "approval_hash": plan["approval_hash"]}
    ]}
    (folder / "batch.json").write_text(json.dumps(batch), encoding="utf-8")
    submit = Mock()
    monkeypatch.setattr(webui_task, "submit_ltx_render", submit)
    params = VideoParams(video_subject="Paper boat", video_source="ltx")
    state.state.update_task(task_id, params=params.model_dump(mode="json"),
                            state=const.TASK_STATE_PROCESSING, phase="awaiting_approval",
                            audio_file="neutral.wav", video_subject="Paper boat")
    app = AppTest.from_file(str(paths.SOURCE_ROOT / "webui/Main.py"), default_timeout=20).run()
    app.button(key=f"view_task_all_{task_id}").click().run()
    send, = (button for button in app.button if button.key.startswith(f"ltx-render-{task_id}-"))
    assert send.label == "Send to ComfyUI" and not send.disabled
    assert not any(button.label == "Approve all scenes" for button in app.button)
    if changed_since_review:
        plan["approval_hash"] = "b" * 64
        plan_file.write_text(json.dumps(plan), encoding="utf-8")
    send.click().run()
    assert not app.exception
    if changed_since_review:
        submit.assert_not_called()
        assert ltx.load_batch(task_id)["phase"] == "awaiting_approval"
    else:
        submit.assert_called_once()
        scene, = ltx.load_batch(task_id)["scenes"]
        assert scene["status"] == "approved" and scene["submitted_approval_hash"] == "a" * 64


@pytest.mark.parametrize("start", [False, True])
def test_manual_scene_mode_saves_before_one_click_submission(ui, monkeypatch, start, verify_fixture_plan):
    from unittest.mock import Mock
    from app.services import webui_task

    task_id = "manual-scenes"
    folder = Path(utils.task_dir(task_id)) / "ltx"
    folder.mkdir()
    plan_file = folder / "plan.json"
    plan = {"approval_hash": "a" * 64, "prompt": "An original paper boat."}
    plan_file.write_text(json.dumps(plan), encoding="utf-8")
    batch = {"task_id": task_id, "phase": "awaiting_approval", "scenes": [
        {"index": 1, "status": "planned", "plan_file": str(plan_file), "approval_hash": plan["approval_hash"]}
    ]}
    events = []

    def persist(phase):
        batch["phase"] = phase
        (folder / "batch.json").write_text(json.dumps(batch), encoding="utf-8")
        state.state.update_task(task_id, phase=phase)

    def open_editor(identifier):
        events.append("open")
        batch["editor"] = {"url": "http://127.0.0.1:8188/?mpt_edit=" + "a" * 32}
        persist("editing")

    def apply_editor(identifier):
        events.append("apply")
        plan.update(approval_hash="b" * 64, prompt="An edited red boat.")
        plan_file.write_text(json.dumps(plan), encoding="utf-8")
        batch["scenes"][0]["approval_hash"] = plan["approval_hash"]
        persist("awaiting_approval")
        return batch

    monkeypatch.setattr(ltx, "open_scene_editor", open_editor)
    monkeypatch.setattr(ltx, "apply_scene_editor", apply_editor)
    monkeypatch.setattr(ltx, "close_scene_editor", lambda identifier: (events.append("close"), persist("awaiting_approval")))
    submit = Mock(side_effect=lambda *a, **kw: events.append("submit"))
    monkeypatch.setattr(webui_task, "submit_ltx_render", submit)
    params = VideoParams(video_subject="Paper boat", video_source="ltx")
    state.state.update_task(task_id, params=params.model_dump(mode="json"), audio_file="neutral.wav",
                            state=const.TASK_STATE_PROCESSING, phase="awaiting_approval", video_subject="Paper boat")
    persist("awaiting_approval")
    app = AppTest.from_file(str(paths.SOURCE_ROOT / "webui/Main.py"), default_timeout=20).run()
    app.button(key=f"view_task_all_{task_id}").click().run()
    app.checkbox(key=f"ltx-manual-{task_id}").set_value(True).run()
    assert not app.exception and events == ["open"]
    submit.assert_not_called()
    app.run()
    assert events == ["open"]
    if start:
        send, = (button for button in app.button if button.key and button.key.startswith(f"ltx-render-{task_id}-"))
        assert send.label == "Load saved edits and start" and not send.disabled
        send.click().run()
        assert not app.exception and events == ["open", "apply", "submit"]
        submit.assert_called_once()
        assert ltx.load_batch(task_id)["scenes"][0]["submitted_approval_hash"] == "b" * 64
    else:
        app.checkbox(key=f"ltx-manual-{task_id}").set_value(False).run()
        assert not app.exception and events == ["open", "close"]
        assert json.loads(plan_file.read_bytes())["prompt"] == "An original paper boat."
        submit.assert_not_called()


def test_long_task_output_stays_bounded_without_losing_text(ui, monkeypatch, tmp_path):
    from unittest.mock import Mock
    from app.services import webui_task

    task_id = "long-task-output"
    prompt = "A paper boat drifts past the reeds. " * 500 + "END OF PROMPT"
    response = json.dumps({"visual_prompt": prompt})
    logs = [f"Log line {index}: " + "render progress " * 20 for index in range(200)]
    plan = tmp_path / "scene.json"
    plan.write_text(json.dumps({"approval_hash": "a" * 64, "prompt": prompt}), encoding="utf-8")
    scenes = [{"index": 1, "status": "planned", "plan_file": str(plan)}]
    batch = {"task_id": task_id, "phase": "awaiting_approval", "scenes": scenes}
    params = VideoParams(video_subject="Paper boats", video_source="ltx")
    state.state.update_task(task_id, params=params.model_dump(mode="json"),
                            state=const.TASK_STATE_PROCESSING, phase="llm_ltx_scene_plan",
                            video_subject="Paper boats", ltx_scene_plan_response=response)
    monkeypatch.setitem(config.ui, "hide_log", False)
    monkeypatch.setattr(ltx, "load_batch", lambda _: batch)
    monkeypatch.setattr(webui_task, "get_task_logs", lambda _: logs)
    submit = Mock(side_effect=AssertionError("Viewing output must not start generation or rendering"))
    monkeypatch.setattr(webui_task, "submit_generation", submit)
    monkeypatch.setattr(webui_task, "submit_ltx_render", submit)
    code_heights = {}
    original_code = st.code

    def code(body, *args, **kwargs):
        code_heights[body] = kwargs.get("height")
        return original_code(body, *args, **kwargs)

    monkeypatch.setattr(st, "code", code)
    app = AppTest.from_file(str(paths.SOURCE_ROOT / "webui/Main.py"), default_timeout=20).run()
    app.text_area(key="video_script").set_value("Keep my unsaved screenplay.").run()
    app.button(key=f"view_task_all_{task_id}").click().run()
    assert not app.exception
    assert next(item for item in app.expander if item.label == "Scene director response").proto.expanded
    assert code_heights[response] == 320
    assert code_heights["\n".join(logs)] == 240
    assert response in [item.value for item in app.code]
    assert "\n".join(logs) in [item.value for item in app.code]

    state.state.update_task(task_id, phase="awaiting_approval", ltx_scenes=scenes)
    app.run()
    assert not app.exception
    assert not next(item for item in app.expander if item.label == "Scene director response").proto.expanded
    scene = next(item for item in app.expander if item.label.startswith("Scene 1 "))
    assert not scene.proto.expanded
    assert json.loads(scene.json[0].value)["prompt"] == prompt
    assert any(container.proto.height_config.pixel_height == 320 and container.json
               for container in scene.get("flex_container"))
    assert response in [item.value for item in app.code]
    assert app.text_area(key="video_script").value == "Keep my unsaved screenplay."
    assert not next(button for button in app.button if button.key and button.key.startswith(f"ltx-render-{task_id}-")).disabled
    submit.assert_not_called()
