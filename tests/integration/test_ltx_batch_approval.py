"""Verify complete approved batches before any scene is submitted."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

from app.services import ltx


ADDON = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ADDON))
sys.path.insert(0, str(ADDON / "tests"))
from smoke_mpt import fixture_object_info
from configure_ltx import configure, MODEL_INPUTS, MODEL_VERSIONS
from workflow_fixtures import sample_graph, write_workflow
from ltx_operator_lib.planning import create_plan
from ltx_operator_lib.preparation import prepare_request


@pytest.fixture
def approved_batch(tmp_path, monkeypatch):
    app = tmp_path / "ComfyUI"
    app.mkdir()
    (app / "main.py").touch()
    (app / "output").mkdir()
    graph = sample_graph()
    for node in graph.values():
        for field, category in MODEL_INPUTS.get(node["class_type"], {}).items():
            weight = app / "models" / category / node["inputs"][field]
            weight.parent.mkdir(parents=True, exist_ok=True)
            weight.touch()
    source = write_workflow(tmp_path / "2.5.api.json", graph)
    template = json.loads(configure(app, tmp_path / "settings", sys.executable, workflow_api=source).read_bytes())
    monkeypatch.setattr(ltx.utils, "task_dir", lambda task_id="": str(tmp_path / task_id))
    monkeypatch.setattr(ltx, "settings", lambda: {
        "python_executable": sys.executable,
        "operator_script": str(ADDON / "runtime/scripts/ltx_operator.py"),
    })
    batch = {"task_id": "neutral", "phase": "approved", "scenes": []}
    for index in (1, 2):
        scene_root = tmp_path / "neutral" / "ltx" / str(index)
        scene_root.mkdir(parents=True)
        request = prepare_request(template, {"prompt": f"A boat crosses pool {index}.", "output_prefix": f"scene-{index}"})
        dependency = scene_root / "dependency.bin"
        dependency.write_bytes(b"AAAA")
        request["requirements"]["required_files"] = [str(dependency)]
        plan = create_plan(request)
        plan_file = scene_root / "plan.json"
        plan_file.write_text(json.dumps(plan), encoding="utf-8")
        batch["scenes"].append({"index": index, "plan_file": str(plan_file),
                                "approval_hash": plan["approval_hash"],
                                "submitted_approval_hash": plan["approval_hash"], "status": "approved"})
    ltx._replace_json(ltx._batch_path("neutral"), batch)
    return batch


@pytest.fixture
def worker(approved_batch, tmp_path, monkeypatch):
    from ltx_operator_lib import runs
    from ltx_operator_lib.api import CancellationResult, CompletionResult
    from ltx_operator_lib.cli import build_parser, _dispatch, _jsonable

    # This fake worker exercises ownership transitions with inert clip bytes.
    monkeypatch.setattr(runs, "_validate_clip", lambda *args, **kwargs: None)

    class Client:
        def __init__(self):
            self.calls = []
            self.submissions = []
            self.after_post = None
            self.failed_id = None
            self.running = ["foreign"]
            self.pending = ["foreign-pending"]
            self.cancellations = []
            self.terminal = {}

        def system_stats(self, **kwargs):
            return {"system": {}}

        def object_info(self, **kwargs):
            return fixture_object_info(tmp_path / "ComfyUI", sample_graph())

        def queue(self, **kwargs):
            return {"queue_running": [[0, value] for value in self.running],
                    "queue_pending": [[1, value] for value in self.pending]}

        def submit(self, prompt, *, client_id, **kwargs):
            self.submissions.append(prompt)
            prompt_id = str(len(self.submissions))
            self.calls.append(("post", prompt_id))
            if self.after_post:
                self.after_post(prompt_id)
            return {"prompt_id": prompt_id}

        def wait_for_completion(self, prompt_id, client_id, **kwargs):
            self.calls.append(("wait", prompt_id))
            if prompt_id == self.failed_id:
                self.terminal[prompt_id] = "failed"
                return CompletionResult(prompt_id, "failed", {})
            return CompletionResult(prompt_id, "completed", self.completed_history(prompt_id))

        def completed_history(self, prompt_id, **kwargs):
            self.terminal[prompt_id] = "completed"
            filename = f"scene-{prompt_id}.mp4"
            (tmp_path / "ComfyUI" / "output" / filename).write_bytes(b"neutral video fixture")
            node_id, = (key for key, node in self.submissions[int(prompt_id) - 1].items()
                        if node["class_type"] == "SaveVideo")
            return {prompt_id: {"outputs": {node_id: {"images": [{"filename": filename, "type": "output"}]}},
                                "status": {"completed": True}}}

        def delete_pending(self, prompt_id, **kwargs):
            self.calls.append(("delete", prompt_id))

        def history(self, prompt_id, **kwargs):
            return {}

        def cancel_and_confirm(self, prompt_id, **kwargs):
            self.cancellations.append(prompt_id)
            self.pending[:] = [value for value in self.pending if value != prompt_id]
            status = self.terminal.get(prompt_id) or ("uncertain" if prompt_id in self.running else "confirmed_cancelled")
            return CancellationResult(prompt_id, status, self.queue(), {})

    client = Client()
    monkeypatch.setattr(ltx.utils, "storage_dir", lambda key: str(tmp_path / key))
    def invoke(arguments, timeout, progress):
        try:
            return _dispatch(build_parser().parse_args(arguments), lambda _: client, progress)
        except Exception as exc:
            raise ltx.LtxError(str(exc)) from exc
    monkeypatch.setattr(ltx, "run_cli_stream", invoke)
    def cancel_command(command, **kwargs):
        from types import SimpleNamespace
        assert command[2] == "cancel"
        def owned_client(endpoint, *, owned_prompt_ids):
            assert owned_prompt_ids <= {str(index + 1) for index in range(len(client.submissions))}
            return client
        payload = _dispatch(build_parser().parse_args(command[2:-1]), owned_client)
        return SimpleNamespace(returncode=0, stdout=json.dumps(_jsonable(payload)))
    monkeypatch.setattr(ltx.subprocess, "run", cancel_command)
    return client


@pytest.mark.parametrize("changed", [None, "plan", "dimensions", "dependency", "missing", "consumed"])
def test_every_scene_is_checked_before_rendering(approved_batch, tmp_path, worker, changed):
    from ltx_operator_lib.state import OperatorState

    batch = approved_batch
    second = Path(batch["scenes"][1]["plan_file"])
    if changed in {"plan", "dimensions"}:
        plan = json.loads(second.read_bytes())
        if changed == "dimensions":
            latent, = (n for n in plan["api_prompt"].values() if n["class_type"] == "EmptyLTXVLatentVideo")
            plan["parameters"]["width"] = latent["inputs"]["width"] = 960
        else:
            plan["prompt"] = "Changed after approval"
        second.write_text(json.dumps(plan), encoding="utf-8")
    elif changed == "dependency":
        (second.parent / "dependency.bin").write_bytes(b"BBBB")
    elif changed == "missing":
        second.unlink()
    elif changed == "consumed":
        OperatorState(tmp_path / "ltx-state").consume_approval(batch["scenes"][1]["approval_hash"],
                                                             run_directory=tmp_path / "old-run")

    if changed:
        with pytest.raises(ltx.LtxError):
            ltx.render_batch("neutral")
        assert worker.calls == []
        assert ltx.load_batch("neutral") == batch
    else:
        assert [Path(p).name for p in ltx.render_batch("neutral")] == ["scene-1.mp4", "scene-2.mp4"]
        assert worker.calls == [("post", "1"), ("post", "2"), ("wait", "1"), ("wait", "2")]
        assert ltx.load_batch("neutral")["phase"] == "complete"
        for scene, submitted in zip(batch["scenes"], worker.submissions):
            plan = json.loads(Path(scene["plan_file"]).read_bytes())
            assert submitted == plan["api_prompt"]
        records = sorted((tmp_path / "neutral" / "ltx" / "scenes").glob("*/run/run.json"))
        from ltx_operator_lib.runs import load_signed_run_record
        assert [load_signed_run_record(r, OperatorState(tmp_path / "ltx-state"))["prompt_id"]
                for r in records] == ["1", "2"]


@pytest.mark.parametrize("cause", ["drift", "missing_weight", "lost_response"])
def test_later_failure_preserves_earlier_inflight_scene(approved_batch, tmp_path, worker, cause):
    import os

    plan_file = Path(approved_batch["scenes"][1]["plan_file"])
    dep = plan_file.with_name("dependency.bin")
    def after_post(prompt_id):
        if prompt_id == "1" and cause == "drift":
            info = dep.stat()
            dep.write_bytes(b"BBBB")
            os.utime(dep, ns=(info.st_atime_ns, info.st_mtime_ns))
        if prompt_id == "1" and cause == "missing_weight":
            weights = json.loads(plan_file.read_bytes())["requirements"]["resolved_models"]
            Path(next(iter(weights.values()))).unlink()
        if prompt_id == "2" and cause == "lost_response":
            raise OSError("POST response lost")
    worker.after_post = after_post
    with pytest.raises(ltx.LtxError, match="drift|response lost"):
        ltx.render_batch("neutral")
    batch = ltx.load_batch("neutral")
    assert batch["phase"] == ("stop_unconfirmed" if cause == "lost_response" else "failed")
    first, second = batch["scenes"]
    assert first["status"] == "complete"
    assert Path(first["output_file"]).read_bytes() == b"neutral video fixture"
    assert second["status"] == "failed"
    report = json.loads((Path(second["run_directory"]) / "report.json").read_bytes())
    if cause in {"drift", "missing_weight"}:
        assert worker.calls == [("post", "1"), ("wait", "1")]
        assert report["status"] == "failed_before_submission"
        assert not (tmp_path / "ltx-state" / "receipts" / f"{second['approval_hash']}.json").exists()
    else:
        assert worker.calls == [("post", "1"), ("post", "2"), ("wait", "1")]
        assert report["status"] == "uncertain_submission"
        assert (tmp_path / "ltx-state" / "receipts" / f"{second['approval_hash']}.json").is_file()
    with pytest.raises(ltx.LtxError, match="not fully approved"):
        ltx.render_batch("neutral")


@pytest.mark.parametrize("after_delete,expected", [
    ("absent", "removed_or_unconfirmed"), ("pending", "cancelled"), ("running", "complete"),
])
def test_failed_scene_deletes_only_its_pending_successor(approved_batch, worker, tmp_path, after_delete, expected):
    second = json.loads(Path(approved_batch["scenes"][1]["plan_file"]).read_bytes())
    third = create_plan(prepare_request(second, {"prompt": "A third boat.", "output_prefix": "scene-3"}))
    path = tmp_path / "third.json"
    path.write_text(json.dumps(third), encoding="utf-8")
    approved_batch["scenes"].append({"index": 3, "status": "approved", "plan_file": str(path),
                                     "submitted_approval_hash": third["approval_hash"], "approval_hash": third["approval_hash"]})
    ltx._replace_json(ltx._batch_path("neutral"), approved_batch)
    if after_delete == "pending":
        worker.pending.append("2")
    if after_delete == "running":
        worker.running.append("2")
    worker.failed_id = "1"
    with pytest.raises(ltx.LtxError, match="scene 1"):
        ltx.render_batch("neutral")
    expected_calls = [("post", "1"), ("post", "2"), ("wait", "1")]
    if after_delete == "running":
        expected_calls.append(("wait", "2"))
    assert worker.calls == expected_calls
    scenes = ltx.load_batch("neutral")["scenes"]
    assert scenes[0]["error"] == "failed"
    assert (scenes[1].get("error") or scenes[1]["status"]) == expected
    if after_delete == "pending":
        report = json.loads((Path(scenes[1]["run_directory"]) / "report.json").read_bytes())
        assert report["cleanup"]["status"] == "confirmed_cancelled"
        assert "2" not in worker.pending and "foreign-pending" in worker.pending
    assert scenes[2]["status"] == "not_submitted"
    assert not (tmp_path / "ltx-state/receipts" / (third["approval_hash"] + ".json")).exists()


def test_scene_completed_before_late_poll_uses_its_history(approved_batch, worker, monkeypatch):
    from ltx_operator_lib import runs

    clock = [100]
    monkeypatch.setattr(runs.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(ltx, "settings", lambda: {"render_timeout_seconds": 1})
    def finish_both(prompt_id):
        if prompt_id == "2":
            clock[0] += 5
    worker.after_post = finish_both
    worker.history = worker.completed_history
    assert [Path(p).name for p in ltx.render_batch("neutral")] == ["scene-1.mp4", "scene-2.mp4"]
    assert worker.calls == [("post", "1"), ("post", "2")]


@pytest.mark.parametrize("legacy", [False, True], ids=["current", "previous-approval"])
def test_batch_checks_shared_weights_without_opening_them(approved_batch, worker, monkeypatch, tmp_path, legacy):
    from ltx_operator_lib.canonical import approval_hash

    second = approved_batch["scenes"][1]
    path = Path(second["plan_file"])
    plan = json.loads(path.read_bytes())
    weights = {Path(p) for p in plan["requirements"]["resolved_models"].values()}
    assert weights
    if legacy:
        covered = plan["covered"]
        covered["approval_format"] = 2
        for name, filename in plan["requirements"]["resolved_models"].items():
            covered["requirements"]["resolved_models"][name] = {
                "path": filename, "content_sha256": hashlib.sha256(Path(filename).read_bytes()).hexdigest(),
            }
        plan["approval_hash"] = approval_hash(covered)
        second["approval_hash"] = second["submitted_approval_hash"] = plan["approval_hash"]
        path.write_text(json.dumps(plan), encoding="utf-8")
        ltx._replace_json(ltx._batch_path("neutral"), approved_batch)

    open_file = Path.open

    def guard_open(path, *args, **kwargs):
        assert path not in weights, f"opened weight file {path}"
        return open_file(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard_open)
    if legacy:
        with pytest.raises(ltx.LtxError, match="new plan"):
            ltx.render_batch("neutral")
        assert not worker.submissions
        assert not list((tmp_path / "ltx-state/receipts").glob("*.json"))
    else:
        assert len(ltx.render_batch("neutral")) == len(worker.submissions) == 2


def test_first_scene_drift_during_later_scene_verification_rejects_entire_batch(approved_batch, worker, monkeypatch):
    import os
    from ltx_operator_lib import runs

    first, later = [Path(s["plan_file"]).with_name("dependency.bin") for s in approved_batch["scenes"]]
    read_later = 0
    digest = runs.sha256_file
    def hash_file(path):
        nonlocal read_later
        value = digest(path)
        if Path(path) == later:
            read_later += 1
            if read_later == 2:
                old = first.stat()
                first.write_bytes(b"BBBB")
                os.utime(first, ns=(old.st_atime_ns, old.st_mtime_ns))
        return value
    monkeypatch.setattr(runs, "sha256_file", hash_file)
    with pytest.raises(ltx.LtxError, match="drift"):
        ltx.render_batch("neutral")
    assert read_later == 2 and worker.calls == []
    assert [s["status"] for s in ltx.load_batch("neutral")["scenes"]] == ["failed", "not_submitted"]


@pytest.fixture(scope="module")
def model_selection_clip(tmp_path_factory):
    clip = tmp_path_factory.mktemp("model-selection") / "fixture.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "color=blue:s=64x64:r=24",
         "-frames:v", "3", "-an", "-c:v", "mpeg4", str(clip)],
        check=True,
    )
    return clip


@pytest.fixture
def local_comfyui(tmp_path, monkeypatch, version, dimensions, model_selection_clip):
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1")
    monkeypatch.syspath_prepend(str(ADDON / "tests"))
    from smoke_mpt import comfy_fixture

    app = tmp_path / "ComfyUI"
    (app / "output").mkdir(parents=True)
    (app / "main.py").touch()
    with comfy_fixture(app, model_selection_clip, sample_graph(version), dimensions) as (endpoint, submissions):
        yield app, endpoint, submissions, model_selection_clip


@pytest.mark.parametrize("version,aspect,dimensions,custom,frames", [
    ("2.3", "9:16", (1088, 1920), False, 121), ("2.5", "9:16", (1088, 1920), False, 121),
    ("h3", "9:16", (768, 1344), False, 124),
    ("2.3", "16:9", (1920, 1088), True, 121), ("2.5", "1:1", (768, 768), True, 121),
    ("h3", "9:16", (768, 1344), True, 124),
])
def test_selected_model_is_pinned_through_review_and_render(tmp_path, monkeypatch, version, aspect, dimensions,
                                                             custom, frames, local_comfyui):
    from app.models.schema import VideoParams

    app, endpoint, submissions, clip = local_comfyui
    templates = {}
    for key in MODEL_VERSIONS:
        graph = sample_graph(key)
        for node in graph.values():
            for field, folder in MODEL_INPUTS.get(node["class_type"], {}).items():
                weight = app / "models" / folder / node["inputs"][field]
                weight.parent.mkdir(parents=True, exist_ok=True)
                weight.touch()
        source = write_workflow(tmp_path / f"{key}.api.json", graph)
        templates[key] = str(configure(app, tmp_path / key, sys.executable,
                                       model_version=key, endpoint=endpoint, workflow_api=source))
    settings = {"enabled": True, "python_executable": sys.executable,
                "operator_script": str(ADDON / "runtime/scripts/ltx_operator.py"),
                "request_templates": templates, "request_template": templates["2.5"]}
    monkeypatch.setattr(ltx, "settings", lambda: settings)
    monkeypatch.setattr(ltx.utils, "task_dir", lambda task_id: str(tmp_path / task_id))
    monkeypatch.setattr(ltx.utils, "storage_dir", lambda name: str(tmp_path / name))

    sizes = {"ltx_generation_width": dimensions[0], "ltx_generation_height": dimensions[1]} if custom else {}
    params = VideoParams(video_subject="A paper boat", video_source="ltx", ltx_model_version=version,
                         ltx_scene_duration_seconds=5, video_aspect=aspect, **sizes)
    second_scene = version == "2.5" and not custom
    script = "A paper boat floats. It passes under a bridge." if second_scene else "A paper boat floats."
    batch = ltx.plan_batch("boat", script, 10 if second_scene else 5, params)
    assert batch["model_version"] == version
    scene = batch["scenes"][0]
    plan = json.loads(Path(scene["plan_file"]).read_bytes())
    assert plan["variant"]["model_version"] == version
    assert (scene["generation_width"], scene["generation_height"]) == dimensions
    first_pass = tuple(value // (1 if version == "h3" else 2) for value in dimensions)
    assert (plan["parameters"]["width"], plan["parameters"]["height"]) == first_pass
    assert (scene["first_pass_width"], scene["first_pass_height"]) == first_pass
    latent, = (node["inputs"] for node in plan["api_prompt"].values()
               if node["class_type"] in {"EmptyLTXVLatentVideo", "MiniMaxH3ImageToVideo"})
    assert (latent["width"], latent["height"]) == first_pass
    assert plan["parameters"]["frame_count"] == frames
    assert batch["scenes"][0]["render_duration_seconds"] == frames / plan["parameters"]["fps"]
    if version == "h3":
        assert plan["parameters"]["fps"] == 24
        assert sum(key.endswith("_seed") for key in plan["parameters"]) == 1
        assert not ltx.scene_prompt_context(plan)["supports_negative_prompt"]

    ltx.approve_batch("boat", [{"scene_index": s["index"], "approval_hash": s["approval_hash"]}
                               for s in batch["scenes"]])
    assert not submissions
    settings["request_templates"] = {version: templates["h3" if version != "h3" else "2.5"]}
    assert ltx.validate_settings(model_version=version)
    Path(templates[version]).unlink()
    outputs = ltx.render_batch("boat")
    assert len(submissions) == len(outputs) == (2 if second_scene else 1)
    assert all(Path(output).read_bytes() == clip.read_bytes() for output in outputs)
    assert ltx.load_batch("boat")["model_version"] == version


@pytest.mark.parametrize("version,step,minimum,maximum", [
    ("h3", 32, 32, 16384), ("2.3", 64, 128, 32768), ("2.5", 64, 128, 32768),
])
def test_generation_size_limits_and_model_presets(version, step, minimum, maximum):
    from app.models.schema import TaskVideoRequest
    from pydantic import ValidationError

    def request(width, height):
        return TaskVideoRequest(video_subject="A paper boat", video_source="ltx", ltx_model_version=version,
                                ltx_generation_width=width, ltx_generation_height=height)

    assert ltx.generation_dimension_limits(version) == (minimum, maximum, step)
    landscape = [(960, 544), (1024, 576), (1344, 768)] if version == "h3" else [
        (1024, 576), (1280, 704), (1920, 1088)]
    square = [(768, 768)] if version == "h3" else [(576, 576), (704, 704), (1088, 1088)]
    for aspect, sizes in (("16:9", landscape), ("9:16", [(h, w) for w, h in landscape]), ("1:1", square)):
        presets = ltx.resolution_presets(version, aspect)
        assert list(presets.values()) == sizes and presets["recommended"] == sizes[-1]
        for width, height in presets.values():
            assert request(width, height).ltx_generation_width == width
    assert request(minimum, maximum).ltx_generation_height == maximum
    for width, height in ((None, minimum), (minimum, None), (minimum - step, minimum), (maximum + step, maximum),
                          (minimum, minimum + step // 2), (True, minimum), (float(minimum), minimum)):
        with pytest.raises(ValidationError):
            request(width, height)


@pytest.mark.parametrize("modified", ["other-upscaler", "disconnected-upscaler", "resized-output", "custom"])
def test_unknown_output_scale_cannot_silently_apply_clip_size(tmp_path, monkeypatch, modified):
    from app.models.schema import VideoParams

    graph = sample_graph()
    template = {"variant": {"model_version": "2.5"}, "api_prompt": graph,
                "parameters": {"fps": 24, "stage_1_seed": 1, "stage_2_seed": 2}}
    if modified == "custom":
        template["variant"] = {}
    elif modified == "other-upscaler":
        graph["upscaler_weight"]["inputs"]["model_name"] = "my-spatial-upscaler-x1.5.safetensors"
    elif modified == "disconnected-upscaler":
        graph["second_joint"]["inputs"]["video_latent"] = ["latent", 0]
    else:
        graph["extra"] = {"class_type": "ImageScale", "inputs": {"image": ["decode", 0]}}
        graph["video"]["inputs"]["images"] = ["extra", 0]
    assert ltx.generation_scale(template) is None
    params = VideoParams(video_subject="A paper boat", video_source="ltx", ltx_model_version="2.5")
    assert ltx.scene_resolution(params, template) == ((352, 640), None)
    params.ltx_generation_width, params.ltx_generation_height = 1920, 1088
    monkeypatch.setattr(ltx.utils, "task_dir", lambda *_: pytest.fail("invalid size reached scene planning"))
    with pytest.raises(ValueError, match="workflow"):
        ltx.plan_batch("boat", "A boat floats.", 5, params, template=template)


@pytest.mark.parametrize("legacy_decoder", [False, True])
def test_ltx23_official_and_legacy_decoders_keep_explicit_clip_size(legacy_decoder):
    from app.models.schema import VideoParams

    template = {"variant": {"model_version": "2.3"},
                "api_prompt": sample_graph("2.3", legacy_decoder=legacy_decoder)}
    params = VideoParams(video_subject="A paper boat", video_source="ltx", ltx_model_version="2.3",
                         ltx_generation_width=1024, ltx_generation_height=576)
    assert ltx.scene_resolution(params, template) == ((512, 288), 2)


@pytest.mark.parametrize("explicit,second_size,error", [
    (False, (640, 640), None), (True, (640, 640), "task requested 768×768"),
    (True, (768, 768), "same dimensions"),
])
def test_composition_checks_all_returned_clip_sizes(tmp_path, monkeypatch, explicit, second_size, error):
    from unittest.mock import Mock
    from app.models.schema import VideoParams
    from app.services import task

    sizes = {"ltx_generation_width": 768, "ltx_generation_height": 768} if explicit else {}
    params = VideoParams(video_subject="A boat", video_source="ltx", ltx_model_version="2.5", **sizes)
    clips = [Mock(size=(640, 640)), Mock(size=second_size)]
    monkeypatch.setattr(task.video, "VideoFileClip", Mock(side_effect=clips))
    combine = Mock()
    monkeypatch.setattr(task.video, "combine_videos", combine)
    monkeypatch.setattr(task.video, "generate_video", Mock())
    monkeypatch.setattr(task.sm.state, "update_task", Mock())
    monkeypatch.setattr(task.utils, "task_dir", lambda *_: str(tmp_path))
    arguments = ("boat", params, ["one.mp4", "two.mp4"], "audio.mp3", "", 3)
    if error:
        with pytest.raises(RuntimeError, match=error):
            task.generate_final_videos(*arguments)
        combine.assert_not_called()
    else:
        task.generate_final_videos(*arguments)
        assert combine.call_args.kwargs["video_size"] == (640, 640)
    for clip in clips:
        clip.close.assert_called_once()


def test_task_rejects_unrecognized_output_size_before_script_generation(monkeypatch):
    from app.models.schema import VideoParams
    from app.services import task

    params = VideoParams(video_subject="A boat", video_source="ltx", ltx_model_version="2.5",
                         ltx_generation_width=1920, ltx_generation_height=1088)
    monkeypatch.setattr(ltx, "request_template", lambda *_: {"variant": {"model_version": "2.5"}})
    monkeypatch.setattr(task.sm.state, "update_task", lambda *args, **kwargs: None)
    monkeypatch.setattr(task, "_mark_task_failed", lambda _, stage, error: (stage, error))
    monkeypatch.setattr(task, "generate_script", lambda *_: pytest.fail("unrecognized clip size reached the LLM"))
    stage, error = task.start("boat", params, stop_at="materials")
    assert stage == "preflight" and "recognized generated clip size" in error


def test_unversioned_workflow_remains_selectable_after_registering_h3(tmp_path):
    custom = tmp_path / "custom.json"
    custom.write_text('{"variant":{"name":"my-workflow"}}', encoding="utf-8")
    h3 = tmp_path / "h3.json"
    h3.write_text('{"variant":{"model_version":"h3"}}', encoding="utf-8")
    settings = {"enabled": True, "operator_script": __file__, "python_executable": sys.executable,
                "request_template": str(h3), "request_templates": {"h3": str(h3), "custom": str(custom)}}
    assert ltx.default_model_version(settings) == "h3"
    assert ltx.request_template("custom", settings)["variant"]["name"] == "my-workflow"
    assert ltx.validate_settings(settings, model_version="2.5")
    h3.unlink()
    assert ltx.validate_settings(settings, model_version="h3")


def test_h3_task_duration_defaults_bounds_and_frame_grid(tmp_path, monkeypatch):
    from app.models.schema import TaskVideoRequest
    from pydantic import ValidationError

    template = tmp_path / "request.json"
    template.write_text('{"variant":{"model_version":"h3"}}', encoding="utf-8")
    monkeypatch.setattr(ltx, "settings", lambda: {"request_template": str(template)})
    fields = {"video_subject": "A paper boat", "video_source": "ltx"}
    implicit = TaskVideoRequest(**fields)
    explicit = TaskVideoRequest(**fields, ltx_model_version="h3")
    assert implicit.ltx_model_version == "h3"
    assert implicit.ltx_scene_duration_seconds == explicit.ltx_scene_duration_seconds == 5
    assert TaskVideoRequest(**fields, ltx_model_version="2.3").ltx_scene_duration_seconds == 3
    assert TaskVideoRequest(**fields, ltx_model_version="custom").ltx_scene_duration_seconds == 3
    for seconds in (3, 16, 4.5):
        with pytest.raises(ValidationError):
            TaskVideoRequest(**fields, ltx_scene_duration_seconds=seconds)
        with pytest.raises(ValueError, match="4 to 15"):
            ltx.validate_scene_plan([{"duration_seconds": seconds}], 5, 1, "h3")
    frames = [ltx.frame_count_for(seconds, 24, "h3") for seconds in range(4, 16)]
    assert (frames[0], frames[1], frames[-1]) == (107, 124, 362)
    for seconds, count in zip(range(4, 16), frames):
        assert (count - 5) % 17 == 0 and seconds * 24 <= count < seconds * 24 + 17

def test_rest_approval_requires_each_reviewed_hash(approved_batch, monkeypatch):
    from fastapi.testclient import TestClient
    from app.asgi import get_application
    from app.services import state

    monkeypatch.setenv("MPT_API_KEY", "local-test-key")
    monkeypatch.setattr(state, "state", state.MemoryState())
    task_id = approved_batch["task_id"]
    state.state.update_task(task_id, state=1, params={"video_source": "ltx"}, phase="awaiting_approval")
    batch_path = ltx._batch_path(task_id)
    plan_path = batch_path.with_name("scene.json")

    def publish(plan):
        ltx._replace_json(plan_path, plan)
        ltx._replace_json(batch_path, {"task_id": task_id, "phase": "awaiting_approval",
                          "scenes": [{"index": 1, "plan_file": str(plan_path),
                                      "approval_hash": plan["approval_hash"]}]})

    api = TestClient(get_application(), headers={"x-api-key": "local-test-key"})
    route = f"/api/v1/tasks/{task_id}/ltx"
    first, second = (json.loads(Path(scene["plan_file"]).read_bytes()) for scene in approved_batch["scenes"])
    publish(first)
    old = api.get(route + "/plans").json()["data"]["scenes"][0]["plan"]["approval_hash"]
    publish(second)
    approval = {"scene_index": 1, "approval_hash": old}
    unchanged = batch_path.read_bytes()
    assert api.post(route + "/approvals").status_code == 400
    for items in ([approval], [], [approval, approval], [{"scene_index": True, "approval_hash": second["approval_hash"]}]):
        assert api.post(route + "/approvals", json={"approvals": items}).status_code == 409
        assert batch_path.read_bytes() == unchanged
    reviewed = api.get(route + "/plans").json()["data"]["scenes"][0]["plan"]["approval_hash"]
    response = api.post(route + "/approvals", json={"approvals": [{"scene_index": 1, "approval_hash": reviewed}]})
    assert response.status_code == 200
    assert ltx.load_batch(task_id)["scenes"][0]["submitted_approval_hash"] == reviewed
    assert state.state.get_task(task_id)["phase"] == "approved"


@pytest.mark.parametrize("broken", ["changed-graph", "nonfinite-plan", "nonfinite-batch"])
def test_rest_approval_rejects_changed_plan_or_batch(approved_batch, monkeypatch, broken):
    from fastapi.testclient import TestClient
    from app.asgi import get_application
    from app.services import state

    monkeypatch.setenv("MPT_API_KEY", "local-test-key")
    monkeypatch.setattr(state, "state", state.MemoryState())
    task_id = approved_batch["task_id"]
    state.state.update_task(task_id, params={"video_source": "ltx"}, phase="awaiting_approval")
    batch = {**approved_batch, "phase": "awaiting_approval"}
    first = batch["scenes"][0]
    if broken != "nonfinite-batch":
        plan_path = Path(first["plan_file"])
        plan = json.loads(plan_path.read_bytes())
        if broken == "changed-graph":
            plan["api_prompt"] = []
        else:
            plan["unexpected"] = float("nan")
        ltx._replace_json(plan_path, plan)
    else:
        first["render_duration_seconds"] = float("nan")
    batch_path = ltx._batch_path(task_id)
    ltx._replace_json(batch_path, batch)
    before = batch_path.read_bytes()
    api = TestClient(get_application(), headers={"x-api-key": "local-test-key"}, raise_server_exceptions=False)
    approvals = [{"scene_index": s["index"], "approval_hash": s["approval_hash"]} for s in batch["scenes"]]
    assert api.post(f"/api/v1/tasks/{task_id}/ltx/approvals", json={"approvals": approvals}).status_code == 409
    assert batch_path.read_bytes() == before
    assert state.state.get_task(task_id)["phase"] == "awaiting_approval"


@pytest.mark.parametrize("broken", ["missing-batch", "invalid-batch", "empty-batch", "missing-scene", "invalid-scene", "nonfinite-scene"])
def test_rest_plans_reports_unavailable_saved_data(tmp_path, monkeypatch, broken):
    from fastapi.testclient import TestClient
    from app.asgi import get_application
    from app.services import state

    monkeypatch.setattr(ltx.utils, "task_dir", lambda task_id="": str(tmp_path / task_id))
    monkeypatch.setenv("MPT_API_KEY", "local-test-key")
    monkeypatch.setattr(state, "state", state.MemoryState())
    state.state.update_task("scene-task", params={"video_source": "ltx"})
    batch_path = ltx._batch_path("scene-task")
    plan_path = batch_path.with_name("scene.json")
    if broken != "missing-batch":
        ltx._replace_json(batch_path, [] if broken == "invalid-batch" else {
            "task_id": "scene-task", "phase": "awaiting_approval",
            "scenes": [] if broken == "empty-batch" else [{"plan_file": str(plan_path), "approval_hash": "a" * 64}],
        })
    if broken == "invalid-scene":
        ltx._replace_json(plan_path, [])
    if broken == "nonfinite-scene":
        ltx._replace_json(plan_path, {"approval_hash": "a" * 64, "fps": float("nan")})
    client = TestClient(get_application(), headers={"x-api-key": "local-test-key"})
    assert client.get("/api/v1/tasks/scene-task/ltx/plans").status_code == 409


@pytest.mark.parametrize("contents", [None, "{", "[]", '{"approval_hash":"%s","fps":NaN}' % ("a" * 64)])
def test_rest_approval_returns_conflict_for_unreadable_plan(tmp_path, monkeypatch, contents):
    from fastapi.testclient import TestClient
    from app.asgi import get_application
    from app.services import state

    monkeypatch.setattr(ltx.utils, "task_dir", lambda task_id="": str(tmp_path / task_id))
    monkeypatch.setattr(ltx.utils, "storage_dir", lambda key: str(tmp_path / key))
    monkeypatch.setenv("MPT_API_KEY", "local-test-key")
    monkeypatch.setattr(state, "state", state.MemoryState())
    task_id = "approval-task"
    state.state.update_task(task_id, params={"video_source": "ltx"}, phase="awaiting_approval")
    batch_path = ltx._batch_path(task_id)
    plan_path = batch_path.with_name("scene.json")
    ltx._replace_json(batch_path, {"task_id": task_id, "phase": "awaiting_approval", "scenes": [
        {"index": 1, "plan_file": str(plan_path), "approval_hash": "a" * 64},
    ]})
    if contents is not None:
        plan_path.write_text(contents, encoding="utf-8")
    before = batch_path.read_bytes()
    api = TestClient(get_application(), headers={"x-api-key": "local-test-key"}, raise_server_exceptions=False)
    response = api.post(f"/api/v1/tasks/{task_id}/ltx/approvals",
                        json={"approvals": [{"scene_index": 1, "approval_hash": "a" * 64}]})
    assert response.status_code == 409
    assert batch_path.read_bytes() == before
    assert state.state.get_task(task_id)["phase"] == "awaiting_approval"


def test_redis_legacy_scene_parameters_use_the_local_script(tmp_path, monkeypatch):
    from app.models.schema import VideoParams
    from app.services import state, task_artifacts

    monkeypatch.setattr(ltx.utils, "task_dir", lambda task_id="": str(tmp_path / task_id))
    backend = state.RedisState.__new__(state.RedisState)
    backend._redis = Mock()
    params = VideoParams(video_subject="Paper boat", video_source="ltx", video_aspect="1:1")
    for mode, with_script, expected in [("python", True, "1:1"), ("json", False, "1:1"), ("python", False, None)]:
        task_id = f"task-{mode}-{with_script}"
        backend._redis.hgetall.return_value = {
            b"task_id": task_id.encode(), b"params": str(params.model_dump(mode=mode)).encode(),
        }
        if with_script:
            (tmp_path / task_id).mkdir()
            task_artifacts.write_script_data(task_id, {"params": params})
        restored = backend.get_task(task_id)["params"]
        assert restored.get("video_aspect") == expected
        assert restored.get("video_source") == ("ltx" if expected else None)


@pytest.mark.parametrize("case", ["missing-batch", "stale-batch", "scene-record", "duration-record", "terminal",
                                  "unsigned-terminal", "signed-terminal", "forged-terminal"])
def test_delete_keeps_unresolved_ownership_even_when_state_is_incomplete(tmp_path, monkeypatch, case):
    from app.services import state
    from ltx_operator_lib.state import OperatorState

    monkeypatch.setattr(ltx.utils, "task_dir", lambda task_id="": str(tmp_path / "tasks" / task_id))
    monkeypatch.setattr(ltx.utils, "storage_dir", lambda name: str(tmp_path / name))
    monkeypatch.setattr(state, "state", state.MemoryState())
    task_id = "uncertain"
    if case in {"missing-batch", "stale-batch"}:
        state.state.update_task(task_id, state=-1, phase="stop_unconfirmed")
    if case == "stale-batch":
        ltx._replace_json(ltx._batch_path(task_id), {"task_id": task_id, "phase": "approved", "scenes": []})
    if case in {"scene-record", "duration-record", "terminal", "unsigned-terminal", "signed-terminal", "forged-terminal"}:
        signed = case in {"signed-terminal", "forged-terminal"}
        folder = ("duration-predictions/attempt/runs/001" if signed or case == "duration-record" else "scenes/001/run")
        run = ltx._batch_path(task_id).parent / folder
        owned = {"prompt_id": "owned", "run_directory": str(run), "approval_hash": "a" * 64,
                 "endpoint": "http://127.0.0.1:9"}
        if signed:
            monkeypatch.setattr(ltx, "settings", lambda: {"python_executable": sys.executable,
                             "operator_script": str(ADDON / "runtime/scripts/ltx_operator.py")})
            envelope = OperatorState(tmp_path / "ltx-state").sign_record({**owned, "terminal_status": "cancelled"})
            if case == "forged-terminal":
                envelope["record"]["terminal_status"] = "completed"
        elif case == "terminal":
            envelope = OperatorState(tmp_path / "ltx-state").sign_record(owned)
        else:
            envelope = {"record": owned}
        ltx._replace_json(run / "run.json", envelope)
        status = "completed" if case in {"terminal", "unsigned-terminal"} else "queued"
        ltx._replace_json(run / "report.json", {"prompt_id": "owned", "status": status})
    if case in {"terminal", "signed-terminal"}:
        with ltx.guard_task_deletion(task_id):
            pass
    else:
        with pytest.raises(ltx.RenderBusy), ltx.guard_task_deletion(task_id):
            pytest.fail("An unresolved task must keep its ownership records")


@pytest.mark.parametrize("legacy", [False, True])
def test_reused_scene_must_match_the_original_output(tmp_path, monkeypatch, legacy):
    monkeypatch.setattr(ltx.utils, "task_dir", lambda task_id: str(tmp_path))
    clip = tmp_path / "scene.mp4"
    clip.write_bytes(b"original test bytes")
    expected = hashlib.sha256(clip.read_bytes()).hexdigest()
    scene = {"approval_hash": "a" * 64, "output_file": str(clip), "run_directory": str(tmp_path / "run")}
    if legacy:
        ltx._replace_json(tmp_path / "run/report.json", {"approval_hash": scene["approval_hash"], "status": "completed",
                          "outputs": [{"path": str(clip), "content_sha256": expected}]})
    else:
        scene["output_sha256"] = expected
    batch = {"task_id": "test", "phase": "complete", "scenes": [scene]}
    ltx._replace_json(ltx._batch_path("test"), batch)
    with ltx.composition_materials("test", batch) as (originals, inputs):
        assert originals == [str(clip)]
        clip.write_bytes(b"changed test bytes")
        snapshot = Path(inputs[0])
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == expected
    assert not snapshot.exists()
    with pytest.raises(ltx.LtxError, match="changed"):
        ltx.completed_materials("test")
    if legacy:
        (tmp_path / "run/report.json").unlink()
        with pytest.raises(ltx.LtxError, match="original output report"):
            ltx.completed_materials("test")


def test_only_one_queued_or_active_composition_can_own_a_task(tmp_path, monkeypatch):
    from app.services import task

    monkeypatch.setattr(ltx.utils, "task_dir", lambda task_id: str(tmp_path))
    monkeypatch.setattr(ltx.utils, "storage_dir", lambda name: str(tmp_path / name))
    task_id = "queue-test"
    batch = {"task_id": task_id, "phase": "complete", "scenes": []}
    ltx._replace_json(ltx._batch_path(task_id), batch)
    queued = []
    with pytest.raises(RuntimeError, match="queue full"):
        task.queue_ltx_render(task_id, Mock(side_effect=RuntimeError("queue full")))
    assert ltx.load_batch(task_id) == batch
    task.queue_ltx_render(task_id, queued.append)
    with pytest.raises(ltx.RenderBusy):
        task.queue_ltx_render(task_id, queued.append)
    assert len(queued) == 1
    with pytest.raises(ltx.RenderBusy), ltx.render_slot(task_id, "stale"):
        pytest.fail("stale ticket acquired the slot")
    with ltx.render_slot(task_id, queued[0]):
        with pytest.raises(ltx.RenderBusy):
            task.queue_ltx_render(task_id, queued.append)
    assert ltx.load_batch(task_id) == batch
    task.queue_ltx_render(task_id, queued.append)
    assert len(queued) == 2 and queued[0] != queued[1]
    ltx.release_reservation(task_id, queued[0])
    with ltx.render_slot(task_id, queued[1]):
        pass

@pytest.mark.parametrize("malformed", ["params", "signature", "unknown_function", "requeue"])
def test_discarded_queue_entry_may_only_fail_its_own_reservation(tmp_path, monkeypatch, malformed):
    from app.controllers.manager.redis_manager import RedisTaskManager
    from app.models import const
    from app.models.schema import VideoParams
    from app.services import state, task

    monkeypatch.setattr(ltx.utils, "task_dir", lambda task_id="": str(tmp_path / "tasks" / task_id))
    monkeypatch.setattr(ltx.utils, "storage_dir", lambda name: str(tmp_path / name))
    monkeypatch.setattr(state, "state", state.MemoryState())
    task_id = "queue-discard"
    ltx._replace_json(ltx._batch_path(task_id), {"task_id": task_id, "phase": "complete", "scenes": []})
    state.state.update_task(task_id, state=const.TASK_STATE_COMPLETE, phase="complete", videos=["final.mp4"])
    manager = RedisTaskManager(0, "redis://localhost:6379")
    manager.redis_client = Mock()
    accepted = []
    original = state.state.get_task(task_id)
    with pytest.raises(RuntimeError, match="queue full"):
        task.queue_ltx_render(task_id, Mock(side_effect=RuntimeError("queue full")))
    assert state.state.get_task(task_id) == original
    task.queue_ltx_render(task_id, accepted.append)
    assert task.is_task_busy(state.state.get_task(task_id))
    entry = {"func": task.render_ltx, "kwargs": {"task_id": task_id, "ticket": accepted[0],
             "audio_file": "audio.mp3", "subtitle_path": "", "params": VideoParams(video_subject="Paper boat")}}

    def discard():
        if malformed == "requeue":
            manager.redis_client.rpush.side_effect = OSError("lost queue")
            with pytest.raises(OSError, match="lost queue"):
                manager.enqueue(entry)
            return
        data = json.loads(manager._serialize_task(entry))
        if malformed == "signature":
            del data["kwargs"]["audio_file"]
        elif malformed == "params":
            data["kwargs"]["params"]["video_count"] = 0
        else:
            data["func"] = "old_worker_name"
        manager.redis_client.lpop.side_effect = [json.dumps(data), None]
        assert manager.dequeue() is None

    discard()
    failed = state.state.get_task(task_id)
    assert failed["state"] == const.TASK_STATE_FAILED and failed["videos"] == original["videos"]
    task.queue_ltx_render(task_id, accepted.append)
    queued = state.state.get_task(task_id)
    assert queued["phase"] == "queued"
    discard()
    assert state.state.get_task(task_id) == queued
    with ltx.render_slot(task_id, accepted[-1]):
        discard()
        assert state.state.get_task(task_id) == queued


def test_queued_and_active_recomposition_block_rest_deletion(tmp_path, monkeypatch):
    from app.asgi import get_application
    from app.models import const
    from app.services import state, task
    from fastapi.testclient import TestClient

    monkeypatch.setattr(ltx.utils, "task_dir", lambda task_id="": str(tmp_path / "tasks" / task_id))
    monkeypatch.setattr(ltx.utils, "storage_dir", lambda name: str(tmp_path / name))
    monkeypatch.setenv("MPT_API_KEY", "local-test-key")
    monkeypatch.setattr(state, "state", state.MemoryState())
    task_id = "queue-delete"
    ltx._replace_json(ltx._batch_path(task_id), {"task_id": task_id, "phase": "complete", "scenes": []})
    state.state.update_task(task_id, state=const.TASK_STATE_COMPLETE, phase="complete")
    api = TestClient(get_application(), headers={"x-api-key": "local-test-key"})
    tickets = []
    task.queue_ltx_render(task_id, tickets.append)

    def delete():
        return api.delete(f"/api/v1/tasks/{task_id}")

    assert delete().status_code == 409
    with ltx.render_slot(task_id, tickets[0]):
        state.state.patch_task(task_id, state=const.TASK_STATE_COMPLETE, phase="complete")
        assert delete().status_code == 409
    assert delete().status_code == 200
    assert state.state.get_task(task_id) is None and not (tmp_path / "tasks" / task_id).exists()


@pytest.mark.parametrize("phase,status", [
    ("awaiting_approval", 200), ("approved", 200), ("editing", 200), ("planning", 409),
])
def test_only_idle_ltx_review_can_be_deleted_through_api(tmp_path, monkeypatch, phase, status):
    from app.asgi import get_application
    from app.services import state
    from fastapi.testclient import TestClient

    monkeypatch.setattr(ltx.utils, "task_dir", lambda task_id="": str(tmp_path / "tasks" / task_id))
    monkeypatch.setattr(ltx.utils, "storage_dir", lambda name: str(tmp_path / name))
    monkeypatch.setenv("MPT_API_KEY", "local-test-key")
    monkeypatch.setattr(state, "state", state.MemoryState())
    task_id = "discard-local-scenes"
    ltx._create_json(ltx._batch_path(task_id), {"task_id": task_id, "phase": phase, "scenes": []})
    state.state.update_task(task_id, phase=phase, params={"video_source": "ltx"})
    api = TestClient(get_application(), headers={"x-api-key": "local-test-key"})

    assert api.delete(f"/api/v1/tasks/{task_id}").status_code == status
    assert (state.state.get_task(task_id) is None) is (status == 200)
    assert (tmp_path / "tasks" / task_id).exists() is (status == 409)


def test_only_dead_in_memory_queue_with_missing_state_releases_history(tmp_path, monkeypatch):
    from app.services import state, task

    monkeypatch.setattr(ltx.utils, "task_dir", lambda task_id="": str(tmp_path / "tasks" / task_id))
    monkeypatch.setattr(ltx.utils, "storage_dir", lambda name: str(tmp_path / name))
    monkeypatch.setattr(state, "state", state.MemoryState())
    task_id = "old-completed-task"
    ltx._replace_json(ltx._batch_path(task_id), {"task_id": task_id, "phase": "complete", "scenes": []})
    claim = ltx._reservation_path(task_id)
    task.queue_ltx_render(task_id, lambda ticket: None)
    pending = ltx._read_json(claim)
    with pytest.raises(ltx.RenderBusy), ltx.guard_task_deletion(task_id):
        pytest.fail("the live local submitter still owns this task")

    # A dead submitter can leave either a lost local queue or a persisted Redis entry.
    monkeypatch.setattr(task, "_ltx_queue_owner_alive", lambda owner: False)
    for overrides in ({"persistent": True}, {"persistent": None}):
        ltx._replace_json(claim, {**pending, **overrides})
        with pytest.raises(ltx.RenderBusy), ltx.guard_task_deletion(task_id):
            pytest.fail("a durable or unidentified queue must stay protected")
        assert claim.is_file()
    ltx._replace_json(claim, pending)
    state.state.update_task(task_id, phase="queued")
    with pytest.raises(ltx.RenderBusy), ltx.guard_task_deletion(task_id):
        pytest.fail("the task still has live state")
    state.state.delete_task(task_id)
    with ltx.guard_task_deletion(task_id):
        assert not claim.exists()
    assert ltx._batch_path(task_id).is_file()


@pytest.mark.parametrize("batch_phase", ["stop_unconfirmed", "failed", "approved", "corrupt", "state-write-fails"])
def test_task_failure_preserves_remote_stop_uncertainty(tmp_path, monkeypatch, batch_phase):
    from contextlib import contextmanager
    from app.models import const
    from app.models.schema import VideoParams
    from app.services import state, task

    monkeypatch.setattr(ltx.utils, "task_dir", lambda task_id="": str(tmp_path / task_id))
    monkeypatch.setattr(state, "state", state.MemoryState())
    task_id = "failed-local-scenes"
    batch = {"task_id": task_id, "phase": "approved", "scenes": []}
    batch_path = ltx._batch_path(task_id)
    ltx._replace_json(batch_path, batch)
    failures = []
    if batch_phase == "state-write-fails":
        original_update = state.state.update_task
        def unavailable(task_id, **values):
            if values.get("state") == const.TASK_STATE_FAILED:
                raise OSError("Task state unavailable")
            return original_update(task_id, **values)
        monkeypatch.setattr(state.state, "update_task", unavailable)
        def unavailable_patch(*args, **kwargs):
            raise OSError("Task state unavailable")
        monkeypatch.setattr(state.state, "patch_task", unavailable_patch)

    @contextmanager
    def interrupted(*args):
        if batch_phase == "corrupt":
            batch_path.write_text("{", encoding="utf-8")
        else:
            ltx._replace_json(batch_path, {**batch, "phase": batch_phase})
        error = ltx.LtxError("Original operator failure")
        if batch_phase in {"stop_unconfirmed", "corrupt", "state-write-fails"}:
            error.remote_stop_unconfirmed = True
        failures.append(error)
        raise error
        yield

    monkeypatch.setattr(ltx, "composition_materials", interrupted)
    with pytest.raises(ltx.LtxError, match="^Original operator failure$") as raised:
        task._finish_ltx(task_id, VideoParams(video_subject="A paper boat", video_source="ltx"), "", "")
    assert raised.value is failures[0]
    if batch_phase == "state-write-fails":
        assert raised.value.remote_stop_unconfirmed is True
        return
    saved = state.state.get_task(task_id)
    assert saved["state"] == const.TASK_STATE_FAILED
    assert saved["phase"] == ("stop_unconfirmed" if batch_phase in {"stop_unconfirmed", "corrupt"} else "failed")
    assert saved["error"] == "Original operator failure"


@pytest.fixture
def scene_editor(approved_batch, tmp_path, monkeypatch):
    """Use the real operator editor behind a serialized in-process CLI boundary."""
    from types import SimpleNamespace
    from urllib.parse import unquote, urlsplit
    from app.services import state
    from ltx_operator_lib.api import ComfyHttpError
    from ltx_operator_lib.cli import build_parser, _dispatch, _jsonable

    monkeypatch.setattr(state, "state", state.MemoryState())
    monkeypatch.setattr(ltx.utils, "storage_dir", lambda name: str(tmp_path / name))
    state.state.update_task("neutral", phase="approved", params={"video_clip_speed": 2})
    batch = ltx.load_batch("neutral")
    batch.update(model_version="2.5", timing={"audio_duration_seconds": 4, "max_playback_seconds": 10})
    ltx._replace_json(ltx._batch_path("neutral"), batch)
    files, requests, commands = {}, [], []
    control = SimpleNamespace(files=files, requests=requests, commands=commands, mutate_result=None)
    graph = sample_graph()

    class Client:
        endpoint = "http://127.0.0.1:8188"

        def __init__(self):
            self.transport = self

        def request(self, method, path, json_body=None):
            requests.append((method, path))
            path = unquote(urlsplit(path).path)
            if method == "GET" and path == "/users":
                return {"storage": "server", "migrated": True}
            if method == "GET" and path == "/extensions":
                return ["/extensions/mpt_scene_editor/editor.js"]
            assert path.startswith("/userdata/"), f"Unexpected editor endpoint: {method} {path}"
            key = path.removeprefix("/userdata/")
            if method == "POST":
                files[key] = json.loads(json.dumps(json_body))
                return {}
            assert method == "GET"
            if key not in files:
                raise ComfyHttpError(404, "No saved draft")
            return json.loads(json.dumps(files[key]))

        def system_stats(self, **kwargs):
            return {"system": {}}

        def object_info(self, **kwargs):
            return fixture_object_info(tmp_path / "ComfyUI", graph)

        def queue(self, **kwargs):
            return {"queue_running": [], "queue_pending": []}

    client = Client()

    def invoke(arguments, timeout):
        commands.append(arguments[:])
        try:
            result = _dispatch(build_parser().parse_args(arguments), lambda _: client)
        except Exception as exc:
            raise ltx.LtxError(str(exc)) from exc
        if control.mutate_result is not None:
            control.mutate_result(arguments[0], result)
        return json.loads(json.dumps(_jsonable(result)))

    def save(index, **overrides):
        batch = ltx.load_batch("neutral")
        session = batch["editor"]["session"]
        manifest = files[f"mpt-scenes/{session}/manifest.json"]
        exported = next(scene for scene in manifest["scenes"] if scene["index"] == index)
        original = json.loads(Path(batch["scenes"][index - 1]["plan_file"]).read_bytes())
        changed = prepare_request(original, overrides)
        files[f"mpt-scenes/{session}/scene-{index:03d}.json"] = {
            "schema_version": 1, "session": session, "revision": manifest["revision"], "index": index,
            "base_approval_hash": exported["base_approval_hash"], "api_prompt": changed["api_prompt"]}

    control.save = save
    monkeypatch.setattr(ltx, "run_cli", invoke)
    return control


def test_scene_editor_adopts_all_persisted_plans_and_recalculates_timing(approved_batch, scene_editor, tmp_path):
    from ltx_operator_lib.planning import verify_approved_plan

    original_bytes = [Path(scene["plan_file"]).read_bytes() for scene in approved_batch["scenes"]]
    assert ltx.open_scene_editor("neutral").startswith("http://127.0.0.1:8188/?mpt_edit=")
    scene_editor.save(1, prompt="First manually edited boat", width=640, height=352, frame_count=105)
    scene_editor.save(2, prompt="Second manually edited boat", width=704, height=384, frame_count=121)
    result = ltx.apply_scene_editor("neutral")
    assert result == ltx.load_batch("neutral")
    assert result["phase"] == "awaiting_approval" and result["edited_in_comfyui"] is True
    assert "editor" not in result and result["timing"]["duration_mode"] == "manual"
    render_seconds = (105 + 121) / 24
    assert result["timing"]["planned_render_seconds"] == pytest.approx(render_seconds)
    assert result["timing"]["planned_playback_seconds"] == pytest.approx(render_seconds / 2)
    for scene, original, dimensions, frames in zip(result["scenes"], approved_batch["scenes"],
                                                   [(640, 352), (704, 384)], [105, 121], strict=True):
        assert scene["status"] == "planned" and scene["submitted_approval_hash"] == ""
        assert scene["approval_hash"] != original["approval_hash"]
        assert (scene["first_pass_width"], scene["first_pass_height"]) == dimensions
        assert (scene["generation_width"], scene["generation_height"]) == tuple(value * 2 for value in dimensions)
        assert scene["render_duration_seconds"] == pytest.approx(frames / 24)
        assert scene["playback_duration_seconds"] == pytest.approx(frames / 48)
        plan = json.loads(Path(scene["plan_file"]).read_bytes())
        assert plan["prompt"] == scene["visual_prompt"] and plan["parameters"]["frame_count"] == frames
        verify_approved_plan(plan, scene["approval_hash"])
    assert original_bytes == [Path(scene["plan_file"]).read_bytes() for scene in approved_batch["scenes"]]
    assert ltx.sm.state.get_task("neutral")["phase"] == "awaiting_approval"
    assert not list((tmp_path / "ltx-state/receipts").glob("*.json"))
    assert [args[0] for args in scene_editor.commands] == ["export-editor", "import-editor"]
    assert all(method == "GET" or path.startswith("/userdata/") for method, path in scene_editor.requests)


@pytest.mark.parametrize("changed,message", [("session", "invalid edited"), ("revision", "invalid edited"),
                                             ("submissions", "invalid edited"), ("scene", "match this batch"),
                                             ("scene_bool", "match this batch"),
                                             ("base_hash", "original plan"), ("plan_path", "original plan"),
                                             ("request_path", "original plan"),
                                             ("plan_hash", "hash mismatch")])
def test_scene_editor_rejects_unbound_cli_results_atomically(approved_batch, scene_editor, changed, message):
    ltx.open_scene_editor("neutral")
    old_batch = ltx._batch_path("neutral").read_bytes()
    old_state = ltx.sm.state.get_task("neutral")

    def corrupt(command, result):
        if command != "import-editor":
            return
        if changed in {"session", "revision"}:
            result[changed] = "c" * (32 if changed == "session" else 64)
        elif changed == "submissions":
            result["submissions"] = 1
        elif changed == "scene":
            result["scenes"][1]["scene_index"] = 3
        elif changed == "scene_bool":
            result["scenes"][0]["scene_index"] = True
        elif changed == "base_hash":
            result["scenes"][1]["base_approval_hash"] = "c" * 64
        elif changed == "plan_path":
            result["scenes"][1]["plan_file"] = approved_batch["scenes"][1]["plan_file"]
        elif changed == "request_path":
            result["scenes"][1]["request_file"] = approved_batch["scenes"][1]["plan_file"]
        else:
            result["scenes"][1]["approval_hash"] = "c" * 64

    scene_editor.mutate_result = corrupt
    with pytest.raises(ltx.LtxError, match=message):
        ltx.apply_scene_editor("neutral")
    assert ltx._batch_path("neutral").read_bytes() == old_batch
    assert ltx.sm.state.get_task("neutral") == old_state
    assert all(scene["status"] == "approved" for scene in ltx.load_batch("neutral")["scenes"])


def test_scene_editor_later_scene_shortage_retains_original_batch(approved_batch, scene_editor):
    ltx.open_scene_editor("neutral")
    batch = ltx.load_batch("neutral")
    batch["timing"]["audio_duration_seconds"] = 3
    ltx._replace_json(ltx._batch_path("neutral"), batch)
    scene_editor.save(1, prompt="A valid long first scene", frame_count=105)
    scene_editor.save(2, prompt="A second scene too short for narration", frame_count=1)
    before = ltx._batch_path("neutral").read_bytes()
    original_bytes = [Path(scene["plan_file"]).read_bytes() for scene in batch["scenes"]]
    with pytest.raises(ltx.LtxError, match="narration needs"):
        ltx.apply_scene_editor("neutral")
    assert ltx._batch_path("neutral").read_bytes() == before
    assert original_bytes == [Path(scene["plan_file"]).read_bytes() for scene in batch["scenes"]]
    assert ltx.sm.state.get_task("neutral")["phase"] == "editing"


def test_scene_editor_stale_saved_draft_retains_current_batch(approved_batch, scene_editor):
    ltx.open_scene_editor("neutral")
    scene_editor.save(2, prompt="A draft from a previous revision", frame_count=105)
    batch = ltx.load_batch("neutral")
    session = batch["editor"]["session"]
    scene_editor.files[f"mpt-scenes/{session}/scene-002.json"]["revision"] = "c" * 64
    before = ltx._batch_path("neutral").read_bytes()
    with pytest.raises(ltx.LtxError, match="stale"):
        ltx.apply_scene_editor("neutral")
    assert ltx._batch_path("neutral").read_bytes() == before
    assert ltx.sm.state.get_task("neutral")["phase"] == "editing"


def test_scene_editor_close_and_reopen_preserve_saved_drafts_and_session(approved_batch, scene_editor):
    url = ltx.open_scene_editor("neutral")
    scene_editor.save(1, prompt="Retain this draft", frame_count=105)
    first = ltx.load_batch("neutral")
    session = first["editor"]["session"]
    draft = json.loads(json.dumps(scene_editor.files[f"mpt-scenes/{session}/scene-001.json"]))
    ltx.close_scene_editor("neutral")
    closed = ltx.load_batch("neutral")
    assert closed["phase"] == "awaiting_approval" and closed["editor"] == first["editor"]
    assert closed["scenes"] == approved_batch["scenes"]
    assert ltx.open_scene_editor("neutral") == url
    assert ltx.load_batch("neutral")["editor"] == first["editor"]
    assert scene_editor.files[f"mpt-scenes/{session}/scene-001.json"] == draft
    ltx.apply_scene_editor("neutral")
    assert ltx.load_batch("neutral")["scenes"][0]["visual_prompt"] == "Retain this draft"


@pytest.mark.parametrize("operation", ["open_scene_editor", "apply_scene_editor", "close_scene_editor"])
@pytest.mark.parametrize("busy", ["reservation", "queued", "rendering", "stop_unconfirmed"])
def test_scene_editor_never_changes_queued_or_active_batches(approved_batch, scene_editor, operation, busy):
    ltx.open_scene_editor("neutral")
    before = ltx._batch_path("neutral").read_bytes()
    calls = len(scene_editor.commands)
    if busy == "reservation":
        ltx._replace_json(ltx._reservation_path("neutral"), {"ticket": "owned-fixture"})
    else:
        ltx.sm.state.patch_task("neutral", phase=busy)
    with pytest.raises(ltx.RenderBusy):
        getattr(ltx, operation)("neutral")
    assert ltx._batch_path("neutral").read_bytes() == before
    assert len(scene_editor.commands) == calls


@pytest.mark.parametrize("failure", ["render", "compose"])
def test_failed_retake_keeps_old_video_and_can_start_a_fresh_run(approved_batch, worker, tmp_path, monkeypatch, failure):
    from app.models import const
    from app.models.schema import VideoParams
    from app.services import execution_control, state, task, webui_task
    from ltx_operator_lib.cli import _dispatch, build_parser

    task_id = "neutral"
    original_clips = ltx.render_batch(task_id)
    original_batch = ltx._batch_path(task_id).read_bytes()
    final = tmp_path / task_id / "final-1.mp4"
    final.write_bytes(b"previous complete video")
    monkeypatch.setattr(state, "state", state.MemoryState())
    state.state.update_task(task_id, state=const.TASK_STATE_COMPLETE, phase="complete", videos=[str(final)])
    monkeypatch.setattr(ltx, "run_cli", lambda args, timeout: _dispatch(build_parser().parse_args(args), lambda _: worker))
    monkeypatch.setattr(task, "_material_duration", lambda _: 5.0)
    monkeypatch.setattr(task.voice, "get_audio_duration", lambda _: 8.0)
    params = VideoParams(video_subject="Paper boat", video_source="ltx", voice_name="en-US-AnaNeural")
    selection = (2, approved_batch["scenes"][1]["approval_hash"], "A different shot of the boat.", "99")

    def replacement_history(prompt_id, **kwargs):
        worker.terminal[prompt_id] = "completed"
        key, = (key for key, node in worker.submissions[-1].items() if node["class_type"] == "SaveVideo")
        prefix = worker.submissions[-1][key]["inputs"]["filename_prefix"]
        filename = f"{prefix}_00001.mp4"
        (tmp_path / "ComfyUI" / "output" / filename).write_bytes(b"new scene fixture")
        return {prompt_id: {"outputs": {key: {"images": [{"filename": filename, "type": "output"}]}},
                            "status": {"completed": True}}}

    worker.completed_history = replacement_history
    compose = Mock(side_effect=RuntimeError("composition failed"))
    monkeypatch.setattr(task, "generate_final_videos", compose)
    if failure == "render":
        worker.failed_id = "3"
    with pytest.raises(RuntimeError):
        task.render_ltx(task_id, params, "audio.wav", "", retake=selection)
    assert compose.call_count == int(failure == "compose")
    assert len(worker.submissions) == 3
    assert ltx._batch_path(task_id).read_bytes() == original_batch
    assert ltx.completed_materials(task_id) == original_clips
    assert not ltx.latest_retake_video(final.parent)
    assert final.read_bytes() == b"previous complete video"
    saved = state.state.get_task(task_id)
    assert saved["state"] == const.TASK_STATE_COMPLETE and saved["videos"] == [str(final)]
    assert saved["retake_error"]

    worker.failed_id = None
    visible_video = str(final)

    def compose_retake(*args, output_prefix, **kwargs):
        assert final.read_bytes() == b"previous complete video"
        assert state.state.get_task(task_id)["videos"] == [visible_video]
        assert [Path(clip).read_bytes() for clip in args[2]] == [b"neutral video fixture", b"new scene fixture"]
        video = final.with_name(f"{output_prefix}final-1.mp4")
        combined = final.with_name(f"{output_prefix}combined-1.mp4")
        for path in (video, combined):
            path.write_bytes(b"new complete video")
        return [str(video)], [str(combined)], []

    monkeypatch.setattr(task, "generate_final_videos", compose_retake)
    token = execution_control.new_token(task_id)
    replace_json = ltx._replace_json

    def stop_after_commit(path, value):
        replace_json(path, value)
        if path == ltx._batch_path(task_id) and value.get("retake_video"):
            token.event.set()

    monkeypatch.setattr(ltx, "_replace_json", stop_after_commit)
    result = webui_task._run_generation(task_id, params, False, cancellation_token=token,
                                        ltx_audio_file="audio.wav", ltx_retake=selection)
    assert len(worker.submissions) == 4
    first_save = next(node for node in worker.submissions[2].values() if node["class_type"] == "SaveVideo")
    second_save = next(node for node in worker.submissions[3].values() if node["class_type"] == "SaveVideo")
    assert first_save["inputs"]["filename_prefix"] != second_save["inputs"]["filename_prefix"]
    assert result["materials"][0] == original_clips[0] and result["materials"][1] != original_clips[1]
    assert Path(original_clips[1]).read_bytes() == b"neutral video fixture"
    assert final.read_bytes() == b"previous complete video"
    assert state.state.get_task(task_id)["videos"] == result["videos"]
    assert state.state.get_task(task_id)["retake_error"] is None
    assert ltx.latest_retake_video(final.parent) == result["videos"][0]
    visible_video = result["videos"][0]
    previous_bytes = Path(visible_video).read_bytes()

    def failed_recompose(*args, output_prefix, **kwargs):
        final.with_name(f"{output_prefix}final-1.mp4").write_bytes(b"unfinished revision")
        raise RuntimeError("recomposition failed")

    monkeypatch.setattr(task, "generate_final_videos", failed_recompose)
    with pytest.raises(RuntimeError, match="recomposition failed"):
        task.render_ltx(task_id, params, "audio.wav", "")
    assert ltx.latest_retake_video(final.parent) == visible_video
    assert Path(visible_video).read_bytes() == previous_bytes
    monkeypatch.setattr(task, "generate_final_videos", compose_retake)
    recomposed = task.render_ltx(task_id, params, "audio.wav", "")
    assert recomposed["videos"] != [visible_video]
    assert ltx.latest_retake_video(final.parent) == recomposed["videos"][0]
    assert Path(visible_video).read_bytes() == previous_bytes
    assert final.read_bytes() == b"previous complete video"
    with pytest.raises(ltx.LtxError, match="changed"):
        task.render_ltx(task_id, params, "audio.wav", "", retake=selection)
    assert len(worker.submissions) == 4
    assert ltx.latest_retake_video(final.parent) == recomposed["videos"][0]


@pytest.mark.parametrize("fail_composition", [False, True])
def test_first_recomposition_keeps_previous_video_and_paid_music(worker, tmp_path, monkeypatch, fail_composition):
    from app.models import const
    from app.models.schema import VideoParams
    from app.services import state, task

    task_id = "neutral"
    originals = ltx.render_batch(task_id)
    directory = tmp_path / task_id
    final = directory / "final-1.mp4"
    final.write_bytes(b"previous complete video")
    music = directory / ("sonilo-bgm-1" + task._VIDEO_MUSIC_PROVIDERS["sonilo"]["suffix"])
    music.write_bytes(b"previous paid music")
    monkeypatch.setattr(state, "state", state.MemoryState())
    state.state.update_task(task_id, state=const.TASK_STATE_COMPLETE, phase="complete", videos=[str(final)])
    monkeypatch.setattr(task, "_material_duration", lambda _: 5.0)
    monkeypatch.setattr(task, "_ltx_material_size", lambda *args, **kwargs: (64, 64))
    monkeypatch.setattr(task.voice, "get_audio_duration", lambda _: 8.0)
    monkeypatch.setattr(task.video, "combine_videos", lambda **kw: Path(kw["combined_video_path"]).write_bytes(b"combined"))
    def compose(**kwargs):
        Path(kwargs["output_file"]).write_bytes(b"new composed video")
        if fail_composition:
            raise RuntimeError("composition failed")
        assert kwargs["bgm_file_override"] == str(music)
        return True
    paid_request = Mock()
    monkeypatch.setattr(task.video, "generate_video", compose)
    monkeypatch.setattr(task._VIDEO_MUSIC_PROVIDERS["sonilo"]["service"], "generate_bgm", paid_request)
    params = VideoParams(video_subject="Paper boat", video_source="ltx", voice_name="en-US-AnaNeural",
                         bgm_type="sonilo", bgm_volume=.5)

    def recompose(ticket):
        return task.render_ltx(task_id, params, "audio.wav", "", ticket=ticket)

    if fail_composition:
        with pytest.raises(RuntimeError, match="composition failed"):
            task.queue_ltx_render(task_id, recompose)
        assert not ltx.latest_retake_video(directory)
        previous = state.state.get_task(task_id)
        assert previous["videos"] == [str(final)]
        assert previous["state"] == const.TASK_STATE_COMPLETE and previous["phase"] == "complete"
    else:
        task.queue_ltx_render(task_id, recompose)
        saved = state.state.get_task(task_id)
        assert saved["videos"] != [str(final)]
        assert ltx.latest_retake_video(directory) == saved["videos"][0]
    assert final.read_bytes() == b"previous complete video"
    assert music.read_bytes() == b"previous paid music"
    paid_request.assert_not_called()
    assert ltx.completed_materials(task_id) == originals
    assert len(worker.submissions) == 2


def test_first_composition_retry_reuses_paid_music(worker, tmp_path, monkeypatch):
    from app.models import const
    from app.models.schema import VideoParams
    from app.services import state, task

    task_id = "neutral"
    ltx.render_batch(task_id)
    directory = tmp_path / task_id
    final = directory / "final-1.mp4"
    provider = task._VIDEO_MUSIC_PROVIDERS["sonilo"]
    music = directory / ("sonilo-bgm-1" + provider["suffix"])
    monkeypatch.setattr(state, "state", state.MemoryState())
    monkeypatch.setattr(task, "_material_duration", lambda _: 5.0)
    monkeypatch.setattr(task, "_ltx_material_size", lambda *args, **kwargs: (64, 64))
    monkeypatch.setattr(task.voice, "get_audio_duration", lambda _: 8.0)
    monkeypatch.setattr(task.video, "combine_videos", lambda **kw: Path(kw["combined_video_path"]).write_bytes(b"combined"))
    paid_request = Mock(side_effect=lambda **kw: Path(kw["output_path"]).write_bytes(b"paid music"))
    monkeypatch.setattr(provider["service"], "generate_bgm", paid_request)
    params = VideoParams(video_subject="Paper boat", video_source="ltx", voice_name="en-US-AnaNeural",
                         bgm_type="sonilo", bgm_volume=.5)

    def recompose(ticket):
        return task.render_ltx(task_id, params, "audio.wav", "", ticket=ticket)

    failed = Mock(side_effect=RuntimeError("composition failed"))
    monkeypatch.setattr(task.video, "generate_video", failed)
    with pytest.raises(RuntimeError, match="composition failed"):
        task.queue_ltx_render(task_id, recompose)
    assert not final.exists() and music.read_bytes() == b"paid music"
    paid_request.assert_called_once()

    def compose(**kwargs):
        assert kwargs["bgm_file_override"] == str(music)
        Path(kwargs["output_file"]).write_bytes(b"composed video")
        return True

    monkeypatch.setattr(task.video, "generate_video", compose)
    task.queue_ltx_render(task_id, recompose)
    assert state.state.get_task(task_id)["state"] == const.TASK_STATE_COMPLETE
    assert final.is_file()
    paid_request.assert_called_once()


@pytest.mark.parametrize("outage", ["transient", "restart"])
def test_completed_revision_is_published_after_state_write_failure(worker, tmp_path, monkeypatch, outage):
    from app.models import const
    from app.models.schema import VideoParams
    from app.services import state, task

    task_id = "neutral"
    monkeypatch.setattr(ltx.utils, "task_dir", lambda task_id="": str(tmp_path / task_id))
    ltx.render_batch(task_id)
    directory = tmp_path / task_id
    old = directory / f"retake-{'a' * 32}-final-1.mp4"
    old.write_bytes(b"previous complete revision")
    batch = ltx.load_batch(task_id)
    ltx._replace_json(ltx._batch_path(task_id), {**batch, "retake_video": old.name})
    memory = state.MemoryState()
    monkeypatch.setattr(state, "state", memory)
    memory.update_task(task_id, state=const.TASK_STATE_COMPLETE, phase="complete", videos=[str(old)])
    monkeypatch.setattr(task, "_material_duration", lambda _: 5.0)
    monkeypatch.setattr(task.voice, "get_audio_duration", lambda _: 8.0)
    params = VideoParams(video_subject="Paper boat", video_source="ltx", voice_name="en-US-AnaNeural")

    def compose(*args, output_prefix, **kwargs):
        final = directory / f"{output_prefix}final-1.mp4"
        combined = directory / f"{output_prefix}combined-1.mp4"
        for output in (final, combined):
            output.write_bytes(b"new complete revision")
        return [str(final)], [str(combined)], []

    original_update = memory.update_task
    original_patch = memory.patch_task
    def failed_update(task_id, **fields):
        if fields.get("phase") == "complete":
            raise OSError("state write unavailable")
        return original_update(task_id, **fields)

    def failed_patch(task_id, **fields):
        if fields.get("phase") == "complete":
            raise OSError("state write unavailable")
        return original_patch(task_id, **fields)

    monkeypatch.setattr(task, "generate_final_videos", compose)
    with monkeypatch.context() as backend:
        backend.setattr(memory, "update_task", failed_update)
        if outage == "restart":
            backend.setattr(memory, "patch_task", failed_patch)
            with pytest.raises(OSError, match="state write unavailable"):
                task.render_ltx(task_id, params, "audio.wav", "")
        else:
            task.render_ltx(task_id, params, "audio.wav", "")
    if outage == "restart":
        ltx.recover_interrupted_renders()
    committed = ltx.latest_retake_video(directory)
    assert committed and committed != str(old)
    saved = memory.get_task(task_id)
    assert saved["phase"] == "complete"
    assert saved["videos"] == [committed]
    assert saved["combined_videos"] == [committed.replace("final-1.mp4", "combined-1.mp4")]
    assert old.read_bytes() == b"previous complete revision"


@pytest.mark.parametrize("output_complete,invalid_commit", [(False, False), (True, False), (False, True)])
def test_partial_initial_output_does_not_trigger_repeat_composition(worker, tmp_path, monkeypatch, output_complete,
                                                                    invalid_commit):
    from app.models import const
    from app.models.schema import VideoParams
    from app.services import state, task

    task_id = "neutral"
    ltx.render_batch(task_id)
    directory = tmp_path / task_id
    final = directory / "final-1.mp4"
    final.write_bytes(b"partial output")
    if invalid_commit:
        batch = ltx.load_batch(task_id)
        batch["composition_result"] = {
            "videos": [str(final)], "combined_videos": [str(directory / "missing.mp4")],
        }
        ltx._replace_json(ltx._batch_path(task_id), batch)
    memory = state.MemoryState()
    monkeypatch.setattr(state, "state", memory)
    memory.update_task(task_id, state=const.TASK_STATE_FAILED, phase="failed", videos=[str(final)])
    monkeypatch.setattr(task, "_material_duration", lambda _: 5.0)
    monkeypatch.setattr(task.voice, "get_audio_duration", lambda _: 8.0)
    params = VideoParams(video_subject="Paper boat", video_source="ltx", voice_name="en-US-AnaNeural")

    def compose(*args, output_prefix="", **kwargs):
        video = directory / f"{output_prefix}final-1.mp4"
        combined = directory / f"{output_prefix}combined-1.mp4"
        video.write_bytes(b"complete output")
        if output_complete:
            combined.write_bytes(b"complete output")
        return [str(video)], [str(combined)], []

    monkeypatch.setattr(task, "generate_final_videos", compose)
    def recompose(ticket):
        return task.render_ltx(task_id, params, "audio.wav", "", ticket=ticket)
    if output_complete:
        task.queue_ltx_render(task_id, recompose)
        assert memory.get_task(task_id)["videos"] == [str(final)]
        assert memory.get_task(task_id)["state"] == const.TASK_STATE_COMPLETE
    else:
        with pytest.raises(RuntimeError, match="did not produce complete videos"):
            task.queue_ltx_render(task_id, recompose)
        assert memory.get_task(task_id)["state"] == const.TASK_STATE_FAILED
        assert ("composition_result" in ltx.load_batch(task_id)) == invalid_commit
    assert final.read_bytes() == b"complete output"
    assert not ltx.latest_retake_video(directory)


def test_retake_recovery_waits_for_proof_before_allowing_retry(tmp_path, monkeypatch):
    from app.models import const
    from app.services import state

    monkeypatch.setattr(ltx.utils, "task_dir", lambda task_id="": str(tmp_path / task_id))
    monkeypatch.setattr(ltx.utils, "storage_dir", lambda key: str(tmp_path / key))
    monkeypatch.setattr(state, "state", state.MemoryState())
    task_id = "retake-recovery"
    take_id = "a" * 32
    batch = {"task_id": task_id, "phase": "complete", "scenes": [], "retake_pending": take_id}
    path = ltx._batch_path(task_id)
    ltx._replace_json(path, batch)
    manifest = ltx._retake_manifest(path, take_id)
    ltx._create_json(manifest, {"phase": "rendering", "render_attempt_started": True})
    state.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, phase="rendering", videos=["old.mp4"])
    ownership = Mock(side_effect=[["unconfirmed"], []])
    monkeypatch.setattr(ltx, "_cancel_owned_runs", ownership)
    ltx.recover_interrupted_renders()
    assert state.state.get_task(task_id)["phase"] == "stop_unconfirmed"
    assert ltx.load_batch(task_id)["retake_pending"] == take_id
    with pytest.raises(ltx.LtxError, match="not confirmed"):
        ltx.reserve_render(task_id, "owner", lambda _: False)
    with pytest.raises(ltx.RenderBusy), ltx.guard_task_deletion(task_id):
        pytest.fail("Unconfirmed retake ownership must be retained")
    ltx.recover_interrupted_renders()
    assert state.state.get_task(task_id)["phase"] == "complete"
    assert state.state.get_task(task_id)["videos"] == ["old.mp4"]
    assert "retake_pending" not in ltx.load_batch(task_id)


def test_remote_plan_cannot_open_local_scene_editor(approved_batch, scene_editor):
    batch = ltx.load_batch("neutral")
    batch["phase"] = "awaiting_approval"
    ltx._replace_json(ltx._batch_path("neutral"), batch)
    plan_path = Path(batch["scenes"][0]["plan_file"])
    plan = ltx._read_json(plan_path)
    plan["profile"]["remote_server"] = True
    ltx._replace_json(plan_path, plan)
    with pytest.raises(ltx.LtxError, match="only on the local server"):
        ltx.open_scene_editor("neutral")
    assert not scene_editor.commands
