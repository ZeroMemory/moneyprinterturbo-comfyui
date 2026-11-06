"""Keep owned work visible when the operator exits before its final response."""
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import wave

import pytest

from app.models.schema import VideoParams
from app.services import execution_control, ltx, state, voice


@pytest.fixture
def process_batch(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "runtime/scripts"))
    monkeypatch.setattr(ltx.utils, "task_dir", lambda task_id="": str(tmp_path / task_id))
    monkeypatch.setattr(ltx.utils, "storage_dir", lambda name: str(tmp_path / name))
    monkeypatch.setattr(state, "state", state.MemoryState())
    child = tmp_path / "fake_operator.py"
    child.write_text('''import json, os, sys, time
from pathlib import Path
from ltx_operator_lib.state import OperatorState
args = sys.argv
mode = os.environ["OPERATOR_FAULT"]
if mode != "before-submission":
    state = OperatorState(args[args.index("--state-root") + 1])
    run = Path(args[args.index("--runs-dir") + 1]) / "001" / "run"
    approval = "a" * 64
    if mode in ("empty-run", "initialized-report"):
        if mode == "empty-run":
            run.mkdir(parents=True)
        else:
            from ltx_operator_lib.runs import _SceneRun
            _SceneRun({"api_prompt": {}, "variant": {}, "profile": {}}, approval, run)
        print(json.dumps({"error": "inert operator failure"}), flush=True)
        sys.exit(1)
    run.mkdir(parents=True)
    state.consume_approval(approval, run_directory=run)
    if mode != "receipt-only":
        record = state.sign_record({"schema_version": 1, "approval_hash": approval,
            "endpoint": "http://127.0.0.1:8188", "prompt_id": "inert-owned",
            "run_directory": str(run)})
        (run / "run.json").write_text(json.dumps(record), encoding="utf-8")
    if mode == "queued-json":
        print(json.dumps({"event": "scene_render", "scene_index": 1, "status": "queued",
            "run_directory": str(run), "outputs": [], "errors": []}), file=sys.stderr, flush=True)
if mode == "timeout": time.sleep(30)
if mode == "exit": os._exit(9)
if mode in ("invalid-json", "queued-json"): print("{", flush=True)
elif mode != "empty-eof":
    print(json.dumps({"error": "inert operator failure"}), flush=True)
    sys.exit(1)
''', encoding="utf-8")
    monkeypatch.setattr(ltx, "settings", lambda: {"python_executable": sys.executable,
                         "operator_script": str(child), "render_timeout_seconds": 1})
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[2] / "runtime/scripts"))
    batch = {"task_id": "neutral", "phase": "approved", "scenes": [{"index": 1,
             "status": "approved", "submitted_approval_hash": "a" * 64, "approval_hash": "a" * 64}]}
    ltx._replace_json(ltx._batch_path("neutral"), batch)
    return batch


@pytest.mark.parametrize(("case", "expected_task", "expected_batch"), [
    ("live", "rendering", "approved"), ("queued", "queued", "approved"),
    ("pre-start", "failed", "approved"), ("complete", "failed", "complete"),
    ("legacy", "stop_unconfirmed", "stop_unconfirmed"), ("unknown", "stop_unconfirmed", "stop_unconfirmed"),
    ("terminal", "failed", "failed"),
])
def test_restart_reconciles_only_proven_ltx_work(process_batch, monkeypatch, case, expected_task, expected_batch):
    from contextlib import nullcontext
    from ltx_operator_lib.state import OperatorState

    task_id = "neutral"
    batch_path = ltx._batch_path(task_id)
    if case == "pre-start":
        process_batch["render_attempt_started"] = False
    if case == "complete":
        process_batch["phase"] = "complete"
    if case in {"unknown", "terminal"}:
        run = batch_path.parent / "scenes/001/run"
        approval = process_batch["scenes"][0]["approval_hash"]
        operator = OperatorState(ltx.utils.storage_dir("ltx-state"))
        operator.consume_approval(approval, run_directory=run)
        ltx._replace_json(run / "run.json", operator.sign_record({
            "approval_hash": approval, "endpoint": "http://127.0.0.1:8188", "prompt_id": "owned",
            "run_directory": str(run),
        }))
    ltx._replace_json(batch_path, process_batch)
    state.state.update_task(task_id, phase="queued" if case == "queued" else "rendering",
                            params={"video_source": "ltx"}, videos=["preserved.mp4"])
    claim = ltx._reservation_path(task_id)
    if case == "queued":
        ltx._replace_json(claim, {"ticket": "queued", "persistent": True})
    calls = []

    def status(command, **kwargs):
        calls.append(command[2])
        return SimpleNamespace(returncode=0, stdout=json.dumps({
            "prompt_id": "owned", "status": "failed" if case == "terminal" else "unknown",
        }))

    monkeypatch.setattr(ltx.subprocess, "run", status)
    with ltx.render_slot(task_id) if case == "live" else nullcontext():
        ltx.recover_interrupted_renders()
    assert calls == (["status"] if case in {"unknown", "terminal"} else [])
    assert state.state.get_task(task_id)["phase"] == expected_task
    assert state.state.get_task(task_id)["videos"] == ["preserved.mp4"]
    assert ltx.load_batch(task_id)["phase"] == expected_batch
    assert claim.exists() is (case == "queued")


@pytest.mark.parametrize(("case", "expected"), [
    ("live", "ltx_duration_prediction"), ("pre-start", "failed"),
    ("partial", "failed"), ("uncertain", "stop_unconfirmed"), ("misbound", "stop_unconfirmed"),
    ("terminal", "failed"), ("remote-terminal", "failed"),
])
def test_restart_leaves_live_prediction_and_preserves_uncertain_work(process_batch, monkeypatch, case, expected):
    from contextlib import nullcontext
    from ltx_operator_lib.state import OperatorState

    task_id = "predict-task"
    folder = ltx._batch_path(task_id).parent / "duration-predictions/attempt"
    state.state.update_task(task_id, phase="predicting_scene_durations" if case == "pre-start" else "ltx_duration_prediction")
    ltx._replace_json(folder / "manifest.json", {"scenes": [{"index": 1}, {"index": 2}]})
    if case == "uncertain":
        ltx._replace_json(folder / "runs/001/report.json", {"prompt_id": None, "status": "submitting"})
    if case in {"partial", "misbound", "terminal", "remote-terminal"}:
        for index in range(1, 2 if case in {"partial", "misbound"} else 3):
            run = folder / "runs" / f"{index:03d}"
            owned = {"prompt_id": f"owned-{index}",
                     "run_directory": str(run.with_name("other") if case == "misbound" else run)}
            envelope = OperatorState(ltx.utils.storage_dir("ltx-state")).sign_record(owned)
            ltx._replace_json(run / "run.json", envelope)
            ltx._replace_json(run / "report.json", {"prompt_id": f"owned-{index}",
                              "status": "queued" if case == "remote-terminal" else "completed"})
    calls = []

    def status(args, timeout):
        assert args[0] == "status" and 0 < timeout <= 4
        record = ltx._read_json(Path(args[2]))["record"]
        calls.append(record["prompt_id"])
        return {"prompt_id": record["prompt_id"], "status": "completed"}

    monkeypatch.setattr(ltx, "run_cli", status)
    with ltx.render_slot(task_id) if case == "live" else nullcontext():
        ltx.recover_interrupted_renders()
    assert state.state.get_task(task_id)["phase"] == expected
    assert set(calls) == ({"owned-1", "owned-2"} if case == "remote-terminal" else set())
    assert (folder / "manifest.json").is_file()


@pytest.mark.parametrize("valid", [False, True])
def test_prediction_recovery_authenticates_definitive_rejection(process_batch, monkeypatch, valid):
    from ltx_operator_lib.state import OperatorState

    task_id = "predict-rejected"
    state.state.update_task(task_id, phase="ltx_duration_prediction")
    folder = ltx._batch_path(task_id).parent / "duration-predictions/attempt"
    run = folder / "runs/001"
    digest = "b" * 64
    owner = OperatorState(ltx.utils.storage_dir("ltx-state"))
    record = owner.sign_record({"approval_hash": digest, "endpoint": "http://127.0.0.1:8188",
                                "run_directory": str(run), "prediction_scene_index": 1,
                                "terminal_status": "submission_rejected"})
    if not valid:
        record["signature"] = "0" * 64
    ltx._replace_json(folder / "manifest.json", {"scenes": [{"index": 1}, {"index": 2}]})
    ltx._replace_json(run / "run.json", record)
    ltx._replace_json(run / "report.json", {"status": "submitting", "scene_index": 1,
                                          "prompt_id": None, "approval_hash": digest})
    operator = Path(__file__).resolve().parents[2] / "runtime/scripts/ltx_operator.py"
    monkeypatch.setattr(ltx, "settings", lambda: {"python_executable": sys.executable,
                                                 "operator_script": str(operator)})
    ltx.recover_interrupted_renders()
    assert state.state.get_task(task_id)["phase"] == ("failed" if valid else "stop_unconfirmed")


@pytest.mark.parametrize("kind", ["render", "prediction"])
def test_cli_reuses_an_owned_persisted_cancellation_without_history(process_batch, tmp_path, kind):
    from io import StringIO
    from ltx_operator_lib.cli import run_cli
    from ltx_operator_lib.state import OperatorState

    run = tmp_path / "cancelled"
    state_root = tmp_path / "cli-state"
    approval = "a" * 64
    owner = OperatorState(state_root)
    record = run / "run.json"
    ltx._replace_json(record, owner.sign_record({
        "approval_hash": approval, "endpoint": "http://127.0.0.1:8188", "prompt_id": "owned",
        "run_directory": str(run),
    }))
    details = ({"approval_hash": approval, "cleanup": {"status": "confirmed_cancelled"}} if kind == "render" else
               {"scene_index": 1, "cancellation": "confirmed_cancelled"})
    ltx._replace_json(run / "report.json", {"status": "uncertain_execution", "prompt_id": "owned", **details})

    def unexpected_network(*args, **kwargs):
        pytest.fail("Already confirmed cancellation must not make another HTTP request")

    for command, expected in [("status", "cancelled"), ("cancel", "confirmed_cancelled")]:
        stdout = StringIO()
        assert run_cli([command, "--run-record", str(record), "--state-root", str(state_root), "--json"],
                       stdout=stdout, client_factory=unexpected_network) == 0
        assert json.loads(stdout.getvalue())["status"] == expected


@pytest.mark.parametrize("mode", ["timeout", "empty-eof", "invalid-json", "error-json", "exit", "queued-json"])
def test_abnormal_exit_checks_owned_work_before_persisting_failure(process_batch, monkeypatch, mode):
    monkeypatch.setenv("OPERATOR_FAULT", mode)
    original = ltx.run_cli_stream
    monkeypatch.setattr(ltx, "run_cli_stream", lambda args, timeout, progress: original(args, .8, progress))
    calls = []
    def cancel(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout='{"status":"uncertain"}')
    monkeypatch.setattr(ltx.subprocess, "run", cancel)
    with pytest.raises(ltx.LtxError):
        ltx._render_approved_batch("neutral", ltx._batch_path("neutral"))
    saved = ltx.load_batch("neutral")
    assert saved["phase"] == "stop_unconfirmed"
    assert "unconfirmed" in saved["stop_error"].lower()
    assert len(calls) == 1 and calls[0][2] == "cancel"
    assert Path(calls[0][calls[0].index("--run-record") + 1]).name == "run.json"


@pytest.mark.parametrize("mode,phase", [("before-submission", "approved"), ("receipt-only", "stop_unconfirmed")])
def test_missing_record_distinguishes_unused_approval_from_submission(process_batch, monkeypatch, mode, phase):
    monkeypatch.setenv("OPERATOR_FAULT", mode)
    monkeypatch.setattr(ltx.subprocess, "run", lambda *args, **kwargs: pytest.fail("no owned prompt may be cancelled"))
    with pytest.raises(ltx.LtxError, match="inert operator failure"):
        ltx._render_approved_batch("neutral", ltx._batch_path("neutral"))
    assert ltx.load_batch("neutral")["phase"] == phase
    receipt = Path(ltx.utils.storage_dir("ltx-state")) / "receipts" / ("a" * 64 + ".json")
    assert receipt.exists() == (mode == "receipt-only")


@pytest.mark.parametrize("tampered", [False, True])
def test_native_rejection_requires_bound_proof_during_render_and_restart(process_batch, monkeypatch, tampered):
    import time

    from ltx_operator_lib.state import OperatorState

    task_id = "neutral"
    batch_path = ltx._batch_path(task_id)
    run = batch_path.parent / "scenes/001/run"
    approval = process_batch["scenes"][0]["approval_hash"]
    operator_state = OperatorState(ltx.utils.storage_dir("ltx-state"))
    script = Path(__file__).resolve().parents[2] / "runtime/scripts/ltx_operator.py"
    monkeypatch.setattr(ltx, "settings", lambda: {"python_executable": sys.executable,
                         "operator_script": str(script), "render_timeout_seconds": 1})

    def reject(*_args):
        operator_state.consume_approval(approval, run_directory=run)
        record = operator_state.sign_record({"approval_hash": approval, "endpoint": "http://127.0.0.1:8188",
                                            "run_directory": str(run), "terminal_status": "submission_rejected"})
        if tampered:
            record["signature"] = "0" * 64
        ltx._replace_json(run / "run.json", record)
        raise ltx.LtxError("ComfyUI HTTP 400")

    monkeypatch.setattr(ltx, "run_cli_stream", reject)
    with pytest.raises(ltx.LtxError, match="HTTP 400") as raised:
        ltx._render_approved_batch(task_id, batch_path)
    expected = "stop_unconfirmed" if tampered else "failed"
    assert getattr(raised.value, "remote_stop_unconfirmed", False) is tampered
    assert ltx.load_batch(task_id)["phase"] == expected
    with pytest.raises(PermissionError):
        operator_state.check_unused(approval)

    process_batch["phase"] = "rendering"
    ltx._replace_json(batch_path, process_batch)
    state.state.update_task(task_id, phase="rendering", params={"video_source": "ltx"})
    ltx._recover_interrupted_render(task_id, time.monotonic() + 10)
    assert ltx.load_batch(task_id)["phase"] == expected
    assert state.state.get_task(task_id)["phase"] == expected
    if tampered:
        with pytest.raises(ltx.RenderBusy):
            with ltx.guard_task_deletion(task_id):
                pytest.fail("a forged rejection must not permit deletion")
    else:
        with ltx.guard_task_deletion(task_id):
            pass


@pytest.mark.parametrize("mode,phase", [("empty-run", "failed"), ("initialized-report", "failed"),
                                       ("before-submission", "approved")])
def test_interrupted_initialization_requires_replanning_without_reusing_evidence(process_batch, monkeypatch, mode, phase):
    from ltx_operator_lib.state import OperatorState

    monkeypatch.setenv("OPERATOR_FAULT", mode)
    batch_path = ltx._batch_path("neutral")
    run = batch_path.parent / "scenes" / "001" / "run"
    run.parent.mkdir(parents=True)
    original = ltx.run_cli_stream
    failures = []
    def capture(*args):
        try:
            return original(*args)
        except ltx.LtxError as error:
            failures.append(error)
            raise
    monkeypatch.setattr(ltx, "run_cli_stream", capture)
    with pytest.raises(ltx.LtxError, match="^inert operator failure$") as observed:
        ltx._render_approved_batch("neutral", batch_path)
    assert observed.value is failures[0]
    batch = ltx.load_batch("neutral")
    assert batch["phase"] == phase
    operator_state = OperatorState(ltx.utils.storage_dir("ltx-state"))
    operator_state.check_unused("a" * 64)
    if phase == "approved":
        assert not run.exists() and "retry_error" not in batch
        ticket = ltx.reserve_render("neutral", "owner", lambda _: False)
        ltx.release_reservation("neutral", ticket)
        return
    assert "new task" in batch["retry_error"] and "scene plans" in batch["retry_error"]
    snapshot = {str(path.relative_to(run)): (path.stat().st_mode, path.read_bytes() if path.is_file() else None)
                for path in (run, *run.rglob("*"))}
    with pytest.raises(ltx.LtxError, match="Create a new task"):
        ltx.reserve_render("neutral", "owner", lambda _: False)
    with pytest.raises(ltx.LtxError, match="Create a new task"):
        ltx._render_approved_batch("neutral", batch_path)
    assert not ltx._reservation_path("neutral").exists() and len(failures) == 1
    assert snapshot == {str(path.relative_to(run)): (path.stat().st_mode, path.read_bytes() if path.is_file() else None)
                        for path in (run, *run.rglob("*"))}
    operator_state.check_unused("a" * 64)


@pytest.mark.parametrize("stop", [False, True])
def test_cleanup_failure_preserves_original_error_and_state(process_batch, monkeypatch, stop):
    error = execution_control.GenerationStopped("original stop") if stop else ltx.LtxError("original failure")
    def fail(*args):
        raise error
    monkeypatch.setattr(ltx, "run_cli_stream", fail)
    monkeypatch.setattr(ltx, "_cancel_owned_runs", lambda _: (_ for _ in ()).throw(OSError("cleanup unavailable")))
    with pytest.raises(type(error)) as observed:
        ltx._render_approved_batch("neutral", ltx._batch_path("neutral"))
    assert observed.value is error
    assert error.remote_stop_unconfirmed
    assert ltx.load_batch("neutral")["phase"] == "stop_unconfirmed"


@pytest.mark.parametrize("status,phase", [("confirmed_cancelled", "failed"), ("completed", "failed"),
                                         ("failed", "failed"), ("uncertain", "stop_unconfirmed"),
                                         (None, "stop_unconfirmed")])
def test_partial_batch_preserves_completed_scene_and_verifies_owned_runs(
        process_batch, monkeypatch, tmp_path, status, phase):
    from ltx_operator_lib.state import OperatorState

    operator_state = OperatorState(ltx.utils.storage_dir("ltx-state"))
    batch_path = ltx._batch_path("neutral")
    first, = process_batch["scenes"]
    clip = tmp_path / "completed.mp4"
    clip.write_bytes(b"existing completed output")
    second = {**first, "index": 2, "approval_hash": "b" * 64, "submitted_approval_hash": "b" * 64}
    process_batch["scenes"].append(second)
    ltx._replace_json(batch_path, process_batch)
    for scene in process_batch["scenes"]:
        run = batch_path.parent / "scenes" / f"{scene['index']:03d}" / "run"
        operator_state.consume_approval(scene["approval_hash"], run_directory=run)
        record = {"approval_hash": scene["approval_hash"], "endpoint": "http://127.0.0.1:8188",
                  "prompt_id": f"owned-{scene['index']}", "run_directory": str(run)}
        ltx._replace_json(run / "run.json", operator_state.sign_record(record))
        ltx._replace_json(run / "report.json", {**record, "status": "completed" if scene is first else "queued"})
    original = ltx.LtxError("operator response lost")
    def fail(args, timeout, progress):
        progress({"scene_index": 1, "status": "completed", "run_directory": str(clip.parent),
                  "outputs": [{"primary": True, "path": str(clip), "content_sha256": "c" * 64}]})
        progress({"scene_index": 2, "status": "queued", "run_directory": "second-run"})
        raise original
    monkeypatch.setattr(ltx, "run_cli_stream", fail)
    calls = []
    def cancel(command, **kwargs):
        calls.append(command)
        if "001" in command[command.index("--run-record") + 1]:
            return SimpleNamespace(returncode=0, stdout='{"status":"completed"}')
        return SimpleNamespace(returncode=0, stdout=json.dumps({"status": status} if status else []))
    monkeypatch.setattr(ltx.subprocess, "run", cancel)
    with pytest.raises(ltx.LtxError) as observed:
        ltx._render_approved_batch("neutral", batch_path)
    saved = ltx.load_batch("neutral")
    assert observed.value is original and saved["phase"] == phase
    assert saved["scenes"][0]["status"] == "complete" and clip.read_bytes() == b"existing completed output"
    assert len(calls) == 2
    assert {Path(call[call.index("--run-record") + 1]).parents[1].name for call in calls} == {"001", "002"}


@pytest.mark.parametrize("with_token", [False, True])
def test_user_stop_preserves_uncertainty_with_optional_token(process_batch, monkeypatch, with_token):
    from contextlib import nullcontext

    original = execution_control.GenerationStopped("user stop")
    def stop(*args):
        raise original
    monkeypatch.setattr(ltx, "run_cli_stream", stop)
    monkeypatch.setattr(ltx, "_cancel_owned_runs", lambda _: ["targeted cancellation unavailable"])
    token = execution_control.new_token("neutral") if with_token else None
    with execution_control.operation(token=token) if with_token else nullcontext():
        with pytest.raises(execution_control.GenerationStopped) as observed:
            ltx._render_approved_batch("neutral", ltx._batch_path("neutral"))
    assert observed.value is original and original.remote_stop_unconfirmed
    assert ltx.load_batch("neutral")["phase"] == "stop_unconfirmed"
    if token:
        assert "unconfirmed" in token.stop_error.lower()


@pytest.mark.parametrize("response", [[], 3, {"status": "uncertain"}])
def test_prediction_cleanup_does_not_mask_original_failure(tmp_path, monkeypatch, response):
    monkeypatch.setattr(ltx.utils, "task_dir", lambda _: str(tmp_path))
    monkeypatch.setattr(ltx.utils, "storage_dir", lambda name: str(tmp_path / name))
    monkeypatch.setattr(state, "state", state.MemoryState())
    monkeypatch.setattr(ltx, "settings", lambda: {"python_executable": sys.executable, "operator_script": __file__})
    original = ltx.LtxError("prediction response lost")
    def fail(args, timeout, progress):
        manifest = ltx._read_json(Path(args[args.index("--manifest") + 1]))
        ltx._replace_json(Path(manifest["runs_dir"]) / "001" / "run.json", {})
        raise original
    monkeypatch.setattr(ltx, "run_cli_stream", fail)
    monkeypatch.setattr(ltx.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout=json.dumps(response)))
    params = VideoParams(video_subject="Boat", voice_name="no-voice", ltx_predict_duration=True)
    request = {"variant": {"model_version": "2.5"}, "parameters": {"fps": 24}}
    scene = {"index": 1, "duration_seconds": 3, "narration": "A boat floats.", "visual_prompt": "A boat floats."}
    with pytest.raises(ltx.LtxError) as observed:
        ltx.predict_scene_durations("neutral", params, request, [scene], {"duration_mode": "predicted"})
    assert observed.value is original and original.remote_stop_unconfirmed
    assert state.state.get_task("neutral")["phase"] == "stop_unconfirmed"


@pytest.mark.parametrize("case,expected_calls", [("wrong-report", 1), ("scalar-status", 1), ("matching-report", 1),
                                                ("foreign-approval", 0), ("extra-record", 0)])
def test_cleanup_checks_current_scene_ownership_before_terminal_reports(process_batch, monkeypatch, case, expected_calls):
    from ltx_operator_lib.state import OperatorState

    run = ltx._batch_path("neutral").parent / "scenes" / ("002" if case == "extra-record" else "001") / "run"
    record = {"approval_hash": ("b" if case == "foreign-approval" else "a") * 64,
              "prompt_id": "owned-active", "endpoint": "http://127.0.0.1:8188", "run_directory": str(run)}
    ltx._replace_json(run / "run.json", OperatorState(ltx.utils.storage_dir("ltx-state")).sign_record(record))
    report = {**record, "status": ["completed"] if case == "scalar-status" else "completed"}
    if case == "wrong-report":
        report.update(approval_hash="b" * 64, prompt_id="another-prompt")
    ltx._replace_json(run / "report.json", report)
    calls = []
    def cancel(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout='{"status":"uncertain"}')
    monkeypatch.setattr(ltx.subprocess, "run", cancel)
    assert ltx._cancel_owned_runs(ltx._batch_path("neutral"))
    assert len(calls) == expected_calls


@pytest.mark.parametrize("phase,rejected", [("stop_unconfirmed", True), ("failed", False)])
def test_reservation_uses_uncertainty_when_batch_persistence_failed(process_batch, phase, rejected):
    state.state.update_task("neutral", phase=phase)
    if rejected:
        with pytest.raises(ltx.LtxError, match="stop.*confirmed"):
            ltx.reserve_render("neutral", "owner", lambda _: False)
        assert not ltx._reservation_path("neutral").exists()
        assert state.state.get_task("neutral")["phase"] == phase
    else:
        ticket = ltx.reserve_render("neutral", "owner", lambda _: False)
        ltx.release_reservation("neutral", ticket)
        assert state.state.get_task("neutral")["phase"] == phase


@pytest.mark.parametrize("fps,speed,samples,rate,version", [
    (24, 1, 124080, 24000, "h3"), (1000, 2, 96097, 192000, "2.5"),
])
def test_measured_audio_shortage_remains_visible(tmp_path, fps, speed, samples, rate, version):
    audio = tmp_path / "narration.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(b"\0\0" * samples)
    params = VideoParams(video_subject="A boat", voice_name="no-voice", custom_audio_file=str(audio),
                         ltx_scene_duration_seconds=5 if version == "h3" else 1,
                         ltx_max_scenes=1, video_clip_speed=speed)
    request = {"variant": {"model_version": version}, "parameters": {"fps": fps}}
    duration = voice.get_audio_duration(str(audio))
    with pytest.raises(ValueError, match="deficit") as observed:
        ltx.local_video_timing(params, request, audio_duration=duration)
    values = re.search(r"Audio is ([\d.e+-]+)s.*cover ([\d.e+-]+)s, a deficit of ([\d.e+-]+)s", str(observed.value))
    assert values and float(values[1]) > float(values[2]) and float(values[3]) > 0
    if version == "h3":
        assert ltx.local_video_timing(params, request, audio_duration=5.166)["required_scenes"] == 1


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("structured", [False, True])
def test_mpt_cli_prefers_structured_error_and_uses_stderr_only_as_fallback(tmp_path, monkeypatch, stream, structured):
    child = tmp_path / "operator.py"
    payload = {"ok": False, "error": "Neutral node execution failed."} if structured else {"ok": False}
    diagnostic = json.dumps({"event": "scene_render", "status": "failed", "scene_index": 1,
                             "run_directory": "neutral/run", "outputs": [], "errors": []}) if structured else "Neutral worker failure."
    child.write_text("import sys\nprint(" + repr(json.dumps(payload)) + ")\nprint(" + repr(diagnostic) + ", file=sys.stderr)\nsys.exit(1)\n", encoding="utf-8")
    monkeypatch.setattr(ltx, "settings", lambda: {"python_executable": sys.executable, "operator_script": str(child)})
    with pytest.raises(ltx.LtxError) as caught:
        if stream:
            ltx.run_cli_stream([], 5, lambda _: None)
        else:
            ltx.run_cli([], 5)
    if structured:
        assert str(caught.value) == "Neutral node execution failed."
    else:
        assert "Neutral worker failure." in str(caught.value)
