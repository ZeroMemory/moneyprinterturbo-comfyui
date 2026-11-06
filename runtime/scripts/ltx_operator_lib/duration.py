"""Predict LTX-2.5 scene lengths without sampling or producing media."""

import copy
import hashlib
import math
from pathlib import Path
import sys
import time
from uuid import uuid4

from .adapters import adapter_for
from .api import (ComfyClient, ComfyHttpError, ComfyPromptNotSubmitted, client_for_snapshot,
                  execution_error, remaining_timeout, remote_client_options)
from .artifacts import create_json_exclusive, replace_json_atomic
from .canonical import canonical_json
from .profiles import load_profile_snapshot
from .semantics import _weight_errors
from .state import OperatorState
from .workflow import load_workflow


DURATION_HEAD = "ltx-2.5-duration-head-bf16.safetensors"


def _frame_bounds(fps, minimum, maximum):
    # Include exact frame boundaries even when multiplication rounds one float below or above them.
    first = math.ceil(math.nextafter(minimum * fps, -math.inf))
    last = math.floor(math.nextafter(maximum * fps, math.inf))
    return first + (1 - first) % 8, last - (last - 1) % 8


def prediction_graph(template, prompt, min_seconds, max_seconds):
    if template.get("variant", {}).get("model_version") != "2.5":
        raise ValueError("Duration prediction requires LTX-2.5")
    graph = template.get("api_prompt", {})

    def one(kind):
        matches = [(key, node) for key, node in graph.items() if node.get("class_type") == kind]
        if len(matches) != 1:
            raise ValueError(f"Duration prediction requires one native {kind}")
        return matches[0]

    model_key, model = one("UNETLoader")
    clip_key, clip = one("CLIPLoader")
    text_keys = [key for key, node in graph.items() if node.get("class_type") == "CLIPTextEncode"]
    if len(text_keys) == 1:
        text_key = text_keys[0]
    else:
        positive = adapter_for("t2v-i2v-two-stage").discover(load_workflow(graph)).prompt
        if positive.derived:
            raise ValueError("Duration prediction requires the workflow's direct text encoder")
        text_key = positive.node_key
    text = graph[text_key]
    if text.get("class_type") != "CLIPTextEncode" or text.get("inputs", {}).get("clip") != [clip_key, 0]:
        raise ValueError("Duration prediction requires the workflow's direct text encoder")
    fps = template.get("parameters", {}).get("fps")
    if type(fps) not in {int, float} or not math.isfinite(fps) or not 1 <= fps <= 120:
        raise ValueError("Duration prediction FPS must be from 1 to 120")
    if any(type(value) not in {int, float} or not math.isfinite(value)
           for value in (min_seconds, max_seconds)) or not .5 <= min_seconds < max_seconds <= 30:
        raise ValueError("Duration prediction bounds require 0.5 <= minimum < maximum <= 30")
    first, last = _frame_bounds(fps, min_seconds, max_seconds)
    if first > last:
        raise ValueError("Duration prediction bounds and FPS cannot fit an 8k+1 frame count")
    # ComfyUI rounds the time bounds to frame integers before it snaps to the grid.
    if round(min_seconds * fps) <= first - 8:
        min_seconds = first / fps
    if round(max_seconds * fps) >= last + 8:
        max_seconds = last / fps
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Duration prediction requires a scene prompt")
    for node in (model, clip):
        if any(isinstance(value, (list, dict)) for value in node.get("inputs", {}).values()):
            raise ValueError("Duration prediction requires direct native model loaders")
    result = copy.deepcopy({model_key: model, clip_key: clip, text_key: text})
    result[text_key]["inputs"]["text"] = prompt
    head_key = "mpt_duration_head"
    predictor_key = "mpt_duration_predictor"
    frames_key, seconds_key = "mpt_duration_frames", "mpt_duration_seconds"
    if set(result) & {head_key, predictor_key, frames_key, seconds_key}:
        raise ValueError("Workflow uses reserved duration prediction node IDs")
    result[head_key] = {"class_type": "ModelPatchLoader", "inputs": {"name": DURATION_HEAD}}
    result[predictor_key] = {"class_type": "LTXVDurationPredictor", "inputs": {
        "model": [model_key, 0], "positive": [text_key, 0], "duration_head": [head_key, 0],
        "frame_rate": fps, "min_seconds": min_seconds, "max_seconds": max_seconds,
    }}
    result[frames_key] = {"class_type": "PreviewAny", "inputs": {"source": [predictor_key, 0]}}
    result[seconds_key] = {"class_type": "PreviewAny", "inputs": {"source": [predictor_key, 1]}}
    return result


def _prediction(history, prompt_id, fps, minimum, maximum):
    try:
        outputs = history[prompt_id]["outputs"]
        frame_values = outputs["mpt_duration_frames"]["text"]
        second_values = outputs["mpt_duration_seconds"]["text"]
        if not isinstance(frame_values, (list, tuple)) or not isinstance(second_values, (list, tuple)):
            raise ValueError
        raw_frames, = frame_values
        raw_seconds, = second_values
        if type(raw_frames) not in {str, int} or type(raw_seconds) not in {str, int, float}:
            raise ValueError
        frames, seconds = int(raw_frames), float(raw_seconds)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("ComfyUI did not return duration predictor outputs") from exc
    first, last = _frame_bounds(fps, minimum, maximum)
    if not first <= frames <= last or (frames - 1) % 8 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("ComfyUI returned an invalid or out-of-bounds duration prediction")
    return {"frame_count": frames, "predicted_duration_seconds": seconds}


def predict_durations(manifest, *, client_factory=ComfyClient, progress=None):
    """Run one owned, bounded prediction at a time; never submit a render graph."""
    template = manifest["template"]
    scenes = manifest.get("scenes")
    if not isinstance(scenes, list) or not 1 <= len(scenes) <= 20:
        raise ValueError("Duration prediction requires 1 to 20 scenes")
    minimum, maximum = manifest.get("min_seconds", .5), manifest["max_seconds"]
    timeout = manifest.get("timeout_seconds", 600)
    if type(timeout) not in {int, float} or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Duration prediction timeout must be positive and finite")
    graphs = []
    for index, scene in enumerate(scenes, 1):
        if type(scene.get("index")) is not int or scene["index"] != index:
            raise ValueError("Duration prediction scene indexes must be sequential")
        graphs.append(prediction_graph(template, scene["visual_prompt"], minimum, maximum))
    state = OperatorState(manifest["state_root"])
    directory = Path(manifest["runs_dir"]).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    endpoint = template["profile"]["endpoint"]
    client = client_for_snapshot(template["profile"], client_factory)
    options = (remote_client_options(load_profile_snapshot(template["profile"]))
               if template["profile"].get("remote_server") is True else {})
    info = client.object_info()
    for kind in {node["class_type"] for node in graphs[0].values()}:
        if kind not in info:
            raise ValueError(f"ComfyUI is missing duration prediction node {kind}")
    names = info.get("ModelPatchLoader", {}).get("input", {}).get("required", {}).get("name", [[]])[0]
    if DURATION_HEAD not in names:
        raise ValueError(f"Put {DURATION_HEAD} in ComfyUI/models/model_patches before predicting duration")
    workflow = load_workflow(graphs[0])
    loaders = [node for node in workflow.nodes if node.node_type in {"UNETLoader", "CLIPLoader"}]
    if errors := _weight_errors(workflow, loaders, template, info, None):
        raise ValueError("; ".join(errors))
    results = []
    for index, graph in enumerate(graphs, 1):
        run_dir = directory / f"{index:03d}"
        run_dir.mkdir()
        report_path = run_dir / "report.json"
        approval_hash = hashlib.sha256(canonical_json(graph).encode()).hexdigest()
        ownership = {"schema_version": 1, "approval_hash": approval_hash, "endpoint": endpoint,
                     "run_directory": str(run_dir), "prediction_scene_index": index, **options}
        report = {"status": "not_submitted", "prompt_id": None, "scene_index": index,
                  "approval_hash": approval_hash}
        create_json_exclusive(report_path, report, private=True)
        prompt_id = None
        completed = False
        try:
            deadline = time.monotonic() + timeout
            client_id = str(uuid4())
            replace_json_atomic(report_path, {**report, "status": "submitting"})
            budget = remaining_timeout(deadline)
            report["status"] = "submitting"
            try:
                prompt_id = client.submit(graph, client_id=client_id, timeout=budget)["prompt_id"]
            except (ComfyHttpError, ComfyPromptNotSubmitted) as exc:
                if isinstance(exc, ComfyPromptNotSubmitted) or exc.prompt_rejected:
                    try:
                        record = state.sign_record({**ownership, "terminal_status": "submission_rejected"})
                        create_json_exclusive(run_dir / "run.json", record, private=True)
                    except Exception as record_error:
                        exc.add_note(f"Could not save prediction rejection: {record_error}")
                    else:
                        report["status"] = "submission_rejected"
                raise
            record = state.sign_record({**ownership, "prompt_id": prompt_id})
            create_json_exclusive(run_dir / "run.json", record, private=True)
            report.update(status="queued", prompt_id=prompt_id)
            replace_json_atomic(report_path, report)
            remaining = remaining_timeout(deadline)
            outcome = client.wait_for_completion(prompt_id, client_id, timeout=remaining, websocket_events=lambda *_: ())
            if outcome.status != "completed":
                error = execution_error(outcome)
                raise RuntimeError(f"Scene {index} duration prediction {outcome.status}" + (f": {error}" if error else ""))
            result = {"index": index, **_prediction(outcome.history, prompt_id,
                                                   template["parameters"]["fps"], minimum, maximum)}
            report.update(status="completed", **result)
            replace_json_atomic(report_path, report)
            results.append(result)
            completed = True
            if progress:
                progress({"event": "scene_planned", "scene_index": index,
                          "status": "duration_predicted", **result})
        except BaseException as exc:
            if not completed and report["status"] not in {"not_submitted", "submission_rejected"}:
                report["status"] = "uncertain_execution" if prompt_id else "uncertain_submission"
            report["error"] = f"{type(exc).__name__}: {exc}"
            try:
                replace_json_atomic(report_path, report)
            except BaseException:
                pass
            raise
        finally:
            if prompt_id and not completed:
                primary_error = sys.exception()
                try:
                    cancellation = client.cancel_and_confirm(prompt_id, timeout=1)
                    report.update(status="failed" if cancellation.status in {"confirmed_cancelled", "completed", "failed"}
                                  else "uncertain_execution", cancellation=cancellation.status,
                                  cleanup_grace_seconds=1)
                except BaseException as exc:
                    report.update(status="uncertain_execution", cancellation="uncertain",
                                  cleanup_grace_seconds=1, cleanup_error=f"{type(exc).__name__}: {exc}")
                    if not isinstance(exc, Exception) and primary_error is None:
                        primary_error = exc
                        raise
                finally:
                    try:
                        replace_json_atomic(report_path, report)
                    except BaseException:
                        if primary_error is None:
                            raise
    return {"ok": True, "scenes": results}
