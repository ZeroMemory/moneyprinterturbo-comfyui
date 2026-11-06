"""Run offline scenes through an installed MPT, its operator subprocess and FFmpeg."""
import argparse
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from threading import Thread
import time
from unittest.mock import patch
from urllib.parse import parse_qs, quote, unquote, urlsplit
from uuid import uuid4

IN_IMAGE = os.environ.get("MPT_SMOKE_CONTAINER") == "1"
ROOT = Path("/opt/mpt-comfyui") if IN_IMAGE else Path(__file__).resolve().parents[1]
IMAGE_COMFYUI = Path("/comfyui")
sys.path.insert(0, str(ROOT))
from configure_ltx import MODEL_INPUTS, MODEL_VERSIONS
from workflow_fixtures import sample_graph, write_workflow


def fixture_object_info(app, graph):
    result = {}
    for node in graph.values():
        kind = node["class_type"]
        inputs = {}
        for field, category in MODEL_INPUTS.get(kind, {}).items():
            if app is None:
                inputs[field] = [sorted({other["inputs"][field] for other in graph.values()
                                         if other["class_type"] == kind})]
            else:
                root = app / "models" / category
                inputs[field] = [sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())]
        result[kind] = {"input": {"required": inputs}}
    return result


@contextmanager
def comfy_fixture(app, clip, graph, expected_clip_size=None, downloads=None, alternate=None):
    submissions, userdata = {}, {}
    outputs = {}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = unquote(urlsplit(self.path).path)
            if path == "/users":
                self.reply({"storage": "server", "migrated": True})
            elif path == "/extensions":
                self.reply(["/extensions/mpt_scene_editor/editor.js"])
            elif path.startswith("/userdata/"):
                key = path.removeprefix("/userdata/")
                if key not in userdata:
                    return self.send_error(404)
                self.reply(userdata[key])
            elif path == "/system_stats":
                self.reply({"system": {"os": sys.platform}})
            elif path == "/object_info":
                self.reply(fixture_object_info(app if downloads is None else None, graph))
            elif path == "/queue":
                self.reply({"queue_running": [], "queue_pending": []})
            elif path == "/history":
                self.reply(submissions)
            elif path.startswith("/history/"):
                key = path.split("/")[-1]
                self.reply({key: submissions[key]} if key in submissions else {})
            elif path == "/view" and downloads is not None:
                query = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
                name, = query.get("filename", [""])
                if name not in outputs or query.get("type") != ["output"] or query.get("subfolder", [""]) != [""]:
                    return self.send_error(404)
                downloads.append(name)
                output = outputs[name]
                self.send_response(200)
                self.send_header("Content-Length", str(output.stat().st_size))
                self.send_header("Content-Type", "video/mp4")
                self.end_headers()
                with output.open("rb") as source:
                    shutil.copyfileobj(source, self.wfile)
            else:
                self.send_error(404)

        def do_POST(self):
            path = unquote(urlsplit(self.path).path)
            if path.startswith("/userdata/"):
                userdata[path.removeprefix("/userdata/")] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                return self.reply({})
            if path != "/prompt":
                return self.send_error(404)
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if expected_clip_size:
                latent, = [node for node in body["prompt"].values()
                           if node["class_type"] in ("EmptyLTXVLatentVideo", "MiniMaxH3ImageToVideo")]
                scale = 2 if latent["class_type"] == "EmptyLTXVLatentVideo" else 1
                assert tuple(latent["inputs"][key] * scale for key in ("width", "height")) == expected_clip_size
            node_id, save = next((k, n) for k, n in body["prompt"].items() if n["class_type"] == "SaveVideo")
            prefix = save["inputs"]["filename_prefix"]
            assert Path(prefix).name == prefix
            name = prefix + ".mp4"
            rendered = clip
            if alternate and any(alternate[0] in node["inputs"].values() for node in body["prompt"].values()):
                rendered = alternate[1]
            if downloads is None:
                shutil.copyfile(rendered, app / "output" / name)
            outputs[name] = rendered
            key = uuid4().hex
            submissions[key] = {"status": {"completed": True, "status_str": "success", "messages": []},
                                "prompt": [0, key, body["prompt"], {"client_id": body["client_id"]}, [node_id]],
                                "outputs": {node_id: {"video": [{"filename": name, "subfolder": "", "type": "output"}]}}}
            self.reply({"prompt_id": key, "number": 1, "node_errors": {}})

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", submissions
        finally:
            server.shutdown()
            thread.join()


def verify(target, work, version, external_server=False, remote=False):
    work.mkdir(parents=True, exist_ok=False)
    os.environ["MPT_HOME"] = str(work / "home")
    app = IMAGE_COMFYUI if IN_IMAGE else work / "ComfyUI"
    if not remote:
        (app / "output").mkdir(parents=True)
        (app / "main.py").write_text("# HTTP fixture owns the server.\n", encoding="utf-8")
    graph = sample_graph(version)
    workflows = work / "home/workflows" if IN_IMAGE else work
    workflows.mkdir(parents=True, exist_ok=True)
    workflow_api = write_workflow(workflows / f"{version}.api.json", graph)
    if not remote:
        for node in graph.values():
            for field, category in MODEL_INPUTS.get(node["class_type"], {}).items():
                path = app / "models" / category / node["inputs"][field]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"Test-only dependency for plan hashing.")
    duration = 5 if version == "h3" else 1
    audio_source = ["-f", "lavfi", "-i", "sine=frequency=660", "-c:a", "aac"] if remote else ["-an"]
    clip = work / "scene.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=256x256:rate=24", *audio_source, "-t", str(duration), "-c:v", "libx264", str(clip)], check=True)
    alternate = None
    if version == "2.5" and not IN_IMAGE:
        replacement = work / "replacement.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                        "color=c=blue:size=256x256:rate=24", *audio_source, "-t", str(duration),
                        "-c:v", "libx264", str(replacement)], check=True)
        alternate = ("The paper boat passes through blue shadows.", replacement)
    downloads = [] if remote else None
    with comfy_fixture(app, clip, graph, expected_clip_size=(256, 256),
                       downloads=downloads, alternate=alternate) as (endpoint, submissions):
        if IN_IMAGE:
            spec = importlib.util.spec_from_file_location("mpt_image_entrypoint", ROOT / "docker/entrypoint.py")
            entrypoint = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(entrypoint)
            assert (entrypoint.ADDON, entrypoint.MPT, entrypoint.COMFYUI) == (ROOT, target, app)
            os.environ.update(MPT_MODELS=version, COMFYUI_ENDPOINT=endpoint)
            with patch.object(os, "execv") as launch:
                entrypoint.main(["api"])
            launch.assert_called_once_with(sys.executable, [sys.executable, "-B", str(target / "main.py")])
        else:
            command = [sys.executable, str(ROOT / "install.py"), "configure", str(target),
                       "--model-version", version, "--workflow-api", str(workflow_api)]
            if remote:
                command.extend(["--remote-comfyui", endpoint, "--allow-insecure-comfyui-http"])
            else:
                command.extend(["--comfyui", str(app), "--comfy-python", sys.executable, "--endpoint", endpoint])
            if external_server:
                command.append("--external-server")
            subprocess.run(command, check=True)
        sys.path.insert(0, str(target))
        from app.asgi import get_application
        from app.config import config
        from app.models.schema import VideoParams
        from app.services import llm, ltx, state, task, upload_post
        from app.utils import utils
        from fastapi.testclient import TestClient
        assert Path(ltx.settings()["operator_script"]) == ROOT / "runtime/scripts/ltx_operator.py"
        if IN_IMAGE:
            editor = Path("mpt_scene_editor/js/editor.js")
            assert (app / "custom_nodes" / editor).read_bytes() == (ROOT / "runtime/comfyui" / editor).read_bytes()
        request = json.loads(Path(ltx.settings()["request_template"]).read_bytes())
        profile = json.loads(Path(request["profile"]["source_path"]).read_bytes())
        assert profile.get("remote_server", False) is remote
        if not remote:
            assert profile.get("external_server", False) is (external_server or IN_IMAGE)
        assert request["variant"]["model_version"] == version and request["prompt"] == ""
        assert not upload_post.upload_post_service.is_configured()
        task_id = str(uuid4())
        audio = Path(utils.task_dir(task_id)) / "input.wav"
        text = "A paper boat floats across a pond."
        visual = "A plain white paper boat drifts across the water. The camera stays at water level."
        if version == "h3":
            visual = ("integrated_multimodal_description: [Shot 1] " + visual +
                      "\noverall_soundscape: Water laps against the paper.\nnon_diegetic_music: N/A")
        scene = [{"index": 1, "duration_seconds": duration, "narration": text, "visual_prompt": visual,
                  "negative_prompt": ""}]
        if version == "2.5":
            scene.append({"index": 2, "duration_seconds": duration, "narration": "It reaches the far bank.",
                          "visual_prompt": "A paper boat nudges the reeds at the water's edge.", "negative_prompt": ""})
            text += " " + scene[1]["narration"]
        if not remote:
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                            f"sine=frequency=440:duration={len(scene) * duration}", str(audio)], check=True)
        params = VideoParams(video_subject="Paper boat", video_script=text, video_source="ltx",
                             ltx_model_version=version,
                             ltx_generation_width=256, ltx_generation_height=256,
                             custom_audio_file="" if remote else str(audio),
                             voice_name="no-voice" if remote else "", subtitle_enabled=False,
                             bgm_type="", n_threads=1, ltx_max_scenes=len(scene), ltx_scene_duration_seconds=duration,
                             video_aspect="9:16")
        with TestClient(get_application()) as api:
            with patch.object(llm, "_generate_response", return_value=json.dumps(scene)) as director:
                task.start(task_id, params)
            director.assert_called_once()
            assert text in director.call_args.args[0]
            assert "1:1 (256×256 pixels)" in director.call_args.args[0]
            route = f"/api/v1/tasks/{task_id}/ltx"
            assert state.state.get_task(task_id)["phase"] == "awaiting_approval", state.state.get_task(task_id)
            review = api.get(route + "/plans")
            assert review.status_code == 200
            plans = [item["plan"] for item in review.json()["data"]["scenes"]]
            assert len(plans) == len(scene)
            for plan, shot in zip(plans, scene):
                assert plan["variant"]["model_version"] == version
                expected_negative = shot["negative_prompt"] or request["negative_prompt"]
                assert plan["negative_prompt"] == expected_negative
                assert plan["prompt"] == shot["visual_prompt"]
                assert any(plan["prompt"] in n["inputs"].values() for n in plan["api_prompt"].values())
            assert api.post(route + "/render").status_code == 409
            if version == "2.5" and not remote:
                from ltx_operator_lib.api import ComfyClient
                from ltx_operator_lib.preparation import prepare_request

                original_batch = ltx.load_batch(task_id)
                originals = [Path(record["plan_file"]).read_bytes() for record in original_batch["scenes"]]
                editor_url = ltx.open_scene_editor(task_id)
                editor = ltx.load_batch(task_id)["editor"]
                assert editor_url == endpoint + "/?mpt_edit=" + editor["session"]
                client = ComfyClient(endpoint)
                prefix = f"mpt-scenes/{editor['session']}/"
                exported = client.transport.request("GET", "/userdata/" + quote(prefix + "manifest.json", safe=""))
                assert len(exported["scenes"]) == len(scene) and not submissions
                for item, plan in zip(exported["scenes"], plans, strict=True):
                    index = item["index"]
                    seeds = [1000 + index * 10 + offset for offset in range(len(
                        [key for key in plan["parameters"] if key == "seed" or key.endswith("_seed")]))]
                    edited = prepare_request(plan, {"prompt": f"Manually edited scene {index}: " + plan["prompt"], "seeds": seeds})
                    envelope = {"schema_version": 1, "session": exported["session"], "revision": exported["revision"],
                                "index": index, "base_approval_hash": item["base_approval_hash"], "api_prompt": edited["api_prompt"]}
                    client.transport.request("POST", "/userdata/" + quote(prefix + f"scene-{index:03d}.json", safe=""), envelope)
                adopted = ltx.apply_scene_editor(task_id)
                assert adopted["phase"] == "awaiting_approval" and adopted["edited_in_comfyui"] is True
                assert all(record["status"] == "planned" and not record["submitted_approval_hash"] for record in adopted["scenes"])
                assert originals == [Path(record["plan_file"]).read_bytes() for record in original_batch["scenes"]]
                assert not submissions
                review = api.get(route + "/plans")
                assert review.status_code == 200
                plans = [item["plan"] for item in review.json()["data"]["scenes"]]
                for old, plan in zip(original_batch["scenes"], plans, strict=True):
                    assert plan["approval_hash"] != old["approval_hash"] and plan["prompt"].startswith("Manually edited scene ")
                assert api.post(route + "/render").status_code == 409
            approvals = [{"scene_index": s["index"], "approval_hash": s["plan"]["approval_hash"]}
                         for s in review.json()["data"]["scenes"]]
            assert api.post(route + "/approvals", json={"approvals": approvals}).status_code == 200
            assert not submissions
            assert api.post(route + "/render").status_code == 200
            until = time.monotonic() + 100
            while time.monotonic() < until:
                result = state.state.get_task(task_id)
                if result.get("phase") in ("complete", "failed"):
                    break
                time.sleep(0.1)
            assert result["phase"] == "complete", result
            assert len(submissions) == len(scene)
            assert [entry["prompt"][2] for entry in submissions.values()] == [plan["api_prompt"] for plan in plans]
            if remote:
                assert len(downloads) == len(scene) and len(set(downloads)) == len(scene)
                assert not app.exists()
                for record in ltx.load_batch(task_id)["scenes"]:
                    output = Path(record["output_file"])
                    assert output.is_relative_to(work / "home") and output.read_bytes() == clip.read_bytes()
            assert len(list((work / "home/storage/ltx-state/receipts").glob("*.json"))) == len(scene)
            video = Path(result["videos"][0])
            streams = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams",
                                                         "-of", "json", str(video)]))["streams"]
            assert {s["codec_type"] for s in streams} >= {"video", "audio"}
            video_stream = next(s for s in streams if s["codec_type"] == "video")
            assert (video_stream["width"], video_stream["height"]) == (256, 256)
            samples = subprocess.check_output(["ffmpeg", "-nostdin", "-v", "error", "-i", str(video),
                                                "-map", "0:a:0", "-f", "s16le", "-ac", "1", "-"])
            assert any(samples), "The final video's audio track is silent"
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(video), "-f", "null", "-"], check=True)
            model = "MiniMax H3" if version == "h3" else f"LTX-{version}"
            manual = "manually edited and " if version == "2.5" and not remote else ""
            delivery = ", downloaded over HTTP" if remote else ""
            sound = "scene audio" if remote else "narration"
            print(f"PASS {model}: {len(scene)} {manual}reviewed scene(s), each submitted once{delivery}, decoded MP4 with {sound}.")
            if alternate:
                before = ltx.load_batch(task_id)["scenes"]
                old_clip = Path(before[0]["output_file"])
                old_video_bytes = video.read_bytes()
                retake = task.render_ltx(task_id, params, result["audio_file"], "",
                                         retake=(1, before[0]["approval_hash"], alternate[0], "42"))
                after = ltx.load_batch(task_id)["scenes"]
                assert len(submissions) == len(scene) + 1
                if remote:
                    assert len(downloads) == len(scene) + 1 and len(set(downloads)) == len(downloads)
                assert after[1:] == before[1:]
                assert after[0]["approval_hash"] != before[0]["approval_hash"]
                assert Path(after[0]["output_file"]).read_bytes() == alternate[1].read_bytes()
                assert old_clip.read_bytes() == clip.read_bytes() and video.read_bytes() == old_video_bytes
                new_video = Path(retake["videos"][0])
                assert new_video != video and new_video.read_bytes() != old_video_bytes
                subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(new_video),
                                "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"], check=True)
                print("PASS scene retake: one replacement rendered, other scenes and previous MP4 preserved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpt", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--model-version", choices=MODEL_VERSIONS)
    parser.add_argument("--external-server", action="store_true")
    parser.add_argument("--remote", action="store_true")
    args = parser.parse_args()
    if args.remote and (not args.model_version or args.external_server or IN_IMAGE):
        parser.error("--remote requires an explicit model and a native test process")
    if IN_IMAGE and not args.model_version:
        parser.error("Run a separate image smoke test per model")
    if args.model_version:
        verify(args.mpt.resolve(), args.work_dir.resolve(), args.model_version, args.external_server, args.remote)
    else:
        args.work_dir.mkdir(parents=True, exist_ok=False)
        for version in MODEL_VERSIONS:
            command = [sys.executable, __file__, "--model-version", version, "--mpt", str(args.mpt),
                       "--work-dir", str(args.work_dir / version)]
            if args.external_server:
                command.append("--external-server")
            subprocess.run(command, check=True)
        if not args.external_server:
            subprocess.run([sys.executable, __file__, "--model-version", "2.5", "--remote", "--mpt", str(args.mpt),
                            "--work-dir", str(args.work_dir / "remote")], check=True)
