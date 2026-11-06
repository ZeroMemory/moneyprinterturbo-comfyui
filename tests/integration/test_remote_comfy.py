"""Check the remote server trust boundary without sharing model files."""
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from threading import Thread
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

ADDON = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ADDON), str(ADDON / "tests")]
import configure_ltx
from ltx_operator_lib.api import ComfyClient, PromptOwnershipError
from ltx_operator_lib.profiles import profile_from_dict, validate_profile
from ltx_operator_lib import runs
from workflow_fixtures import sample_graph, write_workflow


def test_remote_profile_uses_advertised_weights_without_local_install(tmp_path, monkeypatch):
    graph = sample_graph("2.5")
    source = write_workflow(tmp_path / "workflow.json", graph)
    info = {}
    for node in graph.values():
        fields = info.setdefault(node["class_type"], {"input": {"required": {}}})["input"]["required"]
        for field in configure_ltx.MODEL_INPUTS.get(node["class_type"], {}):
            fields.setdefault(field, [[]])[0].append(node["inputs"][field])
    for node in info.values():
        for field in node["input"]["required"].values():
            field[0] = sorted(set(field[0]))

    class Client:
        def object_info(self):
            return info

    monkeypatch.setattr(configure_ltx, "ComfyClient", lambda endpoint, **options: Client())
    profile, request = configure_ltx.build_template(
        None, tmp_path / "profile.json", endpoint="https://comfy.example",
        remote_server=True, workflow_api=source,
    )
    assert profile["remote_server"] is True
    assert not validate_profile(profile_from_dict(profile, tmp_path))
    assert request["profile"]["remote_server"] is True
    assert request["requirements"]["required_models"]
    assert not request["requirements"]["resolved_models"]

    first = next(iter(field for node in info.values() for field in node["input"]["required"].values()))
    first[0].clear()
    with pytest.raises(ValueError, match="does not advertise one matching weight"):
        configure_ltx.build_template(
            None, tmp_path / "profile.json", endpoint="https://comfy.example",
            remote_server=True, workflow_api=source,
        )


@pytest.mark.parametrize(
    "endpoint,allowed_http",
    [("http://192.168.1.20:8188", False), ("http://203.0.113.20:8188", True),
     ("https://user:secret@comfy.example", False), ("https://comfy.example/other", False)],
)
def test_remote_profile_rejects_untrusted_endpoint(tmp_path, endpoint, allowed_http):
    profile = profile_from_dict({
        "name": "remote-comfyui", "endpoint": endpoint,
        "remote_server": True, "allow_insecure_http": allowed_http,
    }, tmp_path)
    assert validate_profile(profile)


def test_only_approved_save_video_output_is_downloaded(tmp_path):
    root = tmp_path / "outputs"
    graph = {"3": {"class_type": "SaveVideo", "inputs": {"format": "mp4", "codec": "h264"}}}
    outputs = [
        {"filename": "unrelated.mp4", "subfolder": "scene", "type": "output"},
        {"filename": "clip_00001_.mp4", "subfolder": "scene", "type": "output"},
    ]
    fetched = []

    def download(file, path):
        fetched.append(file)
        path.parent.mkdir(parents=True)
        path.write_bytes(b"Verified output")

    result = runs._collect_outputs(
        {"outputs": {"3": {"images": outputs}}}, root, graph=graph, prefixes={"3": "scene/clip"},
        expected=["scene/clip*.mp4"], primary_node="3", download=download,
    )
    assert fetched == [outputs[1]]
    assert result == [{
        "path": str(root / "scene" / outputs[1]["filename"]),
        "content_sha256": sha256(b"Verified output").hexdigest(),
        "primary": True,
    }]


def test_interrupted_download_retries_view_without_submitting(tmp_path, monkeypatch):
    seen = []

    class ComfyHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.path)
            body = b"vid" if len(seen) == 1 else b"verified-clip"
            self.send_response(200)
            self.send_header("Content-Length", str(len(b"verified-clip")))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), ComfyHandler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = ComfyClient(
                f"http://127.0.0.1:{server.server_port}", owned_prompt_ids={"owned"},
                remote_server=True, allow_insecure_http=True,
            )
            with pytest.raises(PromptOwnershipError):
                client.download_output("foreign", "clip.mp4", "scene", None)
            clip = tmp_path / "clips" / "clip.mp4"
            monkeypatch.setattr(runs.subprocess, "run", lambda *_args, **_kwargs:
                                SimpleNamespace(returncode=0, stdout=b"video\n"))
            runs._download_clip(client, "owned", {"filename": clip.name, "subfolder": "video/scene"}, clip, None)
        finally:
            server.shutdown()
            thread.join()
    assert clip.read_bytes() == b"verified-clip"
    assert list(clip.parent.iterdir()) == [clip]
    assert [urlsplit(path).path for path in seen] == ["/view", "/view"]
    assert all(parse_qs(urlsplit(path).query) == {
        "filename": ["clip.mp4"], "subfolder": ["video/scene"], "type": ["output"],
    } for path in seen)



def test_json_response_requires_complete_content_length():
    from http.client import IncompleteRead

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = b'{"queue_running":[],"queue_pending":[]}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body) + 100))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with pytest.raises(IncompleteRead):
                ComfyClient(f"http://127.0.0.1:{server.server_port}").queue()
        finally:
            server.shutdown()
            thread.join()


@pytest.mark.parametrize("transport", ["https", "websocket"])
def test_connect_deadline_includes_dns(monkeypatch, transport):
    import socket
    from threading import Event

    release = Event()
    resolved = Event()

    def stalled_dns(*args, **kwargs):
        release.wait(1)
        resolved.set()
        raise socket.gaierror("fixture did not resolve a host")

    monkeypatch.setattr(socket, "getaddrinfo", stalled_dns)
    try:
        with pytest.raises(TimeoutError):
            if transport == "https":
                ComfyClient("https://comfy.example", remote_server=True).queue(timeout=.1)
            else:
                pytest.importorskip("websocket")
                from ltx_operator_lib.api import _default_websocket_events
                list(_default_websocket_events("http://localhost:8188", "fixture", .1))
        assert not resolved.is_set(), "the deadline must elapse while DNS is still waiting"
    finally:
        release.set()


@pytest.mark.parametrize("damaged", [False, True])
@pytest.mark.parametrize("remote", [False, True])
def test_video_is_decodable_before_delivery(tmp_path, damaged, remote):
    import shutil
    import struct
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg executable is unavailable")
    sample = tmp_path / "source.mp4"
    subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=red:s=64x64:r=24:d=1",
                    "-c:v", "libx264", "-movflags", "+faststart", str(sample)], check=True)
    body = bytearray(sample.read_bytes())
    if damaged:
        offset = 0
        while offset < len(body):
            size, kind = struct.unpack_from(">I4s", body, offset)
            if kind == b"mdat":
                body[offset + 8:offset + size] = b"\xff" * (size - 8)
                break
            offset += size
        else:
            pytest.fail("Fixture video does not have an mdat atom")

    def download_output(prompt_id, filename, subfolder, output, *, timeout):
        output.write(body)

    clip = tmp_path / "received" / "clip.mp4"
    if remote:
        client = SimpleNamespace(download_output=download_output)
        deliver = lambda: runs._download_clip(client, "owned", {"filename": clip.name}, clip, None)
    else:
        from ltx_operator_lib.api import CompletionResult

        clip.parent.mkdir()
        clip.write_bytes(body)
        graph = sample_graph()
        graph["save"]["inputs"]["filename_prefix"] = "clip"
        plan = {"api_prompt": graph, "variant": {"adapter_family": "t2v-i2v-two-stage"},
                "profile": {}, "client_id": "fixture", "outputs": ["clip*.mp4"]}
        history = {"owned": {"outputs": {"save": {"images": [{"filename": clip.name, "type": "output"}]}}}}
        client = SimpleNamespace(wait_for_completion=lambda *a, **kw: CompletionResult("owned", "completed", history))
        run = runs._SceneRun(plan, "a" * 64, tmp_path / "run")
        run.report["prompt_id"] = "owned"
        profile = SimpleNamespace(remote_server=False, output_directory=clip.parent)
        deliver = lambda: run.finish(client, profile, None, batch=True, cleanup_budget=lambda: 1)
    if damaged:
        with pytest.raises(ValueError, match="valid video"):
            deliver()
        if remote:
            assert list(clip.parent.iterdir()) == []
    else:
        deliver()
        assert clip.read_bytes() == bytes(body)


def test_lost_response_after_post_is_not_a_pre_send_failure():
    import json
    from http.client import RemoteDisconnected

    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            self.close_connection = True

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = ComfyClient(f"http://127.0.0.1:{server.server_port}")
            with pytest.raises(RemoteDisconnected):
                client.submit({}, client_id="fixture")
        finally:
            server.shutdown()
            thread.join()
    assert received == [("/prompt", {"prompt": {}, "client_id": "fixture"})]


@pytest.mark.parametrize("outcome", ["pre_submit_timeout", "unknown", "rejected", "connect_failed"])
def test_prediction_submission_records_are_bound_to_what_happened(tmp_path, monkeypatch, request, outcome):
    import json
    import socket
    from io import StringIO
    from ltx_operator_lib.api import ComfyHttpError, ComfyPromptNotSubmitted
    from ltx_operator_lib.cli import run_cli
    from ltx_operator_lib import duration
    from smoke_mpt import fixture_object_info

    graph = sample_graph()
    info = fixture_object_info(None, graph)
    info.update({"ModelPatchLoader": {"input": {"required": {"name": [[duration.DURATION_HEAD]]}}},
                 "LTXVDurationPredictor": {}, "PreviewAny": {}})

    def submit(*args, **kwargs):
        assert outcome != "pre_submit_timeout", "deadline expired before the POST"
        raise ComfyHttpError(400, "validation failed", prompt_rejected=outcome == "rejected")

    if outcome == "pre_submit_timeout":
        submitting = False
        write_report = duration.replace_json_atomic

        def persist_report(path, report):
            nonlocal submitting
            write_report(path, report)
            submitting = report["status"] == "submitting"

        def remaining_timeout(deadline):
            if submitting:
                raise TimeoutError("fixture expired while writing submission intent")
            return 1

        monkeypatch.setattr(duration, "replace_json_atomic", persist_report)
        monkeypatch.setattr(duration, "remaining_timeout", remaining_timeout)

    endpoint = "https://comfy.example"
    options = {}
    if outcome == "connect_failed":
        # Reserve the port without listening so that no request can be accepted.
        unlistening = socket.socket()
        request.addfinalizer(unlistening.close)
        unlistening.bind(("127.0.0.1", 0))
        endpoint = f"http://127.0.0.1:{unlistening.getsockname()[1]}"
        options["allow_insecure_http"] = True
        submit = ComfyClient(endpoint, remote_server=True, **options).submit
    client = SimpleNamespace(object_info=lambda: info, submit=submit)
    monkeypatch.setattr(configure_ltx, "ComfyClient", lambda *args, **kwargs: client)
    profile_path = tmp_path / "profile.json"
    profile, template = configure_ltx.build_template(
        None, profile_path, endpoint=endpoint, remote_server=True,
        workflow_api=write_workflow(tmp_path / "export.json", graph), **options,
    )
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    state_root = tmp_path / "state"
    run = tmp_path / "runs" / "001"
    manifest = {"template": template, "scenes": [{"index": 1, "visual_prompt": "A paper boat"}],
                "max_seconds": 3, "runs_dir": str(run.parent), "state_root": str(state_root)}
    expected = {"pre_submit_timeout": TimeoutError, "connect_failed": ComfyPromptNotSubmitted}
    with pytest.raises(expected.get(outcome, ComfyHttpError)) as failure:
        duration.predict_durations(manifest, client_factory=lambda *args, **kwargs: client)
    if outcome == "connect_failed":
        # A bound, non-listening socket can time out on macOS.
        assert isinstance(failure.value.__cause__, (ConnectionRefusedError, TimeoutError))
    report = json.loads((run / "report.json").read_bytes())
    record = run / "run.json"
    if outcome in {"rejected", "connect_failed"}:
        assert report["status"] == "submission_rejected"
        stdout = StringIO()
        assert run_cli(["status", "--run-record", str(record), "--state-root", str(state_root), "--json"],
                       stdout=stdout) == 0
        assert json.loads(stdout.getvalue()) == {"status": "submission_rejected", "prompt_id": None,
                                                 "approval_hash": report["approval_hash"],
                                                 "prediction_scene_index": 1}
    else:
        assert report["status"] == ("not_submitted" if outcome == "pre_submit_timeout" else "uncertain_submission")
        assert not record.exists()

