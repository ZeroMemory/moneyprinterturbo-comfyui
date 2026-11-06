"""The scene director must describe what the selected graph can actually render."""
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from app.config import config
from app.models import const
from app.models.schema import VideoParams
from app.services import llm, ltx, state, task


def template(cfg, *, version=None, media=()):
    variant = {"name": "custom", "adapter_family": "t2v-i2v-two-stage"}
    if version:
        variant.update(model_version=version, supports_negative_prompt=cfg != 1)
    nodes = {
        "model": {"class_type": "UNETLoader", "inputs": {}},
        "select": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
        "g": {"class_type": "CFGGuider", "inputs": {"cfg": cfg, "model": ["model", 0]}},
        "s": {"class_type": "SamplerCustomAdvanced", "inputs": {"guider": ["g", 0], "sampler": ["select", 0]}},
    }
    if media:
        nodes["i"] = {"class_type": "LoadImage", "inputs": {"image": "private-name.png"}}
    return {"variant": variant, "input_media": list(media), "api_prompt": nodes,
            "parameters": {"fps": 24, "seed": 1}}


def configure(monkeypatch, tmp_path, request):
    path = tmp_path / "private-request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    monkeypatch.setattr(config, "ltx", {"enabled": True, "request_template": str(path),
                                        "python_executable": sys.executable, "operator_script": __file__})


@pytest.mark.parametrize("sampler, capability, expected", [
    ("euler_ancestral_cfg_pp", None, True), ("euler_cfg_pp", True, True),
    ("euler_ancestral_cfg_pp", False, False), ("euler_ancestral", None, False),
    ("external_sampler", None, True),
])
def test_scene_writer_respects_sampler_and_explicit_negative_capability(sampler, capability, expected):
    request = template(1)
    request["api_prompt"]["select"]["inputs"]["sampler_name"] = sampler
    if capability is not None:
        request["variant"]["supports_negative_prompt"] = capability
    assert ltx.scene_prompt_context(request)["supports_negative_prompt"] is expected


@pytest.mark.parametrize("version,heading,syntax", [
    ("2.3", "## LTX-2.3", "one continuous take"),
    ("2.5", "## LTX-2.5", "name the transition in prose"),
    ("h3", "## MiniMax H3", "integrated_multimodal_description:"),
])
def test_outbound_scene_director_messages_match_selected_workflow(monkeypatch, tmp_path, version, heading, syntax):
    configure(monkeypatch, tmp_path, template(1, version=version))
    monkeypatch.setattr(config, "app", {"llm_provider": "nvidia", "nvidia_api_key": "offline-test",
                                        "nvidia_base_url": "http://director.invalid/v1",
                                        "nvidia_model_name": "offline/director"})
    narration = "В пекарне начинается смена."
    direction = "Пекарь говорит в кадре: «Доброе утро!»"
    visual = 'A baker opens the shop and says in Russian, "Доброе утро!"'
    if version == "h3":
        visual = ("integrated_multimodal_description: [Shot 1] A baker (S1) opens the shop "
                  "and says <d>[Russian] Доброе утро!</d>\n\n"
                  "overall_soundscape: Keys jingle in the door.\n\nnon_diegetic_music: N/A")
    scene = {"index": 1, "duration_seconds": 5, "narration": narration,
             "visual_prompt": visual}
    captured = []

    def respond(transport, request):
        assert str(request.url) == "http://director.invalid/v1/chat/completions"
        body = json.loads(request.content)
        assert body["stream"] is True
        assert request.extensions["timeout"]["read"] == 600
        captured.append(body["messages"])
        scenes = [dict(scene, negative_prompt="logos" if len(captured) == 1 else "")]
        chunk = {"id": "offline", "model": "offline/director", "object": "chat.completion.chunk", "created": 0,
                 "choices": [{"index": 0, "finish_reason": "stop", "delta": {"content": json.dumps(scenes)}}]}
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode())

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", respond)
    result = llm.generate_ltx_scene_plan("Утро", narration, 5, 5, 1, "16:9", language="ru",
                                         video_script_prompt=direction, workflow=ltx.scene_prompt_context())
    assert result[0]["negative_prompt"] == "" and result[0]["narration"] == narration
    assert result[0]["visual_prompt"] == visual
    assert len(captured) == 2 and captured[0] == captured[1]
    message, = captured[0]
    assert message["role"] == "user"
    prompt = message["content"]
    assert {h for h in ("## LTX-2.3", "## LTX-2.5", "## MiniMax H3") if h in prompt} == {heading}
    assert syntax in prompt and narration in prompt and direction in prompt
    assert "private-request.json" not in prompt and str(tmp_path) not in prompt
    if version == "h3":
        assert "overall_soundscape:" in prompt and "non_diegetic_music:" in prompt
        assert "<d>[Language] exact words</d>" in prompt
        assert "4 to 15" in prompt and "5.17 seconds" in prompt
    else:
        assert "<d>" not in prompt


def test_h3_retries_a_scene_outside_its_duration_range(monkeypatch):
    scene = {"index": 1, "narration": "A boat floats.", "visual_prompt": "A boat crosses the pond."}
    durations = iter([3, 4.5, 5])
    monkeypatch.setattr(llm, "_generate_response",
                        lambda prompt, **kwargs: json.dumps([dict(scene, duration_seconds=next(durations))]))
    result = llm.generate_ltx_scene_plan("Boat", scene["narration"], 5, 5, 1, "16:9",
                                         workflow={"model_version": "h3"})
    assert result[0]["duration_seconds"] == 5


def test_custom_reference_prompt_uses_only_known_capabilities(monkeypatch, tmp_path):
    private = str(tmp_path / "private-name.png")
    request = template(4, version=private, media=[private])
    configure(monkeypatch, tmp_path, request)
    prompt = llm.build_ltx_scene_plan_prompt("Boat", "A boat floats.", 3, 3, 1, "16:9",
                                            workflow=ltx.scene_prompt_context())
    assert "custom workflow" in prompt and "Reference media is present" in prompt
    assert "private-name.png" not in prompt and str(tmp_path) not in prompt
    assert "## LTX-" not in prompt and "## MiniMax" not in prompt
    assert "An explicit exclusion requested by the user" in prompt


@pytest.mark.parametrize("guider,inputs", [
    ("CFGGuider", {"cfg": 1}),
    ("LTXVDualCFGGuider", {"video_cfg": 1, "audio_cfg": 1}),
])
def test_manual_plan_with_ineffective_negative_fails_before_any_scene_write(monkeypatch, tmp_path, guider, inputs):
    request = template(1)  # an existing request without capability metadata
    request["api_prompt"]["g"] = {"class_type": guider, "inputs": {**inputs, "model": ["model", 0]}}
    configure(monkeypatch, tmp_path, request)
    folder = tmp_path / "task"
    monkeypatch.setattr(ltx.utils, "task_dir", lambda task_id: str(folder))
    monkeypatch.setattr(ltx, "run_cli", lambda *args: pytest.fail("unsupported exclusions reached the operator"))
    params = SimpleNamespace(ltx_scene_duration_seconds=3, ltx_max_scenes=1, video_aspect="16:9")
    scenes = [{"index": 1, "narration": "A boat floats.", "visual_prompt": "A blank white boat floats.",
               "negative_prompt": "logos"}]
    with pytest.raises(ltx.LtxError, match="negative_prompt"):
        ltx.plan_batch("task", "A boat floats.", 3, params, scene_plan=scenes)
    assert not folder.exists()


def test_custom_workflow_with_effective_guidance_keeps_negative_prompt(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path, template(4))
    scene = {"index": 1, "duration_seconds": 3, "narration": "A boat floats.",
             "visual_prompt": "A white boat floats.", "negative_prompt": "lettering"}
    monkeypatch.setattr(llm, "_generate_response", lambda prompt, **kwargs: json.dumps([scene]))
    planned = llm.generate_ltx_scene_plan("Boat", scene["narration"], 3, 3, 1, "16:9",
                                          workflow=ltx.scene_prompt_context())
    assert planned[0]["negative_prompt"] == "lettering"


@pytest.mark.parametrize("value", [None, 17, {"text": "unexpected"}, " \t\n"])
def test_scene_requires_visual_text_before_planning_or_approval(monkeypatch, value):
    scene = {"index": 1, "narration": "The train arrives.", "visual_prompt": value}
    monkeypatch.setattr(llm, "_generate_response", lambda prompt, **kwargs: json.dumps([scene]))
    with pytest.raises(ValueError, match="visual_prompt"):
        llm.generate_ltx_scene_plan("Train", scene["narration"], 3, 3, 1, "16:9")
    with pytest.raises(ltx.LtxError, match="visual_prompt"):
        ltx.validate_scene_plan([scene], 3, 1)


def test_director_publishes_partial_response_and_can_stop_before_completion(monkeypatch):
    from app.services import execution_control

    monkeypatch.setattr(config, "app", {"llm_provider": "nvidia", "nvidia_api_key": "offline-test",
                                        "nvidia_base_url": "http://director.invalid/v1",
                                        "nvidia_model_name": "offline/director"})
    updates, requests = [], []
    closed = []

    class Reply(httpx.SyncByteStream):
        def __iter__(self):
            chunk = {"id": "offline", "model": "offline/director", "object": "chat.completion.chunk", "created": 0,
                     "choices": [{"index": 0, "finish_reason": None,
                                  "delta": {"content": '[{"index": 1,'}}]}
            yield ("data: " + json.dumps(chunk) + "\n\n").encode()
            assert updates[-1] == '[{"index": 1,'
            execution_control.stop_all()
            yield b'data: [DONE]\n\n'

        def close(self):
            closed.append(True)

    def respond(transport, request):
        requests.append(request)
        assert request.extensions["timeout"]["read"] == 600
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Reply())

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", respond)
    with pytest.raises(execution_control.GenerationStopped):
        llm.generate_ltx_scene_plan("Boat", "A boat floats.", 3, 3, 1, "16:9", on_progress=updates.append)
    assert len(requests) == 1 and closed
    with execution_control.operation():
        execution_control.checkpoint()


@pytest.fixture
def scene_task(monkeypatch):
    narration = "A paper boat crosses a quiet pond."
    scene = {"index": 1, "duration_seconds": 3, "narration": narration,
             "visual_prompt": "A white paper boat floats slowly across a quiet pond."}
    fixture = SimpleNamespace(narration=narration, scene=scene, requests=[], clients=[],
                              planned=[], error=False)
    monkeypatch.setattr(config, "app", {"llm_provider": "nvidia", "nvidia_api_key": "offline-test",
                                       "nvidia_base_url": "http://director.invalid/v1",
                                       "nvidia_model_name": "minimaxai/minimax-m3.1-flash"})
    monkeypatch.setattr(state, "state", state.MemoryState())
    monkeypatch.setattr(llm, "_max_retries", 1)
    fixture.probe = Mock(side_effect=AssertionError("automatic LLM probe must not run"))
    monkeypatch.setattr(llm, "probe_llm_model", fixture.probe)
    monkeypatch.setattr(task.utils, "check_ffmpeg_ready", lambda: True)
    monkeypatch.setattr(task, "save_script_data", lambda *args, **kwargs: None)
    monkeypatch.setattr(task, "generate_audio", lambda *args, **kwargs: ("unused-audio.mp3", 3, None))
    monkeypatch.setattr(task.voice, "get_audio_duration", lambda *args: 3.0)
    monkeypatch.setattr(task, "generate_subtitle", lambda *args: "")
    template = {"variant": {"model_version": "2.5"}, "parameters": {"fps": 24}, "api_prompt": {}}
    monkeypatch.setattr(ltx, "request_template", lambda *args: template)

    def plan_batch(*args, scene_plan, **kwargs):
        fixture.planned.append(scene_plan)
        return {"scenes": [{"index": 1, "status": "planned"}]}

    monkeypatch.setattr(ltx, "plan_batch", plan_batch)
    real_openai = llm.OpenAI

    def respond(request):
        assert str(request.url) == "http://director.invalid/v1/chat/completions"
        body = json.loads(request.content)
        fixture.requests.append(body)
        if fixture.error:
            return httpx.Response(400, json={"error": {"message": "offline director failed"}})
        if body.get("stream"):
            story_request = "# Video Scene Director" not in body["messages"][0]["content"]
            assert request.extensions["timeout"]["read"] == (60 if story_request else 600)
            payload = {"id": "offline", "model": body["model"], "created": 0,
                       "object": "chat.completion.chunk", "choices": [{"index": 0,
                       "finish_reason": "stop", "delta": {"content": narration if story_request else json.dumps([scene])}}]}
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  content=("data: " + json.dumps(payload) + "\n\ndata: [DONE]\n\n").encode())
        return httpx.Response(200, json={"id": "offline", "model": body["model"], "created": 0,
                              "object": "chat.completion", "choices": [{"index": 0,
                              "finish_reason": "stop", "message": {"role": "assistant", "content": narration}}]})

    def openai(**kwargs):
        fixture.clients.append(kwargs.copy())
        return real_openai(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond)))

    monkeypatch.setattr(llm, "OpenAI", openai)
    return fixture


def test_ltx_task_sends_full_scene_request_without_short_probe(scene_task):
    params = VideoParams(video_subject="Paper boat", video_script=scene_task.narration,
                         video_source="ltx", ltx_model_version="2.5",
                         ltx_scene_duration_seconds=3, ltx_max_scenes=1)
    result = task.start("scene-request", params, stop_at="materials")
    scene_task.probe.assert_not_called()
    assert result is not None, state.state.get_task("scene-request")
    assert result["phase"] == "awaiting_approval"
    assert result["ltx_scene_plan"] == scene_task.planned[0]
    assert result["ltx_scene_plan"][0]["visual_prompt"] == scene_task.scene["visual_prompt"]
    assert len(scene_task.requests) == len(scene_task.clients) == 1
    request, = scene_task.requests
    assert request["stream"] is True
    assert request["model"] == "minimaxai/minimax-m3.1-flash"
    prompt = request["messages"][0]["content"]
    assert "## LTX-2.5" in prompt and scene_task.narration in prompt
    assert "Reply with OK" not in prompt
    assert scene_task.clients[0]["timeout"] == 600
    assert scene_task.clients[0]["max_retries"] == 0
    assert state.state.get_task("scene-request")["phase"] == "awaiting_approval"


@pytest.mark.parametrize("uncertain", [False, True])
def test_predictor_failure_preserves_remote_stop_state(scene_task, monkeypatch, uncertain):
    error = ltx.LtxError("Duration predictor failed")
    error.remote_stop_unconfirmed = uncertain
    monkeypatch.setattr(task.voice, "generate_silent_audio", lambda *args: True)
    monkeypatch.setattr(llm, "generate_ltx_scene_plan", lambda **kwargs: [scene_task.scene])
    predictor = Mock(side_effect=error)
    monkeypatch.setattr(ltx, "predict_scene_durations", predictor)
    params = VideoParams(video_subject="Paper boat", video_script=scene_task.narration,
                         video_source="ltx", voice_name="no-voice", ltx_predict_duration=True,
                         ltx_scene_duration_seconds=3, ltx_max_scenes=1)
    assert task.start("failed-duration", params, stop_at="materials") is None
    saved = state.state.get_task("failed-duration")
    assert saved["state"] == const.TASK_STATE_FAILED
    assert saved["phase"] == ("stop_unconfirmed" if uncertain else "failed")
    assert saved["error"] == str(error)
    predictor.assert_called_once()
    assert not scene_task.planned


def test_script_task_sends_generation_request_without_short_probe(scene_task):
    result = task.start("script-request", VideoParams(video_subject="Paper boat"), stop_at="script")
    scene_task.probe.assert_not_called()
    assert result == {"script": scene_task.narration}
    assert len(scene_task.requests) == 1
    prompt = scene_task.requests[0]["messages"][0]["content"]
    assert "Paper boat" in prompt and "Reply with OK" not in prompt
    assert state.state.get_task("script-request")["state"] == const.TASK_STATE_COMPLETE


def test_visual_story_and_director_use_distinct_contracts_via_real_sdk(scene_task):
    params = VideoParams(video_subject="Paper boat", voice_name="no-voice", video_source="ltx",
                         ltx_model_version="2.5", ltx_scene_duration_seconds=3, ltx_max_scenes=3)
    timing = ltx.local_video_timing(params, template(1, version="2.5"))
    updates = []
    story = llm.generate_script(params.video_subject, local_video_context=timing, on_progress=updates.append)
    assert story == scene_task.narration and updates[-1] == story
    scenes = llm.generate_ltx_scene_plan(params.video_subject, story, 0, 3, 3, "16:9", timing=timing)
    assert scenes[0]["visual_prompt"] == scene_task.scene["visual_prompt"]
    assert len(scene_task.requests) == 2
    writer, director = [request["messages"][0]["content"] for request in scene_task.requests]
    assert "9.12s" in writer
    assert "Do not break the story into scenes or shots" in writer
    assert "Maximum scenes:" not in writer and "Each scene:" not in writer
    assert "camera movement" not in writer and "Return a JSON array" not in writer
    assert story in director and "max scenes: 3" in director and "scene duration seconds: 3" in director
    assert "Return a JSON array" in director
    assert all(request["stream"] for request in scene_task.requests)
    scene_task.probe.assert_not_called()


def test_real_director_error_is_logged_and_marks_task_failed(scene_task):
    scene_task.error = True
    errors = []
    sink = task.logger.add(lambda message: errors.append(str(message)), level="ERROR")
    try:
        params = VideoParams(video_subject="Paper boat", video_script=scene_task.narration,
                             video_source="ltx", ltx_model_version="2.5", ltx_max_scenes=1)
        result = task.start("scene-error", params, stop_at="materials")
    finally:
        task.logger.remove(sink)
    scene_task.probe.assert_not_called()
    assert result is None and not scene_task.planned
    failed = state.state.get_task("scene-error")
    assert failed["state"] == const.TASK_STATE_FAILED and failed["phase"] == "failed"
    assert "NVIDIA inference failed (HTTP 400)" in failed["error"]
    assert any("LTX scene planning failed:" in line and failed["error"] in line for line in errors)
    assert len(scene_task.requests) == 1


def test_local_timing_distinguishes_visual_and_measured_audio_budgets():
    request = template(1, version="2.5")
    params = VideoParams(video_subject="Boat", voice_name="no-voice", ltx_scene_duration_seconds=3, ltx_max_scenes=3,
                         video_clip_speed=2)
    timing = ltx.local_video_timing(params, request, audio_duration=241)
    assert timing["mode"] == "visual" and timing["audio_duration_seconds"] is None
    assert timing["frames_per_scene"] == 73
    assert timing["max_playback_seconds"] == pytest.approx(3 * 73 / 48)
    params.custom_audio_file = "fixture.wav"
    assert ltx.local_video_timing(params, request, audio_duration=3)["required_scenes"] == 2
    with pytest.raises(ValueError, match="needs 4 scenes"):
        ltx.local_video_timing(params, request, audio_duration=6)


def test_visual_director_accepts_fewer_scenes_but_preserves_selected_duration(monkeypatch):
    request = template(1, version="2.5")
    timing = ltx.local_video_timing(VideoParams(video_subject="Boat", voice_name="no-voice", ltx_scene_duration_seconds=3,
                                               ltx_max_scenes=3), request)
    scene = {"index": 1, "duration_seconds": 3, "narration": "A boat drifts.",
             "visual_prompt": "A red paper boat drifts across the pond."}
    monkeypatch.setattr(llm, "_generate_response", lambda *args, **kwargs: json.dumps([scene]))
    screenplay = "A boat drifts, then reaches the far bank."
    assert len(llm.generate_ltx_scene_plan("Boat", screenplay, 241, 3, 3, "16:9",
                                           timing=timing)) == 1
    prompt = llm.build_script_prompt("Boat", local_video_context=timing)
    assert "Video Story Writer" in prompt and "9.12" in prompt
    scene["duration_seconds"] = 4
    monkeypatch.setattr(llm, "_max_retries", 1)
    with pytest.raises(ValueError, match="selected 3 seconds"):
        llm.generate_ltx_scene_plan("Boat", screenplay, 241, 3, 3, "16:9", timing=timing)


def test_duration_prediction_respects_the_cap_and_requires_visual_ltx_25():
    params = VideoParams(video_subject="Boat", voice_name="no-voice", ltx_scene_duration_seconds=3,
                         ltx_max_scenes=8, ltx_predict_duration=True)
    request = template(1, version="2.5")
    timing = ltx.local_video_timing(params, request)
    assert timing["duration_mode"] == "predicted" and timing["frames_per_scene"] == 65
    assert timing["max_playback_seconds"] == pytest.approx(8 * 65 / 24)
    params.custom_audio_file = "voice.wav"
    with pytest.raises(ValueError, match="Voiceover None"):
        ltx.local_video_timing(params, request)
    params.custom_audio_file = None
    with pytest.raises(ValueError, match="LTX-2.5"):
        ltx.local_video_timing(params, template(1, version="2.3"))


@pytest.mark.parametrize("fps, cap, native_min, native_max, frames, raw", [
    (24.2, 3, 12, 65, 65, 6.155),
    (18.4, 3, 17, 55, 17, .3),
    (19.24, 25, 10, 481, 481, 28),
    (18.000000000000004, 1, 9, 18, 9, .3),
    (3, 1, None, None, None, .5),
])
def test_duration_prediction_uses_inclusive_frame_grid_bounds(monkeypatch, fps, cap, native_min, native_max, frames, raw):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "runtime/scripts"))
    from ltx_operator_lib.duration import _prediction, prediction_graph

    request = template(1, version="2.5")
    request["parameters"]["fps"] = fps
    request["api_prompt"].update({
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "encoder.safetensors"}},
        "text": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": ""}},
    })
    params = VideoParams(video_subject="Boat", voice_name="no-voice", ltx_scene_duration_seconds=cap,
                         ltx_predict_duration=True)
    if frames is None:
        with pytest.raises(ValueError, match=r"8k\+1 frame count"):
            prediction_graph(request, "A paper boat crosses a pond.", .5, cap)
        with pytest.raises(ValueError, match="at least 0.5 seconds"):
            ltx.local_video_timing(params, request)
        return
    graph = prediction_graph(request, "A paper boat crosses a pond.", .5, cap)
    inputs = graph["mpt_duration_predictor"]["inputs"]
    assert (round(inputs["min_seconds"] * fps), round(inputs["max_seconds"] * fps)) == (native_min, native_max)
    history = {"id": {"outputs": {"mpt_duration_frames": {"text": [str(frames)]},
                                  "mpt_duration_seconds": {"text": [str(raw)]}}}}
    assert _prediction(history, "id", fps, .5, cap)["frame_count"] == frames
    timing = ltx.local_video_timing(params, request)
    scene = {"index": 1, "duration_seconds": frames / fps, "frame_count": frames,
             "narration": "A paper boat crosses a pond.", "visual_prompt": "A paper boat crosses a pond."}
    assert ltx.validate_scene_plan([scene], cap, 1, "2.5", timing=timing)[0]["frame_count"] == frames


@pytest.mark.parametrize("speed", [0.5, 1.0, 2.0])
def test_scene_audio_preserves_order_speed_and_silent_gaps(monkeypatch, tmp_path, speed):
    import wave
    import numpy as np
    from moviepy import AudioClip

    closed = []

    class Source:
        def __init__(self, name):
            self.name = name
            self.audio = None if name == "silent" else AudioClip(
                lambda t: np.stack([.25 * np.sin(2 * np.pi * 440 * t)] * 2, axis=-1),
                duration=.1 if name == "first" else .2, fps=8000,
            )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            closed.append(self.name)
            if self.audio:
                self.audio.close()

    def open_source(name, *, audio):
        assert audio is True
        return Source(name)

    monkeypatch.setattr(task.video, "_open_video_clip_quietly", open_source)
    monkeypatch.setattr(task.file_security, "ensure_output_path", lambda *args: None)
    output = tmp_path / "scene-audio.wav"
    assert task._ltx_scene_audio(["first", "silent", "last"], [.2, .2, .2], speed, str(output))
    with wave.open(str(output)) as wav:
        rate = wav.getframerate()
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        samples = samples.reshape(-1, wav.getnchannels())[:, 0] / 32768
    assert len(samples) / rate == pytest.approx(.6 / speed, abs=2 / rate)

    def rms(source_time):
        start = int((source_time / speed) * rate)
        length = max(8, int(.02 / speed * rate))
        return float(np.sqrt(np.mean(samples[start:start + length] ** 2)))

    assert rms(.03) > .1 and rms(.45) > .1
    assert rms(.15) < .001 and rms(.3) < .001
    assert sorted(closed) == ["first", "last", "silent"]


def test_scene_audio_without_tracks_falls_back_without_writing(monkeypatch, tmp_path):
    from contextlib import contextmanager

    closed = []

    @contextmanager
    def silent_source(name, *, audio):
        assert audio is True
        try:
            yield SimpleNamespace(audio=None)
        finally:
            closed.append(name)

    monkeypatch.setattr(task.video, "_open_video_clip_quietly", silent_source)
    monkeypatch.setattr(task.file_security, "ensure_output_path", lambda *args: None)
    output = tmp_path / "scene-audio.wav"
    assert not task._ltx_scene_audio(["silent"], [1.0], 1.0, str(output))
    assert not output.exists() and closed == ["silent"]


@pytest.mark.parametrize("voice_name,custom_audio,keep_scene_audio", [
    ("no-voice", None, True),
    ("no-voice", "custom.wav", False),
    ("en-US-JennyNeural", None, False),
])
def test_ltx_no_voice_uses_scene_audio_without_changing_voiceover(tmp_path, monkeypatch, voice_name, custom_audio, keep_scene_audio):
    from contextlib import contextmanager

    @contextmanager
    def materials(*args):
        yield ["original.mp4"], ["snapshot.mp4"]

    scene_audio = Mock(return_value=True)
    final, combined = tmp_path / "final-1.mp4", tmp_path / "combined-1.mp4"
    for output in (final, combined):
        output.write_bytes(b"valid output fixture")
    compose = Mock(return_value=([str(final)], [str(combined)], []))
    silence = Mock(return_value=True)
    monkeypatch.setattr(ltx, "load_batch", lambda _: {"phase": "complete", "scenes": []})
    monkeypatch.setattr(ltx, "composition_materials", materials)
    monkeypatch.setattr(task, "_material_duration", lambda _: 1.5)
    monkeypatch.setattr(task, "_ltx_scene_audio", scene_audio, raising=False)
    monkeypatch.setattr(task, "generate_final_videos", compose)
    monkeypatch.setattr(task.sm, "state", task.sm.MemoryState())
    monkeypatch.setattr(task.utils, "task_dir", lambda _: str(tmp_path))
    monkeypatch.setattr(task.voice, "generate_silent_audio", silence)
    monkeypatch.setattr(task.voice, "get_audio_duration", lambda _: 1.5)
    params = VideoParams(video_subject="A paper boat", video_source="ltx", voice_name=voice_name,
                         custom_audio_file=custom_audio, voice_volume=0)
    result = task._finish_ltx("audio-test", params, "voice.wav", "")
    if keep_scene_audio:
        assert result["audio_file"].endswith("ltx-scene-audio.wav")
    else:
        assert result["audio_file"] == "voice.wav"
    assert scene_audio.call_count == int(keep_scene_audio)
    silence.assert_not_called()
    assert compose.call_args.args[1].voice_volume == (1.0 if keep_scene_audio else 0)


@pytest.mark.parametrize("script,completed", [
    ("A display reads Error: file not found, then the operator restores it.", True),
    ("Error: model unavailable", False),
])
def test_generated_script_error_prefix_controls_completion_status(monkeypatch, script, completed):
    updates = Mock()
    monkeypatch.setattr(task, "_update_llm_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(task.sm, "state", SimpleNamespace(update_task=updates))
    monkeypatch.setattr(task.llm, "generate_script", lambda **kwargs: script)
    assert task.generate_script("script-status", VideoParams(video_subject="A display")) == script
    assert updates.call_count == int(completed)



@pytest.mark.parametrize("first_status, expected", [
    (None, "failed"),
    ("not_submitted", "failed"),
    ("completed", "failed"),
    ("uncertain_submission", "stop_unconfirmed"),
])
def test_duration_recovery_distinguishes_unstarted_scenes(tmp_path, monkeypatch, first_status, expected):
    import time

    monkeypatch.setattr(ltx.sm, "state", state.MemoryState())
    task_id = "test-interrupted-prediction"
    ltx.sm.state.update_task(task_id, phase="ltx_duration_prediction")
    prediction = tmp_path / "attempt"
    first = prediction / "runs" / "001"
    first.mkdir(parents=True)
    (prediction / "manifest.json").write_text(json.dumps({"scenes": [{"index": 1}, {"index": 2}]}), encoding="utf-8")
    if first_status is not None:
        prompt_id = "first-prompt" if first_status == "completed" else None
        (first / "report.json").write_text(json.dumps(
            {"status": first_status, "scene_index": 1, "prompt_id": prompt_id}), encoding="utf-8")
        if prompt_id:
            from pathlib import Path
            monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "runtime/scripts"))
            from ltx_operator_lib.state import OperatorState
            record = OperatorState(tmp_path / "state").sign_record({
                "prompt_id": prompt_id, "run_directory": str(first.resolve()),
                "schema_version": 1, "endpoint": "http://127.0.0.1:8188", "approval_hash": "a" * 64,
            })
            (first / "run.json").write_text(json.dumps(record), encoding="utf-8")
    ltx._recover_interrupted_prediction(task_id, "ltx_duration_prediction", tmp_path, time.monotonic() + 4)
    assert ltx.sm.state.get_task(task_id)["phase"] == expected
