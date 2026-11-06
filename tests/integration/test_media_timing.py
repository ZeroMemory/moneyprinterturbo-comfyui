"""Timestamp and composition boundaries; no video encoding or provider requests."""
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import wave

import numpy as np
import pytest
from moviepy import AudioClip, VideoClip
from moviepy.video.io.ffmpeg_reader import FFMPEG_VideoReader

from app.models import const
from app.models.schema import VideoConcatMode, VideoParams
from app.services import llm, ltx, media_timing, state, task, video, voice


@pytest.fixture(autouse=True)
def clear_timing_cache():
    media_timing._video_timing.cache_clear()
    media_timing._frame_passthrough_args.cache_clear()
    yield
    media_timing._video_timing.cache_clear()
    media_timing._frame_passthrough_args.cache_clear()


@pytest.mark.parametrize("expected,error", [
    (None, "same dimensions"),
    ([(256, 256), (320, 256)], None),
    ([(256, 256), (256, 256)], "edited dimensions"),
    ([(256, 256)], "material count"),
])
def test_edited_materials_verify_each_scene_before_fitting_the_final_video(monkeypatch, expected, error):
    sizes = {"first.mp4": (256, 256), "second.mp4": (320, 256)}
    closed = []
    monkeypatch.setattr(video, "VideoFileClip", lambda name: SimpleNamespace(
        size=sizes[name], close=lambda: closed.append(name)))
    if error:
        with pytest.raises(RuntimeError, match=error):
            task._ltx_material_size(list(sizes), expected_sizes=expected)
    else:
        assert task._ltx_material_size(list(sizes), expected_sizes=expected) == (256, 256)
        assert closed == list(sizes)


@pytest.mark.parametrize("seconds,fits", [(5.16, True), (5.166, True), (5.17, False)])
def test_h3_capacity_uses_real_wav_samples(seconds, fits, tmp_path, monkeypatch):
    folder = tmp_path / "tasks" / "h3-fit"
    folder.mkdir(parents=True)
    source = folder / "narration.wav"
    with wave.open(str(source), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24000)
        output.writeframes(b"\x01\x00" * round(seconds * 24000))
    before = source.read_bytes()
    template = {"variant": {"model_version": "h3"}, "parameters": {"fps": 24}, "api_prompt": {}}
    monkeypatch.setattr(task.utils, "task_dir", lambda task_id="": str(folder.parent / task_id))
    monkeypatch.setattr(state, "state", state.MemoryState())
    monkeypatch.setattr(ltx, "request_template", lambda *a, **kw: template)
    monkeypatch.setattr(task.utils, "check_ffmpeg_ready", lambda: True)
    monkeypatch.setattr(voice, "tts", Mock(side_effect=AssertionError("Custom audio must skip TTS")))
    director = Mock(return_value=[{"index": 1, "duration_seconds": 5, "narration": "A boat drifts.", "visual_prompt": "A red boat drifts."}])
    monkeypatch.setattr(llm, "generate_ltx_scene_plan", director)
    planner = Mock(side_effect=lambda *a, scene_plan, **kw: {"scenes": [{**s, "status": "planned"} for s in scene_plan]})
    monkeypatch.setattr(ltx, "plan_batch", planner)
    params = VideoParams(video_subject="Boat", video_script="A boat drifts.", video_source="ltx", ltx_model_version="h3", voice_name="no-voice", custom_audio_file=str(source), subtitle_enabled=False, ltx_scene_duration_seconds=5, ltx_max_scenes=1)
    result = task.start("h3-fit", params, stop_at="materials")
    assert json.loads((folder / "script.json").read_text(encoding="utf-8"))["script"] == params.video_script
    assert source.read_bytes() == before
    if fits:
        assert result["phase"] == "awaiting_approval", result
        assert result["audio_duration"] == seconds
        assert director.call_args.kwargs["timing"]["required_scenes"] == 1
        assert director.call_count == planner.call_count == 1
    else:
        assert result["state"] == const.TASK_STATE_FAILED
        assert result["failed_stage"] == "audio" and "needs 2 scenes" in result["error"]
        director.assert_not_called()
        planner.assert_not_called()


@pytest.mark.parametrize("options,passthrough", [
    (b"-vsync              set video sync method", ["-vsync", "0"]),
    (b"-fps_mode[:<stream_spec>]  set framerate mode", ["-fps_mode", "passthrough"]),
])
def test_vfr_reader_uses_pts_and_last_frame_end(tmp_path, monkeypatch, options, passthrough):
    source = tmp_path / "source.bin"
    source.write_bytes(b"Timestamp fixture; decoding is replaced.")
    payload = {"streams": [{"time_base": "1/1000", "duration_ts": 600, "start_pts": 100}], "frames": [
        {"pts": 100, "duration": 40}, {"pts": 140, "duration": 110},
        {"pts": 250, "duration": 250}, {"pts": 500, "duration": 200},
    ]}
    monkeypatch.setattr(media_timing, "_probe", lambda *a, **kw: payload)
    timing = media_timing.video_timing(source)
    assert float(timing.duration) == .6 and timing.frame_rate is None
    help_command = Mock(return_value=SimpleNamespace(stdout=options))
    monkeypatch.setattr(media_timing.subprocess, "run", help_command)
    commands = []
    class Pipe:
        def __init__(self, command, **kwargs):
            commands.append(command)
            self.stdout = io.BytesIO(bytes([10] * 3 + [20] * 3 + [30] * 3 + [40] * 3))
            self.stderr = io.BytesIO()
        def terminate(self): pass
        def wait(self): return 0
        def poll(self): return None
    monkeypatch.setattr(media_timing.subprocess, "Popen", Pipe)
    original = FFMPEG_VideoReader.__new__(FFMPEG_VideoReader)
    original.__dict__.update(filename=str(source), size=(1, 1), pixel_format="rgb24", depth=3, resize_algo="bicubic", bufsize=103, fps=6.67, infos={}, duration=.6, n_frames=4, proc=None)
    reader = media_timing.PresentationTimeReader(original, timing)
    try:
        assert [int(reader.get_frame(t)[0, 0, 0]) for t in [0, .05, .2, .59, .01]] == [10, 20, 30, 40, 10]
        assert all(command[-3:-1] == passthrough and "-ss" not in command for command in commands)
        help_command.assert_called_once()
    finally:
        reader.close()


@pytest.mark.parametrize("speed", [.5, 1., 2., .7, 1.3])
@pytest.mark.parametrize("repeat", [False, True])
def test_directed_clips_cover_frame_grid_without_shifting_scene_starts(speed, repeat, tmp_path, monkeypatch):
    source, audio = tmp_path / "source.bin", tmp_path / "audio.bin"
    source.write_bytes(b"Source fixture.")
    audio.write_bytes(b"Audio fixture.")
    monkeypatch.setattr(video, "AudioFileClip", lambda *a: AudioClip(lambda t: np.zeros(2), duration=2, fps=24000))
    monkeypatch.setattr(video, "_open_video_clip_quietly", lambda *a: VideoClip(lambda t: np.zeros((2, 2, 3), dtype=np.uint8), duration=.375).with_fps(24))
    counts, concatenations = [], []
    def writer(clip, *args, **kwargs):
        times = [t for t, frame in clip.iter_frames(fps=kwargs["fps"], with_times=True)]
        counts.append(len(times))
        assert times[-1] < .375 / speed
    monkeypatch.setattr(video, "_write_videofile_with_codec_fallback", writer)
    monkeypatch.setattr(video, "concat_video_clips_with_ffmpeg", lambda **kw: concatenations.append(kw))
    video.combine_videos(str(tmp_path / "not-encoded.mp4"), [str(source)] * 2, str(audio), video_concat_mode=VideoConcatMode.sequential, max_clip_duration=1 / speed, clip_speed=speed, video_size=(2, 2), repeat_clips=repeat)
    expected = math.ceil(.375 / speed * video.fps) if not repeat else int(.375 / speed * video.fps)
    assert counts == [expected] * 2
    if repeat:
        assert "clip_durations" not in concatenations[0]
    else:
        assert concatenations[0]["clip_durations"] == [.375 / speed] * 2
        assert concatenations[0]["max_duration"] == .75 / speed


def test_concat_manifest_and_silence_command_keep_fractional_timing(tmp_path, monkeypatch):
    source = tmp_path / "source.bin"
    source.write_bytes(b"Source fixture.")
    captured = []
    def run(command, output):
        captured.append((Path(command[command.index("-i") + 1]).read_text(encoding="utf-8"), command))
        Path(output).write_bytes(b"Mock encoder boundary, not video.")
        return SimpleNamespace(returncode=0, stderr="", stdout="")
    monkeypatch.setattr(video, "_run_concat_with_heartbeat", run)
    monkeypatch.setattr(video, "_get_effective_video_codec", lambda: "libx264")
    video.concat_video_clips_with_ffmpeg([str(source)] * 2, str(tmp_path / "boundary.bin"), 1, str(tmp_path), max_duration=.75 / .7, clip_durations=[.375 / .7] * 2)
    manifest, command = captured[0]
    assert [float(line.split()[1]) for line in manifest.splitlines() if line.startswith("duration ")] == pytest.approx([.375 / .7] * 2, abs=1e-9)
    assert float(command[command.index("-t") + 1]) == pytest.approx(.75 / .7, abs=1e-9)
    calls = []
    monkeypatch.setattr(voice, "_publish_tts_ffmpeg_output", lambda command, *a: calls.append(command) or True)
    assert voice.generate_silent_audio(124 / 24, str(tmp_path / "not-encoded.mp3"))
    assert float(calls[0][calls[0].index("-t") + 1]) == pytest.approx(124 / 24, abs=1e-9)


@pytest.mark.parametrize("offset", [.05, -.05])
def test_audio_offset_is_normalized_before_scene_composition(offset, tmp_path, monkeypatch):
    source = tmp_path / "source.bin"
    source.write_bytes(b"Timestamp fixture.")
    def probe(filename, stream, **kwargs):
        if stream.startswith("v"):
            return {"streams": [{"time_base": "1/1000"}], "frames": [{"pts": n, "duration": 100} for n in [100, 200, 300, 400]]}
        return {"streams": [{"time_base": "1/1000", "start_pts": round((.1 + offset) * 1000), "duration_ts": 250}]}
    monkeypatch.setattr(media_timing, "_probe", probe)
    audio = AudioClip(lambda t: np.array([1 + t]), duration=.25, fps=24000)
    audio.reader = SimpleNamespace(fps=24000)
    clip = SimpleNamespace(reader=SimpleNamespace(), audio=audio)
    media_timing.apply_video_timing(clip, source)
    assert clip.audio.duration == .4
    assert clip.audio.get_frame(.1)[0] == pytest.approx(1.1 - offset)
    assert clip.audio.get_frame(.39)[0] == 0
    if offset > 0:
        assert clip.audio.get_frame(.01)[0] == 0
