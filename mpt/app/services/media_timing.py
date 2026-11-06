"""Use media timestamps rather than FFmpeg's rounded console duration."""
# PresentationTimeReader.initialize adapts MoviePy's FFMPEG_VideoReader.
# The MIT License (MIT)
# Copyright (c) 2015 Zulko
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
from bisect import bisect_right
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
import json
import math
import shutil
import subprocess

from moviepy import AudioFileClip, CompositeAudioClip
from moviepy.config import FFMPEG_BINARY
from moviepy.tools import cross_platform_popen_params, ffmpeg_escape_filename
from moviepy.video.io.ffmpeg_reader import FFMPEG_VideoReader

from app.utils import utils


def _probe(filename, stream, *, frames=False):
    source = Path(filename).resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"Media source is not a file: {source}")
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        sibling = Path(utils.get_ffmpeg_binary()).with_name(
            "ffprobe.exe" if Path(utils.get_ffmpeg_binary()).suffix == ".exe" else "ffprobe"
        )
        ffprobe = str(sibling) if sibling.is_file() else "ffprobe"
    command = [ffprobe, "-v", "error", "-select_streams", stream, "-show_entries",
               "stream=duration_ts,time_base,start_pts,sample_rate"]
    if frames:
        command[-1] += ":frame=best_effort_timestamp,pts,duration,pkt_duration,nb_samples"
    command += ["-of", "json", str(source)]
    try:
        result = subprocess.run(command, capture_output=True, check=True, timeout=30)
        return json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise RuntimeError(f"Cannot read media timestamps: {source}") from exc


def _stream_duration(stream):
    ticks = stream.get("duration_ts")
    if ticks in (None, "N/A"):
        return None
    duration = int(ticks) * Fraction(stream["time_base"])
    if duration <= 0:
        raise ValueError("Media duration must be positive")
    return duration


def audio_duration(filename):
    """Audio stream ticks retain the exact PCM sample count for WAV files."""
    info = _probe(filename, "a:0")
    if not info.get("streams"):
        return 0.0
    stream = info["streams"][0]
    duration = _stream_duration(stream)
    if duration is None:
        info = _probe(filename, "a:0", frames=True)
        frames = info.get("frames", [])
        if not frames or any(not frame.get("nb_samples") for frame in frames):
            raise ValueError("Audio has no measurable sample count")
        duration = Fraction(sum(int(frame["nb_samples"]) for frame in frames), int(stream["sample_rate"]))
    if duration <= 0:
        raise ValueError("Audio duration must be positive")
    return float(duration)


@dataclass(frozen=True)
class VideoTiming:
    duration: Fraction
    pts: tuple[Fraction, ...]
    frame_rate: Fraction | None
    start: Fraction = Fraction(0)


def video_timing(filename):
    source = Path(filename).resolve(strict=True)
    stat = source.stat()
    return _video_timing(str(source), (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))


@lru_cache(maxsize=16)
def _video_timing(filename, source_stat):
    """Normalize presentation timestamps and include the complete last frame."""
    info = _probe(filename, "v:0", frames=True)
    if not info.get("streams") or not info.get("frames"):
        raise ValueError("Video has no presentation timestamps")
    stream, frames = info["streams"][0], info["frames"]
    time_base = Fraction(stream["time_base"])
    if time_base <= 0:
        raise ValueError("Video time base must be positive")
    absolute = tuple(int(frame.get("best_effort_timestamp", frame.get("pts"))) * time_base for frame in frames)
    pts = tuple(value - absolute[0] for value in absolute)
    if any(right <= left for left, right in zip(pts, pts[1:])):
        raise ValueError("Video presentation timestamps must increase")
    last_ticks = frames[-1].get("duration", frames[-1].get("pkt_duration"))
    if last_ticks not in (None, "N/A", 0, "0"):
        duration = pts[-1] + int(last_ticks) * time_base
    else:
        stream_duration = _stream_duration(stream)
        if stream_duration is None:
            raise ValueError("Video has no last-frame end timestamp")
        duration = int(stream.get("start_pts", absolute[0] / time_base)) * time_base + stream_duration - absolute[0]
    if duration <= pts[-1]:
        raise ValueError("Video last-frame end must follow its presentation timestamp")
    interval = pts[1] if len(pts) > 1 else duration
    regular = all(right - left == interval for left, right in zip(pts, pts[1:])) and duration - pts[-1] == interval
    return VideoTiming(duration, pts, 1 / interval if regular else None, absolute[0])


@lru_cache(maxsize=1)
def _frame_passthrough_args(executable):
    # Older FFmpeg uses -vsync. Current builds can omit it.
    options = subprocess.run([executable, "-hide_banner", "-h", "long"], capture_output=True,
                             check=True, timeout=10,
                             **cross_platform_popen_params({"stdin": subprocess.DEVNULL})).stdout
    if any(line.lstrip().startswith((b"-fps_mode ", b"-fps_mode[")) for line in options.splitlines()):
        return ("-fps_mode", "passthrough")
    return ("-vsync", "0")


class PresentationTimeReader(FFMPEG_VideoReader):
    """Decode VFR frames once each and select the frame displayed at time t."""
    def __init__(self, reader, timing):
        self.__dict__ = reader.__dict__.copy()
        reader.close()
        self.proc = None
        self.presentation_times = tuple(float(value) for value in timing.pts)
        self.duration = self.ffmpeg_duration = float(timing.duration)
        self.n_frames = len(timing.pts)
        try:
            self.initialize()
        except BaseException:
            self.close()
            raise

    def get_frame_number(self, t):
        return max(0, min(len(self.presentation_times) - 1, bisect_right(self.presentation_times, t) - 1))

    def initialize(self, start_time=0):
        # Restarting VFR seeks from zero avoids MoviePy's index/fps seek rounding.
        self.close(delete_lastread=False)
        target = self.get_frame_number(start_time)
        self.pos = 0
        input_args = ["-i", ffmpeg_escape_filename(self.filename)]
        if self.depth == 4 and self.infos.get("video_codec_name") in {"vp8", "vp9"}:
            codec = "libvpx-vp9" if self.infos["video_codec_name"] == "vp9" else "libvpx"
            input_args = ["-c:v", codec, *input_args]
        command = [FFMPEG_BINARY, *input_args, "-map", "0:v:0", "-loglevel", "error",
                   "-f", "image2pipe", "-vf", "scale=%d:%d" % tuple(self.size),
                   "-sws_flags", self.resize_algo, "-pix_fmt", self.pixel_format,
                   "-vcodec", "rawvideo", *_frame_passthrough_args(FFMPEG_BINARY), "-"]
        self.proc = subprocess.Popen(command, **cross_platform_popen_params({
            "bufsize": self.bufsize, "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE, "stdin": subprocess.DEVNULL,
        }))
        if target:
            self.skip_frames(target)
        self.last_read = self.read_frame()


def apply_video_timing(clip, filename):
    """Keep MoviePy's clip, decoder and frame selection on one source timeline."""
    timing = video_timing(filename)
    if timing.frame_rate is None:
        clip.reader = PresentationTimeReader(clip.reader, timing)
    else:
        clip.reader.fps = clip.fps = float(timing.frame_rate)
        clip.reader.duration = clip.reader.ffmpeg_duration = float(timing.duration)
        clip.reader.n_frames = len(timing.pts)
    clip.duration = clip.end = float(timing.duration)
    if clip.audio is not None:
        apply_audio_timing(clip.audio, filename)
        stream = _probe(filename, "a:0")["streams"][0]
        offset = int(stream.get("start_pts", 0)) * Fraction(stream["time_base"]) - timing.start
        if offset:
            # Normalize to video time zero so scene composition retains gaps/trims.
            aligned = CompositeAudioClip([clip.audio.with_start(float(offset))]).with_duration(clip.duration)
            clip.audio.frame_function = aligned.frame_function
            clip.audio.duration = clip.audio.end = clip.duration
    return clip


def apply_audio_timing(clip, filename):
    duration = audio_duration(filename)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Audio duration must be positive and finite")
    clip.duration = clip.end = clip.reader.duration = duration
    clip.reader.n_frames = int(duration * clip.reader.fps)
    return clip


def open_audio_clip(filename, *args, **kwargs):
    clip = AudioFileClip(filename, *args, **kwargs)
    try:
        return apply_audio_timing(clip, filename)
    except BaseException:
        clip.close()
        raise
