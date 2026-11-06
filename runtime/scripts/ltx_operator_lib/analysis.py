"""Evidence-focused video metadata, keyframe, and report collection."""

import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Callable

from .artifacts import create_json_exclusive, ensure_output_directory, replace_json_atomic


class AnalysisToolError(RuntimeError):
    pass


def keyframe_times(duration: float, count: int) -> list[float]:
    """Return evenly distributed timestamps including both endpoints."""
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or duration <= 0
    ):
        raise ValueError("duration must be finite and positive")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    if count == 1:
        return [float(duration) / 2]
    step = float(duration) / (count - 1)
    return [step * index for index in range(count)]


def probe_media(
    path: str | Path,
    *,
    runner: Callable[..., object] = subprocess.run,
    executable: str = "ffprobe",
) -> dict:
    """Collect factual media metadata from ffprobe JSON."""
    media = _existing_file(path)
    command = [
        executable,
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        str(media),
    ]
    result = _run_tool(runner, command, "ffprobe")
    try:
        payload = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise AnalysisToolError(f"ffprobe returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise AnalysisToolError("ffprobe returned a non-object JSON result")

    streams = payload.get("streams", [])
    if not isinstance(streams, list):
        streams = []
    video = next(
        (
            stream
            for stream in streams
            if isinstance(stream, dict) and stream.get("codec_type") == "video"
        ),
        {},
    )
    audio = next(
        (
            stream
            for stream in streams
            if isinstance(stream, dict) and stream.get("codec_type") == "audio"
        ),
        None,
    )
    format_data = payload.get("format")
    if not isinstance(format_data, dict):
        format_data = {}
    duration = _optional_float(format_data.get("duration"))
    if duration is None:
        duration = _optional_float(video.get("duration"))
    return {
        "duration": duration,
        "video_codec": _optional_string(video.get("codec_name")),
        "audio_codec": _optional_string(audio.get("codec_name")) if audio else None,
        "width": _optional_int(video.get("width")),
        "height": _optional_int(video.get("height")),
        "frame_rate": _frame_rate(video),
        "audio_present": audio is not None,
    }


def extract_keyframes(
    path: str | Path,
    output_directory: str | Path,
    times: list[float],
    *,
    runner: Callable[..., object] = subprocess.run,
    executable: str = "ffmpeg",
) -> list[Path]:
    """Extract one JPEG at each exact timestamp without overwriting files."""
    media = _existing_file(path)
    directory = Path(output_directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    frames = []
    for index, timestamp in enumerate(times, start=1):
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(timestamp)
            or timestamp < 0
        ):
            raise ValueError("keyframe timestamps must be finite and non-negative")
        destination = directory / f"frame-{index:03d}.jpg"
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite keyframe: {destination}")
        command = [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-n",
            "-ss",
            f"{float(timestamp):.6f}",
            "-i",
            str(media),
            "-frames:v",
            "1",
            "-pix_fmt",
            "yuvj420p",
            "-q:v",
            "2",
            str(destination),
        ]
        _run_tool(runner, command, "ffmpeg")
        if not destination.is_file():
            raise AnalysisToolError(f"ffmpeg did not create keyframe: {destination}")
        frames.append(destination)
    return frames


def analyze_outputs(
    paths: list[str | Path],
    report_directory: str | Path,
    *,
    probe: Callable[[str | Path], dict] = probe_media,
    extractor: Callable[[str | Path, str | Path, list[float]], list[Path]] = extract_keyframes,
    keyframe_count: int = 5,
) -> dict:
    """Write a report separating collected evidence from interpretation."""
    if not isinstance(paths, list):
        raise TypeError("paths must be a list")
    directory = Path(report_directory).expanduser().resolve()
    ensure_output_directory(directory)
    report_path = directory / "analysis.json"
    create_json_exclusive(report_path, {"status": "collecting"})

    files = []
    limitations = [
        "Metadata and sampled keyframes do not assess prompt adherence, motion quality, identity stability, or audio quality."
    ]
    observations = []
    for index, value in enumerate(paths, start=1):
        path = Path(value).expanduser().resolve()
        record = {"path": str(path), "exists": path.is_file(), "keyframes": []}
        if not path.is_file():
            limitations.append(f"missing output: {path}")
            files.append(record)
            continue
        record["content_sha256"] = _sha256(path)
        try:
            metadata = probe(path)
            if not isinstance(metadata, dict):
                raise AnalysisToolError("probe returned a non-object result")
            record.update(metadata)
            observations.append(f"Collected metadata for {path.name}.")
        except Exception as exc:
            record["probe_error"] = str(exc)
            limitations.append(f"{path.name}: {exc}")
            files.append(record)
            continue

        duration = record.get("duration")
        if isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration > 0:
            try:
                requested_times = keyframe_times(duration, keyframe_count)
                times = _decodable_times(
                    requested_times, duration, record.get("frame_rate")
                )
                frame_dir = directory / f"keyframes-{index:03d}"
                frames = extractor(path, frame_dir, times)
                record["keyframes"] = [str(Path(frame).resolve()) for frame in frames]
                record["requested_keyframe_times"] = requested_times
                record["keyframe_times"] = times
                observations.append(f"Collected {len(frames)} sampled keyframes for {path.name}.")
            except Exception as exc:
                record["keyframe_error"] = str(exc)
                limitations.append(f"{path.name}: keyframe extraction unavailable: {exc}")
        else:
            limitations.append(f"{path.name}: duration unavailable; keyframes not extracted")
        files.append(record)

    report = {
        "files": files,
        "observations": observations,
        "interpretation": [],
        "limitations": limitations,
    }
    replace_json_atomic(report_path, report)
    return report


def _run_tool(runner: Callable[..., object], command: list[str], label: str):
    try:
        result = runner(command, capture_output=True, text=True, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise AnalysisToolError(f"{label} unavailable: {exc}") from exc
    if getattr(result, "returncode", None) != 0:
        error = str(getattr(result, "stderr", "")).strip() or "unknown error"
        raise AnalysisToolError(f"{label} failed: {error}")
    return result


def _existing_file(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"media file is unavailable: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_string(value) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_int(value) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _frame_rate(video: dict) -> float | None:
    value = video.get("avg_frame_rate") or video.get("r_frame_rate")
    if not isinstance(value, str):
        return _optional_float(value)
    if "/" not in value:
        return _optional_float(value)
    numerator, denominator = value.split("/", 1)
    top = _optional_float(numerator)
    bottom = _optional_float(denominator)
    if top is None or bottom in {None, 0}:
        return None
    return top / bottom


def _decodable_times(times: list[float], duration: float, frame_rate) -> list[float]:
    result = list(times)
    if not result or result[-1] < duration:
        return result
    rate = _optional_float(frame_rate)
    interval = 1 / rate if rate and rate > 0 else min(1 / 30, duration / 2)
    result[-1] = max(0.0, duration - interval)
    return result
