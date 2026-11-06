"""Approval-gated LTX material planning and rendering support."""

from contextlib import ExitStack, contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time
from app.services import execution_control
from uuid import NAMESPACE_URL, uuid4, uuid5

from loguru import logger

from app.config import config
from app.models import const
from app.services import state as sm
from app.services.video_project import ProjectError, _lock
from app.utils import utils


PROMPT_PREFIX = "Create one coherent continuous shot illustrating this narration: "
MODEL_LABELS = {"2.5": "LTX-2.5", "2.3": "LTX-2.3", "h3": "MiniMax H3"}
LTX_RESOLUTIONS_BY_ASPECT = {
    "9:16": (352, 640),
    "16:9": (640, 352),
    "1:1": (512, 512),
}
H3_RESOLUTIONS_BY_ASPECT = {"9:16": (480, 864), "16:9": (864, 480), "1:1": (640, 640)}


class LtxError(RuntimeError):
    pass


class RenderBusy(LtxError):
    pass


def settings() -> dict:
    return dict(config.ltx)


def _read_template(values: dict, model_version=None) -> dict:
    templates = values.get("request_templates", {})
    if not isinstance(templates, dict):
        raise LtxError("ComfyUI request_templates must map model versions to request files")
    if model_version is None and not values.get("request_template"):
        model_version = next((key for key in (*MODEL_LABELS, "custom") if key in templates), None)
    path = templates.get(model_version, values.get("request_template"))
    label = MODEL_LABELS.get(model_version, "Local ComfyUI")
    setup = "Export the workflow from ComfyUI with Export (API), then run python install.py install MPT --comfyui COMFYUI --workflow-api WORKFLOW.json"
    if model_version == "custom":
        setup = "Register your request with python install.py configure MPT --request-template SETTINGS/request.json."
    elif model_version in MODEL_LABELS:
        setup += f" --model-version {model_version}"
    if not isinstance(path, str) or not path:
        raise LtxError(f"{label} request_template is not configured. {setup}")
    if not Path(path).is_absolute():
        raise LtxError(f"{label} request_template must be absolute")
    try:
        template = _read_json(Path(path))
    except (OSError, ValueError) as exc:
        raise LtxError(f"{label} request_template is unavailable: {path}. {exc}. {setup}") from exc
    if not isinstance(template, dict) or not isinstance(template.get("variant", {}), dict):
        raise LtxError(f"{label} request_template must contain a request object")
    declared = template.get("variant", {}).get("model_version")
    if declared is not None and not isinstance(declared, str):
        raise LtxError(f"{label} request_template model_version must be a string")
    key = declared if declared in MODEL_LABELS else "custom"
    if model_version is not None and key != model_version:
        raise LtxError(f"No matching request template for {label}. {setup}")
    return template


def default_model_version(values: dict):
    if not values.get("request_template") and not values.get("request_templates"):
        return "2.5"
    try:
        version = _read_template(values).get("variant", {}).get("model_version")
    except LtxError:
        return None
    return version if version in MODEL_LABELS else "custom"


def _runtime_errors(values: dict) -> list[str]:
    errors = []
    if not values.get("enabled"):
        errors.append("LTX source is disabled")
    for name in ("python_executable", "operator_script"):
        value = values.get(name)
        if not isinstance(value, str) or not value:
            errors.append(f"LTX {name} is not configured")
            continue
        path = Path(value)
        if not path.is_absolute():
            errors.append(f"LTX {name} must be absolute")
        elif not path.is_file():
            errors.append(f"LTX {name} is unavailable: {path}")
    return errors


def request_template(model_version=None, values=None) -> dict:
    values = settings() if values is None else values
    errors = _runtime_errors(values)
    if errors:
        raise LtxError("\n".join(errors))
    return _read_template(values, model_version)


def validate_settings(values: dict | None = None, model_version=None) -> tuple[str, ...]:
    try:
        request_template(model_version, values)
    except LtxError as exc:
        return tuple(str(exc).splitlines())
    return ()


def _uses_unit_guidance(graph):
    """Infer a missing capability only for familiar, unmodified sampler paths."""
    def source(inputs, field):
        link = inputs.get(field)
        if isinstance(link, list) and len(link) == 2 and link[1] == 0:
            node = graph.get(str(link[0]))
            if isinstance(node, dict):
                return node
        return {}

    seen = False
    for node in graph.values():
        kind = node.get("class_type", "") if isinstance(node, dict) else ""
        if "Sampler" not in kind or kind == "KSamplerSelect":
            continue
        if kind not in {"SamplerCustomAdvanced", "SamplerCustom", "KSamplerAdvanced", "KSampler"}:
            return False
        seen = True
        inputs = node.get("inputs", {})
        sampler = inputs.get("sampler_name")
        if kind.startswith("SamplerCustom"):
            selector = source(inputs, "sampler")
            if selector.get("class_type") != "KSamplerSelect":
                return False
            sampler = selector.get("inputs", {}).get("sampler_name")
        keys = ("cfg",)
        if kind == "SamplerCustomAdvanced":
            guider = source(inputs, "guider")
            if guider.get("class_type") == "BasicGuider":
                continue
            if guider.get("class_type") not in {"CFGGuider", "LTXVDualCFGGuider"}:
                return False
            if guider["class_type"] == "LTXVDualCFGGuider":
                keys = ("video_cfg", "audio_cfg")
            inputs = guider.get("inputs", {})
        if sampler not in ("euler", "euler_ancestral"):
            return False
        if source(inputs, "model").get("class_type") not in {"UNETLoader", "CheckpointLoaderSimple"}:
            return False
        if not all(type(inputs.get(key)) in (int, float) and inputs[key] == 1 for key in keys):
            return False
    return seen


def scene_prompt_context(template=None) -> dict:
    """Describe conditioning for the director without exposing local files to the LLM."""
    if template is None:
        template = request_template()
    variant = template.get("variant", {})
    graph = template.get("api_prompt", {})
    if not isinstance(variant, dict) or not isinstance(graph, dict):
        raise LtxError("LTX request variant and api_prompt must be objects")
    if "supports_negative_prompt" in variant:
        negative = variant["supports_negative_prompt"]
        if type(negative) is not bool:
            raise LtxError("supports_negative_prompt must be a boolean")
    else:
        negative = not _uses_unit_guidance(graph)

    version = variant.get("model_version")
    return {
        "model_version": version if version in MODEL_LABELS else None,
        "reference_media": version != "h3" and bool(template.get("input_media")),
        "supports_negative_prompt": version != "h3" and negative,
    }


def split_scenes(
    script: str, audio_duration: float, target_seconds: int, max_scenes: int
) -> list[str]:
    sentences = [
        value.strip()
        for value in re.findall(r"[^.!?;:]+(?:[.!?;:]+|$)", script)
        if value.strip()
    ]
    if not sentences:
        raise LtxError("video script produced no LTX scenes")
    count = min(
        len(sentences),
        max_scenes,
        max(1, math.ceil(audio_duration / target_seconds)),
    )
    groups = []
    start = 0
    remaining_chars = sum(map(len, sentences))
    for group_index in range(count):
        remaining_groups = count - group_index
        if remaining_groups == 1:
            end = len(sentences)
        else:
            target = remaining_chars / remaining_groups
            end = start + 1
            chars = len(sentences[start])
            while end < len(sentences) - remaining_groups + 1:
                candidate = chars + len(sentences[end])
                if abs(candidate - target) > abs(chars - target):
                    break
                chars = candidate
                end += 1
        group = " ".join(sentences[start:end])
        groups.append(group)
        remaining_chars -= sum(map(len, sentences[start:end]))
        start = end
    return groups


def _fallback_scene_plan(
    script: str, audio_duration: float, target_seconds: int, max_scenes: int
) -> list[dict]:
    return [
        {
            "index": index,
            "duration_seconds": target_seconds,
            "narration": scene,
            "visual_prompt": PROMPT_PREFIX + scene,
            "camera": "",
            "lighting": "",
            "continuity": "",
            "negative_prompt": "",
        }
        for index, scene in enumerate(
            split_scenes(script, audio_duration, target_seconds, max_scenes), start=1
        )
    ]


def validate_h3_duration(seconds):
    if type(seconds) is not int or not 4 <= seconds <= 15:
        raise ValueError("MiniMax H3 scene duration must be a whole number from 4 to 15 seconds")


def validate_scene_plan(
    scene_plan, default_duration_seconds: int, max_scenes: int, model_version=None, *, script=None, timing=None
) -> list[dict]:
    if not isinstance(scene_plan, list) or not scene_plan:
        raise LtxError("LTX scene plan must be a non-empty list")
    if len(scene_plan) > max_scenes:
        raise LtxError("LTX scene plan exceeds ltx_max_scenes")
    if timing and timing.get("required_scenes") is not None and len(scene_plan) != timing["required_scenes"]:
        raise LtxError(f"LTX scene plan requires exactly {timing['required_scenes']} scenes to cover the audio")

    remaining = (" ".join(script.split()) if script is not None and
                 (not timing or timing["mode"] != "visual") else None)
    normalized = []
    for position, item in enumerate(scene_plan, start=1):
        if not isinstance(item, dict):
            raise LtxError("LTX scene plan items must be objects")
        index = item.get("index", position)
        if type(index) is not int or index != position:
            raise LtxError("LTX scene plan indexes must be sequential")
        duration = item.get("duration_seconds", default_duration_seconds)
        predicted = bool(timing and timing.get("duration_mode") == "predicted" and "frame_count" in item)
        if predicted:
            frames = item["frame_count"]
            if (type(frames) is not int or frames < 1 or (frames - 1) % 8
                    or not _minimum_prediction_frames(timing["fps"]) <= frames <= timing["frames_per_scene"]
                    or type(duration) not in {int, float}
                    or not math.isclose(duration, frames / timing["fps"], abs_tol=1e-8)):
                raise LtxError("Predicted scene duration must match its bounded 8k+1 frame count")
        elif timing and (type(duration) not in {int, float} or duration != timing["scene_duration_seconds"]):
            raise LtxError(f"Every scene must use the selected {timing['scene_duration_seconds']} seconds")
        if model_version == "h3":
            validate_h3_duration(duration)
        if not predicted and (type(duration) not in {int, float} or duration <= 0 or duration > 30):
            raise LtxError("LTX scene duration_seconds must be between 1 and 30")
        narration = item.get("narration")
        visual_prompt = item.get("visual_prompt")
        if not isinstance(narration, str) or not narration.strip():
            raise LtxError("LTX scene narration is required")
        if not isinstance(visual_prompt, str) or not visual_prompt.strip():
            raise LtxError("LTX scene visual_prompt is required")
        if remaining is not None:
            fragment = " ".join(narration.split())
            if not remaining.startswith(fragment):
                raise LtxError(f"Scene {position} narration does not match the next part of the script")
            remaining = remaining[len(fragment):].lstrip()
        details = {}
        for field in ("camera", "lighting", "continuity", "negative_prompt"):
            value = item.get(field)
            if value is not None and not isinstance(value, str):
                raise LtxError(f"LTX scene {field} must be text")
            details[field] = value.strip() if value else ""
        normalized.append(
            {
                "index": position,
                "duration_seconds": duration if predicted else int(math.ceil(duration)),
                "narration": narration.strip(),
                "visual_prompt": visual_prompt.strip(),
                **details,
                **({"frame_count": frames} if predicted else {}),
            }
        )
    if remaining:
        raise LtxError("Scene narration must cover the entire script")
    return normalized


def frame_count_for(seconds: int, fps: float, model_version=None) -> int:
    if model_version == "h3":
        validate_h3_duration(seconds)
        return math.ceil((seconds * fps - 5) / 17) * 17 + 5
    requested = math.ceil(seconds * fps)
    return math.ceil((requested - 1) / 8) * 8 + 1


def _minimum_prediction_frames(fps):
    first = math.ceil(math.nextafter(.5 * fps, -math.inf))
    return first + (1 - first) % 8


def local_video_timing(params, template, audio_duration=None) -> dict:
    """Resolve local clip playback independently of the script's reading time."""
    from app.services import voice

    version = template.get("variant", {}).get("model_version")
    fps = template.get("parameters", {}).get("fps")
    if type(fps) not in {int, float} or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Local video FPS must be a finite positive number")
    if version == "h3" and fps != 24:
        raise ValueError("MiniMax H3 requires 24 FPS")
    seconds = params.ltx_scene_duration_seconds
    if type(seconds) is not int or not 1 <= seconds <= 30:
        raise ValueError("Local scene duration must be a whole number from 1 to 30 seconds")
    max_scenes = params.ltx_max_scenes
    if type(max_scenes) is not int or not 1 <= max_scenes <= 20:
        raise ValueError("Local maximum scenes must be from 1 to 20")
    automatic = bool(getattr(params, "ltx_predict_duration", False))
    visual = voice.is_no_voice(getattr(params, "voice_name", None)) and not str(
        getattr(params, "custom_audio_file", None) or ""
    ).strip()
    if automatic and (version != "2.5" or not visual):
        raise ValueError("Automatic duration requires LTX-2.5 with Voiceover None and no custom audio")
    if automatic:
        last = math.floor(math.nextafter(seconds * fps, math.inf))
        frames = last - (last - 1) % 8
        if frames < _minimum_prediction_frames(fps):
            raise ValueError("The selected duration and FPS cannot fit an automatic scene of at least 0.5 seconds")
    else:
        frames = frame_count_for(seconds, fps, version)
    render_seconds = frames / fps
    playback_seconds = render_seconds / utils.normalize_clip_speed(getattr(params, "video_clip_speed", 1))
    duration = None
    required = None
    maximum = max_scenes * playback_seconds
    if not visual and audio_duration is not None:
        if type(audio_duration) not in {int, float} or not math.isfinite(audio_duration) or audio_duration <= 0:
            raise ValueError("Local narration audio duration must be a finite positive number")
        duration = audio_duration
        required = max(1, math.ceil(duration / playback_seconds - 1e-9))
        if required > max_scenes:
            raise ValueError(
                f"Audio is {duration}s and needs {required} scenes; maximum {max_scenes} scenes "
                f"cover {maximum}s, a deficit of {duration - maximum:g}s. "
                "Increase Maximum scenes or Scene duration, lower playback speed, or shorten the audio."
            )
    return {
        "mode": "visual" if visual else "narrated", "scene_duration_seconds": seconds,
        "frames_per_scene": frames, "fps": fps, "render_scene_seconds": render_seconds,
        "playback_scene_seconds": playback_seconds, "max_scenes": max_scenes,
        "max_playback_seconds": maximum, "required_scenes": required,
        "audio_duration_seconds": duration,
        **({"duration_mode": "predicted"} if automatic else {}),
    }


def ltx_resolution_for_aspect(video_aspect, model_version=None) -> tuple[int, int] | None:
    aspect = getattr(video_aspect, "value", video_aspect)
    resolutions = H3_RESOLUTIONS_BY_ASPECT if model_version == "h3" else LTX_RESOLUTIONS_BY_ASPECT
    return resolutions.get(str(aspect))


def resolution_presets(model_version, video_aspect) -> dict[str, tuple[int, int]]:
    """Output pixels for the local two-pass LTX and native H3 workflows."""
    aspect = getattr(video_aspect, "value", video_aspect)
    if model_version not in MODEL_LABELS or aspect not in LTX_RESOLUTIONS_BY_ASPECT:
        return {}
    if model_version == "h3":
        if aspect == "1:1":
            return {"recommended": (768, 768)}
        sizes = {"draft": (960, 544), "compact": (1024, 576), "recommended": (1344, 768)}
    else:
        sizes = {"540p": (1024, 576), "720p": (1280, 704), "recommended": (1920, 1088)}
    if aspect == "16:9":
        return sizes
    if aspect == "9:16":
        return {key: (height, width) for key, (width, height) in sizes.items()}
    return {key: (height, height) for key, (_, height) in sizes.items()}


def generation_dimension_limits(model_version) -> tuple[int, int, int]:
    """ComfyUI's per-axis minimum, maximum and step, in generated clip pixels."""
    if model_version == "h3":
        return 32, 16384, 32
    if model_version in {"2.3", "2.5"}:
        return 128, 32768, 64
    raise ValueError("Generated clip dimensions require a supported local video workflow")


def validate_generation_dimensions(width, height, model_version):
    if width is None and height is None:
        return
    if width is None or height is None:
        raise ValueError("Set both ltx_generation_width and ltx_generation_height")
    minimum, maximum, step = generation_dimension_limits(model_version)
    if any(type(value) is not int or not minimum <= value <= maximum or value % step for value in (width, height)):
        raise ValueError(f"Generated clip width and height must be multiples of {step} from {minimum} to {maximum}")


def generation_scale(template) -> int | None:
    """Recognize the saved video path used by native H3 and two-pass LTX graphs."""
    version = template.get("variant", {}).get("model_version")
    graph = template.get("api_prompt", {})
    if version not in MODEL_LABELS or not isinstance(graph, dict):
        return None
    saves = [node for node in graph.values() if isinstance(node, dict) and node.get("class_type") == "SaveVideo"]
    if len(saves) != 1:
        return None

    def upstream(node, port, kind, slot=0):
        inputs = node.get("inputs")
        link = inputs.get(port) if isinstance(inputs, dict) else None
        if not isinstance(link, list) or len(link) != 2 or link[1] != slot:
            return {}
        source = graph.get(str(link[0]))
        return source if isinstance(source, dict) and source.get("class_type") == kind else {}

    video = upstream(saves[0], "video", "CreateVideo")
    if version == "h3":
        decoder = upstream(video, "images", "VAEDecode")
        sampler = upstream(decoder, "samples", "SamplerCustomAdvanced")
        latent = upstream(sampler, "latent_image", "MiniMaxH3ImageToVideo", slot=1)
        scale = 1
    else:
        decoder = upstream(video, "images", "VAEDecodeTiled")
        port = "samples"
        if not decoder and version == "2.3":
            decoder = upstream(video, "images", "LTXVTiledVAEDecode")
            port = "latents"

        def sampler_input(separator):
            sampler = upstream(separator, "av_latent", "SamplerCustomAdvanced")
            return upstream(sampler, "latent_image", "LTXVConcatAVLatent")

        second_pass = sampler_input(upstream(decoder, port, "LTXVSeparateAVLatent"))
        upsampler = upstream(second_pass, "video_latent", "LTXVLatentUpsampler")
        loader = upstream(upsampler, "upscale_model", "LatentUpscaleModelLoader")
        loader_inputs = loader.get("inputs")
        model_name = loader_inputs.get("model_name") if isinstance(loader_inputs, dict) else None
        known_x2 = {"2.3": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
                    "2.5": "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"}
        if not isinstance(model_name, str) or model_name.replace("\\", "/").split("/")[-1] != known_x2[version]:
            return None
        first_pass = sampler_input(upstream(upsampler, "samples", "LTXVSeparateAVLatent"))
        latent = upstream(first_pass, "video_latent", "EmptyLTXVLatentVideo")
        scale = 2
    inputs = latent.get("inputs", {})
    if isinstance(inputs, dict) and all(type(inputs.get(key)) is int for key in ("width", "height")):
        return scale
    return None


def scene_resolution(params, template):
    """Resolve first-pass dimensions and the verified scale before planning scenes."""
    version = template.get("variant", {}).get("model_version")
    width = getattr(params, "ltx_generation_width", None)
    height = getattr(params, "ltx_generation_height", None)
    validate_generation_dimensions(width, height, version)
    scale = generation_scale(template)
    if width is not None:
        if scale is None:
            raise ValueError("This workflow does not expose a recognized generated clip size")
        return (width // scale, height // scale), scale
    if scale is not None:
        recommended = resolution_presets(version, params.video_aspect)["recommended"]
        return tuple(side // scale for side in recommended), scale
    parameters = template.get("parameters", {})
    if any(isinstance(field, dict) and field.get("derived") is True
           for field in (parameters.get("width"), parameters.get("height"))):
        return None, None
    return ltx_resolution_for_aspect(params.video_aspect, version), None


def scene_seed(task_id: str, index: int) -> int:
    digest = hashlib.sha256(f"{task_id}:{index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _cli_error(payload, stderr, returncode):
    def reason(value):
        if not isinstance(value, dict):
            return None
        error = value.get("error")
        if isinstance(error, str) and error.strip():
            return error.strip()
        errors = value.get("errors")
        if isinstance(errors, list):
            return next((item.strip() for item in errors if isinstance(item, str) and item.strip()), None)

    error = reason(payload)
    if error:
        return error
    try:
        error = reason(json.loads(stderr))
    except json.JSONDecodeError:
        error = None
    if error:
        return error
    diagnostics = []
    for line in stderr.splitlines():
        try:
            error = reason(json.loads(line))
        except json.JSONDecodeError:
            diagnostics.append(line)
        else:
            if error:
                return error
    tail = "\n".join(diagnostics).strip()[-500:]
    return f"LTX CLI failed with {returncode}" + (f": {tail}" if tail else "")


def run_cli(args: list[str], timeout: int) -> dict:
    execution_control.checkpoint()
    values = settings()
    command = [values["python_executable"], values["operator_script"], *args, "--json"]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    remove_cancel = None
    try:
        token = execution_control.current_token()
        if token is not None:
            remove_cancel = token.add_cancel_callback(process.kill)
        stdout, stderr = process.communicate(timeout=timeout)
        execution_control.checkpoint()
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.communicate()
        raise
    finally:
        if remove_cancel is not None:
            remove_cancel()
    stream = stdout if process.returncode == 0 else stdout or stderr
    try:
        payload = json.loads(stream)
    except json.JSONDecodeError as exc:
        if process.returncode != 0:
            raise LtxError(_cli_error(None, stderr or stdout, process.returncode)) from exc
        raise LtxError(f"LTX CLI returned invalid JSON: {stream[-500:]}") from exc
    if process.returncode != 0:
        raise LtxError(_cli_error(payload, stderr, process.returncode))
    if not isinstance(payload, dict):
        raise LtxError("LTX CLI returned invalid JSON: expected an object")
    return payload


def run_cli_stream(args: list[str], timeout: int, progress) -> dict:
    execution_control.checkpoint()
    values = settings()
    command = [values["python_executable"], values["operator_script"], *args, "--json"]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    stdout_lines = []
    stderr_lines = []
    progress_errors = []

    def read_stdout():
        stdout_lines.extend(process.stdout)

    def read_stderr():
        for line in process.stderr:
            stderr_lines.append(line)
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(event, dict)
                and event.get("event") in {"scene_planned", "scene_render"}
                and callable(progress)
            ):
                try:
                    progress(event)
                except Exception as exc:
                    progress_errors.append(exc)

    readers = [
        threading.Thread(target=read_stdout, daemon=True),
        threading.Thread(target=read_stderr, daemon=True),
    ]
    for reader in readers:
        reader.start()
    remove_cancel = None
    try:
        token = execution_control.current_token()
        if token is not None:
            remove_cancel = token.add_cancel_callback(process.kill)
        process.wait(timeout=timeout)
        execution_control.checkpoint()
    except subprocess.TimeoutExpired as exc:
        process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        raise LtxError(f"LTX CLI timed out after {timeout}s") from exc
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if remove_cancel is not None:
            remove_cancel()
        for reader in readers:
            reader.join(timeout=5)
        for reader, stream in zip(readers, (process.stdout, process.stderr)):
            if not reader.is_alive():
                stream.close()

    stdout = "".join(stdout_lines).strip()
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        if process.returncode != 0:
            raise LtxError(_cli_error(None, "".join(stderr_lines) or stdout, process.returncode)) from exc
        diagnostics = "".join(stderr_lines)[-500:]
        raise LtxError(f"LTX CLI returned invalid JSON: {diagnostics}") from exc
    if process.returncode != 0:
        raise LtxError(_cli_error(payload, "".join(stderr_lines), process.returncode))
    if not isinstance(payload, dict):
        raise LtxError("LTX CLI returned invalid JSON: expected an object")
    if progress_errors:
        raise LtxError(
            f"LTX progress callback failed: {progress_errors[0]}"
        ) from progress_errors[0]
    return payload


def predict_scene_durations(task_id, params, template, scene_plan, timing):
    """Resolve and freeze native duration-head results before scene approval."""
    if timing.get("duration_mode") != "predicted":
        return scene_plan
    with render_slot(task_id):
        sm.state.update_task(task_id, phase="predicting_scene_durations", progress=39,
                             status_message="Preparing local LTX-2.5 duration prediction.")
        result = _predict_scene_durations(task_id, params, template, scene_plan)
        sm.state.patch_task(task_id, phase="planning", status_message="Local scene durations resolved.")
        return result


def _predict_scene_durations(task_id, params, template, scene_plan):
    checked = local_video_timing(params, template)
    if checked.get("duration_mode") != "predicted":
        raise LtxError("Automatic scene duration is not enabled")
    scenes = validate_scene_plan(
        [{key: value for key, value in scene.items() if key not in {"frame_count", "predicted_duration_seconds"}}
         for scene in scene_plan],
        params.ltx_scene_duration_seconds, params.ltx_max_scenes, "2.5", timing=checked,
    )
    folder = Path(utils.task_dir(task_id)) / "ltx" / "duration-predictions" / uuid4().hex
    manifest = {"template": template, "scenes": scenes, "min_seconds": .5,
                "max_seconds": params.ltx_scene_duration_seconds, "timeout_seconds": 600,
                "state_root": utils.storage_dir("ltx-state"), "runs_dir": str(folder / "runs")}
    manifest_path = folder / "manifest.json"
    _create_json(manifest_path, manifest)
    succeeded = False
    failure = None
    try:
        sm.state.update_task(task_id, phase="ltx_duration_prediction", progress=39,
                             status_message=f"Predicting local scene durations: 0/{len(scenes)}")
        result = run_cli_stream(
            ["predict-durations", "--manifest", str(manifest_path)], 600 * len(scenes) + 30,
            lambda event: sm.state.patch_task(task_id, status_message=(
                f"Predicting local scene durations: {event['scene_index']}/{len(scenes)}")),
        )
        predictions = result.get("scenes", [])
        if not result.get("ok") or len(predictions) != len(scenes):
            raise LtxError("Duration predictor did not return every scene")
        output = []
        for scene, prediction in zip(scenes, predictions, strict=True):
            if prediction.get("index") != scene["index"]:
                raise LtxError("Duration predictor returned out-of-order scenes")
            frames = prediction.get("frame_count")
            if type(frames) is not int:
                raise LtxError("Duration predictor returned an invalid frame count")
            output.append({**scene, "frame_count": frames, "duration_seconds": frames / checked["fps"]})
        output = validate_scene_plan(output, params.ltx_scene_duration_seconds, params.ltx_max_scenes,
                                     "2.5", timing=checked)
        _replace_json(folder / "result.json", {"scenes": predictions})
        succeeded = True
        return output
    except BaseException as exc:
        failure = exc
        raise
    finally:
        if not succeeded:
            errors = []
            try:
                values = settings()
                runs = (folder / "runs").resolve()
                for index in range(1, len(scenes) + 1):
                    record = runs / f"{index:03d}" / "run.json"
                    report = record.with_name("report.json")
                    if not record.resolve().is_relative_to(runs):
                        errors.append("Duration prediction ownership path escaped its task")
                        continue
                    if not record.is_file():
                        if report.exists() and not _report_without_submission(_read_json(report)):
                            errors.append(f"Scene {index} prediction began without a signed prompt record")
                        continue
                    try:
                        rejected = _read_json(record)["record"].get("terminal_status") == "submission_rejected"
                        stopped = subprocess.run(
                            [values["python_executable"], values["operator_script"], "status" if rejected else "cancel",
                             "--run-record", str(record),
                             "--state-root", manifest["state_root"], "--json"],
                            capture_output=True, text=True, encoding="utf-8", errors="replace", shell=False, timeout=25,
                        )
                        payload = json.loads(stopped.stdout)
                        if not isinstance(payload, dict):
                            raise ValueError("cancellation response must be an object")
                        terminal = (_matches_prediction_rejection(payload, _read_json(report), record.parent)
                                    if rejected else payload.get("status") in {"confirmed_cancelled", "completed", "failed"})
                        if stopped.returncode or not terminal:
                            errors.append(f"Scene {index} prediction stop was not confirmed")
                    except Exception as exc:
                        errors.append(f"Scene {index} prediction stop: {type(exc).__name__}")
            except Exception as exc:
                errors.append(f"Duration prediction ownership check: {type(exc).__name__}: {exc}")
            if errors:
                if failure is not None:
                    failure.remote_stop_unconfirmed = True
                token = execution_control.current_token()
                message = "Duration prediction stop unconfirmed: " + "; ".join(errors)
                if token is not None:
                    token.stop_error = message
                try:
                    sm.state.update_task(task_id, phase="stop_unconfirmed", status_message=message)
                except Exception as state_error:
                    if failure is not None:
                        failure.add_note(f"Could not persist duration prediction stop state: {state_error}")


def plan_batch(
    task_id: str,
    script: str,
    audio_duration: float,
    params,
    progress=None,
    scene_plan=None,
    template=None,
    timing=None,
) -> dict:
    values = settings()
    if template is None:
        template = request_template(getattr(params, "ltx_model_version", None), values)
    workflow = scene_prompt_context(template)
    version = workflow["model_version"]
    fps = template.get("parameters", {}).get("fps")
    seed_count = len(
        [
            key
            for key in template.get("parameters", {})
            if key == "seed" or key.endswith("_seed")
        ]
    )
    if not isinstance(fps, (int, float)) or fps <= 0 or seed_count < 1:
        raise LtxError("LTX request template must expose FPS and at least one seed")
    if version == "h3" and (fps != 24 or seed_count != 1):
        raise LtxError("MiniMax H3 requires 24 FPS and one seed")
    if version == "h3":
        validate_h3_duration(params.ltx_scene_duration_seconds)
    if timing and timing.get("duration_mode") == "predicted" and (
        not scene_plan or any("frame_count" not in scene for scene in scene_plan)
    ):
        raise LtxError("Automatic scene durations must be predicted before preparing approval plans")
    first_pass, scale = scene_resolution(params, template)

    if scene_plan is None:
        scenes = _fallback_scene_plan(
            script,
            audio_duration,
            params.ltx_scene_duration_seconds,
            params.ltx_max_scenes,
        )
    else:
        scenes = validate_scene_plan(
            scene_plan,
            params.ltx_scene_duration_seconds,
            params.ltx_max_scenes,
            version,
            script=script,
            timing=timing,
        )
    if not workflow["supports_negative_prompt"] and any(s["negative_prompt"] for s in scenes):
        raise LtxError("This workflow cannot use a scene negative_prompt. Describe the intended visible result "
                       "in visual_prompt and leave negative_prompt empty.")
    ltx_dir = Path(utils.task_dir(task_id)) / "ltx"
    # Every scene uses the workflow the director saw, even if a user reconfigures ComfyUI.
    template_path = ltx_dir / "request-template.json"
    _create_json(template_path, template)
    records = []
    for scene in scenes:
        index = scene["index"]
        scene_dir = ltx_dir / "scenes" / f"{index:03d}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        base_seed = scene_seed(task_id, index)
        prefix = f"mpt_{task_id}_scene_{index:03d}"
        overrides = {
            "prompt": scene["visual_prompt"],
            "seeds": [base_seed + offset for offset in range(seed_count)],
            "frame_count": (scene["frame_count"] if "frame_count" in scene else
                            frame_count_for(scene["duration_seconds"], fps, version)),
            "output_prefix": prefix,
            "client_id": str(uuid5(NAMESPACE_URL, f"mpt:{task_id}:{index}")),
        }
        if scene["negative_prompt"]:
            overrides["negative_prompt"] = scene["negative_prompt"]
        if first_pass is not None:
            overrides.update({"width": first_pass[0], "height": first_pass[1]})
        overrides_path = scene_dir / "overrides.json"
        request_path = scene_dir / "request.json"
        _create_json(overrides_path, overrides)
        run_cli(
            [
                "prepare-request",
                "--template-request",
                str(template_path),
                "--overrides",
                str(overrides_path),
                "--output",
                str(request_path),
            ],
            int(values.get("render_timeout_seconds", 3600)),
        )
        records.append(
            {
                "index": index,
                "status": "planned",
                "request_file": str(request_path),
                "plan_file": "",
                "approval_hash": "",
                "submitted_approval_hash": "",
                "run_directory": "",
                "output_file": "",
                "error": "",
                "duration_seconds": scene["duration_seconds"],
                "render_duration_seconds": overrides["frame_count"] / fps,
                "narration": scene["narration"],
                "visual_prompt": scene["visual_prompt"],
                "camera": scene["camera"],
                "lighting": scene["lighting"],
                "continuity": scene["continuity"],
                "negative_prompt": scene["negative_prompt"],
            }
        )

    manifest_path = ltx_dir / "plan-batch.json"
    plans_dir = ltx_dir / "plans"
    manifest = {
        "schema_version": 1,
        "scenes": [
            {
                "scene_index": record["index"],
                "request_file": record["request_file"],
            }
            for record in records
        ],
    }
    _create_json(manifest_path, manifest)
    result = run_cli_stream(
        [
            "plan-batch",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(plans_dir),
        ],
        int(values.get("render_timeout_seconds", 3600)),
        lambda event: progress(event["current"], event["total"])
        if callable(progress)
        else None,
    )

    summaries = result.get("scenes") if isinstance(result, dict) else None
    if not isinstance(summaries, list):
        raise LtxError("LTX batch planner returned an invalid scene set")
    by_index = {}
    for item in summaries:
        if (
            not isinstance(item, dict)
            or type(item.get("scene_index")) is not int
            or item["scene_index"] in by_index
        ):
            raise LtxError("LTX batch planner returned an invalid scene set")
        by_index[item["scene_index"]] = item
    expected = {record["index"] for record in records}
    if set(by_index) != expected or result.get("submissions") != 0:
        raise LtxError("LTX batch planner returned an invalid scene set")

    for record in records:
        summary = by_index[record["index"]]
        plan_file = summary.get("plan_file")
        approval_hash = summary.get("approval_hash")
        if (
            not isinstance(plan_file, str)
            or not plan_file
            or not isinstance(approval_hash, str)
            or not approval_hash
        ):
            raise LtxError("LTX batch planner returned an invalid scene set")
        plan_path = Path(plan_file).expanduser().resolve()
        expected_path = (plans_dir / f"{record['index']:03d}.json").resolve()
        if plan_path != expected_path:
            raise LtxError(
                f"LTX scene {record['index']} returned an unexpected plan path"
            )
        if not plan_path.is_file():
            raise LtxError(f"LTX scene {record['index']} plan is unavailable")
        plan = _read_json(plan_path)
        if plan.get("approval_hash") != approval_hash:
            raise LtxError(f"LTX scene {record['index']} plan hash mismatch")
        if first_pass and scale:
            parameters = plan["parameters"]
            if (parameters.get("width"), parameters.get("height")) != first_pass:
                raise LtxError(f"LTX scene {record['index']} plan dimensions do not match the request")
            record.update(first_pass_width=first_pass[0], first_pass_height=first_pass[1],
                          generation_width=first_pass[0] * scale, generation_height=first_pass[1] * scale)
        record["plan_file"] = str(plan_path)
        record["approval_hash"] = approval_hash

    batch = {"task_id": task_id, "phase": "awaiting_approval", "model_version": version,
             "scenes": records, "render_attempt_started": False}
    if timing is not None:
        speed = utils.normalize_clip_speed(getattr(params, "video_clip_speed", 1))
        for record in records:
            record["playback_duration_seconds"] = record["render_duration_seconds"] / speed
        batch["timing"] = {
            **timing, "planned_scenes": len(records),
            "planned_render_seconds": sum(record["render_duration_seconds"] for record in records),
            "planned_playback_seconds": sum(record["playback_duration_seconds"] for record in records),
        }
    with _dispatch_lock(task_id):
        _replace_json(ltx_dir / "batch.json", batch)
    return batch


def load_batch(task_id: str) -> dict:
    path = Path(utils.task_dir(task_id)) / "ltx" / "batch.json"
    try:
        batch = _read_json(path)
        json.dumps(batch, allow_nan=False)
    except FileNotFoundError as exc:
        raise LtxError(f"LTX batch is unavailable: {path}") from exc
    except (OSError, ValueError, RecursionError) as exc:
        raise LtxError("LTX batch record cannot be read") from exc
    if (not isinstance(batch, dict) or batch.get("task_id") != task_id
            or not isinstance(batch.get("phase"), str) or not isinstance(batch.get("scenes"), list)):
        raise LtxError("LTX batch record is invalid")
    return batch


def approve_batch(task_id: str, approvals: list[dict]) -> dict:
    with _dispatch_lock(task_id):
        batch = _approve_reviewed_batch(task_id, approvals)
        sm.state.patch_task(task_id, phase="approved", ltx_scenes=batch["scenes"])
        return batch


@contextmanager
def _editor_slot(task_id):
    with _dispatch_lock(task_id), ExitStack() as active:
        _take_render_lock(active, _reservation_path(task_id))
        if _reservation_path(task_id).is_file():
            raise RenderBusy("This task is already queued")
        task = sm.state.get_task(task_id) or {}
        if task.get("phase") in {"queued", "rendering", "composing", "stopping", "stop_unconfirmed"}:
            raise RenderBusy("Wait for the current operation or confirmed stop before editing")
        batch = load_batch(task_id)
        if (batch.get("phase") not in {"awaiting_approval", "approved", "editing"}
                or batch.get("retry_error") or not batch["scenes"]
                or any(scene.get("status") not in {"planned", "approved"} for scene in batch["scenes"])):
            raise LtxError(batch.get("retry_error") or "Only scenes that have not started can be edited")
        yield batch


def open_scene_editor(task_id: str) -> str:
    """Publish owned drafts to ComfyUI without submitting a generation job."""
    with _editor_slot(task_id) as batch:
        if any(_read_json(Path(scene["plan_file"])).get("profile", {}).get("remote_server") is True
               for scene in batch["scenes"]):
            raise LtxError("The ComfyUI scene editor is available only on the local server")
        editor = batch.get("editor")
        if editor is None:
            session = uuid4().hex
            scenes = [{"scene_index": scene["index"], "plan_file": scene["plan_file"]}
                      for scene in batch["scenes"]]
            revision = hashlib.sha256(json.dumps(
                [{"index": scene["index"], "approval_hash": scene["approval_hash"]}
                 for scene in batch["scenes"]], sort_keys=True).encode()).hexdigest()
            manifest_path = Path(utils.storage_dir("ltx-editor")) / f"{session}.json"
            _create_json(manifest_path, {"schema_version": 1, "session": session,
                                         "revision": revision, "scenes": scenes})
            editor = {"session": session, "revision": revision, "manifest_file": str(manifest_path)}
        result = run_cli(["export-editor", "--manifest", editor["manifest_file"]], 600)
        if (result.get("ok") is not True or result.get("session") != editor["session"] or result.get("revision") != editor["revision"]
                or result.get("submissions") != 0 or not isinstance(result.get("editor_url"), str)):
            raise LtxError("ComfyUI returned an invalid editor session")
        editor["url"] = result["editor_url"]
        batch.update(phase="editing", editor=editor)
        _replace_json(_batch_path(task_id), batch)
        sm.state.patch_task(task_id, phase="editing", ltx_scenes=batch["scenes"],
                            status_message="Edit and save scenes in ComfyUI, then start the batch here.")
        return editor["url"]


def apply_scene_editor(task_id: str) -> dict:
    """Adopt a validated new plan set; retain the old batch if any scene fails."""
    with _editor_slot(task_id) as batch:
        editor = batch.get("editor")
        if batch.get("phase") != "editing" or not isinstance(editor, dict):
            raise LtxError("Open this batch in ComfyUI before loading edits")
        output = Path(utils.task_dir(task_id)) / "ltx" / "edited" / uuid4().hex
        result = run_cli(["import-editor", "--manifest", editor["manifest_file"],
                          "--output-dir", str(output)], 600)
        summaries = result.get("scenes")
        if (result.get("ok") is not True or result.get("session") != editor["session"] or result.get("revision") != editor["revision"]
                or result.get("submissions") != 0 or not isinstance(summaries, list)):
            raise LtxError("ComfyUI returned an invalid edited scene set")
        if any(not isinstance(item, dict) or type(item.get("scene_index")) is not int
               or item["scene_index"] <= 0 for item in summaries):
            raise LtxError("Edited scenes do not match this batch")
        by_index = {item["scene_index"]: item for item in summaries}
        if len(by_index) != len(summaries) or set(by_index) != {s["index"] for s in batch["scenes"]}:
            raise LtxError("Edited scenes do not match this batch")
        records = []
        task = sm.state.get_task(task_id) or {}
        speed = utils.normalize_clip_speed(task.get("params", {}).get("video_clip_speed", 1))
        for original in batch["scenes"]:
            summary = by_index[original["index"]]
            plan_path = Path(summary.get("plan_file", "")).resolve()
            expected = (output / f"scene-{original['index']:03d}.plans" / f"{original['index']:03d}.json").resolve()
            request_path = Path(summary.get("request_file", "")).resolve()
            if (plan_path != expected or summary.get("base_approval_hash") != original["approval_hash"]
                    or request_path != (output / f"scene-{original['index']:03d}.request.json").resolve()
                    or not request_path.is_file()):
                raise LtxError("Edited scene does not match its original plan")
            plan = _read_json(plan_path)
            if plan.get("approval_hash") != summary.get("approval_hash"):
                raise LtxError("Edited scene plan hash mismatch")
            parameters = plan["parameters"]
            duration = parameters["frame_count"] / parameters["fps"]
            record = {**original, "plan_file": str(plan_path), "request_file": str(request_path),
                      "approval_hash": plan["approval_hash"], "submitted_approval_hash": "",
                      "status": "planned", "visual_prompt": plan["prompt"],
                      "negative_prompt": plan["negative_prompt"], "duration_seconds": duration,
                      "render_duration_seconds": duration, "playback_duration_seconds": duration / speed}
            scale = generation_scale(plan)
            record.update(first_pass_width=parameters["width"], first_pass_height=parameters["height"],
                          generation_width=parameters["width"] * scale if scale else None,
                          generation_height=parameters["height"] * scale if scale else None)
            records.append(record)
        timing = dict(batch.get("timing") or {})
        total = sum(scene["playback_duration_seconds"] for scene in records)
        narration = timing.get("audio_duration_seconds")
        if narration is not None and narration - total > 1e-9:
            raise LtxError(f"Edited scenes cover {total:.3f}s, but narration needs {narration:.3f}s")
        timing.update(duration_mode="manual", planned_scenes=len(records),
                      planned_render_seconds=sum(scene["render_duration_seconds"] for scene in records),
                      planned_playback_seconds=total, max_playback_seconds=max(total, timing.get("max_playback_seconds", 0)))
        batch.update(phase="awaiting_approval", scenes=records, timing=timing, edited_in_comfyui=True)
        batch.pop("editor", None)
        _replace_json(_batch_path(task_id), batch)
        sm.state.patch_task(task_id, phase="awaiting_approval", ltx_scenes=records,
                            status_message="Saved ComfyUI edits are ready.")
        return batch


def close_scene_editor(task_id: str) -> None:
    """Return to the original plans, retaining saved drafts for reopening."""
    with _editor_slot(task_id) as batch:
        if batch.get("phase") == "editing":
            batch["phase"] = "awaiting_approval"
            _replace_json(_batch_path(task_id), batch)
            sm.state.patch_task(task_id, phase="awaiting_approval", ltx_scenes=batch["scenes"])


def _approve_reviewed_batch(task_id, approvals):
    batch = load_batch(task_id)
    if batch.get("phase") != "awaiting_approval":
        raise LtxError("LTX batch is not awaiting approval")
    if not isinstance(approvals, list) or not approvals or any(
        not isinstance(item, dict) or type(item.get("scene_index")) is not int
        or not isinstance(item.get("approval_hash"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", item["approval_hash"]) for item in approvals
    ):
        raise LtxError("Supply the index and reviewed approval hash for every scene")
    indexes = [item["scene_index"] for item in approvals]
    if len(indexes) != len(set(indexes)):
        raise LtxError("duplicate LTX scene approval")
    if any(not isinstance(scene, dict) or type(scene.get("index")) is not int for scene in batch["scenes"]):
        raise LtxError("Saved LTX scene list is invalid")
    expected_indexes = [scene["index"] for scene in batch["scenes"]]
    if sorted(indexes) != sorted(expected_indexes):
        raise LtxError("approval must cover every scene exactly once")
    submitted = {item["scene_index"]: item["approval_hash"] for item in approvals}
    directory = _batch_path(task_id).parent.resolve()
    for scene in batch["scenes"]:
        index = scene["index"]
        if submitted[index] != scene.get("approval_hash"):
            raise LtxError(f"approval hash mismatch for scene {index}")
        try:
            plan = Path(scene["plan_file"]).resolve()
            if not plan.is_relative_to(directory) or not plan.is_file():
                raise ValueError("scene plan must be inside the task")
            verified = run_cli(["verify-plan", "--plan", str(plan), "--approval-hash", submitted[index]], 60)
            if verified.get("ok") is not True or verified.get("approval_hash") != submitted[index]:
                raise ValueError("unexpected plan verification result")
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            raise LtxError(f"Saved LTX plan is unavailable or invalid for scene {index}") from exc
    for scene in batch["scenes"]:
        scene["submitted_approval_hash"] = submitted[scene["index"]]
        scene["status"] = "approved"
    batch["phase"] = "approved"
    _replace_json(_batch_path(task_id), batch)
    return batch


def completed_materials(task_id: str, *, snapshot_dir: Path | None = None, batch=None) -> list[str]:
    if batch is None:
        batch = load_batch(task_id)
    if batch.get("phase") != "complete":
        raise LtxError("LTX batch is not complete")
    if not batch["scenes"]:
        raise LtxError("completed LTX batch has missing outputs")
    outputs = []
    for index, scene in enumerate(batch["scenes"], 1):
        path = Path(scene.get("output_file") or "")
        if not path.is_file():
            raise LtxError(f"Completed LTX scene {index} has no output file")
        expected = scene.get("output_sha256")
        if expected is None and scene.get("run_directory"):
            # Older batches have the checksum only in the operator's completion report.
            try:
                report = _read_json(Path(scene["run_directory"]) / "report.json")
            except (OSError, ValueError) as exc:
                raise LtxError(f"Cannot read original output report for LTX scene {index}") from exc
            if (report.get("status") == "completed" and scene.get("approval_hash")
                    and report.get("approval_hash") == scene["approval_hash"]):
                expected = next((item.get("content_sha256") for item in report.get("outputs", [])
                                 if item.get("path") and Path(item["path"]) == path), None)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise LtxError(f"No original output checksum for LTX scene {index}")
        destination = snapshot_dir / f"{index:03d}{path.suffix}" if snapshot_dir is not None else path
        checksum = hashlib.sha256()
        with ExitStack() as stack:
            source = stack.enter_context(path.open("rb"))
            copy = stack.enter_context(destination.open("xb")) if snapshot_dir is not None else None
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                checksum.update(chunk)
                if copy:
                    copy.write(chunk)
        if checksum.hexdigest() != expected:
            raise LtxError(f"Completed output changed for LTX scene {index}")
        outputs.append(str(destination))
    return outputs


def _take_render_lock(active, claim):
    try:
        active.enter_context(_lock(claim.with_suffix(".render.lock"), timeout=0))
    except ProjectError as exc:
        raise RenderBusy("This task is already rendering or composing") from exc


def _reservation_path(task_id):
    # Keep queue ownership outside downloadable task files and the directory being deleted.
    name = hashlib.sha256(task_id.encode()).hexdigest() + ".json"
    return Path(utils.storage_dir("ltx-state")) / "queue" / name


def _dispatch_lock(task_id, timeout=30):
    return _lock(_reservation_path(task_id).with_suffix(".lock"), timeout=timeout)


@contextmanager
def guard_task_deletion(task_id):
    claim = _reservation_path(task_id)
    folder = Path(utils.task_dir()) / task_id / "ltx"
    batch_path = folder / "batch.json"
    if (not folder.is_dir() and not claim.is_file()
            and (sm.state.get_task(task_id) or {}).get("phase") != "stop_unconfirmed"):
        yield
        return
    with _dispatch_lock(task_id), ExitStack() as active:
        _take_render_lock(active, claim)
        if (sm.state.get_task(task_id) or {}).get("phase") == "stop_unconfirmed":
            raise RenderBusy("Remote stop is not confirmed; keep this task's run records")
        batch = {}
        if batch_path.is_file():
            try:
                batch = load_batch(task_id)
            except LtxError as exc:
                raise RenderBusy("Cannot verify LTX batch state before deletion") from exc
            if batch["phase"] == "stop_unconfirmed" or batch.get("retake_pending"):
                raise RenderBusy("Remote stop is not confirmed; keep this task's run records")
        if (_has_unresolved_runs(folder / "duration-predictions") or (
                batch.get("phase") not in {"complete", "failed", "stopped"}
                and (batch.get("phase") == "rendering" or batch.get("render_attempt_started") is True
                     or _has_unresolved_runs(folder / "scenes")))):
            raise RenderBusy("Remote work could still be active; keep this task's run records")
        if claim.is_file():
            from app.services import task as tm

            pending = _read_json(claim)
            # A restart loses in-memory jobs and their task state, but Redis can still deliver queued jobs.
            if (pending.get("persistent") is not False
                    or not isinstance(sm.state, sm.MemoryState)
                    or sm.state.get_task(task_id) is not None
                    or tm._ltx_queue_owner_alive(pending.get("owner"))):
                raise RenderBusy("This task is already queued")
            claim.unlink()
        yield


def _report_without_submission(report):
    return report.get("status") in {"not_submitted", "failed_before_submission"} and not report.get("prompt_id")


def _matches_prediction_rejection(proof, report, directory):
    index = report.get("scene_index")
    digest = report.get("approval_hash")
    return (type(index) is int and 1 <= index <= 20 and directory.name == f"{index:03d}"
            and type(proof.get("prediction_scene_index")) is int and proof["prediction_scene_index"] == index
            and isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest)
            and report.get("prompt_id") is None and _matches_submission_rejection(proof, digest))


def _has_unresolved_runs(folder, *, status_deadline=None):
    locations = {file.parent for name in ("run.json", "report.json") for file in folder.rglob(name)}
    deadline = status_deadline
    for location in locations:
        try:
            report = _read_json(location / "report.json")
            record = location / "run.json"
            if not record.exists() and _report_without_submission(report):
                continue
            envelope = _read_json(record)
            owned = envelope["record"]
            prompt = owned.get("prompt_id")
            signature = envelope.get("signature")
            if owned.get("terminal_status") == "submission_rejected":
                approval = owned.get("approval_hash")
                remaining = deadline - time.monotonic() if deadline is not None else 2
                if "prediction_scene_index" in owned:
                    if remaining > 0:
                        proof = run_cli(["status", "--run-record", str(record), "--state-root",
                                         utils.storage_dir("ltx-state")], remaining)
                        if _matches_prediction_rejection(proof, report, location):
                            continue
                    return True
                if (remaining > 0 and isinstance(approval, str) and re.fullmatch(r"[0-9a-f]{64}", approval)
                        and _receipt_run_directory(approval) == location.resolve()):
                    proof = run_cli(["status", "--run-record", str(record), "--state-root",
                                     utils.storage_dir("ltx-state")], remaining)
                    if _matches_submission_rejection(proof, approval):
                        continue
                return True
            bound_record = (isinstance(signature, str) and re.fullmatch(r"[0-9a-f]{64}", signature)
                            and isinstance(prompt, str) and prompt
                            and Path(owned.get("run_directory", "")).resolve() == location.resolve())
            if bound_record and report.get("prompt_id") == prompt:
                cleanup = report.get("cleanup")
                if (report.get("status") in ("completed", "failed", "cancelled")
                        or (isinstance(cleanup, dict) and cleanup.get("status") == "confirmed_cancelled"
                            and isinstance(owned.get("approval_hash"), str)
                            and report.get("approval_hash") == owned["approval_hash"])
                        or (type(report.get("scene_index")) is int and report["scene_index"] > 0
                            and report.get("cancellation") == "confirmed_cancelled")):
                    continue
            # Only the operator can authenticate terminal evidence in the signed envelope.
            if deadline is None and owned.get("terminal_status") in ("completed", "failed", "cancelled"):
                deadline = time.monotonic() + 2
            remaining = deadline - time.monotonic() if deadline is not None else 0
            if bound_record and report.get("prompt_id") in (None, prompt) and remaining > 0:
                result = run_cli(["status", "--run-record", str(record), "--state-root", utils.storage_dir("ltx-state")],
                                 remaining)
                if result.get("prompt_id") == prompt and result.get("status") in ("completed", "failed", "cancelled"):
                    continue
        except (AttributeError, KeyError, OSError, TypeError, ValueError, LtxError, subprocess.TimeoutExpired):
            pass
        return True
    return False


def recover_interrupted_renders():
    """Reconcile unlocked LTX work after an API process restart without submitting prompts."""
    prediction_phases = {"predicting_scene_durations", "ltx_duration_prediction"}
    phases = {"rendering", "composing", "stopping", "stop_unconfirmed", *prediction_phases}
    try:
        task_ids = sm.state.list_task_ids()
    except Exception as exc:
        logger.warning("Could not list interrupted LTX tasks ({})", type(exc).__name__)
        return
    # Slow or offline ComfyUI must not delay unrelated API routes per task.
    deadline = time.monotonic() + 4
    for task_id in task_ids:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", task_id):
            continue
        try:
            task = sm.state.get_task(task_id) or {}
            phase = task.get("phase")
            if phase not in phases:
                continue
            params = task.get("params")
            folder = Path(utils.task_dir()) / task_id / "ltx"
            predictions = folder / "duration-predictions"
            is_ltx = isinstance(params, dict) and params.get("video_source") == "ltx"
            if phase not in prediction_phases and not (is_ltx or (folder / "batch.json").is_file() or predictions.is_dir()):
                continue
            with _dispatch_lock(task_id, timeout=0), ExitStack() as active:
                claim = _reservation_path(task_id)
                _take_render_lock(active, claim)
                task = sm.state.get_task(task_id) or {}
                phase = task.get("phase")
                if claim.is_file() or phase not in phases:
                    continue
                if phase in prediction_phases or (phase == "stop_unconfirmed" and predictions.is_dir()
                                                   and not (folder / "batch.json").is_file()):
                    _recover_interrupted_prediction(task_id, phase, predictions, deadline)
                else:
                    _recover_interrupted_render(task_id, deadline)
        except (ProjectError, RenderBusy):
            continue
        except Exception as exc:
            logger.warning("Could not recover interrupted LTX task {} ({})", task_id, type(exc).__name__)


def _recover_interrupted_prediction(task_id, phase, directory, deadline):
    manifests = list(directory.glob("*/manifest.json"))
    terminal = (phase == "predicting_scene_durations" or bool(manifests)) and not _has_unresolved_runs(
        directory, status_deadline=deadline)
    for manifest in manifests:
        if not terminal:
            break
        try:
            scenes = _read_json(manifest)["scenes"]
            # Each submission has a durable report before its POST. Future scenes may have none.
            terminal = isinstance(scenes, list) and bool(scenes)
        except (KeyError, OSError, ValueError, TypeError):
            terminal = False
    message = ("MPT restarted before saving the predicted scene plans. Create a new task." if terminal else
               "MPT restarted during duration prediction. ComfyUI work is unconfirmed; keep the run records.")
    sm.state.patch_task(task_id, state=const.TASK_STATE_FAILED,
                        phase="failed" if terminal else "stop_unconfirmed",
                        failed_stage="ltx_recovery", status_message=message, error=message)


def _recover_interrupted_render(task_id, deadline):
    batch_path = _batch_path(task_id)
    try:
        batch = load_batch(task_id)
    except LtxError:
        batch = None
    if batch and batch.get("retake_pending"):
        try:
            manifest = _retake_manifest(batch_path, batch["retake_pending"])
            candidate = _read_json(manifest)
            terminal = candidate.get("phase") in {"complete", "failed", "stopped"} or (
                candidate.get("phase") == "approved" and candidate.get("render_attempt_started") is False)
            if not terminal:
                terminal = _cancel_owned_runs(manifest, status_deadline=deadline) == []
        except (OSError, KeyError, ValueError, TypeError, AttributeError, LtxError):
            terminal = False
        if terminal:
            batch.pop("retake_pending")
            _replace_json(batch_path, batch)
            record_retake_error(task_id, "MPT restarted during a scene retake. The previous video was kept.")
        else:
            record_retake_error(task_id, "Retake stop is unconfirmed. Keep the run records.", unconfirmed=True)
        return
    phase = batch["phase"] if batch else None
    task = sm.state.get_task(task_id) or {}
    if phase == "complete" and task.get("phase") != "stop_unconfirmed":
        params = task.get("params")
        expected_count = params.get("video_count", 1) if isinstance(params, dict) else 1
        result = _saved_composition_result(batch_path, batch, expected_count)
        if result is not None:
            sm.state.patch_task(task_id, state=const.TASK_STATE_COMPLETE, phase="complete", progress=100,
                                error=None, status_message=None, failed_stage=None, retake_error=None,
                                ltx_scenes=batch["scenes"], **result)
            return
    safe_to_retry = phase == "approved" and batch.get("render_attempt_started") is False
    if not safe_to_retry and phase not in {"complete", "failed", "stopped"}:
        try:
            errors = _cancel_owned_runs(batch_path, status_deadline=deadline) if batch else None
        except (KeyError, OSError, ValueError, TypeError):
            errors = None
        if errors == []:
            phase = ("complete" if batch["scenes"] and all(scene.get("status") == "complete"
                     for scene in batch["scenes"]) else "failed")
        else:
            phase = "stop_unconfirmed"
        if batch:
            batch["phase"] = phase
            if phase != "stop_unconfirmed":
                batch.pop("stop_error", None)
    if safe_to_retry:
        message = "MPT restarted before starting ComfyUI work. Send the reviewed scenes again when ready."
    elif phase == "complete":
        message = "MPT restarted before finishing composition. Recompose video from the verified clips."
    elif phase == "stop_unconfirmed":
        message = "MPT restarted during local generation. ComfyUI work is unconfirmed; keep the run records."
        if batch:
            batch["stop_error"] = message
    else:
        message = "MPT restarted during local generation. Create a new task and review its plans before rendering."
        batch["retry_error"] = message
    if batch:
        try:
            _replace_json(batch_path, batch)
        except OSError:
            phase = "stop_unconfirmed"
            message = "MPT could not save recovered LTX state. Keep this task's run records."
    sm.state.patch_task(task_id, state=const.TASK_STATE_FAILED,
                        phase="failed" if phase in {"approved", "complete"} else phase,
                        failed_stage="ltx_recovery", status_message=message, error=message)


@contextmanager
def render_slot(task_id, ticket=None):
    claim = _reservation_path(task_id)
    with ExitStack() as active:
        with _dispatch_lock(task_id):
            _take_render_lock(active, claim)
            pending = _read_json(claim) if claim.is_file() else {}
            if pending.get("ticket") != ticket:
                raise RenderBusy("This request no longer owns the queued task")
            if pending:
                claim.unlink()
        yield pending.get("previous")


def reserve_render(task_id, owner, owner_alive, *, persistent=False):
    claim = _reservation_path(task_id)
    with _dispatch_lock(task_id), ExitStack() as active:
        _take_render_lock(active, claim)
        batch = load_batch(task_id)
        if batch.get("phase") == "stop_unconfirmed" or batch.get("retake_pending"):
            raise LtxError("Remote stop is not confirmed for this task")
        if batch.get("phase") not in {"approved", "complete"}:
            raise LtxError(batch.get("retry_error") or "LTX batch is not approved or complete")
        pending = _read_json(claim) if claim.is_file() else {}
        if pending and (pending["owner"] == owner or owner_alive(pending["owner"])):
            raise RenderBusy("This task is already queued")
        task = sm.state.get_task(task_id) or {}
        if task.get("phase") == "stop_unconfirmed":
            raise LtxError("Remote stop is not confirmed for this task")
        previous = pending.get("previous", {key: task.get(key) for key in ("state", "phase")})
        ticket = uuid4().hex
        _replace_json(claim, {"owner": owner, "ticket": ticket, "previous": previous, "persistent": persistent})
        try:
            sm.state.patch_task(task_id, state=const.TASK_STATE_PROCESSING, phase="queued")
        except BaseException:
            claim.unlink()
            raise
        return ticket


def release_reservation(task_id, ticket, **state_updates):
    claim = _reservation_path(task_id)
    if not isinstance(ticket, str) or not ticket or not claim.is_file():
        return
    with _dispatch_lock(task_id):
        pending = _read_json(claim) if claim.is_file() else {}
        if pending.get("ticket") == ticket:
            claim.unlink()
            updates = state_updates or pending.get("previous")
            if updates:
                sm.state.patch_task(task_id, **updates)


def render_batch(task_id: str) -> list[str]:
    with render_slot(task_id):
        return _render_approved_batch(task_id, _batch_path(task_id))


@contextmanager
def composition_materials(task_id, batch):
    """Keep the exact bytes checked against the operator report throughout encoding."""
    previous_result = batch.get("composition_result")
    originals = ([scene["output_file"] for scene in batch["scenes"]] if batch["phase"] == "complete"
                 else _render_approved_batch(task_id, _batch_path(task_id), batch=batch))
    with tempfile.TemporaryDirectory(prefix=".ltx-materials-", dir=utils.task_dir(task_id)) as folder:
        yield originals, completed_materials(task_id, snapshot_dir=Path(folder), batch=batch)
    if batch.get("composition_result") is not previous_result:
        _commit_video_revision(_batch_path(task_id), batch)


def composition_outputs_valid(task_path, result, expected_count):
    """Require the complete set of distinct, nonempty video files in the task folder."""
    try:
        videos, combined = result["videos"], result["combined_videos"]
        if (type(expected_count) is not int or not isinstance(videos, list) or not isinstance(combined, list)
                or not len(videos) == len(combined) == expected_count > 0):
            return False
        task_path = Path(task_path).resolve()
        outputs = {Path(filename).resolve() for filename in videos + combined}
        return len(outputs) == 2 * expected_count and all(
            video.parent == task_path and video.suffix == ".mp4"
            and video.is_file() and video.stat().st_size > 0 for video in outputs
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False


def _saved_composition_result(batch_path, batch, expected_count):
    result = batch.get("composition_result")
    if not isinstance(result, dict) or not composition_outputs_valid(batch_path.parent.parent, result, expected_count):
        return None
    fields = ("videos", "combined_videos", "materials", "audio_file", "audio_duration", "subtitle_path", "warnings")
    try:
        return {key: result[key] for key in fields}
    except KeyError:
        return None


def _commit_video_revision(batch_path, batch):
    execution_control.checkpoint()
    _replace_json(batch_path, batch)
    token = execution_control.current_token()
    if token is not None:
        token.result_committed = True


def _retake_manifest(batch_path, take_id):
    if not isinstance(take_id, str) or not re.fullmatch(r"[0-9a-f]{32}", take_id):
        raise LtxError("Invalid retake record")
    manifest = batch_path.parent / "retakes" / take_id / "batch.json"
    if not manifest.resolve().is_relative_to(batch_path.parent.resolve()):
        raise LtxError("Retake record escaped its task")
    return manifest


def latest_retake_video(task_path):
    """Read only the committed retake so interrupted encodes stay out of history."""
    task_path = Path(task_path).resolve()
    try:
        filename = _read_json(task_path / "ltx" / "batch.json")["retake_video"]
        if isinstance(filename, str) and re.fullmatch(r"retake-[0-9a-f]{32}-final-1\.mp4", filename):
            video = (task_path / filename).resolve()
            if video.parent == task_path and video.is_file():
                return str(video)
    except (OSError, KeyError, ValueError, TypeError):
        pass
    return ""


def record_retake_error(task_id, message, *, unconfirmed=False):
    sm.state.patch_task(task_id, state=const.TASK_STATE_FAILED if unconfirmed else const.TASK_STATE_COMPLETE,
                        phase="stop_unconfirmed" if unconfirmed else "complete", progress=100,
                        retake_error=str(message), status_message=None, error=None, failed_stage=None)


@contextmanager
def retake_materials(task_id, batch, selection):
    """Render a separately approved clip and commit it only after successful composition."""
    index, approval, prompt, seed = selection
    if (batch.get("phase") != "complete" or batch.get("retake_pending") or type(index) is not int
            or not 1 <= index <= len(batch["scenes"])):
        raise LtxError("Select a scene from a completed batch")
    original = batch["scenes"][index - 1]
    if original.get("index") != index or original.get("approval_hash") != approval:
        raise LtxError("This scene has changed. Refresh the task before starting a retake.")
    if not isinstance(prompt, str) or not prompt.strip():
        raise LtxError("Enter a scene prompt")
    if not isinstance(seed, str) or (seed and not re.fullmatch(r"[0-9]{1,20}", seed)):
        raise LtxError("The seed must be a non-negative integer, or blank to keep the original")

    batch_path = _batch_path(task_id)
    take_id = uuid4().hex
    manifest = _retake_manifest(batch_path, take_id)
    folder = manifest.parent
    original_plan = Path(original["plan_file"])
    verified = run_cli(["verify-plan", "--plan", str(original_plan), "--approval-hash", approval], 600)
    if verified.get("ok") is not True or verified.get("approval_hash") != approval:
        raise LtxError("The original scene plan could not be verified")
    plan = _read_json(original_plan)
    prefix = f"retake-{take_id}-"
    overrides = {"prompt": prompt.strip(), "client_id": str(uuid4()),
                 "output_prefix": f"mpt_{task_id}_scene_{index:03d}_{prefix.rstrip('-')}"}
    if seed:
        count = sum(key == "seed" or key.endswith("_seed") for key in plan["parameters"])
        base = int(seed)
        if not count or base > 2**64 - count:
            raise LtxError("The seed is too large for this workflow")
        overrides["seeds"] = list(range(base, base + count))
    overrides_path = folder / "overrides.json"
    request_path = folder / "request.json"
    plan_path = folder / "plan.json"
    _create_json(overrides_path, overrides)
    run_cli(["prepare-request", "--template-request", str(original_plan), "--overrides", str(overrides_path),
             "--output", str(request_path)], 600)
    planned = run_cli(["plan", "--request", str(request_path), "--output", str(plan_path)], 600)
    new_approval = planned.get("approval_hash")
    if (not isinstance(new_approval, str) or not re.fullmatch(r"[0-9a-f]{64}", new_approval)
            or new_approval == approval or _read_json(plan_path).get("approval_hash") != new_approval):
        raise LtxError("The retake plan could not be verified")
    replacement = {**original, "index": 1, "status": "approved", "request_file": str(request_path),
                   "plan_file": str(plan_path), "approval_hash": new_approval,
                   "submitted_approval_hash": new_approval, "visual_prompt": overrides["prompt"],
                   "run_directory": "", "output_file": "", "error": ""}
    replacement.pop("output_sha256", None)
    candidate = {"task_id": task_id, "phase": "approved", "scenes": [replacement],
                 "render_attempt_started": False}
    _create_json(manifest, candidate)
    _replace_json(batch_path, {**batch, "retake_pending": take_id})
    try:
        _render_approved_batch(task_id, manifest, batch=candidate)
        updated = {**batch, "scenes": list(batch["scenes"])}
        updated["scenes"][index - 1] = {**replacement, "index": index}
        originals = [scene["output_file"] for scene in updated["scenes"]]
        with tempfile.TemporaryDirectory(prefix=".ltx-materials-", dir=utils.task_dir(task_id)) as scratch:
            materials = completed_materials(task_id, snapshot_dir=Path(scratch), batch=updated)
            yield updated, originals, materials, prefix
        _commit_video_revision(batch_path, updated)
    except (Exception, execution_control.GenerationStopped) as exc:
        if not getattr(exc, "remote_stop_unconfirmed", False):
            try:
                _replace_json(batch_path, batch)
            except OSError as state_error:
                exc.remote_stop_unconfirmed = True
                exc.add_note(f"Could not clear the retake record: {state_error}")
        raise


def _record_scene_progress(batch, batch_path, event):
    scenes = batch["scenes"]
    index = event["scene_index"]
    if type(index) is not int or not 1 <= index <= len(scenes):
        raise LtxError("invalid LTX scene progress")
    scene = scenes[index - 1]
    scene["run_directory"] = event["run_directory"]
    status = event["status"]
    batch["phase"] = "rendering"
    if status == "completed":
        primary = [item for item in event["outputs"] if item.get("primary") is True]
        if len(primary) != 1:
            raise LtxError(f"LTX scene {index} has no unique verified primary video")
        output = primary[0]
        path = Path(output["path"])
        if path.suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm"} or not path.is_file():
            raise LtxError(f"LTX scene {index} has no primary video file")
        scene.update(status="complete", output_file=str(path.resolve()),
                     output_sha256=output["content_sha256"])
    elif status == "queued":
        scene["status"] = "queued"
    elif status == "not_submitted":
        scene["status"] = "not_submitted"
    else:
        scene.update(status="failed", error="; ".join(event["errors"]) or status)
    _replace_json(batch_path, batch)


def _render_approved_batch(task_id, batch_path, *, batch=None):
    if batch is None:
        batch = load_batch(task_id)
    scenes = batch["scenes"]
    if batch.get("phase") != "approved" or not scenes or any(
        scene.get("status") != "approved"
        or scene.get("submitted_approval_hash") != scene.get("approval_hash")
        for scene in scenes
    ):
        raise LtxError(batch.get("retry_error") or "LTX batch is not fully approved")
    try:
        timeout = int(settings().get("render_timeout_seconds", 3600))
    except (ValueError, TypeError, OverflowError) as exc:
        raise LtxError("render_timeout_seconds must be a positive number of seconds") from exc
    if timeout <= 0:
        raise LtxError("render_timeout_seconds must be a positive number of seconds")

    # A restart can permit a retry only when the durable batch proves no operator was started.
    previous_attempt = batch.get("render_attempt_started")
    batch["render_attempt_started"] = True
    _replace_json(batch_path, batch)
    try:
        result = run_cli_stream(
            ["run-batch", "--manifest", str(batch_path), "--runs-dir", str(batch_path.parent / "scenes"),
             "--state-root", utils.storage_dir("ltx-state"), "--timeout", str(timeout)],
            timeout * len(scenes) + 30, lambda event: _record_scene_progress(batch, batch_path, event),
        )
        if not result.get("ok") or any(s["status"] != "complete" for s in scenes):
            failed = next((s for s in scenes if s.get("error")), None)
            if failed:
                raise LtxError(f"LTX scene {failed['index']}: {failed['error']}")
            raise LtxError(f"LTX batch did not complete: {result.get('statuses')}")
    except (Exception, execution_control.GenerationStopped) as exc:
        initialized = False
        try:
            errors = _cancel_owned_runs(batch_path)
            run_paths = (batch_path.parent / "scenes" / f"{scene['index']:03d}" / "run" for scene in scenes)
            initialized = errors is None and any(path.exists() or path.is_symlink() for path in run_paths)
        except Exception as cleanup_error:
            errors = [f"Owned run check failed: {type(cleanup_error).__name__}: {cleanup_error}"]
        stopped = isinstance(exc, execution_control.GenerationStopped)
        if errors:
            exc.remote_stop_unconfirmed = True
            batch["phase"] = "stop_unconfirmed"
            batch["stop_error"] = "Remote stop unconfirmed: " + "; ".join(errors)
            token = execution_control.current_token()
            if token is not None:
                token.stop_error = batch["stop_error"]
        elif stopped:
            batch["phase"] = "stopped"
        elif errors is not None or batch["phase"] == "rendering" or initialized:
            batch["phase"] = "failed"
            if initialized:
                batch["retry_error"] = ("This render attempt cannot be resumed. Create a new task and review "
                                        "its scene plans before rendering again.")
        elif previous_attempt is None:
            batch.pop("render_attempt_started", None)
        else:
            batch["render_attempt_started"] = previous_attempt
        try:
            _replace_json(batch_path, batch)
        except Exception as state_error:
            exc.add_note(f"Could not persist LTX failure state: {state_error}")
        raise
    batch["phase"] = "complete"
    _replace_json(batch_path, batch)
    return [scene["output_file"] for scene in scenes]


def _receipt_run_directory(approval):
    receipt = Path(utils.storage_dir("ltx-state")) / "receipts" / f"{approval}.json"
    if not receipt.is_file():
        return None
    consumed = _read_json(receipt)
    if (not isinstance(consumed, dict) or consumed.get("approval_hash") != approval
            or not isinstance(consumed.get("run_directory"), str) or not consumed["run_directory"]):
        raise ValueError("approval receipt has no bound run directory")
    return Path(consumed["run_directory"]).resolve()


def _matches_submission_rejection(proof, approval):
    return (proof.get("status") == "submission_rejected" and proof.get("approval_hash") == approval
            and "prompt_id" in proof and proof["prompt_id"] is None)


def _cancel_owned_runs(batch_path, *, status_deadline=None):
    """Cancel owned jobs, or verify every scene's disposition when recovering after restart."""
    values = settings()
    errors = []
    runs = (batch_path.parent / "scenes").resolve()
    if not runs.is_relative_to(batch_path.parent.resolve()):
        return ["Run ownership directory escaped its task"]
    records = list(runs.rglob("run.json"))
    batch = _read_json(batch_path)
    started = bool(records) or any(scene.get("status") in ("queued", "complete") for scene in batch["scenes"])
    if status_deadline is not None and not batch["scenes"]:
        return ["No persisted scene dispositions to verify"]
    expected_records = {}
    receipt_runs = {}
    for scene in batch["scenes"]:
        approval = scene.get("approval_hash", "")
        if (not isinstance(approval, str) or not re.fullmatch(r"[0-9a-f]{64}", approval)
                or approval != scene.get("submitted_approval_hash")):
            errors.append("Scene approval ownership could not be verified")
            continue
        record = runs / f"{scene['index']:03d}" / "run" / "run.json"
        expected_records[record] = approval
        try:
            receipt_runs[approval] = _receipt_run_directory(approval)
        except (OSError, ValueError) as exc:
            errors.append(f"Scene {scene['index']}: approval receipt could not be verified: {exc}")
            continue
        if receipt_runs[approval] == record.parent.resolve():
            started = True
            if not record.is_file():
                errors.append(f"Scene {scene['index']}: approval consumed without signed prompt ownership")
    if status_deadline is not None:
        for record, approval in expected_records.items():
            if record in records:
                continue
            try:
                report = _read_json(record.with_name("report.json"))
                if (isinstance(report, dict) and report.get("approval_hash") == approval
                        and report.get("status") in ("not_submitted", "failed_before_submission")
                        and not report.get("prompt_id") and receipt_runs.get(approval) != record.parent.resolve()):
                    continue
            except (OSError, ValueError):
                pass
            errors.append(f"{record.parent}: no proof this scene was never submitted")
        if errors:
            return errors
    if started and not records and not errors:
        errors.append("No signed run record; remote submission could not be verified.")
    for report in runs.rglob("report.json"):
        if report.with_name("run.json").is_file():
            continue
        try:
            if not report.resolve().is_relative_to(runs):
                raise ValueError("submission report escaped its task")
            data = _read_json(report)
            if not isinstance(data, dict):
                raise ValueError("submission report must be an object")
            approval = data.get("approval_hash", "")
            consumed = isinstance(approval, str) and receipt_runs.get(approval) == report.parent.resolve()
            status = data.get("status")
            if not isinstance(status, str):
                raise ValueError("submission report status must be a string")
            if data.get("prompt_id") or consumed or status not in {"initializing", "failed_before_submission", "not_submitted"}:
                errors.append(f"{report.parent}: submission began without signed prompt ownership")
        except (OSError, ValueError):
            errors.append(f"{report.parent}: submission report could not be verified")
    for record in records:
        try:
            if not record.resolve().is_relative_to(runs) or record not in expected_records:
                raise ValueError("ownership record escaped its approved scene")
            envelope = _read_json(record)
            owned = envelope.get("record") if isinstance(envelope, dict) else None
            if (not isinstance(owned, dict) or owned.get("approval_hash") != expected_records[record]
                    or not isinstance(owned.get("run_directory"), str)
                    or Path(owned["run_directory"]).resolve() != record.parent.resolve()):
                raise ValueError("ownership record does not match its approved scene")
            rejected = owned.get("terminal_status") == "submission_rejected"
            if rejected:
                if "prompt_id" in owned or receipt_runs.get(owned["approval_hash"]) != record.parent.resolve():
                    raise ValueError("submission rejection does not match its approval receipt")
            elif not isinstance(owned.get("prompt_id"), str) or not owned["prompt_id"]:
                raise ValueError("ownership record has invalid prompt_id")
        except (OSError, ValueError) as exc:
            errors.append(f"{record.parent}: {exc}")
            continue
        try:
            timeout = 15 if status_deadline is None else status_deadline - time.monotonic()
            if timeout <= 0:
                errors.append("ComfyUI recovery check reached its deadline")
                break
            action = "status" if rejected or status_deadline is not None else "cancel"
            command = [values["python_executable"], values["operator_script"], action,
                       "--run-record", str(record), "--state-root", utils.storage_dir("ltx-state"), "--json"]
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                    errors="replace", shell=False, timeout=timeout)
            payload = json.loads(result.stdout)
            if not isinstance(payload, dict):
                raise ValueError("cancellation response must be an object")
            status = payload.get("status")
            if rejected:
                if result.returncode or not _matches_submission_rejection(payload, owned["approval_hash"]):
                    errors.append(f"{record.parent}: ComfyUI submission rejection could not be verified")
            elif status_deadline is not None:
                if (result.returncode or payload.get("prompt_id") != owned["prompt_id"]
                        or status not in ("completed", "failed", "cancelled")):
                    errors.append(f"{record.parent}: owned ComfyUI job has no terminal history")
            elif result.returncode or not isinstance(status, str) or status not in {"confirmed_cancelled", "completed", "failed"}:
                message = payload.get("error")
                errors.append(message if isinstance(message, str) and message else f"{record.parent.name}: cancellation not confirmed")
        except (KeyError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{record.parent.name}: {type(exc).__name__}")
    return errors if status_deadline is not None or started or errors else None


def _batch_path(task_id: str) -> Path:
    return Path(utils.task_dir(task_id)) / "ltx" / "batch.json"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _create_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True)


def _replace_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, encoding="utf-8", newline="\n", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True)
        except Exception:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
