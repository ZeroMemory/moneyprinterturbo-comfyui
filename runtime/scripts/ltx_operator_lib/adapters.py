"""Discover the editable inputs and outputs of supported ComfyUI workflows."""

from dataclasses import asdict, dataclass
from typing import Any

from .workflow import (
    MEDIA_INPUTS,
    NodeRef,
    Workflow,
    WorkflowFormat,
    input_link,
    linked_source,
    nodes_by_type,
    output_ancestors,
)


H3_FAMILY = "minimax-h3-t2v"
FAMILIES = frozenset(
    {
        "t2v-i2v-single-stage",
        "t2v-i2v-two-stage",
        "hdr",
        "lipdub",
        "motion-track",
        "union-control",
        H3_FAMILY,
    }
)


def video_suffix(inputs):
    container = inputs.get("format", "auto")
    codec = inputs.get("format.codec", inputs.get("codec", "auto"))
    if isinstance(container, dict):
        codec = container.get("codec", codec)
        container = container.get("format")
    if isinstance(codec, dict):
        codec = codec.get("codec")
    if container == "auto" and isinstance(codec, str):
        container = "webm" if codec == "av1" else "mp4"
    return "." + container if container in ("mp4", "mkv", "webm") else None


class AmbiguousSemanticField(ValueError):
    pass


class MissingSemanticField(ValueError):
    pass


class InvalidSemanticValue(ValueError):
    pass


@dataclass(frozen=True)
class FieldRef:
    node_key: str
    node_type: str
    input_name: str
    value: Any = None
    derived: bool = False
    evidence: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SemanticMapping:
    prompt: FieldRef
    negative_prompt: FieldRef | None
    seeds: tuple[FieldRef, ...]
    input_media: tuple[FieldRef, ...]
    fps: FieldRef | None
    frame_count: FieldRef | None
    width: FieldRef | None
    height: FieldRef | None
    outputs: tuple[FieldRef, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class WorkflowAdapter:
    family: str

    def discover(self, workflow: Workflow) -> SemanticMapping:
        if self.family == H3_FAMILY:
            return _discover_h3(workflow)
        prompt, negative = _conditioning_prompts(workflow)
        # Displayed seed slots keep their node identity when JSON keys are reordered.
        seeds = tuple(_field(workflow, node, "noise_seed")
                      for node in sorted(nodes_by_type(workflow, "RandomNoise"), key=lambda node: node.key))
        for seed in seeds:
            if not seed.derived and (type(seed.value) is not int or not 0 <= seed.value < 2**64):
                raise InvalidSemanticValue("noise_seed must be an unsigned 64-bit integer")
        media = media_fields(workflow, workflow.nodes)
        fps = _first_equivalent(
            [_field(workflow, node, "fps") for node in nodes_by_type(workflow, "CreateVideo")],
            "fps",
        )
        latents = nodes_by_type(workflow, "EmptyLTXVLatentVideo")
        latent = _one_or_none(latents, "latent dimensions")
        width = _field(workflow, latent, "width") if latent else None
        height = _field(workflow, latent, "height") if latent else None
        frames = _field(workflow, latent, "length") if latent else None
        outputs = tuple(
            _field(workflow, node, "filename_prefix")
            for node in nodes_by_type(workflow, "SaveVideo")
        )
        if not outputs:
            raise MissingSemanticField("outputs")
        for name, field in (("width", width), ("height", height), ("frame_count", frames), ("fps", fps)):
            if field and not field.derived and (error := parameter_error(self.family, name, field.value)):
                raise InvalidSemanticValue(error)
        return SemanticMapping(
            prompt, negative, seeds, media, fps, frames, width, height, outputs
        )


def adapter_for(family: str) -> WorkflowAdapter:
    if family not in FAMILIES:
        raise ValueError(f"unsupported workflow family: {family}")
    return WorkflowAdapter(family)


def _discover_h3(workflow: Workflow) -> SemanticMapping:
    if workflow.format is not WorkflowFormat.API:
        raise ValueError("MiniMax H3 requires a flattened ComfyUI API workflow")

    def source(node, port, node_type, slot=0):
        upstream = linked_source(workflow, node, port)
        if upstream is None or upstream.node_type != node_type or node.inputs[port][1] != slot:
            raise MissingSemanticField(f"{node.node_type}.{port} must use {node_type} output {slot}")
        return upstream

    save = _one_or_none(nodes_by_type(workflow, "SaveVideo"), "video output")
    if not save:
        raise MissingSemanticField("video output")
    video = source(save, "video", "CreateVideo")
    images = source(video, "images", "VAEDecode")
    audio = source(video, "audio", "VAEDecodeAudio")
    sampler = source(images, "samples", "SamplerCustomAdvanced")
    if source(audio, "samples", "SamplerCustomAdvanced") != sampler:
        raise ValueError("H3 audio and video must decode the same joint latent")
    noise = source(sampler, "noise", "RandomNoise")
    guider = source(sampler, "guider", "BasicGuider")
    scene = source(guider, "conditioning", "MiniMaxH3ImageToVideo")
    if source(sampler, "latent_image", "MiniMaxH3ImageToVideo", 1) != scene:
        raise ValueError("H3 conditioning and latent must come from the same scene")
    if (scene.inputs.get("first_frame") is not None or scene.inputs.get("last_frame") is not None
            or any(node.node_type in {"LoadImage", "LoadVideo", "VHS_LoadVideo"} for node in workflow.nodes)):
        raise ValueError("MiniMax H3 text-to-video does not take frame images")
    fps = _field(workflow, video, "fps")
    frames = _field(workflow, scene, "length")
    width = _field(workflow, scene, "width")
    height = _field(workflow, scene, "height")
    for name, field in (("fps", fps), ("frame_count", frames), ("width", width), ("height", height)):
        if error := parameter_error(H3_FAMILY, name, field.value):
            raise InvalidSemanticValue(error)
    prompt = _field(workflow, scene, "prompt")
    seed = _field(workflow, noise, "noise_seed")
    if not isinstance(prompt.value, str):
        raise InvalidSemanticValue("H3 requires a text prompt field")
    if type(seed.value) is not int or not 0 <= seed.value < 2**64:
        raise InvalidSemanticValue("H3 noise_seed must be an unsigned 64-bit integer")
    return SemanticMapping(prompt, None, (seed,), media_fields(workflow, workflow.nodes), fps, frames, width, height,
                           (_field(workflow, save, "filename_prefix"),))


def _conditioning_prompts(workflow: Workflow):
    active = output_ancestors(workflow, media_only=True)
    paired = {"LTXVConditioning", "LTXVCropGuides", "LTXAddVideoICLoRAGuide", "LTXVSetAudioRefTokens"}
    single = {"ConditioningSetArea", "ConditioningSetAreaPercentage", "ConditioningSetMask",
              "ConditioningSetTimestepRange"}

    def encoder(node, port):
        link = input_link(workflow, node, port)
        seen = set()
        while link is not None and link not in seen and link[0] in active:
            seen.add(link)
            node, slot = active[link[0]], link[1]
            if node.node_type == "CLIPTextEncode" and slot == 0:
                return node.key
            if node.node_type in paired and slot in (0, 1):
                port = ("positive", "negative")[slot]
            elif node.node_type in single and slot == 0:
                port = "conditioning"
            elif node.node_type == "Reroute" and slot == 0 and len(node.inputs) == 1:
                port = next(iter(node.inputs))
            else:
                break
            link = input_link(workflow, node, port)
        raise MissingSemanticField("cannot trace sampler conditioning to an active text prompt")

    positive, negative = set(), set()
    for node in active.values():
        if node.node_type == "SamplerCustomAdvanced":
            link = input_link(workflow, node, "guider")
            guider = active.get(link[0]) if link and link[1] == 0 else None
            if guider is None or guider.node_type not in {"BasicGuider", "CFGGuider", "LTXVDualCFGGuider", "MultimodalGuider"}:
                raise MissingSemanticField("cannot identify the active sampler guider")
            node = guider
        elif node.node_type not in {"KSampler", "KSamplerAdvanced", "SamplerCustom"}:
            continue
        if node.node_type == "BasicGuider":
            positive.add(encoder(node, "conditioning"))
        else:
            positive.add(encoder(node, "positive"))
            negative.add(encoder(node, "negative"))
    encoders = {node.key for node in active.values() if node.node_type == "CLIPTextEncode"}
    if encoders - positive - negative:
        raise AmbiguousSemanticField("active text prompt is outside the supported sampler conditioning paths")
    if not positive:
        raise MissingSemanticField("no active sampler positive prompt feeds the video output")
    if len(positive) != 1 or len(negative) > 1:
        raise AmbiguousSemanticField("ambiguous sampler conditioning: multiple active prompt encoders")
    prompt = _field(workflow, active[next(iter(positive))], "text")
    if not negative or negative == positive:
        return prompt, None
    return prompt, _field(workflow, active[next(iter(negative))], "text")


def _field(workflow: Workflow, node: NodeRef, input_name: str) -> FieldRef:
    source = linked_source(workflow, node, input_name)
    if source:
        if (source.node_type in {"PrimitiveInt", "PrimitiveFloat", "PrimitiveString"}
                and input_link(workflow, node, input_name)[1] == 0):
            value = (
                source.inputs.get("value")
                if workflow.format is WorkflowFormat.API
                else (source.widgets[0] if source.widgets else None)
            )
            return FieldRef(
                source.key,
                source.node_type,
                "value",
                value,
                False,
                f"linked editable scalar for {node.node_type}.{input_name}",
            )
        return FieldRef(
            node.key,
            node.node_type,
            input_name,
            derived=True,
            evidence=f"linked from {source.node_type}",
        )
    value = _direct_value(workflow, node, input_name)
    return FieldRef(node.key, node.node_type, input_name, value, False, "direct value")


def _direct_value(workflow: Workflow, node: NodeRef, input_name: str):
    if workflow.format is WorkflowFormat.API:
        value = node.inputs.get(input_name)
        return None if isinstance(value, list) else value
    indexes = {
        "CLIPTextEncode": {"text": 0},
        "RandomNoise": {"noise_seed": 0},
        "EmptyLTXVLatentVideo": {"width": 0, "height": 1, "length": 2},
        **{kind: {field: 0} for kind, field in MEDIA_INPUTS.items()},
        "CreateVideo": {"fps": 0},
        "SaveVideo": {"filename_prefix": 0},
    }
    index = indexes.get(node.node_type, {}).get(input_name)
    return node.widgets[index] if index is not None and index < len(node.widgets) else None


def media_fields(workflow, nodes):
    return tuple(_field(workflow, node, MEDIA_INPUTS[node.node_type])
                 for node in nodes if node.node_type in MEDIA_INPUTS)


def _one_or_none(nodes: list[NodeRef], field: str) -> NodeRef | None:
    if len(nodes) > 1:
        raise AmbiguousSemanticField(field)
    return nodes[0] if nodes else None


def _first_equivalent(fields: list[FieldRef], name: str) -> FieldRef | None:
    if not fields:
        return None
    signatures = {(field.value, field.derived, field.evidence) for field in fields}
    if len(signatures) > 1:
        raise AmbiguousSemanticField(name)
    return fields[0]


def parameter_error(family, name, value, *, min_dimension=64):
    h3 = family == H3_FAMILY
    if name in {"width", "height"}:
        if h3 and (type(value) is not int or not 32 <= value <= 16384 or value % 32):
            return f"H3 {name} must be a multiple of 32 from 32 to 16384"
        if not h3 and (type(value) is not int or not min_dimension <= value <= 16384 or value % 32):
            return f"{name} must be a multiple of 32 from {min_dimension} to 16384"
    elif name == "frame_count":
        if h3:
            if type(value) is not int or not 5 <= value <= 3600 or (value - 5) % 17:
                return "H3 frame_count must match 17k+5 from 5 to 3592"
        elif type(value) is not int or not 1 <= value <= 16377 or (value - 1) % 8:
            return "frame_count must match 8n+1 from 1 to 16377"
    elif name == "fps":
        if h3 and value != 24:
            return "H3 requires 24 fps"
        if type(value) not in (float, int) or not 1 <= value <= 120:
            return "fps must be from 1 to 120"
    return None
