"""ComfyUI UI-workflow and API-prompt parsing without semantic node IDs."""

from dataclasses import dataclass
from enum import StrEnum
import json
from pathlib import Path, PureWindowsPath
from typing import Any


MODEL_SUFFIXES = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf")
MEDIA_INPUTS = {
    "LoadAudio": "audio", "LoadImage": "image", "LoadImageMask": "image",
    "LoadVideo": "file", "VHS_LoadVideo": "video",
}
MODEL_INPUTS = {
    "UNETLoader": {"unet_name": "diffusion_models"},
    "VAELoader": {"vae_name": "vae"},
    "CLIPLoader": {"clip_name": "text_encoders"},
    "CheckpointLoaderSimple": {"ckpt_name": "checkpoints"},
    "unCLIPCheckpointLoader": {"ckpt_name": "checkpoints"},
    "CheckpointLoader": {"config_name": "configs", "ckpt_name": "checkpoints"},
    "LTXVAudioVAELoader": {"ckpt_name": "checkpoints"},
    "LTXAVTextEncoderLoader": {"ckpt_name": "checkpoints", "text_encoder": "text_encoders"},
    "LoraLoaderModelOnly": {"lora_name": "loras"},
    "LoraLoader": {"lora_name": "loras"},
    "LTXICLoRALoaderModelOnly": {"lora_name": "loras"},
    "DualCLIPLoader": {"clip_name1": "text_encoders", "clip_name2": "text_encoders"},
    "TripleCLIPLoader": {f"clip_name{i}": "text_encoders" for i in range(1, 4)},
    "QuadrupleCLIPLoader": {f"clip_name{i}": "text_encoders" for i in range(1, 5)},
    "ControlNetLoader": {"control_net_name": "controlnet"},
    "DiffControlNetLoader": {"control_net_name": "controlnet"},
    "CLIPVisionLoader": {"clip_name": "clip_vision"},
    "StyleModelLoader": {"style_model_name": "style_models"},
    "GLIGENLoader": {"gligen_name": "gligen"},
    "UpscaleModelLoader": {"model_name": "upscale_models"},
    "LatentUpscaleModelLoader": {"model_name": "latent_upscale_models"},
}


def model_directories(roots, category):
    # ComfyUI also exposes legacy model folders under these loader categories.
    categories = {"diffusion_models": ("unet", "diffusion_models"),
                  "text_encoders": ("text_encoders", "clip")}.get(category, (category,))
    return (root / name for root in roots for name in categories)


def model_files(roots, category, filename):
    parts = PureWindowsPath(filename).parts
    return {path.resolve() for directory in model_directories(roots, category)
            if (path := directory.joinpath(*parts)).is_file()}


class WorkflowFormat(StrEnum):
    UI = "ui"
    API = "api"


@dataclass(frozen=True)
class NodeRef:
    key: str
    node_type: str
    title: str
    inputs: dict[str, Any]
    widgets: tuple[Any, ...]


@dataclass(frozen=True)
class Workflow:
    format: WorkflowFormat
    nodes: tuple[NodeRef, ...]
    links: dict[Any, tuple[str, int]]


def detect_format(data: dict) -> WorkflowFormat:
    if not isinstance(data, dict):
        raise ValueError("workflow JSON must be an object")
    if isinstance(data.get("nodes"), list) and isinstance(data.get("links"), list):
        return WorkflowFormat.UI
    if data and all(
        isinstance(node, dict) and isinstance(node.get("class_type"), str)
        for node in data.values()
    ):
        return WorkflowFormat.API
    raise ValueError("unknown ComfyUI workflow format")


def load_workflow(source: str | Path | dict) -> Workflow:
    if isinstance(source, dict):
        data = source
    else:
        data = json.loads(Path(source).read_text(encoding="utf-8"))
    workflow_format = detect_format(data)
    return _parse_ui(data) if workflow_format is WorkflowFormat.UI else _parse_api(data)


def nodes_by_type(workflow: Workflow, node_type: str) -> list[NodeRef]:
    return [node for node in workflow.nodes if node.node_type == node_type]


def titled_nodes(workflow: Workflow, title: str) -> list[NodeRef]:
    return [node for node in workflow.nodes if node.title == title]


def linked_source(
    workflow: Workflow, node: NodeRef, input_name: str
) -> NodeRef | None:
    link = input_link(workflow, node, input_name)
    if link is None:
        return None
    return next((item for item in workflow.nodes if item.key == link[0]), None)


def input_link(workflow: Workflow, node: NodeRef, input_name: str):
    value = node.inputs.get(input_name)
    if workflow.format is WorkflowFormat.UI and value in workflow.links:
        return workflow.links[value]
    elif (
        workflow.format is WorkflowFormat.API
        and isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], (str, int)) and type(value[1]) is int
    ):
        return str(value[0]), value[1]
    return None


def output_ancestors(workflow: Workflow, *, media_only=False) -> dict[str, NodeRef]:
    nodes = {node.key: node for node in workflow.nodes}
    pending = [node for node in workflow.nodes if node.node_type == "SaveVideo"]
    active = {}
    while pending:
        node = pending.pop()
        if node.key in active:
            continue
        active[node.key] = node
        inputs = node.inputs
        if media_only and node.node_type == "SaveVideo":
            inputs = ("video",)
        elif media_only and node.node_type == "CreateVideo":
            inputs = ("images", "audio")
        for name in inputs:
            link = input_link(workflow, node, name)
            if link is not None and link[0] in nodes:
                pending.append(nodes[link[0]])
    return active


def ltx_tiled_decoder_extents(workflow: Workflow):
    """Resolve decoder paths to known native outputs; leave custom geometry unknown."""
    if workflow.format is not WorkflowFormat.API:
        return
    nodes = {node.key: node for node in workflow.nodes}

    def source(node, name, kind):
        link = input_link(workflow, node, name)
        upstream = nodes.get(link[0]) if link and link[1] == 0 else None
        return upstream if upstream and upstream.node_type == kind else None

    def scalar(node, name):
        link = input_link(workflow, node, name)
        if link is None:
            return node.inputs.get(name)
        upstream = nodes.get(link[0])
        if link[1] == 0 and upstream and upstream.node_type in {"PrimitiveInt", "PrimitiveString"}:
            return upstream.inputs.get("value")
        return None

    def extent(link, seen):
        if link is None or link in seen or not (node := nodes.get(link[0])):
            return None
        kind, slot = node.node_type, link[1]
        if kind == "EmptyLTXVLatentVideo" and slot == 0:
            width, height = scalar(node, "width"), scalar(node, "height")
            if type(width) is int and type(height) is int and width >= 32 and height >= 32:
                return width // 32, height // 32
            return None
        scale = 1
        if kind == "SamplerCustomAdvanced" and slot in (0, 1):
            port = "latent_image"
        elif kind == "LTXVConcatAVLatent" and slot == 0:
            port = "video_latent"
        elif kind == "LTXVSeparateAVLatent" and slot == 0:
            port = "av_latent"
        elif kind == "LTXVLatentUpsampler" and slot == 0:
            loader = source(node, "upscale_model", "LatentUpscaleModelLoader")
            model = scalar(loader, "model_name") if loader else None
            if not isinstance(model, str) or model.replace("\\", "/").split("/")[-1] not in {
                "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
                "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
            }:
                return None
            port, scale = "samples", 2
        else:
            return None
        size = extent(input_link(workflow, node, port), seen | {link})
        return tuple(side * scale for side in size) if size else None

    found = set()
    for output in workflow.nodes:
        if output.node_type in {"SaveVideo", "VideoTrim", "VideoCrop"}:
            video = source(output, "video", "CreateVideo")
            decoder = source(video, "images", "LTXVTiledVAEDecode") if video else None
        elif output.node_type in {"SaveImage", "PreviewImage", "SaveAnimatedPNG", "SaveAnimatedWEBP",
                                  "SaveImageAdvanced", "SaveWEBM"}:
            decoder = source(output, "images", "LTXVTiledVAEDecode")
        else:
            continue
        if decoder and decoder.key not in found:
            found.add(decoder.key)
            size = extent(input_link(workflow, decoder, "latents"), set())
            if size:
                yield decoder, size


def _parse_ui(data: dict) -> Workflow:
    nodes = []
    for raw in data["nodes"]:
        inputs = {
            item["name"]: item.get("link")
            for item in raw.get("inputs", [])
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        nodes.append(
            NodeRef(
                key=str(raw.get("id")),
                node_type=str(raw.get("type", "")),
                title=str(raw.get("title", "")),
                inputs=inputs,
                widgets=tuple(raw.get("widgets_values") or ()),
            )
        )
    links = {
        link[0]: (str(link[1]), int(link[2]))
        for link in data["links"]
        if isinstance(link, list) and len(link) >= 3
    }
    return Workflow(WorkflowFormat.UI, tuple(nodes), links)


def _parse_api(data: dict) -> Workflow:
    nodes = tuple(
        NodeRef(
            key=str(key),
            node_type=raw["class_type"],
            title=str(meta.get("title", "")) if isinstance(meta := raw.get("_meta"), dict) else "",
            inputs=dict(raw.get("inputs", {})),
            widgets=(),
        )
        for key, raw in data.items()
    )
    return Workflow(WorkflowFormat.API, nodes, {})
