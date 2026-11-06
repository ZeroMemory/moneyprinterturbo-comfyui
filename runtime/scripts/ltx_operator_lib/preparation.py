"""Prepare executable requests by changing registered direct semantic fields."""

import copy
import os
from pathlib import Path

from .adapters import FieldRef, adapter_for, parameter_error, video_suffix
from .semantics import validate_plan_semantics
from .workflow import load_workflow, ltx_tiled_decoder_extents


ALLOWED_OVERRIDES = frozenset(
    {"prompt", "negative_prompt", "seeds", "frame_count", "width", "height", "output_prefix", "client_id"}
)


def prepare_request(template: dict, overrides: dict) -> dict:
    if not isinstance(template, dict) or not isinstance(overrides, dict):
        raise TypeError("template request and overrides must be dictionaries")
    unknown = sorted(set(overrides) - ALLOWED_OVERRIDES)
    if unknown:
        raise ValueError("unsupported request overrides: " + ", ".join(unknown))

    request = copy.deepcopy(template)
    variant = request.get("variant")
    api_prompt = request.get("api_prompt")
    if not isinstance(variant, dict) or not isinstance(
        variant.get("adapter_family"), str
    ):
        raise ValueError("template variant must include adapter_family")
    if not isinstance(api_prompt, dict):
        raise ValueError("template request must include an API prompt")
    mapping = adapter_for(variant["adapter_family"]).discover(
        load_workflow(api_prompt)
    )

    if overrides.get("negative_prompt"):
        if "supports_negative_prompt" in variant:
            accepts_negative = variant["supports_negative_prompt"]
            if type(accepts_negative) is not bool:
                raise ValueError("supports_negative_prompt must be a boolean")
        else:
            accepts_negative = not _uses_unit_guidance(api_prompt)
        if not accepts_negative:
            raise ValueError("This workflow does not accept a separate negative prompt")
    for name in ("prompt", "negative_prompt"):
        if name in overrides:
            if not isinstance(overrides[name], str):
                raise ValueError(f"{name} must be text")
            field = getattr(mapping, name)
            if not (name == "negative_prompt" and field is None and overrides[name] == ""):
                _set_direct(api_prompt, field, overrides[name], name)
            request[name] = overrides[name]
    if "seeds" in overrides:
        _set_seeds(request, mapping.seeds, overrides["seeds"])
    for name in ("frame_count", "width", "height"):
        if name in overrides:
            value = overrides[name]
            if error := parameter_error(variant["adapter_family"], name, value, min_dimension=64):
                raise ValueError(error)
            _set_direct(api_prompt, getattr(mapping, name), value, name)
            _parameters(request)[name] = value
    if "width" in overrides or "height" in overrides:
        for decoder, size in ltx_tiled_decoder_extents(load_workflow(api_prompt)):
            overlap = decoder.inputs.get("overlap")
            if type(overlap) is int and overlap > 0:
                for axis, extent in zip(("horizontal_tiles", "vertical_tiles"), size):
                    count = decoder.inputs.get(axis)
                    if type(count) is int and count == 2 and extent < overlap:
                        api_prompt[decoder.key]["inputs"][axis] = 1
    if "output_prefix" in overrides:
        prefix = overrides["output_prefix"]
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("output_prefix must be a non-empty string")
        _set_output_prefix(request, mapping.outputs, prefix)
    if "client_id" in overrides:
        client_id = overrides["client_id"]
        if not isinstance(client_id, str) or not client_id:
            raise ValueError("client_id must be a non-empty string")
        request["client_id"] = client_id

    errors = validate_plan_semantics(request)
    if errors:
        raise ValueError("; ".join(errors))
    return request


def _set_output_prefix(request, fields, prefix):
    graph = request["api_prompt"]
    saves = (node for node in graph.values() if node["class_type"] == "SaveVideo")
    suffixes = set()
    for field, node in zip(fields, saves, strict=True):
        _set_direct(graph, field, prefix, "output_prefix")
        suffix = video_suffix(node["inputs"])
        if suffix:
            suffixes.add("*" + suffix)
        else:
            # A linked format is resolved by ComfyUI. Retain the template's matching filename patterns.
            old = Path(os.path.normpath(field.value)).name
            for path in request.get("outputs", []):
                name = Path(path).name
                if os.path.normcase(name).startswith(os.path.normcase(old)):
                    suffixes.add(name[len(old):])
    request["outputs"] = [os.path.normpath(prefix) + suffix for suffix in sorted(suffixes)]


def _set_seeds(request: dict, fields: tuple[FieldRef, ...], values: object) -> None:
    if not isinstance(values, list) or len(values) != len(fields):
        raise ValueError(f"seed override count must be {len(fields)}")
    parameters = _parameters(request)
    seed_keys = sorted(
        key for key in parameters if key == "seed" or key.endswith("_seed")
    )
    if len(seed_keys) != len(fields):
        raise ValueError("template displayed seed count does not match API prompt")
    for field, key, value in zip(fields, seed_keys, values, strict=True):
        if not isinstance(value, int) or value < 0:
            raise ValueError("seed overrides must be non-negative integers")
        _set_direct(request["api_prompt"], field, value, "seed")
        parameters[key] = value


def _set_direct(api_prompt: dict, field: FieldRef | None, value: object, name: str):
    if field is None:
        raise ValueError(f"{name} mapping is missing")
    if field.derived:
        raise ValueError(f"{name} is derived and cannot be overridden")
    api_prompt[str(field.node_key)]["inputs"][field.input_name] = value


def _parameters(request: dict) -> dict:
    parameters = request.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("template parameters must be a dictionary")
    return parameters


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
