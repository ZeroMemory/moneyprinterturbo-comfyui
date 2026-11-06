"""Fail-closed comparison of displayed plan values and API-prompt semantics."""

import math
import os
from pathlib import Path, PureWindowsPath

from .adapters import FieldRef, _field, adapter_for, media_fields
from .profiles import load_profile_snapshot
from .workflow import (MEDIA_INPUTS, MODEL_INPUTS, MODEL_SUFFIXES, input_link, linked_source, load_workflow,
                       ltx_tiled_decoder_extents, model_files, output_ancestors)
from .workflow_import import _constant


def primary_video_node(plan: dict) -> str:
    saves = [key for key, node in plan["api_prompt"].items() if node["class_type"] == "SaveVideo"]
    if "primary_output_node" in plan["variant"]:
        key = plan["variant"]["primary_output_node"]
        if key in saves:
            return key
        raise ValueError("variant.primary_output_node must be the ID of a SaveVideo node")
    if len(saves) == 1:
        return saves[0]
    raise ValueError("Select the final SaveVideo node in variant.primary_output_node for this batch")


def validate_plan_semantics(plan: object, *, allow_empty_prompt: bool = False, object_info=None, launch_profile=None) -> tuple[str, ...]:
    """Return concrete mismatches without mutating the plan."""
    if not isinstance(plan, dict):
        return ("semantic plan must be a dictionary",)
    variant = plan.get("variant")
    api_prompt = plan.get("api_prompt")
    if not isinstance(variant, dict) or not isinstance(
        variant.get("adapter_family"), str
    ):
        return ("semantic mapping requires variant adapter_family",)
    if not isinstance(api_prompt, dict):
        return ("semantic mapping requires an API prompt dictionary",)
    try:
        workflow = load_workflow(api_prompt)
        mapping = adapter_for(variant["adapter_family"]).discover(workflow)
        if "primary_output_node" in variant:
            primary_video_node(plan)
    except Exception as exc:
        return (f"semantic mapping unavailable: {exc}",)

    errors: list[str] = []
    for decoder, size in ltx_tiled_decoder_extents(workflow):
        axes = ("horizontal_tiles", "vertical_tiles")
        try:
            overlap = _field(workflow, decoder, "overlap").value
            counts = [_field(workflow, decoder, axis).value for axis in axes]
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"decoder {decoder.key} tile controls cannot be resolved: {exc}")
            continue
        if type(overlap) is not int or not 1 <= overlap <= 8:
            errors.append(f"decoder {decoder.key} tile overlap must resolve to an integer from 1 to 8")
            continue
        for axis, extent, count in zip(axes, size, counts):
            if type(count) is not int or not 1 <= count <= 6:
                errors.append(f"decoder {decoder.key} {axis} must resolve to an integer from 1 to 6")
            elif count > 1 and extent < overlap:
                errors.append(f"decoder {decoder.key} {axis}: tile overlap exceeds the latent extent; "
                              "prepare explicit dimensions or adjust the decoder tiles before approval")
    dependencies = _dependency_nodes(workflow, object_info)
    errors.extend(_weight_errors(workflow, dependencies, plan, object_info, launch_profile))
    media = media_fields(workflow, dependencies)
    snapshot = plan.get("profile")
    remote = (launch_profile.remote_server if launch_profile else
              isinstance(snapshot, dict) and snapshot.get("remote_server") is True)
    requirements = plan.get("requirements", {})
    if remote and (media or plan.get("input_media") or any(
            isinstance(requirements, dict) and requirements.get(key)
            for key in ("required_files", "resolved_models", "resolved_loras")
    )):
        errors.append("Remote ComfyUI supports text-to-video without input media or local file dependencies")
    if media:
        try:
            profile = launch_profile or load_profile_snapshot(plan.get("profile", {}))
            input_root = (profile.application_directory / "input").resolve()
            covered = {
                Path(value).expanduser().resolve() for value in plan.get("input_media", [])
            }
            for field in media:
                value = _field_value(api_prompt, field)
                if field.derived or not isinstance(value, str) or not value:
                    errors.append(f"semantic input media cannot be resolved: {field.node_key}")
                    continue
                if value.endswith((" [temp]", " [output]")):
                    errors.append(f"semantic input media must use the ComfyUI input directory: {value}")
                    continue
                path = (input_root / value.removesuffix(" [input]")).resolve()
                if not path.is_relative_to(input_root) or not path.is_file():
                    errors.append(f"semantic input media is unavailable in ComfyUI input: {value}")
                elif path not in covered:
                    errors.append(f"semantic input media is not approval-covered: {path}")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"semantic input media cannot be resolved: {exc}")
    prompt = plan.get("prompt")
    if not isinstance(prompt, str):
        errors.append("prompt must be a string")
    elif not allow_empty_prompt and not prompt.strip():
        errors.append("a nonblank scene prompt is required")
    if mapping.prompt.derived and not allow_empty_prompt:
        errors.append("the actual scene prompt cannot be verified from a derived field")
    _compare(mapping.prompt, prompt, api_prompt, "prompt", errors)
    if mapping.negative_prompt is not None:
        _compare(
            mapping.negative_prompt,
            plan.get("negative_prompt"),
            api_prompt,
            "negative_prompt",
            errors,
        )
    elif plan.get("negative_prompt"):
        errors.append("semantic negative_prompt mapping is missing")

    parameters = plan.get("parameters")
    if not isinstance(parameters, dict):
        errors.append("semantic parameters must be a dictionary")
        parameters = {}
    if any(field.derived for field in mapping.seeds):
        errors.append("an executable seed cannot be verified from a derived field")
    expected_seeds = [
        parameters[key]
        for key in sorted(parameters)
        if key == "seed" or key.endswith("_seed")
    ]
    if any(type(value) is not int or not 0 <= value < 2**64 for value in expected_seeds):
        errors.append("displayed seeds must be unsigned 64-bit integers")
    actual_seeds = [
        _field_value(api_prompt, field)
        for field in mapping.seeds
        if not field.derived
    ]
    if expected_seeds != actual_seeds:
        errors.append(
            f"semantic seeds mismatch: displayed {expected_seeds!r}, API prompt {actual_seeds!r}"
        )

    for name, field in (
        ("fps", mapping.fps),
        ("frame_count", mapping.frame_count),
        ("width", mapping.width),
        ("height", mapping.height),
    ):
        if field is None:
            errors.append(f"semantic {name} mapping is missing")
            continue
        _compare(field, parameters.get(name), api_prompt, name, errors)

    fps = _field_value(api_prompt, mapping.fps) if mapping.fps and not mapping.fps.derived else None
    if type(fps) in {int, float} and math.isfinite(fps):
        for node in output_ancestors(workflow, media_only=True).values():
            if node.node_type not in {"LTXVConditioning", "LTXVEmptyLatentAudio"}:
                continue
            try:
                rate = _constant(api_prompt, node.inputs["frame_rate"])
            except (KeyError, TypeError, ValueError, ArithmeticError):
                rate = None
            # The native audio node uses the integer output of the shared rate control.
            expected_rate = int(fps) if node.node_type == "LTXVEmptyLatentAudio" else fps
            if type(rate) not in {int, float} or not math.isclose(rate, expected_rate):
                errors.append(f"{node.node_type}.frame_rate must match video FPS ({expected_rate:g})")

    outputs = plan.get("outputs")
    try:
        profile = launch_profile or load_profile_snapshot(plan.get("profile", {}))
        root = profile.output_directory.expanduser().resolve()
        values = [("output", value) for value in outputs] if isinstance(outputs, list) else []
        values.extend(("output prefix", _field_value(api_prompt, field))
                      for field in mapping.outputs if not field.derived)
        for label, value in values:
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{label} must be a non-empty path")
                continue
            if remote:
                relative = PureWindowsPath(value)
                # The final wildcard belongs to the approved output pattern, not the path.
                if (relative.anchor or ".." in relative.parts or not relative.parts
                        or value.startswith("~")):
                    errors.append(f"{label} must be relative to the remote output directory: {value}")
                continue
            path = Path(value).expanduser()
            resolved = (root / path).resolve()
            if (PureWindowsPath(value).anchor and not path.is_absolute()
                    or resolved == root or not resolved.is_relative_to(root)):
                errors.append(f"{label} is outside the ComfyUI output directory: {value}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"semantic output path cannot be resolved: {exc}")
    output_paths = (
        [PureWindowsPath(os.path.normcase(value)) for value in outputs if isinstance(value, str)]
        if isinstance(outputs, list)
        else []
    )
    for field in mapping.outputs:
        if field.derived:
            errors.append("an executable output prefix cannot be verified from a derived field")
            continue
        prefix = _field_value(api_prompt, field)
        prefix_path = PureWindowsPath(os.path.normcase(os.path.normpath(prefix))) if isinstance(prefix, str) else None
        prefix_parts = (prefix_path.parts or (prefix,)) if prefix_path is not None else ()
        if prefix_path is None or not any(
            (path.root or len(path.parts) == len(prefix_parts))
            and path.parts[-len(prefix_parts):-1] == prefix_parts[:-1]
            and path.name.startswith(prefix_path.name or prefix)
            for path in output_paths
        ):
            errors.append(
                f"semantic output mismatch: API prefix {prefix!r} is not covered by displayed outputs"
            )
    return tuple(errors)


def _dependency_nodes(workflow, object_info):
    # Any custom node can be a ComfyUI output. Only skip known file sources with no consumers.
    referenced = {link[0] for node in workflow.nodes for name in node.inputs
                  if (link := input_link(workflow, node, name)) is not None}
    source_types = MODEL_INPUTS.keys() | MEDIA_INPUTS.keys()
    result = []
    for node in workflow.nodes:
        info = object_info.get(node.node_type) if isinstance(object_info, dict) else None
        if (node.key in referenced or node.node_type not in source_types
                or isinstance(info, dict) and info.get("output_node")):
            result.append(node)
    return result


def _server_choices(info, field):
    inputs = info.get("input", {}) if isinstance(info, dict) else {}
    for section in ("required", "optional"):
        fields = inputs.get(section) if isinstance(inputs, dict) else None
        spec = fields.get(field) if isinstance(fields, dict) else None
        if isinstance(spec, (list, tuple)) and spec:
            if isinstance(spec[0], (list, tuple)):
                return spec[0]
            if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
                options = spec[1].get("options")
                if isinstance(options, (list, tuple)):
                    return options
    return None


def _weight_selectors(workflow, nodes):
    selectors = {field for fields in MODEL_INPUTS.values() for field in fields}
    for node in nodes:
        fields = MODEL_INPUTS.get(node.node_type)
        if fields is None:
            if node.node_type in MEDIA_INPUTS:
                continue
            kind = node.node_type.lower()
            if "loader" in kind or selectors.intersection(node.inputs) or (
                    "load" in kind and any(isinstance(v, str) and v.lower().endswith(MODEL_SUFFIXES)
                                           for v in node.inputs.values())):
                raise ValueError(f"unsupported active weight loader: {node.node_type}")
            for key, value in node.inputs.items():
                if key in {"image", "audio", "video", "latent"}:
                    source = linked_source(workflow, node, key)
                    if not (isinstance(value, str) or "load" in kind
                            or source and source.node_type == "PrimitiveString"):
                        continue
                elif key not in {"file", "filename", "path", "directory"} and not key.endswith(("_path", "_file", "_filename")):
                    continue
                raise ValueError(f"unsupported input media loader: {node.node_type}.{key}")
            continue
        for name, category in fields.items():
            value = node.inputs.get(name)
            source = linked_source(workflow, node, name)
            if source and source.node_type == "PrimitiveString" and value[1] == 0:
                value = source.inputs.get("value")
            yield node, name, value, category


def _weight_errors(workflow, active, plan, object_info, launch_profile):
    try:
        selections = list(_weight_selectors(workflow, active))
    except ValueError as exc:
        return [str(exc)]
    if not selections:
        return []
    requirements = plan.get("requirements", {})
    if not isinstance(requirements, dict):
        return ["requirements must be a dictionary"]
    try:
        profile = launch_profile or load_profile_snapshot(plan.get("profile", {}))
        roots = () if profile.remote_server else (profile.application_directory / "models", *profile.model_roots)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [f"invalid weight profile: {exc}"]
    errors = []
    for node, name, value, category in selections:
        label = f"{node.node_type}.{name}"
        if not isinstance(value, str) or not value.strip():
            errors.append(f"cannot resolve the selected weight for {label}")
            continue
        group = "loras" if "lora" in node.node_type.lower() or name == "lora_name" else "models"
        declared = requirements.get("required_" + group, [])
        resolved = requirements.get("resolved_" + group, {})
        path = resolved.get(value) if isinstance(resolved, dict) else None
        selected = PureWindowsPath(value)
        covered = (isinstance(declared, list) and value in declared
                   and selected.parts and not selected.anchor and ".." not in selected.parts)
        if not profile.remote_server and covered:
            covered = (isinstance(path, str)
                       and model_files(roots, category, selected) == {Path(path).expanduser().resolve()})
        if not covered:
            errors.append(f"selected weight is not approval-covered: {label} = {value!r}. "
                          "Check the required and resolved weights and register any external --model-root.")
        if object_info is not None:
            info = object_info.get(node.node_type) if isinstance(object_info, dict) else None
            choices = _server_choices(info, name)
            if choices is None:
                errors.append(f"ComfyUI did not report selectable weights for {label}")
            elif value not in choices:
                errors.append(f"weight unavailable on ComfyUI: {label} = {value!r}")
    return errors


def _compare(
    field: FieldRef,
    expected: object,
    api_prompt: dict,
    name: str,
    errors: list[str],
) -> None:
    if field.derived:
        source = field.evidence.removeprefix("linked from ")
        if (not isinstance(expected, dict) or expected.get("derived") is not True
                or expected.get("value") is not None
                or expected.get("evidence") not in (field.evidence, source)):
            errors.append(f"derived semantic {name} must be displayed as derived from {source}")
        return
    actual = _field_value(api_prompt, field)
    if actual != expected:
        errors.append(
            f"semantic {name} mismatch: displayed {expected!r}, API prompt {actual!r}"
        )


def _field_value(api_prompt: dict, field: FieldRef):
    node = api_prompt.get(str(field.node_key))
    if not isinstance(node, dict):
        return None
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        return None
    return inputs.get(field.input_name)
