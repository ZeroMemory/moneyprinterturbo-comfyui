"""Make the scene controls in supported ComfyUI exports editable by MPT."""

import math

from .workflow import load_workflow, output_ancestors


def check_ltx_upscaler_version(graph, model_version):
    if model_version not in ("2.3", "2.5", "h3"):
        return
    versions = {"ltx-2.3-spatial-upscaler-x2-1.1.safetensors": "2.3",
                "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors": "2.5"}
    for node in graph.values():
        if node["class_type"] == "LatentUpscaleModelLoader":
            inputs = node.get("inputs")
            name = inputs.get("model_name") if isinstance(inputs, dict) else None
            version = versions.get(name.replace("\\", "/").rsplit("/", 1)[-1]) if isinstance(name, str) else None
            if version and version != model_version:
                raise ValueError(f"This workflow uses the LTX-{version} upscaler. Select model version {version}.")


def _link(graph, value):
    return (isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)
            and type(value[1]) is int and value[0] in graph)


def _constant(graph, value, depth=0):
    if not _link(graph, value):
        return value
    if depth >= 16:
        raise ValueError("Scene input has a cyclic or unsupported control chain")
    key, slot = value
    node = graph[key]
    kind, inputs = node["class_type"], node["inputs"]
    if kind in {"PrimitiveInt", "PrimitiveFloat", "PrimitiveString", "PrimitiveStringMultiline", "PrimitiveBoolean"} and slot == 0:
        return _constant(graph, inputs["value"], depth + 1)
    if kind == "ResolutionSelector" and slot in (0, 1):
        ratio = inputs["aspect_ratio"]
        if not isinstance(ratio, str) or not ratio.strip():
            raise ValueError("ResolutionSelector needs a width:height aspect ratio")
        width, height = map(int, ratio.split()[0].split(":"))
        multiple = inputs["multiple"]
        scale = math.sqrt(inputs["megapixels"] * 1024**2 / (width * height))
        return round((width, height)[slot] * scale / multiple) * multiple
    if kind == "ComfyMathExpression" and slot in (0, 1):
        expression = inputs["expression"]
        if not isinstance(expression, str):
            raise ValueError("A scene calculation must be text")
        expression = "".join(expression.split())
        a = _constant(graph, inputs["values.a"], depth + 1)
        if type(a) not in (int, float) or not math.isfinite(a):
            raise ValueError("Scene input needs a finite number")
        if expression == "a":
            result = a
        elif expression == "a/2":
            result = a / 2
        elif expression == "a*b+1":
            b = _constant(graph, inputs["values.b"], depth + 1)
            if type(b) not in (int, float) or not math.isfinite(b):
                raise ValueError("Scene input needs a finite frame rate")
            result = a * b + 1
        elif expression == "max(5,round(a*24))+(5-(max(5,round(a*24))%17))%17":
            frames = max(5, round(a * 24))
            result = frames + (5 - frames) % 17
        else:
            raise ValueError("Unsupported calculation on a scene control")
        return int(result) if slot == 1 else float(result)
    raise ValueError(f"Scene control linked to {kind} cannot be edited by MPT")


def _control(graph, key, kind, value):
    while key in graph:
        key += "_"
    graph[key] = {"class_type": kind, "inputs": {"value": value}}
    return [key, 0]


def editable_scene_graph(graph, version):
    """Resolve the official templates' static branches and bind per-scene controls."""
    aliases = {}
    for key, node in graph.items():
        kind, inputs = node["class_type"], node["inputs"]
        if kind == "ComfySwitchNode":
            flag = _constant(graph, inputs["switch"])
            if type(flag) is not bool:
                raise ValueError("A template switch must have a fixed Boolean selection")
            on_true = inputs.get("on_true")
            # MPT supplies the finished scene prompt. Keep the template's direct text branch.
            prompt_switch = (_link(graph, on_true)
                             and graph[on_true[0]]["class_type"] == "TextGenerateLTX2Prompt")
            aliases[key] = inputs["on_true" if flag and not prompt_switch else "on_false"]
        elif kind == "LTXVImgToVideoInplace" and _constant(graph, inputs["bypass"]) is True:
            aliases[key] = inputs["latent"]

    for node in graph.values():
        for field, value in node["inputs"].items():
            seen = set()
            while _link(graph, value) and value[1] == 0 and value[0] in aliases:
                if value[0] in seen:
                    raise ValueError("Cyclic workflow switch")
                seen.add(value[0])
                value = aliases[value[0]]
            node["inputs"][field] = value

    live = output_ancestors(load_workflow(graph))
    if not live:
        raise ValueError("The exported graph must include a SaveVideo output")
    graph = {key: node for key, node in graph.items() if key in live}
    scene_kind = "MiniMaxH3ImageToVideo" if version == "h3" else "EmptyLTXVLatentVideo"
    for node in list(graph.values()):
        kind, inputs = node["class_type"], node["inputs"]
        if kind == scene_kind:
            for field in ("width", "height", "length"):
                value = inputs[field]
                if _link(graph, value):
                    number = _constant(graph, value)
                    if type(number) not in (int, float) or not math.isfinite(number):
                        raise ValueError(f"Scene {field} must resolve to a finite number")
                    if field == "length":
                        origin, step = (5, 17) if version == "h3" else (1, 8)
                        inputs[field] = origin + max(0, (int(number) - origin) // step) * step
                    else:
                        inputs[field] = int(number) // 32 * 32
            if version != "h3":
                audio = [item["inputs"] for item in graph.values()
                         if item["class_type"] == "LTXVEmptyLatentAudio"]
                if len(audio) > 1:
                    raise ValueError("The workflow has more than one scene audio length")
                if audio:
                    frames = _control(graph, "mpt_import_frames", "PrimitiveInt", inputs["length"])
                    inputs["length"] = audio[0]["frames_number"] = frames
        elif kind == "CreateVideo" and _link(graph, inputs["fps"]):
            rate = graph[inputs["fps"][0]]
            expression = rate["inputs"].get("expression")
            if (version != "h3" and rate["class_type"] == "ComfyMathExpression"
                    and isinstance(expression, str) and expression.strip() == "a"):
                # Keep the float conditioning rate and integer audio rate on the same control.
                fps = _constant(graph, inputs["fps"])
                control = _control(graph, "mpt_import_fps", "PrimitiveFloat", fps)
                rate["inputs"]["values.a"] = inputs["fps"] = control
            elif rate["class_type"] not in {"PrimitiveFloat", "PrimitiveInt"}:
                inputs["fps"] = _constant(graph, inputs["fps"])
        elif kind == "CLIPTextEncode" and _link(graph, inputs["text"]):
            inputs["text"] = _constant(graph, inputs["text"])
    live = output_ancestors(load_workflow(graph))
    graph = {key: node for key, node in graph.items() if key in live}
    # ComfyUI replaces ':' in flattened IDs when it opens an API graph on the canvas.
    taken = {key for key in graph if ":" not in key}
    renamed = {}
    for key in graph:
        if ":" in key:
            candidate = key.replace(":", "_")
            while candidate in taken:
                candidate += "_"
            taken.add(candidate)
            renamed[key] = candidate
    for node in graph.values():
        for field, value in node["inputs"].items():
            if _link(graph, value) and value[0] in renamed:
                node["inputs"][field] = [renamed[value[0]], value[1]]
    return {renamed.get(key, key): node for key, node in graph.items()}
