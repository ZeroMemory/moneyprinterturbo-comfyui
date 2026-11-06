"""Approval-bound ComfyUI scene editing through its existing userdata API."""

import copy
from glob import escape, glob
import json
import os
from pathlib import Path, PureWindowsPath
import re
from urllib.parse import quote

from .adapters import adapter_for, video_suffix
from .api import ComfyClient, ComfyHttpError
from .artifacts import create_json_exclusive, ignore_private_directory, sha256_file
from .batch_planning import CachedReadOnlyClient, plan_request_batch
from .planning import REQUEST_FIELDS, verify_approved_plan
from .preparation import prepare_request
from .profiles import load_profile_snapshot
from .semantics import _dependency_nodes, _weight_selectors, primary_video_node
from .workflow import load_workflow, model_files, nodes_by_type


EXTENSION = "/extensions/mpt_scene_editor/editor.js"
MAX_EDITOR_SEED = 2**53 - 1


def export_editor(manifest, *, client_factory=ComfyClient):
    scenes = _load_bound_scenes(manifest)
    for _, _, plan in scenes:
        if any(":" in node.key for node in nodes_by_type(load_workflow(plan["api_prompt"]), "SaveVideo")):
            raise ValueError("Manual editing requires SaveVideo IDs without ':'; ComfyUI renames flattened subgraph IDs during import")
    client = _editor_client(scenes, client_factory)
    payload = {"schema_version": 1, "session": manifest["session"], "revision": manifest["revision"],
               "scenes": [{"index": index, "base_approval_hash": plan["approval_hash"],
                           "title": f"Scene {index}", "api_prompt": plan["api_prompt"]}
                          for index, _, plan in scenes]}
    _recheck(scenes)
    client.transport.request("POST", _userdata_path(manifest, "manifest.json") + "?overwrite=true", payload)
    return {"ok": True, "session": manifest["session"], "revision": manifest["revision"],
            "editor_url": client.endpoint + "/?mpt_edit=" + manifest["session"], "submissions": 0}


def import_editor(manifest, output_dir, *, client_factory=ComfyClient, progress=None):
    """Create a complete new private batch, preserving the current batch on failure."""
    scenes = _load_bound_scenes(manifest)
    destination = Path(output_dir).expanduser().absolute()
    if os.path.lexists(destination):
        raise FileExistsError(f"editor output already exists: {destination}")
    destination = destination.resolve()
    _recheck(scenes, fresh=True)
    client = _editor_client(scenes, client_factory)
    cached = CachedReadOnlyClient(client)
    object_info = cached.object_info()
    if not isinstance(object_info, dict):
        raise ValueError("ComfyUI object_info must be an object")
    edited = []
    for index, _, original in scenes:
        try:
            saved = client.transport.request("GET", _userdata_path(manifest, f"scene-{index:03d}.json"))
        except ComfyHttpError as exc:
            if exc.status != 404:
                raise
            graph = original["api_prompt"]
        else:
            graph = _saved_graph(saved, manifest, index, original["approval_hash"])
        edited.append((index, original, _rebuild_request(original, graph, object_info)))
    _recheck(scenes, fresh=True)

    created_files, created_dirs = [], []
    try:
        destination.mkdir(parents=True, exist_ok=False)
        created_dirs.append(destination)
        ignore_private_directory(destination)
        created_files.append(destination / ".gitignore")
        requests = []
        for index, original, request in edited:
            source = destination / f"scene-{index:03d}.workflow.json"
            create_json_exclusive(source, request["api_prompt"])
            created_files.append(source)
            request["variant"]["source_path"] = str(source)
            if "source_sha256" in request["variant"]:
                request["variant"]["source_sha256"] = sha256_file(source)
            path = destination / f"scene-{index:03d}.request.json"
            create_json_exclusive(path, request)
            created_files.append(path)
            requests.append({"scene_index": index, "request_file": str(path)})

        # Edited scenes can differ, so reuse the batch planner per scene with one cached server snapshot.
        summaries = []
        for scene in requests:
            plan_dir = destination / f"scene-{scene['scene_index']:03d}.plans"
            result = plan_request_batch({"schema_version": 1, "scenes": [scene]}, plan_dir,
                                        client_factory=lambda _: cached, progress=progress)
            created_dirs.append(plan_dir)
            created_files.extend(plan_dir.iterdir())
            summary = result["scenes"][0]
            _, original, request = next(item for item in edited if item[0] == scene["scene_index"])
            summary.update(base_approval_hash=original["approval_hash"], prompt=request["prompt"],
                           negative_prompt=request["negative_prompt"], parameters=request["parameters"],
                           request_file=scene["request_file"], source_path=request["variant"]["source_path"])
            summaries.append(summary)
        _recheck(scenes, fresh=True)
        for summary in summaries:
            plan = _load_json(summary["plan_file"])
            verify_approved_plan(plan, summary["approval_hash"])
            _fresh_outputs(plan)
        return {"ok": True, "session": manifest["session"], "revision": manifest["revision"],
                "scenes": summaries, "submissions": 0}
    except BaseException:
        # Only remove our known files; never recursively follow an untrusted path.
        if destination.resolve() == destination:
            for path in reversed(created_files):
                if path.parent.resolve().is_relative_to(destination) and not path.is_dir():
                    path.unlink(missing_ok=True)
            for path in reversed(created_dirs):
                if path.resolve().is_relative_to(destination) and path.is_dir():
                    path.rmdir()
        raise


def _load_bound_scenes(manifest):
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("editor manifest schema_version must be 1")
    for name, length in (("session", 32), ("revision", 64)):
        if not isinstance(manifest.get(name), str) or not re.fullmatch(f"[0-9a-f]{{{length}}}", manifest[name]):
            raise ValueError(f"editor manifest {name} must contain {length} lowercase hexadecimal characters")
    entries = manifest.get("scenes")
    if not isinstance(entries, list) or not entries:
        raise ValueError("editor manifest scenes must be a non-empty list")
    scenes = []
    for entry in entries:
        if not isinstance(entry, dict) or type(entry.get("scene_index")) is not int or entry["scene_index"] <= 0:
            raise ValueError("editor scene_index must be a positive integer")
        value = entry.get("plan_file")
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError("editor plan_file must be an absolute path")
        path = Path(value).resolve()
        plan = _load_json(path)
        verify_approved_plan(plan, plan.get("approval_hash"))
        primary_video_node(plan)
        scenes.append((entry["scene_index"], path, plan))
    if len({index for index, _, _ in scenes}) != len(scenes):
        raise ValueError("editor scene_index values must be unique")
    if any(plan["profile"] != scenes[0][2]["profile"] for _, _, plan in scenes[1:]):
        raise ValueError("editor scenes must share the same launch profile and endpoint")
    return sorted(scenes)


def _editor_client(scenes, client_factory):
    profile = scenes[0][2]["profile"]
    if profile.get("remote_server") is True:
        raise ValueError("Manual scene editing is not available for remote ComfyUI")
    endpoint = profile["endpoint"]
    ComfyClient(endpoint)  # Validate loopback even for an injected client factory.
    client = client_factory(endpoint)
    users = client.transport.request("GET", "/users")
    if not isinstance(users, dict) or users.get("storage") != "server" or "users" in users:
        raise ValueError("Manual scene editing requires single-user ComfyUI; multi-user storage is unsupported")
    extensions = client.transport.request("GET", "/extensions")
    if not isinstance(extensions, list) or EXTENSION not in extensions:
        raise ValueError("MPT scene editor extension is unavailable; install the addon and restart ComfyUI, then reopen the editor")
    return client


def _userdata_path(manifest, filename):
    return "/userdata/" + quote(f"mpt-scenes/{manifest['session']}/{filename}", safe="")


def _saved_graph(saved, manifest, index, base_hash):
    allowed = {"schema_version", "session", "revision", "index", "base_approval_hash", "api_prompt", "workflow"}
    if not isinstance(saved, dict) or set(saved) - allowed:
        raise ValueError(f"scene {index} saved envelope contains unsupported fields")
    expected = {"schema_version": 1, "session": manifest["session"], "revision": manifest["revision"],
                "index": index, "base_approval_hash": base_hash}
    if any(saved.get(key) != value or type(saved.get(key)) is not type(value) for key, value in expected.items()):
        raise ValueError(f"scene {index} saved envelope is stale or belongs to another editing session")
    graph = saved.get("api_prompt")
    if not isinstance(graph, dict) or not graph:
        raise ValueError(f"scene {index} saved API prompt must be a non-empty object")
    return graph


def _rebuild_request(original, graph, object_info):
    request = copy.deepcopy({key: original[key] for key in REQUEST_FIELDS})
    request["api_prompt"] = copy.deepcopy(graph)
    workflow = load_workflow(request["api_prompt"])
    old_workflow = load_workflow(original["api_prompt"])
    old_mapping = adapter_for(original["variant"]["adapter_family"]).discover(old_workflow)
    old_seeds = {(field.node_key, field.node_type, field.input_name): field.value for field in old_mapping.seeds}
    for field in old_mapping.seeds:
        target = request["api_prompt"].get(field.node_key, {})
        if field.derived or field.value <= MAX_EDITOR_SEED or target.get("class_type") != field.node_type:
            continue
        value = target.get("inputs", {}).get(field.input_name)
        # Browser JSON can round UINT64_MAX up to a decimal just above 2**64.
        if type(value) is int and value.bit_length() <= 65 and float(value) == float(field.value):
            target["inputs"][field.input_name] = field.value
    prefixes = {node.key: field.value for node, field in
                zip(nodes_by_type(old_workflow, "SaveVideo"), old_mapping.outputs, strict=True)}
    if {node.key for node in nodes_by_type(workflow, "SaveVideo")} != set(prefixes):
        raise ValueError("Edited workflow must preserve the original SaveVideo nodes and output ownership")
    for key, prefix in prefixes.items():
        request["api_prompt"][key]["inputs"]["filename_prefix"] = prefix
    workflow = load_workflow(request["api_prompt"])
    mapping = adapter_for(request["variant"]["adapter_family"]).discover(workflow)
    fields = (mapping.prompt, *mapping.seeds, mapping.fps, mapping.frame_count, mapping.width, mapping.height,
              *mapping.outputs, *mapping.input_media)
    if any(field is None or field.derived for field in fields) or mapping.negative_prompt and mapping.negative_prompt.derived:
        raise ValueError("Edited workflow uses derived or unsupported scene fields; use direct editable values")
    if any(seed.value > MAX_EDITOR_SEED and old_seeds.get((seed.node_key, seed.node_type, seed.input_name)) != seed.value
           for seed in mapping.seeds):
        raise ValueError(f"A new seed in the ComfyUI editor must be from 0 to {MAX_EDITOR_SEED:,}")
    for node in workflow.nodes:
        info = object_info.get(node.node_type)
        if ((isinstance(info, dict) and info.get("output_node")) or node.node_type.startswith("Save")) and node.node_type != "SaveVideo":
            raise ValueError(f"Edited workflow has an unsupported output node: {node.node_type}")
    request["prompt"] = mapping.prompt.value
    request["negative_prompt"] = mapping.negative_prompt.value if mapping.negative_prompt else ""
    parameters = {key: value for key, value in request["parameters"].items()
                  if key != "seed" and not key.endswith("_seed")}
    parameters.update({name: getattr(mapping, name).value for name in ("fps", "frame_count", "width", "height")})
    parameters.update({f"stage_{index}_seed": field.value for index, field in enumerate(mapping.seeds, 1)})
    request["parameters"] = parameters
    profile = load_profile_snapshot(original["profile"])
    roots = (profile.application_directory / "models", *profile.model_roots)
    resolved = {"models": {}, "loras": {}}
    for node, name, value, category in _weight_selectors(workflow, _dependency_nodes(workflow, object_info)):
        selected = _safe_relative(value, "weight")
        group = "loras" if "lora" in node.node_type.lower() or name == "lora_name" else "models"
        matches = model_files(roots, category, selected)
        previous = original["requirements"].get("resolved_" + group, {}).get(value)
        path = next(iter(matches)) if len(matches) == 1 else None
        if path is None or (str(path) != previous and not any(path.is_relative_to(root.resolve()) for root in roots)):
            raise ValueError(f"Edited weight is unavailable, ambiguous, or outside registered model roots: {node.node_type}.{name} = {value!r}")
        path = str(path)
        if value in resolved[group] and resolved[group][value] != path:
            raise ValueError(f"Edited weight name identifies different files: {value}")
        resolved[group][value] = path
    old_weights = {value for group in ("resolved_models", "resolved_loras")
                   for value in original["requirements"].get(group, {}).values()}
    request["requirements"].update(required_custom_nodes=sorted({node.node_type for node in workflow.nodes}),
                                   required_files=[path for path in original["requirements"].get("required_files", []) if path not in old_weights])
    for group, values in resolved.items():
        request["requirements"]["required_" + group] = sorted(values)
        request["requirements"]["resolved_" + group] = values
    input_root = (profile.application_directory / "input").resolve()
    media = []
    for field in mapping.input_media:
        value = field.value
        if isinstance(value, str) and value.endswith((" [temp]", " [output]")):
            raise ValueError("Edited media must come from the ComfyUI input directory")
        selected = _safe_relative(value.removesuffix(" [input]") if isinstance(value, str) else value, "input media")
        path = input_root.joinpath(*selected.parts).resolve()
        if not path.is_relative_to(input_root) or not path.is_file():
            raise ValueError(f"Edited input media is unavailable in ComfyUI input: {value}")
        media.append(str(path))
    request["input_media"] = sorted(set(media))
    # Reuse output suffix coverage and semantic validation without trusting browser paths.
    if len(old_mapping.outputs) == 1:
        request = prepare_request(request, {"output_prefix": old_mapping.outputs[0].value})
    else:
        outputs = set()
        for node, field in zip(nodes_by_type(workflow, "SaveVideo"), mapping.outputs, strict=True):
            suffix = video_suffix(node.inputs)
            if suffix is None:
                raise ValueError(f"Edited SaveVideo {node.key} format cannot be verified; select an explicit MP4, MKV, or WebM format")
            outputs.add(os.path.normpath(field.value) + "*" + suffix)
        request["outputs"] = sorted(outputs)
        request = prepare_request(request, {})
    return request


def _safe_relative(value, label):
    if not isinstance(value, str) or not value.strip() or any(char in value for char in ':*?"<>|\0'):
        raise ValueError(f"Edited {label} must be a safe category-relative filename")
    selected = PureWindowsPath(value)
    if selected.anchor or not selected.parts or ".." in selected.parts:
        raise ValueError(f"Edited {label} path escapes its registered directory")
    return selected


def _recheck(scenes, *, fresh=False):
    for index, path, plan in scenes:
        current = _load_json(path)
        if current != plan:
            raise ValueError(f"scene {index} current plan changed during editing; reopen the editor")
        verify_approved_plan(current, plan["approval_hash"])
        if fresh:
            _fresh_outputs(current)


def _fresh_outputs(plan):
    root = load_profile_snapshot(plan["profile"]).output_directory.resolve()
    for pattern in plan["outputs"]:
        path = Path(pattern).expanduser()
        relative = (root / path).resolve().relative_to(root)
        if glob(str(Path(escape(str(root))) / relative)):
            raise FileExistsError("Scene outputs already exist; create a fresh scene batch before importing edits")


def _load_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("editor JSON document must be an object")
    return value
