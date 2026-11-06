"""Atomic multi-request planning in one operator process."""

import json
import os
from pathlib import Path
import secrets
import shutil

from .artifacts import create_json_exclusive, ignore_private_directory
from .api import client_for_snapshot
from .planning import create_plan, preflight
from .semantics import primary_video_node


class CachedReadOnlyClient:
    def __init__(self, client):
        self._client = client
        self._responses = {}

    def system_stats(self, *, timeout=None):
        return self._read_once("system_stats", timeout)

    def object_info(self, *, timeout=None):
        return self._read_once("object_info", timeout)

    def queue(self, *, timeout=None):
        return self._read_once("queue", timeout)

    def _read_once(self, method_name, timeout):
        if method_name not in self._responses:
            self._responses[method_name] = getattr(self._client, method_name)(
                **({"timeout": timeout} if timeout is not None else {}))
        return self._responses[method_name]


def plan_request_batch(manifest, output_dir, *, client_factory, progress=None):
    """Preflight and create every scene plan without submitting a prompt."""
    scenes = _validate_manifest(manifest)
    requests = [
        (scene, _load_request(scene["request_file"])) for scene in scenes
    ]
    _validate_compatible_requests(requests)
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"batch plan output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{secrets.token_hex(8)}.tmp"
    clients = {}
    summaries = []

    try:
        staging.mkdir()
        ignore_private_directory(staging)
        for current, (scene, request) in enumerate(requests, start=1):
            endpoint = request["profile"]["endpoint"]
            client = clients.get(endpoint)
            if client is None:
                client = CachedReadOnlyClient(client_for_snapshot(request["profile"], client_factory))
                clients[endpoint] = client

            report = preflight(request, client)
            if not report["ok"]:
                raise RuntimeError(
                    f"scene {scene['scene_index']} preflight failed: "
                    + "; ".join(report["errors"])
                )
            primary_video_node(request)

            warnings = request.get("warnings")
            if not isinstance(warnings, list):
                raise ValueError(f"scene {scene['scene_index']} warnings must be a list")
            warnings.extend(w for w in report["warnings"] if w not in warnings)
            plan = create_plan(request)
            filename = f"{scene['scene_index']:03d}.json"
            create_json_exclusive(staging / filename, plan)
            summaries.append(
                {
                    "scene_index": scene["scene_index"],
                    "plan_file": str(destination / filename),
                    "approval_hash": plan["approval_hash"],
                }
            )
            if callable(progress):
                progress(
                    {
                        "event": "scene_planned",
                        "scene_index": scene["scene_index"],
                        "current": current,
                        "total": len(requests),
                    }
                )

        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    return {"ok": True, "scenes": summaries, "submissions": 0}


def _validate_manifest(manifest):
    if not isinstance(manifest, dict):
        raise ValueError("batch manifest must be a dictionary")
    if manifest.get("schema_version") != 1:
        raise ValueError("batch manifest schema_version must be 1")

    scenes = manifest.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("batch manifest scenes must be a non-empty list")
    if not all(isinstance(scene, dict) for scene in scenes):
        raise ValueError("batch manifest scenes must be dictionaries")

    indexes = [scene.get("scene_index") for scene in scenes]
    if not all(type(index) is int and index > 0 for index in indexes):
        raise ValueError("scene_index must be a positive integer")
    if len(indexes) != len(set(indexes)):
        raise ValueError("scene_index values must be unique")

    validated = []
    for scene in scenes:
        request_file = scene.get("request_file")
        if not isinstance(request_file, str) or not request_file:
            raise ValueError("request_file must be a non-empty string")
        path = Path(request_file).expanduser()
        if not path.is_absolute():
            raise ValueError(f"request_file must be absolute: {request_file}")
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"request_file is unavailable: {path}")
        validated.append(
            {"scene_index": scene["scene_index"], "request_file": str(path)}
        )

    return sorted(validated, key=lambda scene: scene["scene_index"])


def _load_request(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        request = json.load(stream)
    if not isinstance(request, dict):
        raise ValueError(f"generation request must be a dictionary: {path}")
    return request


def _validate_compatible_requests(requests):
    baseline = _compatibility_identity(requests[0][1])
    for scene, request in requests[1:]:
        identity = _compatibility_identity(request)
        if identity["endpoint"] != baseline["endpoint"]:
            raise ValueError(
                "batch requests must use a compatible profile endpoint: "
                f"scene {scene['scene_index']}"
            )
        if identity["dependencies"] != baseline["dependencies"]:
            raise ValueError(
                "batch requests must use compatible shared dependencies: "
                f"scene {scene['scene_index']}"
            )


def _compatibility_identity(request):
    profile = request.get("profile")
    variant = request.get("variant")
    requirements = request.get("requirements")
    profile = profile if isinstance(profile, dict) else {}
    variant = variant if isinstance(variant, dict) else {}
    requirements = requirements if isinstance(requirements, dict) else {}
    resolved_models = requirements.get("resolved_models", {})
    resolved_loras = requirements.get("resolved_loras", {})
    required_files = requirements.get("required_files", [])
    return {
        "endpoint": profile.get("endpoint"),
        "dependencies": {
            "adapter_family": variant.get("adapter_family"),
            "workflow": _path_identity(variant.get("source_path")),
            "profile": _path_identity(profile.get("source_path")),
            "required_files": (
                tuple(sorted(_path_identity(path) for path in required_files))
                if isinstance(required_files, list)
                else required_files
            ),
            "resolved_models": _mapping_identity(resolved_models),
            "resolved_loras": _mapping_identity(resolved_loras),
            "remote_weights": (
                (requirements.get("required_models"), requirements.get("required_loras"))
                if profile.get("remote_server") is True else None
            ),
        },
    }


def _mapping_identity(value):
    if not isinstance(value, dict):
        return value
    return tuple(
        sorted((name, _path_identity(path)) for name, path in value.items())
    )


def _path_identity(value):
    if not isinstance(value, str) or not value:
        return value
    return os.path.normcase(str(Path(value).expanduser().resolve()))
