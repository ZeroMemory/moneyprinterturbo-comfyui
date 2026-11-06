"""Read-only preflight and immutable approval-covered generation plans."""

import json
from pathlib import Path
import secrets
from urllib.parse import urlsplit

from .adapters import H3_FAMILY, parameter_error
from .api import ComfyHttpError, queue_counts, remaining_timeout
from .artifacts import sha256_file
from .canonical import approval_hash, canonical_json, verify_approval
from .preparation import prepare_request
from .profiles import load_profile_snapshot, validate_profile
from .semantics import primary_video_node, validate_plan_semantics
from .workflow import WorkflowFormat, detect_format
from .workflow_import import check_ltx_upscaler_version


REQUEST_FIELDS = (
    "prompt",
    "negative_prompt",
    "variant",
    "profile",
    "input_media",
    "parameters",
    "requirements",
    "warnings",
    "outputs",
    "api_prompt",
    "client_id",
)
MATERIAL_FIELDS = (*REQUEST_FIELDS, "approval_nonce")
APPROVAL_FORMAT = 3
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class ApprovalError(PermissionError):
    pass


class PreflightError(RuntimeError):
    pass


def model_reuse_warnings(stats) -> list[str]:
    system = stats.get("system") if isinstance(stats, dict) else None
    argv = system.get("argv") if isinstance(system, dict) else None
    if isinstance(argv, list):
        flags = [flag for flag in ("--cache-none", "--disable-smart-memory") if flag in argv]
        if flags:
            return [f"Restart ComfyUI without {' and '.join(flags)} to improve model reuse between scenes."]
    return []


def preflight_checks(system_stats, object_info, queue):
    return {
        "system_stats": {"available": True} if isinstance(system_stats, dict) else None,
        "object_info": {"node_count": len(object_info)} if isinstance(object_info, dict) else None,
        "queue": queue_counts(queue),
    }


def unavailable_nodes(object_info, required):
    return [f"required node unavailable: {name}" for name in sorted(required)
            if not isinstance(object_info.get(name), dict)]


def preflight(request: dict, client, *, allow_empty_prompt: bool = False, deadline=None) -> dict:
    """Perform static checks and, when given a client, read-only server checks."""
    errors: list[str] = []
    if not isinstance(request, dict):
        raise TypeError("preflight request must be a dictionary")
    if not isinstance(request.get("negative_prompt"), str):
        errors.append("negative_prompt must be a string")

    api_prompt = request.get("api_prompt")
    if not isinstance(api_prompt, dict) or not api_prompt:
        errors.append("api_prompt must be a non-empty dictionary")
    else:
        try:
            if detect_format(api_prompt) is not WorkflowFormat.API:
                errors.append("api_prompt is not in ComfyUI API format")
        except ValueError as exc:
            errors.append(f"invalid api_prompt: {exc}")

    variant = request.get("variant")
    if not isinstance(variant, dict) or not isinstance(
        variant.get("adapter_family"), str
    ):
        errors.append("variant must include adapter_family")
    elif "source_path" in variant:
        _check_paths([variant["source_path"]], "variant source", errors)

    launch = _check_profile(request.get("profile"), errors)
    remote = launch is not None and launch.remote_server
    _check_parameters(request.get("parameters"), variant.get("adapter_family") if isinstance(variant, dict) else None, errors)
    _check_paths(request.get("input_media"), "input media", errors)

    requirements = request.get("requirements")
    if not isinstance(requirements, dict):
        errors.append("requirements must be a dictionary")
        requirements = {}
    _check_paths(requirements.get("required_files", []), "required file", errors)

    outputs = request.get("outputs")
    if not isinstance(outputs, list) or not outputs or not all(
        isinstance(item, str) and item for item in outputs
    ):
        errors.append("outputs must be a non-empty list of strings")

    stats = object_info = queue = None
    if client is not None:
        stats = read_server_object(client.system_stats, "system_stats", errors, deadline=deadline)
        object_info = read_server_object(client.object_info, "object_info", errors, deadline=deadline)
        queue = read_server_object(client.queue, "queue", errors, deadline=deadline)

    required_nodes = requirements.get("required_custom_nodes", [])
    if not isinstance(required_nodes, list) or not all(
        isinstance(item, str) and item for item in required_nodes
    ):
        errors.append("required_custom_nodes must be a list of strings")
    elif isinstance(object_info, dict):
        api_nodes = {
            node.get("class_type")
            for node in api_prompt.values()
            if isinstance(node, dict) and isinstance(node.get("class_type"), str)
        } if isinstance(api_prompt, dict) else set()
        errors.extend(unavailable_nodes(object_info, set(required_nodes) | api_nodes))

    for label in ("required_models", "required_loras"):
        values = requirements.get(label, [])
        if not isinstance(values, list) or not all(
            isinstance(item, str) and item for item in values
        ):
            errors.append(f"{label} must be a list of strings")
    if not remote:
        _check_resolved_dependencies(
            requirements.get("required_models", []), requirements.get("resolved_models", {}), "model", errors,
        )
        _check_resolved_dependencies(
            requirements.get("required_loras", []), requirements.get("resolved_loras", {}), "lora", errors,
        )
    errors.extend(validate_plan_semantics(request, allow_empty_prompt=allow_empty_prompt, object_info=object_info))

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": model_reuse_warnings(stats),
        "checks": preflight_checks(stats, object_info, queue),
    }


def validate_request_template(request: dict) -> None:
    """Check that MPT can use this request without contacting its server."""
    missing = [field for field in REQUEST_FIELDS if field not in request]
    if missing:
        raise ValueError("missing request fields: " + ", ".join(missing))
    report = preflight(request, None, allow_empty_prompt=True)
    if report["errors"]:
        raise ValueError("; ".join(report["errors"]))
    variant = request["variant"]
    if variant.get("model_version") is not None and not isinstance(variant["model_version"], str):
        raise ValueError("variant.model_version must be a string")
    if type(variant.get("supports_negative_prompt", False)) is not bool:
        raise ValueError("variant.supports_negative_prompt must be a boolean")
    parameters = request["parameters"]
    seeds = sum(key == "seed" or key.endswith("_seed") for key in parameters)
    if type(parameters["fps"]) not in (int, float) or not seeds:
        raise ValueError("the request must expose numeric FPS and at least one seed")
    version = variant.get("model_version")
    if version == "h3" and (parameters["fps"] != 24 or seeds != 1):
        raise ValueError("MiniMax H3 requires 24 FPS and one seed")
    if (variant["adapter_family"] == H3_FAMILY) != (version == "h3"):
        raise ValueError("MiniMax H3 requests must use the H3 adapter and model version together")
    check_ltx_upscaler_version(request["api_prompt"], version)
    profile = load_profile_snapshot(request["profile"])
    if not profile.remote_server:
        errors = validate_profile(profile)
        if errors:
            raise ValueError("; ".join(errors))
    if not isinstance(request["client_id"], str) or not request["client_id"].strip():
        raise ValueError("client_id must be a non-empty string")
    if not isinstance(request["warnings"], list) or not all(isinstance(value, str) for value in request["warnings"]):
        raise ValueError("warnings must be a list of strings")
    primary_video_node(request)
    _normalize_material({**request, "approval_nonce": "validation"})
    prepare_request(request, {"prompt": "A scene", "frame_count": parameters["frame_count"]})


def create_plan(request: dict) -> dict:
    """Bind a scene to its graph, inputs and selected local weights."""
    if not isinstance(request, dict):
        raise TypeError("generation request must be a dictionary")
    missing = [field for field in REQUEST_FIELDS if field not in request]
    if missing:
        raise ValueError("generation request is missing: " + ", ".join(missing))
    semantic_errors = validate_plan_semantics(request)
    if semantic_errors:
        raise ValueError("semantic mismatch: " + "; ".join(semantic_errors))
    material = _normalize_material(
        {
            **{field: request[field] for field in REQUEST_FIELDS},
            "approval_nonce": secrets.token_hex(16),
        }
    )
    covered = _build_covered(material, sha256_file)
    plan = {"schema_version": 1, **material, "covered": covered}
    plan["approval_hash"] = approval_hash(covered)
    return _json_clone(plan)


def run_approved_plan(
    plan: dict, expected_hash: str, client, *, consume_approval=None
):
    """Recheck approval coverage and read-only preflight before one submission."""
    verify_approved_plan(plan, expected_hash)
    if not callable(consume_approval):
        raise TypeError("consume_approval must be callable")

    report = preflight(plan, client)
    if not report["ok"]:
        raise PreflightError("; ".join(report["errors"]))

    verify_approved_plan(plan, expected_hash)
    consume_approval(expected_hash)
    return client.submit(plan["api_prompt"], client_id=plan["client_id"])


def verify_approved_plan(plan: dict, expected_hash: str, *, hash_file=sha256_file) -> None:
    """Verify approval and immutable dependencies without contacting ComfyUI."""
    if not isinstance(plan, dict) or not isinstance(plan.get("covered"), dict):
        raise ApprovalError("plan has no approval-covered snapshot")
    try:
        canonical_json(plan)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ApprovalError("plan contains invalid JSON values") from exc
    if plan["covered"].get("approval_format") != APPROVAL_FORMAT:
        raise ApprovalError("Approval format changed. Create a new plan and approve it before rendering.")
    current = _rebuild_covered(plan, hash_file)
    _verify_current(plan, current, expected_hash)
    errors = validate_plan_semantics(plan)
    if errors:
        raise PreflightError("; ".join(errors))


def _verify_current(plan: dict, current: dict, expected_hash: str) -> None:
    if canonical_json(current) != canonical_json(plan["covered"]):
        raise ApprovalError("approved plan drift detected")
    if plan.get("approval_hash") != expected_hash or not verify_approval(
        current, expected_hash
    ):
        raise ApprovalError("approval hash does not match current plan")


def _rebuild_covered(plan: dict, hash_file=sha256_file) -> dict:
    try:
        return _build_covered(_material_from_plan(plan), hash_file)
    except ApprovalError:
        raise
    except Exception as exc:
        raise ApprovalError(f"approved plan drift detected: {exc}") from exc


def _material_from_plan(plan: dict) -> dict:
    missing = [field for field in MATERIAL_FIELDS if field not in plan]
    if missing:
        raise ApprovalError("plan is missing covered fields: " + ", ".join(missing))
    return {field: plan[field] for field in MATERIAL_FIELDS}


def _normalize_material(request: dict) -> dict:
    material = _json_clone({field: request[field] for field in MATERIAL_FIELDS})
    for field in ("variant", "profile"):
        snapshot = material[field]
        if not isinstance(snapshot, dict):
            raise TypeError(f"{field} must be a dictionary")
        snapshot["source_path"] = str(_existing_file(snapshot.get("source_path"), field))
    if not isinstance(material["input_media"], list):
        raise TypeError("input_media must be a list")
    material["input_media"] = [
        str(_existing_file(path, "input media")) for path in material["input_media"]
    ]
    requirements = material["requirements"]
    if not isinstance(requirements, dict):
        raise TypeError("requirements must be a dictionary")
    required_files = requirements.get("required_files", [])
    if not isinstance(required_files, list):
        raise TypeError("required_files must be a list")
    requirements["required_files"] = [
        str(_existing_file(path, "required file")) for path in required_files
    ]
    for key, label in (("resolved_models", "model"), ("resolved_loras", "lora")):
        resolved = requirements.setdefault(key, {})
        if not isinstance(resolved, dict):
            raise TypeError(f"{key} must be a dictionary")
        normalized = {}
        for name, path in resolved.items():
            if not isinstance(name, str) or not name:
                raise TypeError(f"{key} names must be non-empty strings")
            normalized[name] = str(_existing_file(path, label))
        requirements[key] = normalized
    return material


def _build_covered(material: dict, hash_file) -> dict:
    covered = _json_clone(material)
    covered["approval_format"] = APPROVAL_FORMAT
    covered["variant"] = _snapshot_file_dict(material["variant"], "variant", hash_file)
    covered["profile"] = _snapshot_file_dict(material["profile"], "profile", hash_file)
    covered["input_media"] = [
        _file_record(_existing_file(path, "input media"), hash_file) for path in material["input_media"]
    ]
    weight_files = {}
    for key, label in (("resolved_models", "model"), ("resolved_loras", "lora")):
        records = {}
        for name, value in material["requirements"].get(key, {}).items():
            path = _existing_file(value, label)
            stat = path.stat()
            record = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            records[name] = record
            weight_files[path] = record
        covered["requirements"][key] = records
    # A workflow can also list its weights as generic file dependencies.
    required = (
        _existing_file(value, "required file")
        for value in material["requirements"].get("required_files", [])
    )
    covered["requirements"]["required_files"] = [
        weight_files[path] if path in weight_files else _file_record(path, hash_file)
        for path in required
    ]
    return covered


def _snapshot_file_dict(snapshot: dict, label: str, hash_file) -> dict:
    result = _json_clone(snapshot)
    result["source_path"] = str(_existing_file(result.get("source_path"), label))
    if label == "profile":
        load_profile_snapshot(result)
    result["content_sha256"] = hash_file(result["source_path"])
    return result


def _file_record(path: Path, hash_file) -> dict:
    return {"path": str(path), "content_sha256": hash_file(str(path))}


def _existing_file(value, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path must be a non-empty string")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{label} is unavailable: {path}")
    return path


def _json_clone(value):
    return json.loads(canonical_json(value))


def _check_profile(profile, errors: list[str]):
    if not isinstance(profile, dict):
        errors.append("profile must be a dictionary")
        return
    endpoint = profile.get("endpoint")
    if not isinstance(endpoint, str):
        errors.append("profile endpoint must be a string")
        return
    launch = None
    if "source_path" in profile:
        _check_paths([profile["source_path"]], "profile source", errors)
        try:
            launch = load_profile_snapshot(profile)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"invalid profile snapshot: {exc}")
    if profile.get("remote_server") is True or (launch is not None and launch.remote_server):
        if launch is None or not launch.remote_server or profile.get("remote_server") is not True:
            errors.append("Remote ComfyUI requires its configured profile")
        else:
            errors.extend(validate_profile(launch))
        return launch
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in LOOPBACK_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        errors.append("profile endpoint must be an uncredentialed loopback root")
    return launch


def _check_parameters(parameters, family, errors: list[str]) -> None:
    if not isinstance(parameters, dict):
        errors.append("parameters must be a dictionary")
        return
    for name in ("width", "height", "frame_count", "fps"):
        value = parameters.get(name)
        if _is_derived(value, name, errors):
            continue
        if error := parameter_error(family, name, value):
            errors.append(error)


def _is_derived(value, name: str, errors: list[str]) -> bool:
    if not isinstance(value, dict) or value.get("derived") is not True:
        return False
    evidence = value.get("evidence")
    if not isinstance(evidence, str) or not evidence:
        errors.append(f"derived {name} must include evidence")
    return True


def _check_paths(values, label: str, errors: list[str]) -> None:
    if not isinstance(values, list):
        errors.append(f"{label} paths must be a list")
        return
    for value in values:
        try:
            _existing_file(value, label)
        except ValueError as exc:
            errors.append(str(exc))


def _check_resolved_dependencies(
    required, resolved, label: str, errors: list[str]
) -> None:
    if not isinstance(required, list):
        return
    if not isinstance(resolved, dict):
        errors.append(f"resolved_{label}s must be a dictionary")
        return
    for name in required:
        path = resolved.get(name)
        if path is None:
            errors.append(f"required {label} unresolved: {name}")
            continue
        try:
            _existing_file(path, label)
        except ValueError as exc:
            errors.append(str(exc))


def read_server_object(operation, label: str, errors: list[str], *, deadline=None):
    timeout = remaining_timeout(deadline)
    try:
        value = operation(**({"timeout": timeout} if timeout is not None else {}))
    except Exception as exc:
        reason = f"HTTP {exc.status}" if isinstance(exc, ComfyHttpError) else type(exc).__name__
        errors.append(f"{label} unavailable ({reason})")
        return None
    if not isinstance(value, dict):
        errors.append(f"{label}: expected a JSON object from ComfyUI")
        return None
    if label == "system_stats" and not isinstance(value.get("system"), dict):
        errors.append("system_stats: expected a system object from ComfyUI")
        return None
    if label == "queue" and not all(isinstance(value.get(key), list) for key in ("queue_running", "queue_pending")):
        errors.append("queue: expected queue_running and queue_pending lists from ComfyUI")
        return None
    return value
