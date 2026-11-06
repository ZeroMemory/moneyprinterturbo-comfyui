"""Thin argparse composition for the deterministic LTX operator."""

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import sys


from .adapters import adapter_for
from .analysis import analyze_outputs
from .api import CancellationResult, ComfyClient, client_for_snapshot, history_status, remote_client_options
from .artifacts import create_json_exclusive, replace_json_atomic, sha256_file
from .batch_planning import plan_request_batch
from .duration import predict_durations
from .editor import export_editor, import_editor
from .inventory import inspect_inventory
from .planning import (create_plan, model_reuse_warnings, preflight, preflight_checks,
                       read_server_object, unavailable_nodes, verify_approved_plan)
from .preparation import prepare_request
from .profiles import load_profile, validate_profile
from .registry import SAFE_NAME, register_workflow
from .runs import execute_run, execute_runs, load_signed_run_record
from .state import OperatorState
from .workflow import load_workflow


PROJECT_ROOT = Path(__file__).parents[2]
DEFAULT_PROFILE = Path("ltx-settings/profile.json")
DEFAULT_CUSTOM_REGISTRY = PROJECT_ROOT / "workflows" / "custom"
FRONTEND_ONLY_NODE_TYPES = frozenset({"Note", "MarkdownNote"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ltx_operator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--profile", required=True)
    _json_flag(inventory)

    flight = subparsers.add_parser("preflight")
    source = flight.add_mutually_exclusive_group(required=True)
    source.add_argument("--request", help="Check a render-ready scene request")
    source.add_argument("--template-request", help="Check setup before a scene prompt has been supplied")
    source.add_argument("--variant")
    flight.add_argument("--profile", default=str(DEFAULT_PROFILE), help="Profile for --variant (default: ltx-settings/profile.json)")
    flight.add_argument("--registry-root", default=str(DEFAULT_CUSTOM_REGISTRY))
    _json_flag(flight)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--request", required=True)
    plan.add_argument("--output")
    _json_flag(plan)

    verify = subparsers.add_parser("verify-plan")
    verify.add_argument("--plan", required=True)
    verify.add_argument("--approval-hash", required=True)
    _json_flag(verify)

    batch = subparsers.add_parser("plan-batch")
    batch.add_argument("--manifest", required=True)
    batch.add_argument("--output-dir", required=True)
    _json_flag(batch)

    predict = subparsers.add_parser("predict-durations", help="Predict scene lengths without rendering")
    predict.add_argument("--manifest", required=True)
    _json_flag(predict)

    editor_export = subparsers.add_parser("export-editor", help="Open approval-bound scenes for manual editing")
    editor_export.add_argument("--manifest", required=True)
    _json_flag(editor_export)

    editor_import = subparsers.add_parser("import-editor", help="Validate saved edits and create a new scene batch")
    editor_import.add_argument("--manifest", required=True)
    editor_import.add_argument("--output-dir", required=True)
    _json_flag(editor_import)

    prepare = subparsers.add_parser("prepare-request")
    prepare.add_argument("--template-request", required=True)
    prepare.add_argument("--overrides", required=True)
    prepare.add_argument("--output", required=True)
    _json_flag(prepare)

    run = subparsers.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--approval-hash", required=True)
    run.add_argument("--run-dir", required=True)
    run.add_argument("--state-root", required=True, help="Directory for approval receipts and ownership keys")
    run.add_argument("--timeout", type=float, help="Execution budget in seconds, including startup; owned-job cleanup may take one additional second; default completion wait is 300 seconds")
    _json_flag(run)

    render = subparsers.add_parser("run-batch")
    render.add_argument("--manifest", required=True, help="Approved MPT scene batch")
    render.add_argument("--runs-dir", required=True)
    render.add_argument("--state-root", required=True)
    render.add_argument("--timeout", type=float, required=True, help="Run budget in seconds per scene")
    _json_flag(render)

    status = subparsers.add_parser("status")
    status.add_argument("--run-record", required=True)
    status.add_argument("--state-root", required=True)
    _json_flag(status)

    cancel = subparsers.add_parser("cancel")
    cancel.add_argument("--run-record", required=True)
    cancel.add_argument("--state-root", required=True)
    _json_flag(cancel)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--output", action="append", required=True)
    analyze.add_argument("--report-dir", required=True)
    analyze.add_argument("--keyframe-count", type=int, default=5)
    _json_flag(analyze)

    register = subparsers.add_parser("register-workflow")
    register.add_argument("--source", required=True)
    register.add_argument("--family", required=True)
    register.add_argument("--name", required=True)
    register.add_argument("--registry-root", default=str(DEFAULT_CUSTOM_REGISTRY))
    _json_flag(register)

    return parser


def run_cli(
    argv: list[str] | None = None,
    *,
    stdout=None,
    stderr=None,
    client_factory=ComfyClient,
) -> int:
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    args = build_parser().parse_args(argv)

    def progress(event):
        stderr.write(json.dumps(event, sort_keys=True) + "\n")
        stderr.flush()

    try:
        result = _dispatch(
            args,
            client_factory,
            progress=progress if args.json else None,
        )
    except Exception as exc:
        payload = {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
        _emit(payload, args.json, stdout, stderr, error=True)
        return 1
    _emit(result, args.json, stdout, stderr)
    if args.command == "run":
        return int(result["status"] != "completed")
    return int(isinstance(result, dict) and result.get("ok") is False)


def _dispatch(args, client_factory, progress=None):
    if args.command == "inventory":
        return inspect_inventory(load_profile(args.profile))
    if args.command == "preflight":
        if args.request or args.template_request:
            request = _load_json(args.request or args.template_request)
            client = client_for_snapshot(request["profile"], client_factory)
            return preflight(request, client, allow_empty_prompt=args.template_request is not None)
        return _variant_preflight(
            args.variant, args.profile, args.registry_root, client_factory
        )
    if args.command == "plan":
        plan = create_plan(_load_json(args.request))
        if args.output:
            _write_canonical(args.output, plan)
        return plan
    if args.command == "verify-plan":
        verify_approved_plan(_load_json(args.plan), args.approval_hash)
        return {"ok": True, "approval_hash": args.approval_hash, "submissions": 0}
    if args.command == "plan-batch":
        return plan_request_batch(
            _load_json(args.manifest),
            args.output_dir,
            client_factory=client_factory,
            progress=progress,
        )
    if args.command == "prepare-request":
        request = prepare_request(
            _load_json(args.template_request), _load_json(args.overrides)
        )
        _write_canonical(args.output, request)
        return request
    if args.command == "predict-durations":
        return predict_durations(_load_json(args.manifest), client_factory=client_factory, progress=progress)
    if args.command == "export-editor":
        return export_editor(_load_json(args.manifest), client_factory=client_factory)
    if args.command == "import-editor":
        return import_editor(_load_json(args.manifest), args.output_dir,
                             client_factory=client_factory, progress=progress)
    if args.command == "run":
        plan = _load_json(args.plan)
        return execute_run(
            plan,
            args.approval_hash,
            args.run_dir,
            OperatorState(args.state_root),
            timeout=args.timeout,
            client_factory=client_factory,
        )
    if args.command == "run-batch":
        batch = _load_json(args.manifest)
        scenes = batch.get("scenes")
        if batch.get("phase") != "approved" or not isinstance(scenes, list) or not scenes:
            raise ValueError("a fully approved scene batch is required")
        entries = []
        for index, scene in enumerate(scenes, start=1):
            if (not isinstance(scene, dict) or type(scene.get("index")) is not int or scene["index"] != index
                    or scene.get("status") != "approved" or not scene.get("submitted_approval_hash")
                    or scene["submitted_approval_hash"] != scene.get("approval_hash")):
                raise ValueError(f"invalid approval for scene {index}")
            entries.append((_load_json(scene["plan_file"]), scene["submitted_approval_hash"],
                            Path(args.runs_dir) / f"{index:03d}" / "run"))
        reports = execute_runs(entries, OperatorState(args.state_root), timeout=args.timeout,
                               client_factory=client_factory, progress=progress)
        statuses = [r["status"] for r in reports]
        result = {"ok": all(status == "completed" for status in statuses), "statuses": statuses}
        if not result["ok"]:
            result["error"] = next((f"Scene {index}: {error}" for index, report in enumerate(reports, 1)
                                    for error in report["errors"] if isinstance(error, str) and error.strip()),
                                   "Scene batch did not complete: " + ", ".join(statuses))
        return result
    if args.command in {"status", "cancel"}:
        state = OperatorState(args.state_root)
        record = load_signed_run_record(args.run_record, state, allow_rejected=args.command == "status")
        terminal = record.get("terminal_status")
        if terminal == "submission_rejected":
            return {"prompt_id": None, "approval_hash": record["approval_hash"], "status": terminal,
                    **({"prediction_scene_index": record["prediction_scene_index"]}
                       if "prediction_scene_index" in record else {})}
        prompt_id = record["prompt_id"]
        if terminal not in ("completed", "failed", "cancelled") and _confirmed_cancel_record(args.run_record, record):
            terminal = "cancelled"
            replace_json_atomic(args.run_record, state.sign_record({**record, "terminal_status": terminal}))
        if terminal in ("completed", "failed", "cancelled"):
            if args.command == "status":
                return {"prompt_id": prompt_id, "history": {}, "status": terminal}
            return CancellationResult(prompt_id, "confirmed_cancelled" if terminal == "cancelled" else terminal, {}, {})
        endpoint = record["endpoint"]
        options = ({key: record[key] for key in ("remote_server", "allow_insecure_http", "token_env") if key in record}
                   if record.get("remote_server") is True else {})
        client = client_factory(endpoint, owned_prompt_ids={prompt_id}, **options)
        if args.command == "status":
            history = client.history(prompt_id)
            entry = history.get(prompt_id) if isinstance(history, dict) else None
            terminal = history_status(entry, prompt_id)
            result = {"prompt_id": prompt_id, "history": history, "status": terminal}
        else:
            result = client.cancel_and_confirm(prompt_id)
            terminal = "cancelled" if result.status == "confirmed_cancelled" else result.status
        if terminal in ("completed", "failed", "cancelled"):
            replace_json_atomic(args.run_record, state.sign_record({**record, "terminal_status": terminal}))
        return result
    if args.command == "analyze":
        return analyze_outputs(
            args.output, args.report_dir, keyframe_count=args.keyframe_count
        )
    if args.command == "register-workflow":
        return register_workflow(
            args.source, args.family, args.name, args.registry_root
        )
    raise ValueError(f"unsupported command: {args.command}")


def _confirmed_cancel_record(path, record):
    try:
        report = _load_json(Path(path).expanduser().resolve().with_name("report.json"))
    except (OSError, ValueError):
        return False
    if not isinstance(report, dict) or report.get("prompt_id") != record["prompt_id"]:
        return False
    cleanup = report.get("cleanup")
    return (report.get("approval_hash") == record["approval_hash"]
            and isinstance(cleanup, dict) and cleanup.get("status") == "confirmed_cancelled") or (
                type(report.get("scene_index")) is int
                and report["scene_index"] > 0 and report.get("cancellation") == "confirmed_cancelled"
            )


def _variant_preflight(
    name: str, profile_path: str, registry_root: str, client_factory
) -> dict:
    variant = _resolve_variant(name, registry_root)
    source = Path(variant["source_path"])
    mapping = adapter_for(variant["adapter_family"]).discover(load_workflow(source))
    profile = load_profile(profile_path)
    client = client_factory(profile.endpoint, **remote_client_options(profile))
    errors = list(validate_profile(profile))
    actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual_hash != variant["source_sha256"]:
        errors.append("variant source hash does not match its registry manifest")
    checks = {}
    for label, operation in (
        ("system_stats", client.system_stats),
        ("object_info", client.object_info),
        ("queue", client.queue),
    ):
        checks[label] = read_server_object(operation, label, errors)
    object_info = checks.get("object_info")
    required_nodes = sorted(
        {
            node.node_type
            for node in load_workflow(source).nodes
            if node.node_type not in FRONTEND_ONLY_NODE_TYPES
        }
    )
    if isinstance(object_info, dict):
        errors.extend(unavailable_nodes(object_info, required_nodes))
    return {
        "ok": not errors,
        "variant": variant["name"],
        "variant_status": variant.get("status", "validated"),
        "adapter_family": variant["adapter_family"],
        "source_sha256": variant["source_sha256"],
        "mapping": mapping.to_dict(),
        "required_nodes": required_nodes,
        "warnings": model_reuse_warnings(checks["system_stats"]),
        "errors": errors,
        "checks": preflight_checks(**checks),
        "submissions": 0,
    }


def _resolve_variant(name: str, registry_root: str) -> dict:
    if not isinstance(name, str) or not SAFE_NAME.fullmatch(name):
        raise ValueError(f"unknown or unsafe workflow variant: {name}")
    root = Path(registry_root).expanduser().resolve()
    variant_dir = (root / name).resolve()
    if variant_dir.parent != root:
        raise ValueError("custom variant must be a direct registry child")
    manifest = _load_json(variant_dir / "manifest.json")
    if manifest.get("name") != name:
        raise ValueError("custom variant manifest name mismatch")
    workflow_file = manifest.get("workflow_file")
    if (
        not isinstance(workflow_file, str)
        or not workflow_file
        or Path(workflow_file).is_absolute()
    ):
        raise ValueError("custom variant workflow_file is invalid")
    source = (variant_dir / workflow_file).resolve()
    if source.parent != variant_dir:
        raise ValueError("custom variant workflow must be contained by its registry directory")
    expected_hash = manifest.get("source_sha256")
    if not isinstance(expected_hash, str) or sha256_file(source) != expected_hash:
        raise ValueError("custom variant workflow hash mismatch")
    return {**manifest, "source_path": str(source)}


def _load_json(path: str | Path) -> dict:
    value = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON document must be an object")
    return value


def _write_canonical(path: str | Path, value) -> None:
    create_json_exclusive(path, _jsonable(value), private=True)


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _emit(value, as_json: bool, stdout, stderr, *, error: bool = False) -> None:
    output = json.dumps(_jsonable(value), ensure_ascii=True, indent=2, sort_keys=True)
    stream = stdout if as_json or not error else stderr
    stream.write(output + "\n")


def _json_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true")
