"""Registration of immutable, project-owned custom workflow variants."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil

from .adapters import adapter_for
from .artifacts import create_json_exclusive, ignore_private_directory
from .workflow import MODEL_SUFFIXES, Workflow, detect_format, load_workflow


SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class RegistrationError(ValueError):
    pass


@dataclass(frozen=True)
class RegistrationResult:
    workflow_path: Path
    manifest_path: Path
    manifest: dict


def register_workflow(
    source: str | Path,
    family: str,
    name: str,
    registry_root: str | Path,
) -> RegistrationResult:
    """Validate and copy a custom workflow without modifying its source."""
    reserved = {'con', 'prn', 'aux', 'nul', *(f'com{i}' for i in range(1, 10)), *(f'lpt{i}' for i in range(1, 10))}
    if (not isinstance(name, str) or not SAFE_NAME.fullmatch(name)
            or name.endswith('.') or name.split('.')[0] in reserved):
        raise RegistrationError("variant name must use lowercase letters, numbers, '.', '-', or '_'")
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise RegistrationError(f"source workflow does not exist: {source_path}")
    root = Path(registry_root).expanduser().resolve()
    variant_dir = root / name
    if variant_dir.exists():
        raise RegistrationError(f"variant already exists: {name}")

    before = source_path.read_bytes()
    try:
        data = json.loads(before.decode("utf-8"))
        workflow = load_workflow(data)
        mapping = adapter_for(family).discover(workflow)
    except Exception as exc:
        raise RegistrationError(f"semantic discovery failed: {exc}") from exc

    if source_path.read_bytes() != before:
        raise RegistrationError("source changed during registration")

    manifest = _json_value(
        {
            "name": name,
            "adapter_family": family,
            "status": "experimental",
            "source_path": str(source_path),
            "source_sha256": hashlib.sha256(before).hexdigest(),
            "workflow_format": detect_format(data).value,
            "workflow_file": "workflow.json",
            "mapping": mapping.to_dict(),
            **_dependencies(workflow),
        }
    )

    created_variant_dir = False
    try:
        variant_dir.mkdir(parents=True, exist_ok=False)
        created_variant_dir = True
        ignore_private_directory(variant_dir)
        workflow_path = variant_dir / "workflow.json"
        manifest_path = variant_dir / "manifest.json"
        with workflow_path.open("xb") as stream:
            stream.write(before)
        create_json_exclusive(manifest_path, manifest)
    except FileExistsError as exc:
        if created_variant_dir and variant_dir.exists():
            shutil.rmtree(variant_dir)
        raise RegistrationError(f"variant already exists: {name}") from exc
    except Exception:
        if created_variant_dir and variant_dir.exists():
            shutil.rmtree(variant_dir)
        raise
    return RegistrationResult(workflow_path, manifest_path, manifest)


def _dependencies(workflow: Workflow) -> dict[str, list[str]]:
    loras: set[str] = set()
    models: set[str] = set()
    node_types: set[str] = set()
    for node in workflow.nodes:
        node_types.add(node.node_type)
        values = list(node.widgets) + list(node.inputs.values())
        for value in values:
            if not isinstance(value, str) or not value.lower().endswith(MODEL_SUFFIXES):
                continue
            if "lora" in node.node_type.lower():
                loras.add(value)
            elif "loader" in node.node_type.lower():
                models.add(value)
    return {
        "required_loras": sorted(loras),
        "required_models": sorted(models),
        "required_custom_nodes": sorted(node_types),
    }


def _json_value(value) -> dict:
    return json.loads(json.dumps(value, sort_keys=True))
