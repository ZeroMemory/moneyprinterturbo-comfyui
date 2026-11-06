"""Create a local profile and a text-to-video request for an existing ComfyUI."""
import argparse
import json
from pathlib import Path
import sys
from uuid import uuid4

OPERATOR = Path(__file__).resolve().parent / "runtime"
sys.path.insert(0, str(OPERATOR / "scripts"))
from ltx_operator_lib.adapters import H3_FAMILY, adapter_for, video_suffix
from ltx_operator_lib.api import ComfyApiError, ComfyClient, remote_client_options
from ltx_operator_lib.artifacts import ignore_private_directory
from ltx_operator_lib.profiles import profile_from_dict, validate_profile
from ltx_operator_lib.semantics import _server_choices, validate_plan_semantics
from ltx_operator_lib.workflow import (MODEL_INPUTS, WorkflowFormat, detect_format, load_workflow,
                                       model_directories, model_files, nodes_by_type)
from ltx_operator_lib.workflow_import import check_ltx_upscaler_version, editable_scene_graph

MODEL_VERSIONS = ("2.5", "2.3", "h3")



class MissingComfyPython(ValueError):
    pass


class MissingWeights(ValueError):
    def __init__(self, filenames, roots):
        self.searched_roots = {path.resolve() for path in roots}
        super().__init__("Missing workflow weights:\n" + "\n".join(sorted(filenames)) +
                         "\nFor weights stored separately, pass --model-root with the ComfyUI model root.")


def _local_installation(comfyui, python, external_server):
    root = Path(comfyui).expanduser().resolve(strict=True)
    portable = not (root / "main.py").is_file() and (root / "ComfyUI/main.py").is_file()
    app = root / "ComfyUI" if portable else root
    if not (app / "main.py").is_file():
        raise ValueError("Choose the directory containing main.py or ComfyUI/main.py")
    if type(external_server) is not bool:
        raise ValueError("external_server must be a boolean")
    if external_server and python is None:
        python = sys.executable
    if python is None:
        paths = [root / "python_embeded/python.exe"] if portable else []
        paths += [app / name for name in (".venv/bin/python", ".venv/Scripts/python.exe",
                                           "venv/bin/python", "venv/Scripts/python.exe")]
        python = next((path for path in paths if path.is_file()), None)
    if python is None or not external_server and not Path(python).expanduser().is_file():
        raise MissingComfyPython("ComfyUI's Python executable was not found. "
                                 "Use --comfy-python with install.py or --python with configure_ltx.py.")
    # Keep the venv interpreter path even when bin/python is a symlink.
    return root, app, Path(python).expanduser().absolute(), portable


def build_template(comfyui, profile_path, python=None, model_roots=(), endpoint="http://127.0.0.1:8188",
                   weights=None, model_version="2.5", external_server=False, *, workflow_api,
                   remote_server=False, allow_insecure_http=False, token_env=None):
    """Build a profile and request without creating their destination files."""
    profile_path = Path(profile_path).expanduser().absolute()
    if remote_server:
        if comfyui is not None or python is not None or model_roots or external_server:
            raise ValueError("Remote ComfyUI cannot use local installation options")
        profile = {"name": "remote-comfyui", "endpoint": endpoint, "remote_server": True}
        if allow_insecure_http:
            profile["allow_insecure_http"] = allow_insecure_http
        if token_env is not None:
            profile["token_env"] = token_env
        roots = []
    else:
        if allow_insecure_http or token_env is not None:
            raise ValueError("Remote ComfyUI options require --remote-comfyui")
        root, app, interpreter, portable = _local_installation(comfyui, python, external_server)
        roots = [app / "models", *(Path(p).expanduser().resolve(strict=True) for p in model_roots)]
        profile = {"name": "local-comfyui", "comfyui_root": str(root),
                   "python_executable": str(interpreter), "endpoint": endpoint,
                   "output_directory": str(app / "output"),
                   "launch_args": ["-s", "ComfyUI/main.py", "--windows-standalone-build"] if portable and sys.platform == "win32"
                                  else (["ComfyUI/main.py"] if portable else ["main.py"]),
                   "optional_modules": [], "startup_timeout_seconds": 120,
                   "attention_backend": "default", "allow_operator_start": False,
                   "allow_operator_stop_owned": False}
        if external_server:
            profile["external_server"] = True
        if roots[1:]:
            profile["model_roots"] = [str(path) for path in roots[1:]]
    launch = profile_from_dict(profile, profile_path.parent)
    errors = validate_profile(launch)
    if errors:
        raise ValueError("; ".join(errors))
    if model_version not in MODEL_VERSIONS:
        raise ValueError(f"Unsupported model version: {model_version}")
    family = H3_FAMILY if model_version == "h3" else "t2v-i2v-two-stage"
    variant_name = H3_FAMILY if model_version == "h3" else f"ltx-{model_version}-t2v"
    source = Path(workflow_api).expanduser().resolve(strict=True)
    graph = json.loads(source.read_text(encoding="utf-8-sig"))
    try:
        is_api_graph = detect_format(graph) is WorkflowFormat.API
    except ValueError:
        is_api_graph = False
    if not is_api_graph or any(not isinstance(node.get("inputs"), dict) for node in graph.values()):
        raise ValueError("Choose a ComfyUI workflow saved with Export (API).")
    try:
        graph = editable_scene_graph(graph, model_version)
    except (KeyError, TypeError, ArithmeticError) as error:
        raise ValueError(f"Incomplete or invalid ComfyUI export: {error}") from error
    check_ltx_upscaler_version(graph, model_version)
    # Both VAEs live in models/vae but feed different decoders.
    vae_roles = {n["inputs"][field][0]: role for n in graph.values()
                 for decoder, field, role in (("VAEDecodeTiled", "vae", "video_vae"),
                                              ("VAEDecode", "vae", "video_vae"),
                                              ("VAEDecodeAudio", "vae", "audio_vae"),
                                              ("LTXVAudioVAEDecode", "audio_vae", "audio_vae"))
                 if n["class_type"] == decoder and isinstance(link := n["inputs"].get(field), list)
                 and len(link) == 2 and isinstance(link[0], str)
                 and graph.get(link[0], {}).get("class_type") == "VAELoader"}
    weights = weights or {}
    supported = {vae_roles.get(key, category) for key, n in graph.items()
                 for category in MODEL_INPUTS.get(n["class_type"], {}).values()}
    unknown = set(weights) - supported
    if unknown:
        raise ValueError(f"Weight overrides unavailable for {variant_name}: " + ", ".join(sorted(unknown)))
    for filename in weights.values():
        if (not filename or filename in {".", ".."} or
                any(ch in filename for ch in '/\\:*?[]"<>|')):
            raise ValueError("Weight overrides must be filenames, without paths or wildcards")
    resolved = {"models": {}, "loras": {}}
    remote_weights = {"models": set(), "loras": set()}
    missing = set()
    try:
        object_info = (ComfyClient(endpoint, **remote_client_options(launch)).object_info()
                       if external_server or remote_server else None)
    except (ComfyApiError, OSError) as error:
        raise ValueError(f"Cannot read ComfyUI's weight list: {error}") from error
    if (external_server or remote_server) and not isinstance(object_info, dict):
        raise ValueError("ComfyUI object_info must be an object")
    if remote_server:
        unavailable = {node["class_type"] for node in graph.values()
                       if not isinstance(object_info.get(node["class_type"]), dict)}
        if unavailable:
            raise ValueError("Remote ComfyUI is missing nodes: " + ", ".join(sorted(unavailable)))
    for node_id, node in graph.items():
        for field, category in MODEL_INPUTS.get(node["class_type"], {}).items():
            name = weights.get(vae_roles.get(node_id, category)) or node["inputs"][field]
            if not isinstance(name, str) or not name:
                raise ValueError(f"{category} loader {node_id} needs a filename")
            relative = name.replace("\\", "/")
            if (any(char in relative for char in ':*?[]"<>|')
                    or any(part in {"", ".", ".."} for part in relative.split("/"))):
                raise ValueError(f"Invalid {category} filename: {name}")
            if remote_server:
                key = _advertised_weight(object_info, node, field, category, relative)
                node["inputs"][field] = key
                remote_weights["loras" if category == "loras" else "models"].add(key)
                continue
            matches = model_files(roots, category, relative)
            if not matches and "/" not in relative:
                # A bare filename can also select one uniquely named nested weight.
                nested = {file.relative_to(folder).as_posix() for folder in model_directories(roots, category)
                          for file in folder.rglob(relative) if file.is_file()}
                if len(nested) > 1:
                    raise ValueError(f"Multiple files match {category}/{name}; use a category-relative filename")
                if nested:
                    relative = nested.pop()
                    matches = model_files(roots, category, relative)
            if len(matches) > 1:
                raise ValueError(f"Multiple files match {category}/{relative}; give each weight a distinct filename or subfolder")
            if not matches:
                missing.add(f"{category}/{name}")
                continue
            path, key = next(iter(matches)), str(Path(relative))
            if external_server:
                key = _advertised_weight(object_info, node, field, category, key)
            files = resolved["loras" if category == "loras" else "models"]
            if key in files and Path(files[key]) != path:
                raise ValueError(f"Weight name {key!r} identifies different files: {files[key]} and {path}. "
                                 "Give one weight a distinct filename or category-relative subfolder.")
            node["inputs"][field] = key
            files[key] = str(path)
    if len(vae_roles) == 2 and not missing:
        if len({graph[key]["inputs"]["vae_name"] for key in vae_roles}) < 2:
            raise ValueError("Audio and video VAE must use different weights")
    if missing:
        raise MissingWeights(missing, roots)
    workflow = load_workflow(graph)
    mapping = adapter_for(family).discover(workflow)
    for field in ("prompt", "width", "height", "frame_count", "fps"):
        ref = getattr(mapping, field)
        if ref is None or ref.derived or ref.value is None:
            raise ValueError(f"The exported graph needs an editable {field} control")
    if not mapping.seeds or mapping.input_media:
        raise ValueError("Choose a seeded text-to-video export without required input media")
    if any(field.derived or not isinstance(field.value, str) or not field.value.strip() for field in mapping.outputs):
        raise ValueError("The exported graph needs a direct SaveVideo filename prefix")
    outputs = [field.value + "*" + (video_suffix(node.inputs) or "")
               for node, field in zip(nodes_by_type(workflow, "SaveVideo"), mapping.outputs, strict=True)]
    request = {
        "api_prompt": graph, "client_id": uuid4().hex, "input_media": [],
        "prompt": mapping.prompt.value,
        "negative_prompt": mapping.negative_prompt.value if mapping.negative_prompt else "",
        "profile": {"source_path": str(profile_path), "endpoint": endpoint,
                    **({"remote_server": True} if remote_server else {})},
        "variant": {"name": variant_name, "source_path": str(source),
                    "model_version": model_version, "supports_negative_prompt": model_version == "2.3",
                    "adapter_family": family},
        "parameters": {"fps": mapping.fps.value, "width": mapping.width.value,
                       "height": mapping.height.value, "frame_count": mapping.frame_count.value,
                       **{f"stage_{i}_seed": seed.value for i, seed in enumerate(mapping.seeds, 1)}},
        "requirements": {"required_custom_nodes": sorted({n["class_type"] for n in graph.values()}),
                         "required_models": sorted(remote_weights["models"]) if remote_server else list(resolved["models"]),
                         "resolved_models": resolved["models"],
                         "required_loras": sorted(remote_weights["loras"]) if remote_server else list(resolved["loras"]),
                         "resolved_loras": resolved["loras"],
                         "required_files": []},
        "warnings": [], "outputs": outputs,
    }
    errors = validate_plan_semantics(request, allow_empty_prompt=True, launch_profile=launch,
                                     object_info=object_info if remote_server else None)
    if errors:
        raise ValueError("; ".join(errors))
    return profile, request


def _advertised_weight(object_info, node, field, category, name):
    choices = _server_choices(object_info.get(node["class_type"]), field)
    matches = [value for value in choices or [] if isinstance(value, str)
               and value.replace("\\", "/") == name.replace("\\", "/")]
    if len(matches) != 1:
        raise ValueError(f"ComfyUI does not advertise one matching weight for {category}/{name}")
    return matches[0]


def configure(comfyui, destination, python=None, model_roots=(), endpoint="http://127.0.0.1:8188",
              weights=None, model_version="2.5", external_server=False, *, workflow_api,
              remote_server=False, allow_insecure_http=False, token_env=None):
    target = Path(destination).expanduser().absolute()
    profile_path = target / "profile.json"
    request_path = target / "request.json"
    graph_path = target / "workflow.json"
    profile, request = build_template(comfyui, profile_path, python, model_roots, endpoint, weights,
                                      model_version, external_server, workflow_api=workflow_api,
                                      remote_server=remote_server, allow_insecure_http=allow_insecure_http, token_env=token_env)
    target.mkdir(parents=True, exist_ok=False)
    try:
        ignore_private_directory(target)
        graph_path.write_text(json.dumps(request["api_prompt"], indent=2) + "\n", encoding="utf-8")
        request["variant"]["source_path"] = str(graph_path)
        profile_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
        request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
        return request_path
    except BaseException:
        for created in (profile_path, request_path, graph_path, target / ".gitignore"):
            created.unlink(missing_ok=True)
        target.rmdir()
        raise


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--comfyui", type=Path)
    sources.add_argument("--remote-comfyui", metavar="URL", help="Connect a trusted ComfyUI server over HTTP(S)")
    parser.add_argument("--output", type=Path, required=True, help="New folder for profile.json and request.json")
    parser.add_argument("--python", type=Path, help="ComfyUI interpreter outside its usual venv")
    parser.add_argument("--model-root", type=Path, action="append", default=[],
                        help="Extra model root already registered with this ComfyUI (repeatable)")
    parser.add_argument("--endpoint")
    parser.add_argument("--external-server", action="store_true", help="Use an already running server without executing its host Python")
    parser.add_argument("--allow-insecure-comfyui-http", action="store_true",
                        help="Allow token-free HTTP for an explicit private IP or loopback")
    parser.add_argument("--comfyui-token-env", metavar="NAME",
                        help="Read the HTTPS server's bearer token from this environment variable")
    parser.add_argument("--workflow-api", type=Path, required=True, help="Local JSON saved with ComfyUI Export (API)")
    parser.add_argument("--model-version", choices=MODEL_VERSIONS, default="2.5", help="LTX release or h3 for MiniMax H3 (default: 2.5)")
    for flag in ("checkpoint", "text-encoder", "upscaler", "lora", "transformer", "audio-vae", "video-vae"):
        parser.add_argument("--" + flag, help="Filename of a compatible weight for the selected workflow")
    args = parser.parse_args()
    if args.remote_comfyui and args.endpoint:
        parser.error("--endpoint applies only to a local installation")
    try:
        weights = {"checkpoints": args.checkpoint, "text_encoders": args.text_encoder,
                   "latent_upscale_models": args.upscaler, "loras": args.lora,
                   "diffusion_models": args.transformer, "audio_vae": args.audio_vae, "video_vae": args.video_vae}
        print(configure(args.comfyui, args.output, args.python, args.model_root,
                        args.remote_comfyui or args.endpoint or "http://127.0.0.1:8188",
                        {category: name for category, name in weights.items() if name is not None}, args.model_version,
                        args.external_server, workflow_api=args.workflow_api, remote_server=bool(args.remote_comfyui),
                        allow_insecure_http=args.allow_insecure_comfyui_http, token_env=args.comfyui_token_env))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
