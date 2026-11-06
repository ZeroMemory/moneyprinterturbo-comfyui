"""Connect the mounted host ComfyUI and start MPT without changing its host server."""
import argparse
import os
from pathlib import Path
import sys
import tomllib

MPT = Path("/MoneyPrinterTurbo")
ADDON = Path("/opt/mpt-comfyui")
COMFYUI = Path("/comfyui")
sys.path.insert(0, str(ADDON))
import install as installer

MODEL_STATE = ".ltx-models"


def registry_settings(state):
    path = installer.checked(state, "settings.toml")
    data = installer.contents(path)
    if data is None:
        return path, None, {}
    values = tomllib.loads(data.decode("utf-8")).get("ltx")
    templates = values.get("request_templates") if isinstance(values, dict) else None
    if (not isinstance(templates, dict) or any(not isinstance(value, str) for value in templates.values())
            or any(not isinstance(values.get(key), str)
                   for key in ("python_executable", "operator_script", "request_template"))
            or type(values.get("enabled")) is not bool):
        raise ValueError(f"Invalid model registry settings: {path}")
    return path, data, values


def register_model(data, model, workflow, endpoint):
    with installer.install_lock(data):
        state = installer.checked(data, MODEL_STATE)
        settings, previous, saved = registry_settings(state)
        builtin, files = installer.prepare_model(state, COMFYUI, model, None, (), endpoint,
                                                  external_server=True, workflow_api=workflow)
        new = installer.operator_settings(None, None, saved, builtin)
        state.mkdir(mode=0o700, exist_ok=True)
        ignore = installer.checked(state, ".gitignore")
        if installer.contents(ignore) is None:
            installer.write_file(ignore, b"*\n", 0o600)
        pending = {}
        for name, data in files.items():
            path = installer.generated_file(state, name, allow_missing=True)
            if not path.exists():
                pending[path] = data
        # Publish immutable snapshots before making the new request visible to MPT.
        for path, data in pending.items():
            installer.write_atomic(path, data, 0o600)
        if new != previous:
            installer.write_atomic(settings, new, 0o600)
    print(f"Local model {model} is registered." + installer.scene_editor_status(COMFYUI))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["cli"]:
        mode, cli_args = "cli", argv[1:]
    else:
        parser = argparse.ArgumentParser(description="MoneyPrinterTurbo with host ComfyUI")
        parser.add_argument("mode", choices=("webui", "api", "cli"), nargs="?", default="webui")
        mode, cli_args = parser.parse_args(argv).mode, []
    models = list(dict.fromkeys(value.strip() for value in os.environ.get("MPT_MODELS", "2.5").split(",")))
    if not models or any(model not in {"2.3", "2.5", "h3"} for model in models):
        raise ValueError("MPT_MODELS must list 2.5, 2.3 or h3, separated by commas")
    port = int(os.environ.get("MPT_WEBUI_PORT", "8501")) if mode == "webui" else 8501
    if not 1 <= port <= 65535:
        raise ValueError("MPT_WEBUI_PORT must be between 1 and 65535")
    if not (mode == "cli" and any(arg in {"--help", "-h"} for arg in cli_args)):
        if not (COMFYUI / "main.py").is_file():
            raise ValueError("Mount the ComfyUI folder containing main.py at /comfyui")
        data = Path(os.environ.get("MPT_HOME", "/data")).absolute()
        data.mkdir(parents=True, exist_ok=True)
        os.environ.pop("MPT_LTX_SETTINGS_PATH", None)
        try:
            state = installer.checked(data, MODEL_STATE)
            settings, _, _ = registry_settings(state)
        except (OSError, ValueError, installer.InstallError) as error:
            print(f"Local model registry is unavailable: {error}", file=sys.stderr)
            state = None
        if state is not None:
            os.environ["MPT_LTX_SETTINGS_PATH"] = str(settings)
        workflows = data / "workflows"
        for model in models:
            if state is None:
                break
            workflow = workflows / f"{model}.api.json"
            if not workflow.is_file():
                print(f"Local model {model} was not registered. Save the ComfyUI API export to {workflow} "
                      "and restart the container.", file=sys.stderr)
                continue
            try:
                register_model(data, model, workflow, os.environ.get("COMFYUI_ENDPOINT", "http://127.0.0.1:8188"))
            except Exception as error:
                print(f"Local model {model} was not registered: {error}. Resolve the setup error and restart the "
                      "container to retry. Existing models and other MPT sources remain available.", file=sys.stderr)
    if mode == "api":
        os.environ.setdefault("MPT_LISTEN_HOST", "127.0.0.1")
        command = [str(MPT / "main.py")]
    elif mode == "cli":
        command = [str(MPT / "cli.py"), *cli_args]
    else:
        command = ["-m", "streamlit", "run", str(MPT / "webui/Main.py"),
                   "--server.address=127.0.0.1", f"--server.port={port}", "--server.headless=true",
                   "--browser.serverAddress=127.0.0.1", "--server.enableCORS=true",
                   "--browser.gatherUsageStats=false"]
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, [sys.executable, "-B", *command])


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(f"Container setup failed: {error}")
