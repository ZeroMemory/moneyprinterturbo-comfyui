"""Run approved ComfyUI scenes and record each submission and output."""

from fnmatch import fnmatch
from functools import cache
from glob import escape
from http.client import HTTPException
import json
import math
import os
from pathlib import Path, PureWindowsPath
import subprocess
import time
from uuid import uuid4

from .api import (ComfyClient, ComfyHttpError, ComfyPromptNotSubmitted, CompletionResult, execution_error, history_status,
                  remaining_timeout, remote_client_options)
from .adapters import adapter_for, video_suffix
from .artifacts import create_json_exclusive, ignore_private_directory, replace_json_atomic, sha256_file
from .batch_planning import CachedReadOnlyClient
from .planning import PreflightError, preflight, verify_approved_plan
from .processes import (
    ServerOwnership,
    may_stop_server,
    start_owned_server,
    stop_owned_server,
)
from .profiles import load_profile_snapshot, local_endpoint_error, remote_endpoint_error, validate_profile
from .semantics import primary_video_node
from .state import OperatorState, OwnershipRecordError
from .workflow import load_workflow, nodes_by_type


CLEANUP_GRACE_SECONDS = 1


class _SceneRun:
    def __init__(self, plan, expected_hash, directory):
        self.plan = plan
        self.approval = expected_hash
        self.directory = _initialize_run_directory(directory, plan)
        self.report = {
            "schema_version": 1, "approval_hash": expected_hash, "status": "initializing",
            "started_at_unix": time.time(), "prompt_id": None, "history": {}, "outputs": [], "errors": [],
        }
        create_json_exclusive(self.directory / "report.json", self.report)
        self.consumed = False
        self.rejected = False
        self.deadline = None
        self.error = None
        self.handled = False

    def record(self, status, **fields):
        self.report.update(status=status, **fields)
        replace_json_atomic(self.directory / "report.json", self.report)

    def failed(self, error):
        if self.error is None:
            self.error = error
        status = "uncertain_execution" if self.report["prompt_id"] else (
            "submission_rejected" if self.rejected else
            "uncertain_submission" if self.consumed else "failed_before_submission"
        )
        message = f"{type(self.error).__name__}: {self.error}"
        errors = self.report["errors"]
        self.record(status, errors=errors if message in errors else [*errors, message], completed_at_unix=time.time())

    def submit(self, client, state, deadline, profile):
        remaining_timeout(deadline)
        state.consume_approval(self.approval, run_directory=self.directory)
        self.consumed = True
        try:
            try:
                budget = remaining_timeout(deadline)
            except TimeoutError as exc:
                raise ComfyPromptNotSubmitted("ComfyUI prompt deadline elapsed before sending") from exc
            result = client.submit(self.plan["api_prompt"], client_id=self.plan["client_id"],
                                   **({"timeout": budget} if budget is not None else {}))
        except (ComfyHttpError, ComfyPromptNotSubmitted) as exc:
            if isinstance(exc, ComfyPromptNotSubmitted) or exc.prompt_rejected:
                try:
                    self._sign_run(state, profile, terminal_status="submission_rejected")
                except Exception as record_error:
                    exc.add_note(f"Could not save prompt rejection: {record_error}")
                else:
                    self.rejected = True
            raise
        prompt_id = result["prompt_id"]
        self.report["prompt_id"] = prompt_id
        self.deadline = deadline
        self._sign_run(state, profile, prompt_id=prompt_id)
        self.record("queued")

    def _sign_run(self, state, profile, **disposition):
        ownership = state.sign_record({
            "schema_version": 1, "approval_hash": self.approval, "endpoint": self.plan["profile"]["endpoint"],
            "run_directory": str(self.directory), **remote_client_options(profile), **disposition,
        })
        create_json_exclusive(self.directory / "run.json", ownership)

    def cleanup(self, client, cleanup_budget):
        prompt_id = self.report["prompt_id"]
        self.report["cleanup"] = {"status": "unconfirmed", "grace_seconds": CLEANUP_GRACE_SECONDS}
        try:
            try:
                history = client.history(prompt_id, timeout=cleanup_budget())
            except Exception as exc:
                self.report["cleanup"]["history_error"] = f"{type(exc).__name__}: {exc}"
                history = {}
            status = history_status(history[prompt_id], prompt_id) if isinstance(history, dict) and prompt_id in history else None
            if status in {"completed", "failed", "cancelled"}:
                self.report["cleanup"]["status"] = "terminal_history"
                return CompletionResult(prompt_id, status, history)
            cancellation = client.cancel_and_confirm(prompt_id, timeout=cleanup_budget())
            self.report["cleanup"]["status"] = cancellation.status
            status = "cancelled" if cancellation.status == "confirmed_cancelled" else cancellation.status
            if status in {"completed", "failed", "cancelled"}:
                return CompletionResult(prompt_id, status, cancellation.history)
        except Exception as exc:
            self.report["cleanup"]["error"] = f"{type(exc).__name__}: {exc}"

    def finish(self, client, profile, websocket_events, *, batch, cleanup_budget):
        prompt_id = self.report["prompt_id"]
        remaining = self.deadline - time.monotonic() if self.deadline is not None else 300
        result = client.wait_for_completion(
            prompt_id, self.plan["client_id"], timeout=remaining, websocket_events=websocket_events
        ) if remaining > 0 else CompletionResult(prompt_id, "timeout", {})
        if result.status == "timeout":
            result = self.cleanup(client, cleanup_budget) or result
        self.report["history"] = result.history
        error = execution_error(result)
        if error:
            self.report["errors"] = [error]
        completed = result.status == "completed"
        prefixes = {}
        if completed:
            workflow = load_workflow(self.plan["api_prompt"])
            mapping = adapter_for(self.plan["variant"]["adapter_family"]).discover(workflow)
            prefixes = {node.key: field.value for node, field in
                        zip(nodes_by_type(workflow, "SaveVideo"), mapping.outputs, strict=True)}
        entry = result.history.get(prompt_id, {}) if isinstance(result.history, dict) else {}
        destination, download = profile.output_directory, None
        if profile.remote_server:
            destination = self.directory / "outputs"
            download = lambda file, path: _download_clip(client, prompt_id, file, path, self.deadline)
        outputs = _collect_outputs(
            entry, destination, graph=self.plan["api_prompt"], prefixes=prefixes,
            expected=self.plan["outputs"] if completed else (),
            primary_node=primary_video_node(self.plan) if (batch or profile.remote_server) and completed else None,
            download=download,
            validate_video=lambda path: _validate_clip(path, self.deadline),
        )
        self.record(result.status, outputs=outputs, completed_at_unix=time.time())
        self.handled = True

    def discard_pending(self, client, cleanup_budget):
        prompt_id = self.report["prompt_id"]
        self.report["cleanup"] = {"status": "unconfirmed", "grace_seconds": CLEANUP_GRACE_SECONDS}
        queue = client.queue(timeout=cleanup_budget())
        if prompt_id in _queue_ids(queue["queue_running"]) or prompt_id in client.history(prompt_id, timeout=cleanup_budget()):
            return False
        status = "removed_or_unconfirmed"
        if prompt_id in _queue_ids(queue["queue_pending"]):
            result = client.cancel_and_confirm(prompt_id, timeout=cleanup_budget())
            if result.status in {"completed", "failed"}:
                return False
            status = "cancelled" if result.status == "confirmed_cancelled" else "pending_removal_unconfirmed"
            self.report["history"] = result.history
            self.report["cleanup"]["status"] = result.status
        self.record(status, completed_at_unix=time.time())
        self.handled = True
        return True


def _queue_ids(rows):
    return {row[1] for row in rows if isinstance(row, (tuple, list)) and len(row) >= 2}


def _stop_after_run(owned, client, stopper, cleanup_budget):
    if not may_stop_server(owned):
        return
    # Other clients may have joined even during startup or a single-scene run.
    try:
        queue = client.queue(timeout=cleanup_budget())
        if isinstance(queue, dict) and queue.get("queue_running") == [] and queue.get("queue_pending") == []:
            # stop_owned_server can wait once after terminate and once after kill.
            stopper(owned, timeout=cleanup_budget() / 2)
    except Exception:
        return "unconfirmed"
    return "checked"


def execute_run(plan, approval_hash, run_directory, state, **kwargs) -> dict:
    """Execute exactly one approved submission and record its complete lifecycle."""
    return execute_runs([(plan, approval_hash, run_directory)], state, batch=False, **kwargs)[0]


def execute_runs(
    entries, state: OperatorState, *, timeout=None, client_factory=ComfyClient,
    server_starter=start_owned_server, server_stopper=stop_owned_server,
    websocket_events=None, sleep=time.sleep, progress=None, batch=True,
) -> list[dict]:
    """Queue up to two approved scenes on one ComfyUI server, keeping one report per scene."""
    if timeout is not None and (
        isinstance(timeout, bool) or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout) or timeout <= 0
    ):
        raise ValueError("run timeout must be a positive finite number of seconds")
    first_deadline = time.monotonic() + timeout if timeout is not None else None
    if not entries or len({expected for _, expected, _ in entries}) != len(entries):
        raise ValueError("a batch requires distinct scene approvals")
    # One snapshot for the all-scene gate. No cached digest crosses a submission.
    hash_file = cache(sha256_file)
    for plan, expected, directory in entries:
        verify_approved_plan(plan, expected, hash_file=hash_file)
        if batch:
            primary_video_node(plan)
        if plan["profile"] != entries[0][0]["profile"]:
            raise ValueError("batch scenes must use the same launch profile")
        if Path(directory).exists():
            raise FileExistsError(f"run directory already exists: {directory}")
        state.check_unused(expected)

    scenes = [_SceneRun(*entry) for entry in entries]
    first = scenes[0]
    # A queued scene may finish before it is observed; /history retains its result.
    if batch and websocket_events is None:
        websocket_events = lambda *_: ()
    owned = ServerOwnership(started_by_operator=False)
    client = None
    active = []
    current = first
    cleanup_deadline = None
    primary_error = None
    final_error = None
    abort_owned = False

    def cleanup_budget():
        nonlocal cleanup_deadline
        if cleanup_deadline is None:
            cleanup_deadline = time.monotonic() + CLEANUP_GRACE_SECONDS
        return remaining_timeout(cleanup_deadline)

    def notify(scene):
        if progress:
            progress({"event": "scene_render", "scene_index": scenes.index(scene) + 1,
                      "status": scene.report["status"], "run_directory": str(scene.directory),
                      "outputs": scene.report["outputs"], "errors": scene.report["errors"]})

    def remember_interrupt(error):
        nonlocal final_error
        if not isinstance(error, Exception) and primary_error is None and final_error is None:
            final_error = error

    def failed(scene, error):
        if scene.error is None:
            scene.error = error
        try:
            scene.failed(scene.error)
        except BaseException as exc:
            remember_interrupt(exc)
        try:
            notify(scene)
        except BaseException as exc:
            remember_interrupt(exc)

    def cleanup(scene):
        if not scene.report["prompt_id"] or scene.handled:
            return
        try:
            result = scene.cleanup(client, cleanup_budget)
            if result is not None:
                scene.report["history"] = result.history
        except BaseException as exc:
            remember_interrupt(exc)
            scene.report["cleanup"]["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            scene.handled = True
            try:
                scene.record(scene.report["status"], completed_at_unix=time.time())
            except BaseException as exc:
                remember_interrupt(exc)

    def finish(scene):
        try:
            scene.finish(client, profile, websocket_events, batch=batch, cleanup_budget=cleanup_budget)
        except BaseException as exc:
            failed(scene, exc)
            raise
        notify(scene)

    try:
        profile = load_profile_snapshot(first.plan["profile"])
        errors = validate_profile(profile)
        if errors:
            raise ValueError("invalid launch profile: " + "; ".join(errors))
        client = client_factory(profile.endpoint, **remote_client_options(profile))
        owned = _ensure_server(client, profile, first.directory, server_starter=server_starter,
                               server_stopper=server_stopper, sleep=sleep, deadline=first_deadline,
                               cleanup_budget=cleanup_budget)
        checks = CachedReadOnlyClient(client)
        for current in scenes:
            report = preflight(current.plan, checks, deadline=first_deadline)
            if not report["ok"]:
                raise PreflightError("; ".join(report["errors"]))
        # Check later scenes first so scene one's files are read immediately before its POST.
        hash_file.cache_clear()
        for current in scenes[1:]:
            verify_approved_plan(current.plan, current.approval, hash_file=hash_file)

        for scene in scenes:
            current = None
            if len(active) == 2:
                previous = active.pop(0)
                finish(previous)
                if previous.report["status"] != "completed":
                    return [scene.report for scene in scenes]
            current = scene
            deadline = first_deadline
            if current is not first:
                if timeout is not None:
                    deadline = time.monotonic() + (1 + len(active)) * timeout
            verify_approved_plan(current.plan, current.approval, hash_file=sha256_file)
            current.submit(client, state, deadline, profile)
            active.append(current)
            notify(current)
        current = None
        while active:
            previous = active.pop(0)
            finish(previous)
            if previous.report["status"] != "completed":
                break
    except BaseException as exc:
        primary_error = exc
        abort_owned = not isinstance(exc, Exception) or current is None or bool(current.report["prompt_id"])
        if current is not None and not current.handled:
            failed(current, exc)
        raise
    finally:
        # Preserve earlier in-flight results on a failed POST. Only remove a later pending sibling.
        try:
            # Known ownership survives failed signing/writes and removal from active.
            for scene in scenes:
                if abort_owned and scene.report["prompt_id"] and not scene.handled:
                    failed(scene, primary_error)
                if scene.error is not None:
                    cleanup(scene)
            for scene in active:
                if scene.handled:
                    continue
                try:
                    earlier = scenes[:scenes.index(scene)]
                    if any(s.report["status"] != "completed" for s in earlier) and scene.discard_pending(client, cleanup_budget):
                        notify(scene)
                    else:
                        finish(scene)
                except BaseException as exc:
                    remember_interrupt(exc)
                    if not scene.handled:
                        failed(scene, exc)
                    cleanup(scene)
            for scene in scenes:
                cleanup(scene)
            for scene in scenes:
                if scene.report["status"] == "initializing":
                    try:
                        scene.record("not_submitted", completed_at_unix=time.time())
                        notify(scene)
                    except BaseException as exc:
                        if final_error is None:
                            final_error = exc
        finally:
            try:
                cleanup_status = _stop_after_run(owned, client, server_stopper, cleanup_budget)
                if cleanup_status:
                    first.record(first.report["status"], server_cleanup={"status": cleanup_status,
                                 "grace_seconds": CLEANUP_GRACE_SECONDS})
            except BaseException as exc:
                if final_error is None:
                    final_error = exc
        if primary_error is None and final_error is not None:
            raise final_error
    return [scene.report for scene in scenes]


def load_signed_run_record(path: str | Path, state: OperatorState, *, allow_rejected=False) -> dict:
    """Verify a signed run record and its binding to the containing directory."""
    record_path = Path(path).expanduser().resolve()
    try:
        envelope = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OwnershipRecordError(f"ownership record is unavailable: {exc}") from exc
    record = state.verify_record(envelope)
    for name in ("approval_hash", "endpoint", "run_directory"):
        if not isinstance(record.get(name), str) or not record[name]:
            raise OwnershipRecordError(f"ownership record has invalid {name}")
    if record.get("terminal_status") == "submission_rejected":
        if not allow_rejected or "prompt_id" in record:
            raise OwnershipRecordError("a rejected submission has no owned prompt")
    elif not isinstance(record.get("prompt_id"), str) or not record["prompt_id"]:
        raise OwnershipRecordError("ownership record has invalid prompt_id")
    if Path(record["run_directory"]).expanduser().resolve() != record_path.parent:
        raise OwnershipRecordError("ownership record directory binding is invalid")
    if "prediction_scene_index" in record:
        index = record["prediction_scene_index"]
        if type(index) is not int or not 1 <= index <= 20 or record_path.parent.name != f"{index:03d}":
            raise OwnershipRecordError("prediction ownership record scene binding is invalid")
    if record.get("remote_server") is True:
        if error := remote_endpoint_error(record["endpoint"], record.get("allow_insecure_http", False), record.get("token_env")):
            raise OwnershipRecordError(error)
    elif error := local_endpoint_error(record["endpoint"]):
        raise OwnershipRecordError(error)
    return record


def _initialize_run_directory(path: str | Path, plan: dict) -> Path:
    directory = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    ignore_private_directory(directory)
    create_json_exclusive(directory / "plan.json", plan)
    create_json_exclusive(directory / "api-prompt.json", plan["api_prompt"])
    create_json_exclusive(directory / "variant.json", plan["variant"])
    create_json_exclusive(directory / "profile.json", plan["profile"])
    return directory


def _ensure_server(
    client, profile, run_dir: Path, *, server_starter, server_stopper, sleep, deadline=None,
    cleanup_budget,
):
    try:
        budget = remaining_timeout(deadline)
        client.system_stats(**({"timeout": budget} if budget is not None else {}))
        return ServerOwnership(started_by_operator=False)
    except TimeoutError:
        if deadline is not None:
            raise
    except Exception:
        pass
    if profile.remote_server:
        raise ConnectionError("Remote ComfyUI did not respond to its system_stats request")
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("run timeout expired before server startup")
    stdout = (run_dir / "comfyui.stdout.log").open("x", encoding="utf-8")
    stderr = (run_dir / "comfyui.stderr.log").open("x", encoding="utf-8")
    try:
        ownership = server_starter(profile, stdout_log=stdout, stderr_log=stderr)
    finally:
        stdout.close()
        stderr.close()
    startup_deadline = time.monotonic() + profile.startup_timeout_seconds
    deadline = min(deadline, startup_deadline) if deadline is not None else startup_deadline
    last_error = None
    while time.monotonic() < deadline:
        try:
            client.system_stats(timeout=remaining_timeout(deadline))
            return ownership
        except Exception as exc:
            last_error = exc
            remaining = deadline - time.monotonic()
            if remaining > 0:
                sleep(min(0.1, remaining))
    _stop_after_run(ownership, client, server_stopper, cleanup_budget)
    raise TimeoutError(f"ComfyUI did not become reachable: {last_error}")


def _reported_files(entry, graph):
    reported = entry.get("outputs") if isinstance(entry, dict) else None
    if not isinstance(reported, dict):
        return
    for node_id, slots in reported.items():
        if node_id not in graph or not isinstance(slots, dict):
            continue
        for slot, files in slots.items():
            if graph[node_id]["class_type"] == "SaveVideo" and slot not in {"images", "video", "videos", "gifs"}:
                continue
            if isinstance(files, list):
                for file in files:
                    if isinstance(file, dict) and file.get("type") == "output":
                        yield node_id, file


def _collect_outputs(entry, output_directory: Path, *, graph, prefixes, expected,
                     primary_node=None, download=None, validate_video=None) -> list[dict]:
    if download is not None and not expected:
        return []
    root = output_directory.expanduser().resolve()
    sources = {}
    remote_files = {}
    for node_id, file in _reported_files(entry, graph):
        if download is not None and node_id not in prefixes:
            continue
        filename, subfolder = file.get("filename"), file.get("subfolder", "")
        if not isinstance(filename, str) or not isinstance(subfolder, str):
            continue
        folder = PureWindowsPath(subfolder)
        if (not filename or "/" in filename or "\\" in filename or filename in {".", ".."}
                or folder.anchor or ".." in folder.parts):
            if expected:
                raise ValueError("reported output must have a filename and a relative ComfyUI output subfolder")
            continue
        if download is not None and any(
                any(char in '<>:"|?*' or ord(char) < 32 for char in component)
                or component.endswith((" ", ".")) or PureWindowsPath(component).is_reserved()
                for component in (*folder.parts, filename)):
            raise ValueError("Remote ComfyUI reported an invalid video filename")
        path = root.joinpath(*folder.parts, filename).resolve()
        if not path.is_relative_to(root):
            if expected:
                raise ValueError(f"reported output is outside ComfyUI output directory: {path}")
            continue
        if download is not None:
            if path in remote_files and remote_files[path] != file:
                raise ValueError("Remote ComfyUI reported conflicting video paths")
            remote_files[path] = file
        elif not path.is_file():
            if expected:
                raise FileNotFoundError(f"reported output is unavailable: {path}")
            continue
        sources.setdefault(path, set()).add(node_id)
    prefix_paths = {key: (root / value).resolve() for key, value in prefixes.items()}
    primary = set()
    approved = set()
    primary_suffix = video_suffix(graph[primary_node]["inputs"]) if primary_node is not None else None
    for value in expected:
        pattern = (root / Path(value).expanduser()).resolve()
        prefix = max((p for p in prefix_paths.values() if p.parent == pattern.parent
                      and os.path.normcase(pattern.name).startswith(os.path.normcase(p.name))),
                     key=lambda p: len(p.name), default=None)
        if not pattern.suffix or pattern == prefix:
            pattern = Path(str(pattern) + "*")
        name = escape(prefix.name) + pattern.name[len(prefix.name):] if prefix else pattern.name
        owners = {key for key, path in prefix_paths.items() if path == prefix}
        matches = {path for path, nodes in sources.items() if path.parent == pattern.parent
                   and fnmatch(path.name, name) and (not owners or owners & nodes)}
        if not pattern.is_relative_to(root) or not matches:
            raise FileNotFoundError(f"expected output is unavailable: {value}")
        approved.update(matches)
        if primary_node in owners:
            primary.update(path for path in matches if primary_node in sources[path]
                           and path.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}
                           and (primary_suffix is None or path.suffix.lower() == primary_suffix))
    if primary_node is not None and len(primary) != 1:
        raise ValueError(f"expected one approved video from SaveVideo node {primary_node}, found {len(primary)}")
    for path in primary:
        if sources[path] != {primary_node}:
            raise ValueError(f"multiple output nodes reported the primary video: {path}")
    records = []
    for path in sorted(approved if download is not None else sources):
        if download is not None:
            download(remote_files[path], path)
        elif validate_video is not None and path in approved and sources[path].intersection(prefixes):
            validate_video(path)
        record = {"path": str(path), "content_sha256": sha256_file(path)}
        if path in primary:
            record["primary"] = True
        records.append(record)
    return records


_VIDEO_DEMUXERS = {".mp4": "mov", ".mov": "mov", ".mkv": "matroska", ".webm": "matroska"}


def _download_clip(client, prompt_id, file, path, deadline):
    demuxer = _VIDEO_DEMUXERS.get(path.suffix.lower())
    if demuxer is None:
        raise ValueError("Remote ComfyUI did not report a supported video container")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{uuid4().hex}.part")
    try:
        for attempt in range(2):
            try:
                with temporary.open("xb") as output:
                    client.download_output(prompt_id, file["filename"], file.get("subfolder", ""), output,
                                           timeout=remaining_timeout(deadline))
                break
            except (ConnectionError, TimeoutError, HTTPException):
                if attempt or (deadline is not None and time.monotonic() >= deadline):
                    raise
                temporary.unlink(missing_ok=True)
        _validate_clip(temporary, deadline, demuxer=demuxer)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_clip(path, deadline, *, demuxer=None):
    demuxer = demuxer or _VIDEO_DEMUXERS.get(path.suffix.lower())
    if demuxer is None:
        raise ValueError("ComfyUI did not report a supported video container")
    remaining = remaining_timeout(deadline)
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-abort_on", "empty_output_stream",
         "-protocol_whitelist", "file", "-f", demuxer, "-i", str(path),
         "-map", "0:V", "-map", "0:a?", "-f", "null", "-"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        timeout=remaining if remaining is not None else 120,
    )
    if result.returncode:
        raise ValueError("ComfyUI did not return a valid video")
