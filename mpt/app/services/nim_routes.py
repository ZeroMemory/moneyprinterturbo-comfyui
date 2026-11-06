"""Resolve text inference URLs from NVIDIA's public model specifications."""
import functools
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from loguru import logger


NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
_CHAT = NVIDIA_NIM_BASE_URL + "/chat/completions"
_RESPONSES = NVIDIA_NIM_BASE_URL + "/responses"
_NGC = "https://api.ngc.nvidia.com/v2"
_MODEL_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*/[a-zA-Z0-9][a-zA-Z0-9_.-]*")


def _public_json(url):
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "MoneyPrinterTurbo"})
    with urllib.request.urlopen(request, timeout=5) as response:
        raw = response.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("NVIDIA catalog response is too large")
    return json.loads(raw)


def get_nvidia_model_options() -> list[tuple[str, str]]:
    """Intersect the live model IDs with NVIDIA's free chat listings."""
    try:
        ids = {}
        for item in _public_json(NVIDIA_NIM_BASE_URL + "/models")["data"]:
            model = item["id"]
            ids.setdefault(model.replace(".", "-"), []).append(model)
        models = set()
        for page in range(20):
            query = {"page": page, "pageSize": 100, "scoredSize": 0}
            url = _NGC + "/search/catalog/resources/ENDPOINT?" + urllib.parse.urlencode({"q": json.dumps(query)})
            catalog = _public_json(url)
            for group in catalog["results"]:
                if group.get("groupValue") != "ENDPOINT":
                    continue
                for card in group["resources"]:
                    if any(isinstance(item, dict) and item.get("key") == "DEPRECATION" and item.get("value")
                           for item in card.get("attributes") or ()):
                        continue
                    labels = {label["key"]: label.get("values", []) for label in card.get("labels", [])}
                    tags = {str(tag).casefold() for key in ("general", "nimType") for tag in labels.get(key, [])}
                    if {"chat", "free endpoint"} <= tags:
                        for publisher in labels.get("publisher", []):
                            name = f"{publisher}/{card['name']}".replace(".", "-")
                            models.update(ids.get(name, ()))
            if page + 1 >= catalog["resultPageTotal"]:
                return [(model, model) for model in sorted(models)]
        raise ValueError("NVIDIA catalog exceeded the page limit")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        logger.warning("NVIDIA model catalog is unavailable ({})", type(exc).__name__)
        return []


def _operation_route(spec, model):
    """Use only documented operations on the two NVIDIA inference hosts."""
    direct = "https://ai.api.nvidia.com/v1/vlm/" + model
    allowed = {_CHAT, _RESPONSES, direct}
    routes = {}
    for path, item in spec["paths"].items():
        if "post" not in item:
            continue
        op = item["post"]
        meta = op.get("x-nvai-meta", {})
        examples = set()
        templates = list(meta.get("templates", []))
        for group in meta.get("templateGroups", {}).values():
            templates.extend(group)
        for template in templates:
            curl = template.get("requestEjs", {}).get("curl", "")
            examples.update(u for u in re.findall(r"https://[\w./?%&=:#@+-]+", curl) if u in allowed)

        schema = op.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema", {})
        if "$ref" in schema:
            schema = spec.get("components", {}).get("schemas", {}).get(schema["$ref"].split("/")[-1], {})
        fields = schema.get("properties", {})
        model_field = fields.get("model", {})
        identifies_model = model_field.get("default") == model or model in model_field.get("enum", [])
        text_example = False
        for example in meta.get("examples", []):
            try:
                request = json.loads(example.get("requestJson", ""))
            except (TypeError, ValueError):
                continue
            if isinstance(request, dict):
                identifies_model |= request.get("model") == model
                messages = request.get("messages") or []
                if messages and all(isinstance(m, dict) and isinstance(m.get("content"), str)
                                    and m["content"].strip() and "<" not in m["content"] for m in messages):
                    text_example = True
        for server in op.get("servers", item.get("servers", spec.get("servers", []))):
            base = server["url"].rstrip("/")
            url = base + "/" + path.lstrip("/")
            if (url == direct and direct in examples and "messages" in fields and text_example
                    and set(schema.get("required", ())) <= {"messages", "model"}):
                routes[direct] = "model" in fields
            if not identifies_model or base not in (NVIDIA_NIM_BASE_URL, "https://integrate.api.nvidia.com"):
                continue
            # Some cards give an obsolete path but publish the working chat URL in the same operation's cURL.
            if "messages" in fields and (_CHAT in examples or (url == _CHAT and not templates)):
                routes[_CHAT] = True
            if url == _RESPONSES and "input" in fields and (not templates or _RESPONSES in examples):
                routes[_RESPONSES] = True
    for url in (_RESPONSES, direct, _CHAT):
        if url in routes:
            return url, routes[url]
    raise ValueError("NVIDIA does not document a supported text endpoint for this model")


@functools.lru_cache(maxsize=128)
def _hosted_route(model, cache_period):
    publisher, name = model.split("/")
    for slug in dict.fromkeys((name, name.replace(".", "-"), publisher + "-" + name)):
        try:
            card = _public_json(f"{_NGC}/endpoints/qc69jvmznzxy/{slug}/spec")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                continue
            raise
        if card.get("artifactName") != slug or card.get("artifactType") != "ENDPOINT":
            raise ValueError("Unexpected NVIDIA model specification")
        if json.loads(card.get("attributes") or "{}").get("deprecationNotice"):
            raise ValueError("NVIDIA marks this model endpoint as deprecated")
        return _operation_route(json.loads(card["openAPISpec"]), model)
    raise ValueError("NVIDIA does not publish a specification for this model ID")


def is_nvidia_hosted_endpoint(value):
    """Recognize saved NVIDIA base and operation addresses, including a pasted bare host."""
    if not value or not value.strip():
        return True
    base = value.strip().rstrip("/")
    if "://" not in base:
        base = "https://" + base
    if base in {NVIDIA_NIM_BASE_URL, NVIDIA_NIM_BASE_URL.removesuffix("/v1"), _CHAT, _RESPONSES}:
        return True
    prefix = "https://ai.api.nvidia.com/v1/vlm/"
    return base.startswith(prefix) and bool(_MODEL_ID.fullmatch(base[len(prefix):]))


def nvidia_api_key_setting(base_url):
    if is_nvidia_hosted_endpoint(base_url):
        return "nvidia_api_key"
    endpoint = base_url.strip().rstrip("/")
    return f"nvidia_custom_{hashlib.sha256(endpoint.encode()).hexdigest()}_api_key"


def resolve_nvidia_api_key(app_config, base_url):
    setting = nvidia_api_key_setting(base_url)
    key = (app_config.get(setting) or "").strip()
    if setting == "nvidia_api_key":
        return key or os.getenv("NVIDIA_API_KEY", "").strip()
    return key


def nvidia_chat_route(model, base_url=None):
    base = (base_url or NVIDIA_NIM_BASE_URL).strip().rstrip("/")
    if not is_nvidia_hosted_endpoint(base):
        parts = urllib.parse.urlsplit(base)
        if (parts.scheme not in {"http", "https"} or not parts.hostname or parts.username is not None
                or parts.password is not None or parts.query or parts.fragment):
            raise ValueError("Enter an HTTP(S) server base URL without credentials, query or fragment")
        return base + "/chat/completions", True
    if not isinstance(model, str) or not _MODEL_ID.fullmatch(model):
        raise ValueError("Choose an NVIDIA model ID in publisher/model format")
    try:
        return _hosted_route(model, int(time.monotonic() // 600))
    except (OSError, KeyError, TypeError, AttributeError, json.JSONDecodeError) as exc:
        raise ValueError("Could not read the model's public NVIDIA endpoint specification") from exc


def nvidia_chat_url(model, base_url=None):
    return nvidia_chat_route(model, base_url)[0]


def clear_nvidia_routes():
    _hosted_route.cache_clear()
