import json
import time
from io import BytesIO
from threading import Barrier
from urllib.error import URLError
from urllib.parse import parse_qs, urlsplit

import httpx
from openai import OpenAI
import streamlit as st
from streamlit.testing.v1 import AppTest
from app import paths
from app.config import config
from app.services import llm


def _open_llm_settings(app):
    app.button(key="open_settings_dialog_button").click().run()
    if any(button.key == "settings_general_configure_llm" for button in app.button):
        app.button(key="settings_general_configure_llm").click().run()


def test_nim_only_lists_live_free_chat_endpoints_and_reads_all_pages(monkeypatch):
    def card(name, tags, publisher="sample"):
        return {"resourceType": "ENDPOINT", "name": name, "displayName": name.title(),
                "labels": [{"key": "general", "values": tags},
                           {"key": "publisher", "values": [publisher]}]}

    live = ["sample/new", "sample/paid", "sample/embedding", "other/new", "sample/next", "sample/retiring"]
    cards = [[card("new", ["Free Endpoint", "Chat"]), card("paid", ["chat"]),
              card("embedding", ["Free Endpoint", "Text-to-Embedding"]),
              card("delisted", ["Free Endpoint", "chat"])],
             [card("next", ["chat", "Free Endpoint"]), card("new", ["Free Endpoint", "chat"]),
              {**card("retiring", ["chat", "Free Endpoint"]),
               "attributes": [{"key": "DEPRECATION", "value": "12/31/2030"}]}]]
    seen = []

    def request(req, timeout):
        url = urlsplit(req.full_url)
        assert url.scheme == "https" and timeout > 0
        assert req.get_header("Authorization") is None
        seen.append(url.hostname)
        if url.hostname == "integrate.api.nvidia.com":
            assert url.path == "/v1/models"
            result = {"data": [{"id": name} for name in live]}
        else:
            assert url.hostname == "api.ngc.nvidia.com"
            query = json.loads(parse_qs(url.query)["q"][0])
            result = {"resultPageTotal": 2, "results": [
                {"groupValue": "ENDPOINT", "resources": cards[query["page"]]}]}
        return BytesIO(json.dumps(result).encode())

    monkeypatch.setattr("urllib.request.urlopen", request)
    assert sorted(value for _, value in llm.get_nvidia_model_options()) == ["sample/new", "sample/next"]
    assert seen.count("api.ngc.nvidia.com") == 2


def test_public_nim_catalog_failure_does_not_present_unverified_ids_as_free(monkeypatch):
    def offline(*args, **kwargs):
        raise URLError("offline")

    monkeypatch.setattr("urllib.request.urlopen", offline)
    assert llm.get_nvidia_model_options() == []


def test_check_all_models_reaches_each_hosted_route_only_on_click(monkeypatch):
    chat, responses = "sample/chat", "sample/responses"
    urls = {model: llm.NVIDIA_NIM_BASE_URL + "/chat/completions"
            for model in (chat, "sample/chat-b", "sample/chat-c")}
    urls[responses] = llm.NVIDIA_NIM_BASE_URL + "/responses"
    all_started = Barrier(len(urls), timeout=10)
    sent = []

    def serve(request):
        body = json.loads(request.content)
        model = body["model"]
        assert request.method == "POST" and str(request.url) == urls[model]
        assert request.headers["authorization"] == "Bearer test-catalog-key"
        sent.append(model)
        all_started.wait()  # Each model must start before the first answer arrives.
        if model == responses:
            assert body["input"][0]["content"][0]["type"] == "input_text"
            return httpx.Response(429, json={"error": {"message": "busy"}})
        assert body["messages"][0]["role"] == "user"
        return httpx.Response(200, json={"id": "local", "object": "chat.completion", "model": model,
            "created": 1, "choices": [{"index": 0, "finish_reason": "stop",
                                       "message": {"role": "assistant", "content": "OK"}}]})

    monkeypatch.setattr(llm, "OpenAI", lambda **kw: OpenAI(**kw, http_client=httpx.Client(
        transport=httpx.MockTransport(serve))))
    monkeypatch.setattr(llm, "get_nvidia_model_options", lambda: [(m, m) for m in urls])
    monkeypatch.setattr(llm, "nvidia_chat_url", lambda model, base: urls[model])
    monkeypatch.setattr(llm, "nvidia_chat_route", lambda model, *a, **kw: (urls[model], True))
    monkeypatch.setattr(config, "app", {**config.app, "llm_provider": "nvidia",
                        "nvidia_model_name": responses, "nvidia_base_url": "", "nvidia_api_key": "test-catalog-key"})
    monkeypatch.setattr(config, "ui", {**config.ui, "language": "en"})
    monkeypatch.setattr(config, "save_config", lambda: None)
    st.cache_data.clear()
    try:
        app = AppTest.from_file(str(paths.SOURCE_ROOT / "webui/Main.py"), default_timeout=45).run()
        _open_llm_settings(app)
        assert not app.exception and sent == []
        app.button(key="probe_nvidia_catalog").click().run()
        deadline = time.monotonic() + 20
        while True:
            assert not app.exception
            rows = next((frame.value for frame in app.dataframe if "Latency (ms)" in frame.value.columns), None)
            if rows is not None and len(rows) == len(urls):
                break
            assert time.monotonic() < deadline, "catalog results did not finish"
            time.sleep(0.1)
            app.run()  # AppTest does not schedule the browser's periodic fragment refresh.
        assert sorted(sent) == sorted(urls)
        result = rows.set_index("Model Name")["Status"]
        assert result[chat].startswith("Text received")
        assert result[responses].startswith("Rate limited") and "429" in result[responses]
        assert app.selectbox(key="nvidia_model_name_select").value == responses
        app.run()
        assert sorted(sent) == sorted(urls)
    finally:
        st.cache_data.clear()


def test_nim_settings_refresh_preserves_a_saved_id_and_bypasses_private_servers(monkeypatch):
    choices = ["sample/chat", "sample/shared"]
    calls = []
    gateway = "https://integrate.api.nvidia.com/v1/chat/completions"
    dedicated = "https://ai.api.nvidia.com/v1/sample/chat"
    custom = "http://127.0.0.1:8188/v1"
    refreshes = []
    resolve_url = llm.nvidia_chat_url

    def fetch():
        calls.append(1)
        return [(choice, choice) for choice in choices]

    def chat_url(model, base_url):
        if base_url:
            return resolve_url(model, base_url)
        if model == "sample/manual":
            raise ValueError("private error detail")
        return dedicated if model == "sample/chat" else gateway

    monkeypatch.setattr(llm, "get_nvidia_model_options", fetch)
    monkeypatch.setattr(llm, "nvidia_chat_url", chat_url)
    monkeypatch.setattr(llm, "clear_nvidia_routes", lambda: refreshes.append(1))
    monkeypatch.setattr(config, "app", {**config.app, "llm_provider": "nvidia", "nvidia_api_key": "saved-test-key",
                        "nvidia_model_name": "sample/manual", "nvidia_base_url": ""})
    monkeypatch.setattr(config, "ui", {**config.ui, "language": "en"})
    monkeypatch.setattr(config, "save_config", lambda: None)
    st.cache_data.clear()
    try:
        app = AppTest.from_file(str(paths.SOURCE_ROOT / "webui/Main.py"), default_timeout=45).run()
        _open_llm_settings(app)
        model = app.selectbox(key="nvidia_model_name_select")
        assert model.options == ["sample/manual", "sample/chat", "sample/shared"]
        assert model.value == "sample/manual" and len(calls) == 1
        warnings = [block.value for block in app.warning]
        assert "Could not resolve a chat endpoint for this model." in warnings
        assert not any("private error detail" in message for message in warnings)
        assert not app.button(key="test_llm_connection_button").disabled
        model.select("sample/shared").run()
        assert gateway in [block.value for block in app.code]
        app.selectbox(key="nvidia_model_name_select").select("sample/chat")
        app.text_input(key="nvidia_api_key_input").set_value("edited-test-key")
        app.text_input(key="nvidia_base_url_custom_input").set_value(custom + "?token=private-value").run()
        assert not app.exception and app.button(key="test_llm_connection_button").disabled
        assert config.app["nvidia_base_url"] == ""
        assert config.app["nvidia_model_name"] == "sample/shared"
        assert config.app["nvidia_api_key"] == "saved-test-key"
        assert not any("private-value" in block.value for block in app.warning)
        app.selectbox(key="nvidia_model_name_select").select("sample/shared")
        app.text_input(key="nvidia_base_url_custom_input").set_value("").run()
        assert not app.button(key="test_llm_connection_button").disabled
        app.text_input(key="nvidia_api_key_input").set_value("edited-test-key").run()
        assert config.app["nvidia_api_key"] == "edited-test-key"
        app.selectbox(key="nvidia_model_name_select").select("sample/chat").run()
        assert dedicated in [block.value for block in app.code]
        assert gateway not in [block.value for block in app.code]
        assert config.app["nvidia_base_url"] == ""
        assert config.app["nvidia_model_name"] == "sample/chat"
        assert len(calls) == 1
        choices.clear()
        app.button(key="refresh_nvidia_catalog").click().run()
        assert not app.exception
        model = app.selectbox(key="nvidia_model_name_select")
        assert model.options == ["sample/chat"] and model.value == "sample/chat"
        assert len(calls) == 2
        assert len(refreshes) == 1
        app.text_input(key="nvidia_base_url_custom_input").set_value(custom).run()
        assert not app.exception and len(calls) == 2
        assert config.app["nvidia_base_url"] == custom
        assert "refresh_nvidia_catalog" not in [widget.key for widget in app.button]
        assert custom + "/chat/completions" in [block.value for block in app.code]
        assert "nvidia_api_key_input" not in [widget.key for widget in app.text_input]
        custom_key = next(widget for widget in app.text_input if widget.key.startswith("nvidia_custom_"))
        assert custom_key.value == ""
        custom_key.set_value("custom-test-key").run()
        assert config.app[custom_key.key.removesuffix("_input")] == "custom-test-key"
        app.text_input(key="nvidia_base_url_custom_input").set_value(custom + "/other").run()
        other_key = next(widget for widget in app.text_input if widget.key.startswith("nvidia_custom_"))
        assert other_key.key != custom_key.key and other_key.value == ""
        app.text_input(key="nvidia_base_url_custom_input").set_value(custom).run()
        # Reopening with an existing custom server must skip even an uncached catalog.
        st.cache_data.clear()
        app = AppTest.from_file(str(paths.SOURCE_ROOT / "webui/Main.py"), default_timeout=45).run()
        _open_llm_settings(app)
        assert app.selectbox(key="nvidia_model_name_select").value == "sample/chat"
        assert app.text_input(key="nvidia_base_url_custom_input").value == custom
        assert app.text_input(key=custom_key.key).value == "custom-test-key"
        assert not app.exception and len(calls) == 2
        app.text_input(key="nvidia_base_url_custom_input").set_value("").run()
        assert not app.exception and len(calls) == 3
        assert dedicated in [block.value for block in app.code]
        assert config.app["nvidia_base_url"] == ""
        assert app.text_input(key="nvidia_api_key_input").value == "edited-test-key"
    finally:
        st.cache_data.clear()
