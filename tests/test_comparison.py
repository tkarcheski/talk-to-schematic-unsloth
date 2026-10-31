"""CPU-only comparison transport and privacy contracts; no model calls."""
# cspell:words delenv

import copy
import json
import threading
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from schematic_model.comparison import ComparisonRuntime
from schematic_model.deployment import DeploymentConfig, make_server
from schematic_model.inference import InferenceError


def config():
    return {"profiles": [
        {"id": "baseline", "label": "Baseline", "endpoint": "http://127.0.0.1:8892/v1", "model": "baseline"},
        {"id": "candidate", "label": "Candidate", "endpoint": "http://127.0.0.1:8894/v1", "model": "candidate"},
    ], "max_tokens": 64}


def request():
    return {"profile": "baseline", "messages": [{"role": "user", "content": "Read R1."}]}


def answer():
    return {"model": "baseline", "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "R1 is 10k."}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "model_details": {"adapter_sha256": "a" * 64, "private_path": "/private/models"}}


def test_profile_config_is_private_and_request_is_frozen(monkeypatch):
    settings = config()
    settings["profiles"][0].update(api_key_env="COMPARISON_TEST_KEY", expected_adapter_sha256="a" * 64)
    monkeypatch.setenv("COMPARISON_TEST_KEY", "test-only-secret")
    calls = []
    def transport(url, payload, key):
        calls.append((url, copy.deepcopy(payload), key))
        payload["messages"][0]["content"] = "Changed by transport"
        return answer()
    runtime = ComparisonRuntime(settings, transport)
    described = json.dumps(runtime.describe())
    assert "endpoint" not in described and "api_key_env" not in described and "test-only-secret" not in described
    payload = request()
    result = runtime.complete(payload)
    assert payload == request()
    assert calls[0][0] == "http://127.0.0.1:8892/v1/chat/completions"
    assert calls[0][1] == {"model": "baseline", "messages": request()["messages"], "temperature": 0, "max_tokens": 64, "stream": False}
    assert calls[0][2] == "test-only-secret"
    assert result["identity"] == "pinned_adapter_match"
    assert result["metrics"]["completion_tokens"] == 5
    assert result["metrics"]["timing_scope"] == "upstream_wall_time"
    assert "private_path" not in result["model_details"]


@pytest.mark.parametrize("change", [
    lambda c: c.update(unknown=True),
    lambda c: c.update(max_tokens=True),
    lambda c: c.update(max_tokens=5000),
    lambda c: c.update(profiles=[]),
    lambda c: c["profiles"][0].update(endpoint="https://example.com:443/v1"),
    lambda c: c["profiles"][0].update(endpoint="http://8.8.8.8:8080/v1"),
    lambda c: c["profiles"][0].update(endpoint="http://127.0.0.1:8080/v1?key=secret"),
    lambda c: c["profiles"][0].update(unknown=True),
    lambda c: c["profiles"][1].update(id="baseline"),
    lambda c: c["profiles"][1].update(endpoint="http://127.0.0.1:8892/v1", model="baseline"),
    lambda c: c["profiles"][0].update(api_key_env="bad name"),
    lambda c: c["profiles"][0].update(expected_adapter_sha256="wrong"),
])
def test_invalid_config_fails_closed(change):
    settings = config()
    change(settings)
    with pytest.raises(ValueError):
        ComparisonRuntime(settings)


@pytest.mark.parametrize("change", [
    lambda p: p.update(endpoint="http://127.0.0.1:1"),
    lambda p: p.update(profile="missing"),
    lambda p: p.update(profile=[]),
    lambda p: p.update(tools=[]),
    lambda p: p.update(max_tokens=4000),
    lambda p: p["messages"][0].update(tool_calls=[]),
    lambda p: p["messages"][0].update(content=[{"type": "image_url", "image_url": {"url": "http://127.0.0.1/private"}}]),
])
def test_browser_cannot_change_destination_permissions_or_read_files(change):
    payload = request()
    change(payload)
    def unexpected(*args):
        pytest.fail("Invalid request contacted upstream")
    with pytest.raises(ValueError):
        ComparisonRuntime(config(), unexpected).complete(payload)


@pytest.mark.parametrize("change", [
    lambda r: r.update(model="other-model"),
    lambda r: r.update(choices=[]),
    lambda r: r["choices"][0].update(finish_reason="length"),
    lambda r: r["choices"][0]["message"].update(tool_calls=[{"function": "show_label"}]),
    lambda r: r["choices"][0]["message"].update(content=""),
])
def test_incomplete_wrong_model_and_tool_responses_are_rejected(change):
    result = answer()
    change(result)
    with pytest.raises(InferenceError):
        ComparisonRuntime(config(), lambda *args: result).complete(request())


def test_identity_pin_missing_credential_and_reported_metrics(monkeypatch):
    settings = config()
    settings["profiles"][0]["expected_adapter_sha256"] = "b" * 64
    with pytest.raises(InferenceError, match="identity"):
        ComparisonRuntime(settings, lambda *args: answer()).complete(request())
    settings["profiles"][0]["api_key_env"] = "COMPARISON_MISSING_KEY"
    monkeypatch.delenv("COMPARISON_MISSING_KEY", raising=False)
    with pytest.raises(RuntimeError, match="credential"):
        ComparisonRuntime(settings, lambda *args: pytest.fail("Missing credential contacted upstream")).complete(request())
    response = answer()
    response.update(metrics={"inference_seconds": 2, "timing_scope": "model_inference_including_prefill"})
    response["choices"][0]["message"]["reasoning_content"] = "Provider-supplied text"
    result = ComparisonRuntime(config(), lambda *args: response).complete(request())
    assert result["metrics"]["tokens_per_second"] == 2.5
    assert result["reasoning_content"] == "Provider-supplied text"
    assert result["identity"] == "endpoint_reported"


def test_comparison_http_reuses_chat_lock_and_rejects_cross_origin():
    started, release = threading.Event(), threading.Event()
    def transport(*args):
        started.set()
        assert release.wait(3)
        return answer()
    runtime = ComparisonRuntime(config(), transport)
    engine = SimpleNamespace(config=DeploymentConfig(model=""), fingerprint="f" * 64, model=True)
    server = make_server(engine, 0, comparison=runtime)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    responses = []
    def post(path="/api/comparison/complete", origin=None):
        headers = {"Content-Type": "application/json"}
        if origin:
            headers["Origin"] = origin
        with urlopen(Request(base + path, json.dumps(request()).encode(), headers), timeout=4) as response:
            return json.load(response)
    try:
        assert json.load(urlopen(base + "/api/comparison/config"))["enabled"]
        first = threading.Thread(target=lambda: responses.append(post()))
        first.start()
        assert started.wait(2)
        assert json.load(urlopen(base + "/health"))["busy"]
        with pytest.raises(HTTPError) as busy:
            post()
        assert busy.value.code == 503
        with pytest.raises(HTTPError) as origin:
            post(origin="http://untrusted.invalid")
        assert origin.value.code == 400
        release.set()
        first.join(3)
        assert len(responses) == 1 and responses[0]["answer"] == "R1 is 10k."
        assert not json.load(urlopen(base + "/health"))["busy"]
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join()
