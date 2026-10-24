"""Long-context budgets and request-scoped cancellation without GPU imports."""
import copy
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from schematic_model.agent import AgentConfig, AgentRuntime, GatewayEngine
from schematic_model.deployment import DeploymentConfig, make_server, validate_request
from schematic_model.generation_jobs import GenerationCancelled, GenerationJobs, check_cancelled


def answer():
    return {"choices": [{"message": {"role": "assistant", "content": "Ready"}, "finish_reason": "stop"}]}


def request():
    return {"model": "test", "messages": [{"role": "user", "content": "Read the evidence."}]}


def test_gateway_profile_reaches_upstream_and_health_without_legacy_output_cap():
    seen = []
    def transport(url, data=None, key=None):
        seen.append(data)
        return answer()
    config = AgentConfig("http://127.0.0.1:8000/v1", "test", max_seq=131072,
                         max_tokens=4096, max_text_chars=2_000_000, max_messages=256)
    runtime = AgentRuntime(config, transport)
    payload = request()
    payload["messages"][0]["content"] = "source fact\n" * 20000
    runtime.chat(payload)
    assert seen[0]["max_tokens"] == 4096
    assert seen[0]["messages"][-1]["content"] == payload["messages"][0]["content"]
    assert GatewayEngine(runtime).config.max_seq == 131072
    payload["max_tokens"] = 4097
    with pytest.raises(ValueError, match="generation limit"):
        runtime.chat(payload)
    assert len(seen) == 1


@pytest.mark.parametrize("options", [{"max_seq": 262145}, {"max_seq": True},
    {"max_seq": 8192, "max_tokens": 8192}, {"max_images": 33},
    {"max_messages": 513}, {"max_text_chars": 4000001}, {"request_timeout": float("inf")},
    {"backend": "cloud"}, {"reasoning_effort": "unbounded"}])
def test_profile_rejects_unknown_or_unbounded_limits(options):
    with pytest.raises(ValueError):
        AgentConfig("http://127.0.0.1:8000/v1", "test", **options)


def test_native_256k_configuration_remains_bounded(tmp_path):
    config = DeploymentConfig(model=str(tmp_path), max_seq=262144, max_tokens=8192,
                              max_text_chars=2_000_000, max_messages=256)
    config.validate()
    payload = {**request(), "model": "schematic"}
    payload["messages"][0]["content"] = "x" * 200001
    assert validate_request(payload, config)["max_tokens"] == 8192
    with pytest.raises(ValueError, match="text size"):
        validate_request(payload, DeploymentConfig(model=str(tmp_path)))


@pytest.mark.parametrize("status", [
    {"active_model": "different", "context_length": 131072},
    {"active_model": "test", "context_length": 8192},
    {"active_model": "test", "context_length": 131072, "loading": ["another"]},
])
def test_studio_never_completes_with_wrong_resident_model_or_context(status):
    seen = []
    def transport(url, data=None, key=None):
        seen.append(url)
        return status
    agent = AgentRuntime(AgentConfig("http://127.0.0.1:8888/api/inference", "test",
                                    backend="studio", max_seq=131072), transport)
    with pytest.raises(ValueError):
        agent.chat(request())
    assert seen == [agent.config.endpoint + "/status"]


def test_studio_cancellation_targets_only_its_own_generation():
    entered, released, stop = threading.Event(), threading.Event(), threading.Event()
    seen = []
    def transport(url, data=None, key=None):
        seen.append((url, copy.deepcopy(data)))
        if url.endswith("/status"):
            return {"active_model": "test", "context_length": 131072}
        if url.endswith("/cancel"):
            released.set()
            return {"cancelled": 1}
        entered.set()
        assert released.wait(3)
        return answer()
    agent = AgentRuntime(AgentConfig("http://127.0.0.1:8888/api/inference", "test",
                                    backend="studio", max_seq=131072), transport)
    errors = []
    def run():
        try:
            agent.chat(request(), cancel_event=stop)
        except GenerationCancelled as error:
            errors.append(error)
    worker = threading.Thread(target=run)
    worker.start()
    assert entered.wait(3)
    stop.set()
    worker.join(3)
    assert not worker.is_alive() and len(errors) == 1
    completion = next(data for url, data in seen if url.endswith("/chat/completions"))
    cancelled = next(data for url, data in seen if url.endswith("/cancel"))
    assert cancelled == {"cancel_id": completion["cancel_id"]}
    assert len(completion["cancel_id"]) == 32
    assert completion["context_overflow"] == "error"
    assert completion["enable_tools"] is False
    assert completion["enabled_tools"] == []
    assert completion["mcp_enabled"] is False
    assert completion["auto_heal_tool_calls"] is False
    assert completion["nudge_tool_calls"] is False


def test_studio_refuses_silently_truncated_evidence():
    def transport(url, data=None, key=None):
        if url.endswith("/status"):
            return {"active_model": "test", "context_length": 131072}
        return {**answer(), "context_truncated": True}
    agent = AgentRuntime(AgentConfig("http://127.0.0.1:8888/api/inference", "test",
                                    backend="studio", max_seq=131072), transport)
    with pytest.raises(ValueError, match="truncated"):
        agent.chat(request())


def test_cancel_before_registration_is_scoped_expiring_and_bounded(monkeypatch):
    jobs = GenerationJobs()
    token = "a" * 32
    jobs.cancel(token)
    assert jobs.start(token).is_set()
    assert not jobs.start("b" * 32).is_set()
    jobs.finish(token)
    for index in range(32):
        jobs.cancel(f"{index:032x}")
    with pytest.raises(ValueError, match="limit"):
        jobs.cancel("c" * 32)
    monkeypatch.setattr("schematic_model.generation_jobs.time.monotonic", lambda: float("inf"))
    assert not jobs.start("0" * 32).is_set()


def test_http_stop_releases_busy_only_after_worker_exits():
    entered = threading.Event()
    class Engine:
        config = DeploymentConfig(model="", served_model_name="test")
        fingerprint = "test"
        model = True
        def complete(self, model, messages, *, cancel_event, **options):
            entered.set()
            assert cancel_event.wait(3)
            check_cancelled(cancel_event)
    server = make_server(Engine(), 0)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    base = f"http://127.0.0.1:{server.server_port}"
    errors = []
    def post(path, payload):
        with urlopen(Request(base + path, data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"}), timeout=5) as response:
            return json.load(response)
    def run():
        try:
            post("/v1/chat/completions", {**request(), "request_id": "a" * 32})
        except HTTPError as error:
            errors.append((error.code, json.load(error)))
    worker = threading.Thread(target=run)
    try:
        worker.start()
        assert entered.wait(3)
        assert post("/api/jobs/cancel", {"request_id": "b" * 32})["cancel_requested"] is True
        assert worker.is_alive()
        with urlopen(base + "/health") as response:
            assert json.load(response)["busy"] is True
        assert post("/api/jobs/cancel", {"request_id": "a" * 32})["cancel_requested"] is True
        worker.join(3)
        assert not worker.is_alive()
        assert errors[0][0] == 409
        assert errors[0][1]["error"]["code"] == "cancelled"
        with urlopen(base + "/health") as response:
            assert json.load(response)["busy"] is False
    finally:
        server.shutdown()
        server.server_close()
        serving.join()


def test_compose_gateway_stop_reaches_native_generation():
    entered, exited = threading.Event(), threading.Event()
    class Engine:
        config = DeploymentConfig(model="", served_model_name="test")
        fingerprint = "test"
        model = True
        def complete(self, model, messages, *, cancel_event, **options):
            entered.set()
            try:
                assert cancel_event.wait(3)
                check_cancelled(cancel_event)
            finally:
                exited.set()

    native = make_server(Engine(), 0)
    runtime = AgentRuntime(AgentConfig(f"http://127.0.0.1:{native.server_port}/v1", "test",
                                      backend="native", permission={"*": "deny"}))
    gateway = make_server(GatewayEngine(runtime), 0, agent=runtime)
    threads = [threading.Thread(target=server.serve_forever, daemon=True) for server in (native, gateway)]
    for thread in threads:
        thread.start()
    origin = f"http://127.0.0.1:{gateway.server_port}"
    errors = []
    def post(path, payload):
        with urlopen(Request(origin + path, data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"}), timeout=5) as response:
            return json.load(response)
    def run():
        try:
            post("/api/agent/chat", {**request(), "request_id": "c" * 32})
        except HTTPError as error:
            errors.append((error.code, json.load(error)))
    worker = threading.Thread(target=run)
    try:
        worker.start()
        assert entered.wait(3)
        assert post("/api/jobs/cancel", {"request_id": "c" * 32})["cancel_requested"]
        worker.join(3)
        assert not worker.is_alive() and exited.is_set()
        assert errors[0][0] == 409
        assert errors[0][1]["error"]["code"] == "cancelled"
        with urlopen(origin + "/health") as response:
            health = json.load(response)
        assert health["busy"] is False and health["cancellation"] == "cooperative"
    finally:
        for server in (gateway, native):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join()


@pytest.mark.parametrize("status", [None, {},
    {"provider": "embedded", "model": "wrong", "loaded": True, "busy": False},
    {"provider": "self_hosted", "model": "test", "loaded": True, "busy": False},
    {"provider": "embedded", "model": "test", "loaded": 1, "busy": False},
    {"provider": "embedded", "model": "test", "loaded": True, "busy": "false"},
])
def test_native_health_malformed_or_mismatched_disables_ready(status):
    runtime = AgentRuntime(AgentConfig("http://127.0.0.1:8000/v1", "test", backend="native"),
                           lambda *args: status)
    assert runtime.native_health() == {"loaded": False, "busy": False, "readiness": "native_unavailable"}


def test_native_health_unavailable_disables_ready():
    def transport(*args):
        raise TimeoutError()
    runtime = AgentRuntime(AgentConfig("http://127.0.0.1:8000/v1", "test", backend="native"), transport)
    assert runtime.native_health()["loaded"] is False


def test_native_busy_survives_gateway_completion_timeout():
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    class Engine:
        config = DeploymentConfig(model="", served_model_name="test")
        fingerprint = "test"
        model = True
        def complete(self, model, messages, **options):
            entered.set()
            try:
                assert release.wait(5)
                return {"output": "Ready", "usage": {}, "elapsed_seconds": 0.1}
            finally:
                exited.set()
    native = make_server(Engine(), 0)
    runtime = AgentRuntime(AgentConfig(f"http://127.0.0.1:{native.server_port}/v1", "test",
                                      backend="native", request_timeout=1, permission={"*": "deny"}))
    gateway = make_server(GatewayEngine(runtime), 0, agent=runtime)
    threads = [threading.Thread(target=server.serve_forever, daemon=True) for server in (native, gateway)]
    for thread in threads:
        thread.start()
    origin = f"http://127.0.0.1:{gateway.server_port}"
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(Request(origin + "/api/agent/chat", data=json.dumps(request()).encode(),
                            headers={"Content-Type": "application/json"}), timeout=3)
        assert error.value.code == 503 and entered.is_set() and not exited.is_set()
        with urlopen(origin + "/health", timeout=2) as response:
            health = json.load(response)
        assert health["loaded"] is True and health["busy"] is True
        release.set()
        assert exited.wait(2)
        # The engine signal precedes the handler's lock release; wait for health.
        import time
        for _ in range(40):
            with urlopen(origin + "/health", timeout=2) as response:
                health = json.load(response)
            if not health["busy"]:
                break
            time.sleep(0.01)
        assert health["loaded"] is True and health["busy"] is False
        with pytest.raises(HTTPError) as probe:
            urlopen(Request(origin + "/health", headers={"X-Schematic-Native-Health": "1"}), timeout=2)
        assert probe.value.code == 409
    finally:
        release.set()
        for server in (gateway, native):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join()
