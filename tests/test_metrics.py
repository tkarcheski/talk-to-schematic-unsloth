"""Inference timing is measured, bounded and contains no prompt or file paths."""
from types import SimpleNamespace

import pytest

from schematic_model.metrics import InferenceMetrics, native_model_details, safe_model_details
from schematic_model.agent import AgentRuntime, AgentConfig


def response(text="ready", calls=None, seconds=2):
    return {"choices": [{"finish_reason": "tool_calls" if calls else "stop", "message": {"role": "assistant", "content": text, **({"tool_calls": calls} if calls else {})}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "metrics": {"inference_seconds": seconds, "timing_scope": "model_inference_including_prefill"},
            "model_details": {"backend": "unsloth-library", "backend_version": "2026.1", "model_fingerprint": "a" * 64, "path": "/private/model"}}


def test_native_metadata_strips_sensitive_paths():
    engine = SimpleNamespace(config=SimpleNamespace(served_model_name="schematic-v1"), fingerprint="b" * 64,
        provenance={"backend": "unsloth-library", "packages": {"unsloth": "2026.1"}, "model": {"path": "/private/adapter", "files": [{"name": "adapter_model.safetensors", "sha256": "c" * 64}], "base_model": {"path": "/private/models/Qwen3.5-4B", "sha256": "d" * 64}}})
    result = native_model_details(engine)
    assert result["base_model_name"] == "Qwen3.5-4B"
    assert result["adapter_sha256"] == "c" * 64
    assert "/private" not in str(result)
    assert safe_model_details({"backend": "/private/path", "model_fingerprint": "not-a-hash", "prompt": "secret"}) == {}


def test_aggregate_measured_counts_and_unavailable_fields():
    metrics = InferenceMetrics()
    assert metrics.result()["tokens_per_second"] is None
    metrics.add({"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}, 2, "upstream_wall_time")
    metrics.add({"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}, 4, "upstream_wall_time")
    assert metrics.result()["tokens_per_second"] == 5
    assert metrics.result()["total_tokens"] == 45
    metrics.add({}, 2, "upstream_wall_time")
    assert metrics.result()["completion_tokens"] is None
    assert metrics.result()["tokens_per_second"] is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, 0, True, None])
def test_invalid_durations_do_not_fabricate_rate(bad):
    metrics = InferenceMetrics()
    metrics.add({"completion_tokens": 20}, bad, "model_inference_including_prefill")
    assert metrics.result()["tokens_per_second"] is None


def test_gateway_aggregates_all_model_steps_and_retains_safe_version():
    calls = [{"id": "one", "type": "function", "function": {"name": "calculate", "arguments": '{"operation":"add","a":1,"b":2}'}}]
    replies = [response(None, calls), response()]
    agent = AgentRuntime(AgentConfig("http://127.0.0.1:8000/v1", "test"), lambda *args: replies.pop(0))
    result = agent.chat({"model": "test", "messages": [{"role": "user", "content": "Add"}]})
    assert result["metrics"]["completion_tokens"] == 40
    assert result["metrics"]["inference_seconds"] == 4
    assert result["metrics"]["tokens_per_second"] == 10
    assert result["metrics"]["model_steps"] == 2
    assert agent.model_details["backend_version"] == "2026.1"
    assert "path" not in result["model_details"]


def test_approval_continuation_counts_only_new_model_work():
    calls = [{"id": "one", "type": "function", "function": {"name": "web_search", "arguments": '{"query":"datasheet"}'}}]
    replies = [response(None, calls), response()]
    agent = AgentRuntime(AgentConfig("http://127.0.0.1:8000/v1", "test", search_endpoint="http://127.0.0.1:8888"), lambda *args: replies.pop(0))
    first = agent.chat({"model": "test", "messages": [{"role": "user", "content": "Search"}]})
    second = agent.chat({"approval_id": first["pending_approval"]["id"], "decision": "deny"})
    assert first["metrics"]["completion_tokens"] == second["metrics"]["completion_tokens"] == 20
    assert second["metrics"]["aggregation"] == "response"


def test_external_provider_without_usage_reports_rate_unavailable():
    agent = AgentRuntime(AgentConfig("http://127.0.0.1:8000/v1", "test"), lambda *args: {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ready"}}]})
    result = agent.chat({"model": "test", "messages": [{"role": "user", "content": "Hi"}]})
    assert result["metrics"]["timing_scope"] == "upstream_wall_time"
    assert result["metrics"]["tokens_per_second"] is None
    assert result["metrics"]["inference_seconds"] is not None


@pytest.mark.parametrize("present,metadata", [
    (False, None), (True, None), (True, {}), (True, []),
    (True, {"adapter_sha256": "invalid", "model_fingerprint": "invalid"}),
    (True, {"backend_version": "2026.2"}),
])
def test_gateway_never_reuses_artifacts_when_response_metadata_is_unavailable(present, metadata):
    first = response()
    first["model_details"].update(adapter_sha256="b" * 64, base_model_sha256="c" * 64)
    second = response()
    if present:
        second["model_details"] = metadata
    else:
        second.pop("model_details")
    replies = [first, second]
    agent = AgentRuntime(AgentConfig("http://127.0.0.1:8000/v1", "test"), lambda *args: replies.pop(0))
    request = {"model": "test", "messages": [{"role": "user", "content": "Review the circuit."}]}
    previous = agent.chat(request)
    current = agent.chat(request)
    expected = {"served_model_name": "test", "backend": "self_hosted", **safe_model_details(metadata)}
    assert current["model_details"] == agent.model_details == expected
    assert current["trace"][-1]["data"]["model_details"] == expected
    assert previous["model_details"]["adapter_sha256"] == "b" * 64
    assert previous["model_details"]["base_model_sha256"] == "c" * 64


def test_gateway_reports_changed_artifact_without_retaining_old_metadata():
    first, second = response(), response()
    first["model_details"]["adapter_sha256"] = "b" * 64
    second["model_details"] = {"model_fingerprint": "c" * 64}
    replies = [first, second]
    agent = AgentRuntime(AgentConfig("http://127.0.0.1:8000/v1", "test"), lambda *args: replies.pop(0))
    request = {"model": "test", "messages": [{"role": "user", "content": "Review the circuit."}]}
    agent.chat(request)
    current = agent.chat(request)
    assert current["model_details"] == {"served_model_name": "test", "backend": "self_hosted",
                                         "model_fingerprint": "c" * 64}


def test_metadata_tracks_each_model_step_in_a_single_request():
    calls = [{"id": "one", "type": "function", "function": {
        "name": "calculate", "arguments": '{"operation":"add","a":1,"b":2}'}}]
    first, second = response(None, calls), response()
    second.pop("model_details")
    replies = [first, second]
    agent = AgentRuntime(AgentConfig("http://127.0.0.1:8000/v1", "test"), lambda *args: replies.pop(0))
    result = agent.chat({"model": "test", "messages": [{"role": "user", "content": "Add"}]})
    steps = [event["data"]["model_details"] for event in result["trace"] if event["kind"] == "model_response"]
    assert steps[0]["model_fingerprint"] == "a" * 64
    assert steps[1] == result["model_details"] == {"served_model_name": "test", "backend": "self_hosted"}


def test_viewer_approval_retains_original_identity_after_another_request():
    calls = [{"id": "one", "type": "function", "function": {
        "name": "show_label", "arguments": '{"label":"U1"}'}}]
    proposed, unrelated = response(None, calls), response()
    unrelated["model_details"] = {"model_fingerprint": "b" * 64}
    replies = [proposed, unrelated]
    agent = AgentRuntime(AgentConfig("http://127.0.0.1:8000/v1", "test"), lambda *args: replies.pop(0))
    request = {"model": "test", "messages": [{"role": "user", "content": "Show U1."}]}
    first = agent.chat(request)
    agent.chat(request)
    released = agent.chat({"approval_id": first["pending_approval"]["id"], "decision": "allow"})
    assert released["metrics"]["model_steps"] == 0
    assert released["model_details"] == first["model_details"]
    assert released["model_details"]["model_fingerprint"] == "a" * 64
    assert agent.model_details["model_fingerprint"] == "b" * 64


def test_rejected_response_cannot_publish_artifact_metadata():
    good, incomplete = response(), response()
    incomplete["choices"][0]["finish_reason"] = "length"
    incomplete["model_details"] = {"model_fingerprint": "b" * 64}
    replies = [good, incomplete]
    agent = AgentRuntime(AgentConfig("http://127.0.0.1:8000/v1", "test"), lambda *args: replies.pop(0))
    request = {"model": "test", "messages": [{"role": "user", "content": "Review the circuit."}]}
    first = agent.chat(request)
    with pytest.raises(ValueError, match="finish reason"):
        agent.chat(request)
    assert agent.model_details == first["model_details"]
