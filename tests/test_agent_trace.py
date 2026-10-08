"""Trace display must preserve actions without disclosing transport credentials."""
from schematic_model.agent_trace import message_trace
from tests.test_agent import answer, call, payload, runtime


def test_trace_removes_image_payload_and_marks_truncation():
    result = message_trace([{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,PRIVATE_IMAGE_BYTES"}},
        {"type": "text", "text": "x" * 130_000},
    ]}])
    assert "PRIVATE_IMAGE_BYTES" not in str(result)
    assert result[0]["content"][0]["media_type"] == "image/png"
    assert result[0]["content"][1]["truncated"] is True


def test_trace_records_actual_tool_and_returned_reasoning_only():
    reply = answer()
    reply["choices"][0]["message"]["reasoning_content"] = "Provider-returned explanation."
    agent, _ = runtime([call("calculate", {"operation": "add", "a": 1, "b": 2}), reply])
    result = agent.chat(payload())
    assert [event["kind"] for event in result["trace"]] == [
        "model_request", "model_response", "tool_call", "tool_result", "model_request", "model_response"]
    assert result["trace"][3]["data"]["result"] == {"value": 3}
    assert result["trace"][-1]["data"]["reasoning_content"] == "Provider-returned explanation."


def test_approval_trace_is_delta_and_never_contains_credential(monkeypatch):
    monkeypatch.setenv("TRACE_TEST_KEY", "private-credential-not-for-display")
    agent, _ = runtime([call("web_search", {"query": "public chip"}), answer()],
                       search_endpoint="http://127.0.0.1:8894", api_key_env="TRACE_TEST_KEY")
    first = agent.chat(payload())
    second = agent.chat({"approval_id": first["pending_approval"]["id"], "decision": "deny"})
    assert second["trace"][0]["kind"] == "approval"
    assert second["trace"][0]["data"]["decision"] == "deny"
    assert not {e["id"] for e in first["trace"]} & {e["id"] for e in second["trace"]}
    assert "private-credential-not-for-display" not in str(first) + str(second)
