"""Agent privacy boundaries and deterministic approval lifecycle tests."""
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from schematic_model.agent import AgentConfig, AgentRuntime, GatewayEngine, private_endpoint
from schematic_model.deployment import make_server
from schematic_model.view_tools import VIEW_TOOLS


def call(name, arguments):
    return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}}]}


def answer(text="Ready"):
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def payload():
    return {"model": "test", "messages": [{"role": "user", "content": "What is the voltage?"}], "tools": VIEW_TOOLS}


def runtime(responses, **options):
    seen = []
    def transport(url, data=None, key=None):
        seen.append((url, data))
        return responses.pop(0)
    return AgentRuntime(AgentConfig("http://127.0.0.1:8000/v1", "test", **options), transport), seen


@pytest.mark.parametrize("url", ["https://api.example.com:443/v1", "http://8.8.8.8:8000", "http://169.254.169.254:80", "http://127.0.0.1:8000/v1?x=y", "http://user:password@127.0.0.1:8000", "http://0.0.0.0:8000", "file:///tmp/a"])
def test_endpoint_rejects_egress_metadata_and_credentials(url):
    with pytest.raises(ValueError):
        private_endpoint(url)


def test_search_requires_precise_single_use_approval_and_sends_only_query():
    agent, seen = runtime([call("web_search", {"query": "LM358 datasheet"}), {"results": [{"url": "https://example.com/chip", "title": "Chip", "content": "spec"}]}, answer()], search_endpoint="http://127.0.0.1:8888")
    result = agent.chat(payload())
    approval = result["pending_approval"]
    assert len(seen) == 1
    assert approval["arguments"] == {"query": "LM358 datasheet"}
    assert approval["destination"] == "http://127.0.0.1:8888"
    result = agent.chat({"approval_id": approval["id"], "decision": "allow"})
    assert seen[1] == ("http://127.0.0.1:8888/search?q=LM358+datasheet&format=json", None)
    assert result["events"][0]["status"] == "completed"
    assert result["messages"][-1]["role"] == "tool"
    with pytest.raises(ValueError):
        agent.chat({"approval_id": approval["id"], "decision": "allow"})


def test_denial_never_calls_search_and_missing_search_is_denied():
    agent, seen = runtime([call("web_search", {"query": "secret"}), answer()], search_endpoint="http://127.0.0.1:8888")
    token = agent.chat(payload())["pending_approval"]["id"]
    result = agent.chat({"approval_id": token, "decision": "deny"})
    assert len(seen) == 2
    assert all("/search?" not in url for url, _ in seen)
    assert result["events"][0]["status"] == "denied"
    agent, seen = runtime([call("web_search", {"query": "secret"}), answer()])
    assert agent.chat(payload())["events"][0]["status"] == "denied"
    assert all("/search?" not in url for url, _ in seen)


def test_approval_is_expiring_isolated_and_cannot_change_arguments(monkeypatch):
    agent, _ = runtime([call("web_search", {"query": "public"})], search_endpoint="http://127.0.0.1:8888")
    result = agent.chat(payload())["pending_approval"]
    other, _ = runtime([])
    with pytest.raises(ValueError):
        other.chat({"approval_id": result["id"], "decision": "allow"})
    with pytest.raises(ValueError):
        agent.chat({"approval_id": result["id"], "decision": "allow", "arguments": {"query": "secret"}})
    monkeypatch.setattr("schematic_model.agent.time.time", lambda: result["expires_at"] + 1)
    with pytest.raises(ValueError):
        agent.chat({"approval_id": result["id"], "decision": "allow"})


def test_calculator_and_bound():
    agent, _ = runtime([call("calculate", {"operation": "divide", "a": 5, "b": 2}), answer()])
    result = agent.chat(payload())
    assert result["events"][0]["result"] == {"value": 2.5}
    agent, seen = runtime([call("calculate", {"operation": "add", "a": 1, "b": 2})], max_steps=1)
    assert "step limit" in agent.chat(payload())["choices"][0]["message"]["content"]
    assert len(seen) == 1


def test_no_unknown_tools_or_remote_images():
    agent, _ = runtime([call("bash", {"command": "cat /etc/passwd"})])
    with pytest.raises(ValueError):
        agent.chat(payload())
    agent, seen = runtime([])
    request = payload()
    request["messages"][0]["content"] = [{"type": "image_url", "image_url": {"url": "https://example.com/private.png"}}]
    with pytest.raises(ValueError):
        agent.chat(request)
    assert seen == []


def test_gateway_http_policy_and_completion():
    agent, _ = runtime([answer()])
    server = make_server(GatewayEngine(agent), 0, agent=agent)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/api/agent/config") as response:
            assert json.load(response)["network"] == "disabled"
        req = Request(base + "/api/agent/chat", data=json.dumps(payload()).encode(), headers={"Content-Type": "application/json", "Origin": "https://evil.example"})
        with pytest.raises(HTTPError) as error:
            urlopen(req)
        assert error.value.code == 400
        req.remove_header("Origin")
        with urlopen(req) as response:
            assert json.load(response)["choices"][0]["message"]["content"] == "Ready"
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


@pytest.mark.parametrize("permission", [{"*": "allow"}, {"web_search": "allow"}])
def test_web_search_cannot_inherit_allow(permission):
    with pytest.raises(ValueError):
        AgentConfig("http://127.0.0.1:8000/v1", "test", search_endpoint="http://127.0.0.1:8888", permission=permission)


def test_truncated_and_malformed_completions_rejected():
    for response in [{}, {"choices": [{}]}, {"choices": [{"message": {"role": "assistant", "content": "partial"}, "finish_reason": "length"}]}]:
        agent, _ = runtime([response])
        with pytest.raises(ValueError):
            agent.chat(payload())


def test_text_block_system_and_retained_tool_context():
    agent, seen = runtime([call("calculate", {"operation": "multiply", "a": 2, "b": 3}), answer("6"), answer("Yes")])
    request = payload()
    request["messages"].insert(0, {"role": "system", "content": [{"type": "text", "text": "Read evidence."}]})
    result = agent.chat(request)
    followup = {"model": "test", "messages": [*result["messages"], result["choices"][0]["message"], {"role": "user", "content": "Is that six?"}]}
    assert agent.chat(followup)["choices"][0]["message"]["content"] == "Yes"
    assert any(m["role"] == "tool" for m in seen[-1][1]["messages"])


def test_opt_in_registry_keeps_default_viewer_contract():
    from schematic_model.agent import AGENT_TOOLS
    from schematic_model.deployment import DeploymentConfig, validate_request, assistant_message
    from schematic_model.view_tools import ToolFormatError, parse_tool_calls
    request = payload()
    request["tools"] = AGENT_TOOLS
    with pytest.raises(ToolFormatError):
        validate_request(request, DeploymentConfig(model="", served_model_name="test"))
    validate_request(request, DeploymentConfig(model="", served_model_name="test", agent_tools=True))
    text = '<tool_call>\n<function=calculate>\n<parameter=operation>multiply</parameter>\n<parameter=a>2</parameter>\n<parameter=b>3</parameter>\n</function>\n</tool_call>'
    with pytest.raises(ToolFormatError):
        parse_tool_calls(text)
    message, finish = assistant_message(text, AGENT_TOOLS, "call", registry=AGENT_TOOLS)
    assert finish == "tool_calls"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"])["a"] == 2


def test_all_tools_denied_omits_tools_for_native_endpoint():
    from schematic_model.deployment import DeploymentConfig, validate_request
    agent, seen = runtime([answer()], permission={"*": "deny"})
    assert agent.chat(payload())["choices"][0]["message"]["content"] == "Ready"
    assert "tools" not in seen[0][1]
    validate_request(seen[0][1], DeploymentConfig(model="", served_model_name="test", agent_tools=True))


def test_container_gateway_bind_preserves_host_and_origin_guards():
    agent, _ = runtime([answer()])
    server = make_server(GatewayEngine(agent), 0, agent=agent, bind="0.0.0.0")
    assert server.server_address[0] == "0.0.0.0"
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/health") as response:
            assert json.load(response)["status"] == "ok"
        request = Request(base + "/health", headers={"Host": "192.168.1.10"})
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 400
        request = Request(base + "/api/agent/chat", data=json.dumps(payload()).encode(), headers={"Content-Type": "application/json", "Origin": "http://192.168.1.10"})
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
    with pytest.raises(ValueError):
        make_server(GatewayEngine(agent), 0, bind="0.0.0.0")
    with pytest.raises(ValueError):
        make_server(GatewayEngine(agent), 0, agent=agent, bind="192.168.1.10")


@pytest.mark.parametrize(('name', 'arguments'), [
    ('show_label', {'label': 'U1'}),
    ('show_region', {'box': [100, 200, 300, 400], 'label': 'Regulator'}),
    ('show_full_sheet', {}),
])
@pytest.mark.parametrize('question', ['Review this circuit.', 'Discuss U1.', 'Search for this part.', 'Show me U1.'])
def test_every_model_navigation_needs_exact_consent(name, arguments, question):
    agent, seen = runtime([call(name, arguments)])
    request = payload()
    request['messages'][0]['content'] = question
    proposed = agent.chat(request)
    assert proposed['choices'] == []
    approval = proposed['pending_approval']
    assert approval['tool'] == name
    assert approval['arguments'] == arguments
    assert approval['destination'] == 'schematic viewer'
    assert next(t for t in agent.describe()['tools'] if t['name'] == name)['permission'] == 'ask'
    # A caller cannot use an existing token to authorize a different action.
    with pytest.raises(ValueError):
        agent.chat({'approval_id': approval['id'], 'decision': 'allow', 'arguments': {'label': 'U2'}})
    released = agent.chat({'approval_id': approval['id'], 'decision': 'allow'})
    assert len(seen) == 1  # Approval must not generate a replacement command.
    command = released['choices'][0]['message']['tool_calls'][0]
    assert command['function']['name'] == name
    assert json.loads(command['function']['arguments']) == arguments
    assert released['viewer_authorization'] == {'tool_call_id': command['id'], 'tool': name, 'arguments': arguments}
    assert released['metrics']['model_steps'] == 0
    assert not any(m['role'] == 'tool' for m in released['messages'])
    assert released['events'][-1]['status'] == 'approved'
    with pytest.raises(ValueError):
        agent.chat({'approval_id': approval['id'], 'decision': 'allow'})


def test_navigation_denial_returns_actual_denial_and_keeps_answering():
    agent, seen = runtime([call('show_label', {'label': 'U1'}), answer('I will discuss the circuit without moving the view.')])
    proposed = agent.chat(payload())
    result = agent.chat({'approval_id': proposed['pending_approval']['id'], 'decision': 'deny'})
    assert not result['choices'][0]['message'].get('tool_calls')
    assert 'viewer_authorization' not in result
    assert result['events'][-1]['status'] == 'denied'
    actual_result = json.loads(seen[-1][1]['messages'][-1]['content'])
    assert actual_result == {'error': 'User denied tool execution'}
    assert all('/search?' not in url for url, _ in seen)


def test_earlier_navigation_and_result_cannot_authorize_next_navigation():
    agent, seen = runtime([call('show_label', {'label': 'U1'}), answer('U1 is selected.'), call('show_region', {'box': [0, 0, 200, 200]})])
    request = payload()
    request['messages'][0]['content'] = 'Show me U1.'
    proposed = agent.chat(request)
    approved = agent.chat({'approval_id': proposed['pending_approval']['id'], 'decision': 'allow'})
    assistant_call = approved['choices'][0]['message']
    actual_browser_result = {'role': 'tool', 'tool_call_id': assistant_call['tool_calls'][0]['id'],
                             'content': json.dumps({'status': 'shown', 'label': 'U1', 'kind': 'part', 'box': [10, 10, 100, 100]})}
    continuation = {'model': 'test', 'messages': [*approved['messages'], assistant_call, actual_browser_result]}
    answered = agent.chat(continuation)
    followup = {'model': 'test', 'messages': [*answered['messages'], answered['choices'][0]['message'],
                                             {'role': 'user', 'content': 'Discuss the regulator. Keep this view.'}]}
    next_action = agent.chat(followup)
    assert next_action['choices'] == []
    assert next_action['pending_approval']['tool'] == 'show_region'
    assert next_action['pending_approval']['id'] != proposed['pending_approval']['id']
    assert 'viewer_authorization' not in next_action
    assert len(seen) == 3


def test_search_approval_never_authorizes_viewer_side_effect():
    agent, seen = runtime([call('web_search', {'query': 'AP2112 datasheet'}), {'results': []},
                           call('show_full_sheet', {}), answer('No results were returned.')],
                          search_endpoint='http://127.0.0.1:8888')
    search = agent.chat(payload())['pending_approval']
    proposed_view = agent.chat({'approval_id': search['id'], 'decision': 'allow'})
    assert len([url for url, _ in seen if '/search?' in url]) == 1
    assert proposed_view['choices'] == []
    assert proposed_view['pending_approval']['tool'] == 'show_full_sheet'
    assert proposed_view['pending_approval']['id'] != search['id']
    answer_result = agent.chat({'approval_id': proposed_view['pending_approval']['id'], 'decision': 'deny'})
    assert not answer_result['choices'][0]['message'].get('tool_calls')


def test_viewer_deny_remains_hard_deny_and_ask_configuration_is_valid():
    agent, _ = runtime([call('show_label', {'label': 'U1'}), answer()], permission={'*': 'deny', 'show_label': 'deny'})
    result = agent.chat(payload())
    assert 'pending_approval' not in result
    assert result['events'][-1]['status'] == 'denied'
    agent, _ = runtime([call('show_label', {'label': 'U1'})], permission={'*': 'deny', 'show_label': 'ask'})
    pending = agent.chat(payload())['pending_approval']
    assert pending['tool'] == 'show_label'


def test_viewer_approval_expires_and_stays_with_own_runtime(monkeypatch):
    agent, _ = runtime([call('show_label', {'label': 'U1'})])
    other, _ = runtime([])
    pending = agent.chat(payload())['pending_approval']
    with pytest.raises(ValueError):
        other.chat({'approval_id': pending['id'], 'decision': 'allow'})
    monkeypatch.setattr('schematic_model.agent.time.time', lambda: pending['expires_at'] + 1)
    with pytest.raises(ValueError):
        agent.chat({'approval_id': pending['id'], 'decision': 'allow'})
