"""Viewer command protocol: template equivalence, parsing, server transport and scoring."""

import copy
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment
import pytest

from evaluate import PROFILES, EvaluationError, gate_failures, grade, score, validate_gold_rows
from schematic_model.deployment import DeploymentConfig, assistant_message, make_server, validate_request
from schematic_model.inference import InferenceError
from schematic_model.view_tools import (VIEW_TOOLS, ToolFormatError, box_iou, parse_tool_calls, plain_messages,
                                        validate_arguments)

TEMPLATE = Path(__file__).parent / "fixtures/templates/unsloth-qwen3.5-4b-3764fa35.jinja"


def render(messages, tools=None):
    """Render like transformers: sandboxed Jinja with its tojson filter."""
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = lambda value, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
        value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    env.globals["raise_exception"] = raise_exception
    return env.from_string(TEMPLATE.read_text()).render(messages=messages, tools=tools, add_generation_prompt=True,
                                                       enable_thinking=False)


def call(name, arguments, index=0):
    return {"id": f"call-{index}", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


CONVERSATION = [
    {"role": "system", "content": "Answer from the image."},
    {"role": "user", "content": "Show me R1."},
    {"role": "assistant", "content": None, "tool_calls": [call("show_label", {"label": "R1"})]},
    {"role": "tool", "tool_call_id": "call-0", "content": '{"status": "shown", "box": [1, 2, 30, 40]}'},
    {"role": "assistant", "content": "Showing R1."},
    {"role": "user", "content": "Frame R1 and C2, then zoom out."},
    {"role": "assistant", "content": "Two steps.", "tool_calls": [
        call("show_region", {"box": [1, 2, 300, 400], "label": "R1 and C2"}, 1), call("show_full_sheet", {}, 2)]},
    {"role": "tool", "tool_call_id": "call-1", "content": '{"status": "shown"}'},
    {"role": "tool", "tool_call_id": "call-2", "content": '{"status": "shown", "view": "full_sheet"}'},
]


def native(messages):
    """The same conversation with arguments as mappings, as transformers passes them."""
    result = copy.deepcopy(messages)
    for message in result:
        for item in message.get("tool_calls", []):
            item["function"]["arguments"] = json.loads(item["function"]["arguments"])
    return result


def test_plain_wire_format_matches_pinned_template_tool_rendering():
    assert render(plain_messages(CONVERSATION, VIEW_TOOLS)) == render(native(CONVERSATION), VIEW_TOOLS)
    without_system = CONVERSATION[1:5]
    assert render(plain_messages(without_system, VIEW_TOOLS)) == render(native(without_system), VIEW_TOOLS)


def test_generated_calls_round_trip_and_reject_malformed_markup():
    text = plain_messages(CONVERSATION, VIEW_TOOLS)[6]["content"]
    content, calls = parse_tool_calls(text)
    assert content == "Two steps."
    assert calls == [{"name": "show_region", "arguments": {"box": [1, 2, 300, 400], "label": "R1 and C2"}},
                     {"name": "show_full_sheet", "arguments": {}}]
    assert parse_tool_calls("R1 is marked 10k.") == ("R1 is marked 10k.", [])
    bad = [
        "<tool_call>\n<function=show_label>\n<parameter=label>\nR1\n</parameter>\n</function>",  # unbalanced
        "<tool_call>\n<function=delete_files>\n</function>\n</tool_call>",  # unknown function
        "<tool_call>\n<function=show_region>\n<parameter=box>\n[5, 5, 1, 1]\n</parameter>\n</function>\n</tool_call>",
        "<tool_call>\n<function=show_region>\n<parameter=box>\n[0, 0, 1001, 9]\n</parameter>\n</function>\n</tool_call>",
        "<tool_call>\n<function=show_label>\n<parameter=label>\nR1\n</parameter>\n</function>\n</tool_call> done",
        "<tool_call>\n<function=show_label>\nR1\n</function>\n</tool_call>",
    ]
    for text in bad:
        with pytest.raises(ToolFormatError):
            parse_tool_calls(text)


@pytest.mark.parametrize("name,arguments", [
    ("show_label", {"label": "R1; rm"}), ("show_label", {}), ("show_full_sheet", {"page": 2}),
    ("show_region", {"box": [0, 0, 10]}), ("show_region", {"box": [0, 0, 10, 10.5]}),
    ("show_region", {"box": [True, 0, 10, 10]}), ("open_url", {"url": "https://example.com"}),
])
def test_arguments_are_validated_not_clamped(name, arguments):
    with pytest.raises(ToolFormatError):
        validate_arguments(name, arguments)


def test_tool_messages_require_published_tools_and_answered_calls():
    with pytest.raises(ToolFormatError):
        plain_messages(CONVERSATION)  # tool traffic without definitions
    with pytest.raises(ToolFormatError):
        plain_messages(CONVERSATION, VIEW_TOOLS[:1])
    with pytest.raises(ToolFormatError):
        plain_messages(CONVERSATION[:3] + [{"role": "user", "content": "next"}], VIEW_TOOLS)
    with pytest.raises(ToolFormatError):
        plain_messages(CONVERSATION[:2] + [CONVERSATION[3]], VIEW_TOOLS)


def test_request_validation_and_assistant_message_shape(tmp_path):
    config = DeploymentConfig(model=str(tmp_path))
    request = validate_request({"model": "schematic", "messages": CONVERSATION[:4], "tools": VIEW_TOOLS}, config)
    assert [m["role"] for m in request["messages"]] == ["system", "user", "assistant", "user"]
    assert request["messages"][3]["content"][0]["text"].startswith("<tool_response>")
    with pytest.raises(ValueError):
        validate_request({"model": "schematic", "messages": CONVERSATION[:4]}, config)
    text = plain_messages(CONVERSATION, VIEW_TOOLS)[2]["content"]
    message, finish = assistant_message(text, VIEW_TOOLS, "call-x")
    assert finish == "tool_calls" and message["content"] is None
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"label": "R1"}
    assert assistant_message(text, None, "call-x") == ({"role": "assistant", "content": text}, "stop")
    with pytest.raises(InferenceError):
        assistant_message("<tool_call>broken", VIEW_TOOLS, "call-x")


@pytest.fixture
def tool_server(tmp_path):
    engine = SimpleNamespace(config=DeploymentConfig(model=str(tmp_path)), model=object(), fingerprint="test")
    calls = []
    replies = iter(["<tool_call>\n<function=show_label>\n<parameter=label>\nR1\n</parameter>\n</function>\n</tool_call>",
                    "Showing R1."])

    def complete(model, messages, **kwargs):
        calls.append((messages, kwargs))
        return {"output": next(replies), "usage": {"total_tokens": 4}}

    engine.complete = complete
    server = make_server(engine, port=0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield server, calls
    server.shutdown()
    server.server_close()
    worker.join(timeout=3)


def post(server, payload):
    url = f"http://127.0.0.1:{server.server_port}"
    request = Request(url + "/v1/chat/completions", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json", "Origin": url})
    try:
        with urlopen(request, timeout=3) as response:
            return response.status, json.loads(response.read())
    except HTTPError as error:
        return error.code, json.loads(error.read())


def test_server_publishes_tools_and_returns_openai_tool_calls(tool_server):
    server, calls = tool_server
    with urlopen(f"http://127.0.0.1:{server.server_port}/api/view-tools", timeout=3) as response:
        published = json.loads(response.read())
    assert published["tools"] == VIEW_TOOLS and "show_label" in published["instructions"]
    code, body = post(server, {"model": "schematic", "messages": CONVERSATION[:2], "tools": VIEW_TOOLS})
    assert code == 200 and body["choices"][0]["finish_reason"] == "tool_calls"
    tool_call = body["choices"][0]["message"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "show_label" and calls[0][1]["tools"] == VIEW_TOOLS
    history = CONVERSATION[:2] + [body["choices"][0]["message"],
                                  {"role": "tool", "tool_call_id": tool_call["id"], "content": '{"status": "shown"}'}]
    code, body = post(server, {"model": "schematic", "messages": history, "tools": VIEW_TOOLS})
    assert code == 200 and body["choices"][0]["message"] == {"role": "assistant", "content": "Showing R1."}
    assert post(server, {"model": "schematic", "messages": CONVERSATION[:2],
                         "tools": [{"type": "function", "function": {"name": "shell"}}]})[0] == 400
    assert len(calls) == 2


def view_row():
    region = {"type": "view", "name": "show_region", "arguments": {"box": [100, 100, 300, 300]}, "min_iou": 0.5}
    return {"id": "board-view", "gold": [
        {"type": "view", "name": "show_label", "arguments": {"label": "R1"}}, {"type": "view_reply", "label": "R1"},
        region, {"type": "view_reply"},
        {"type": "view", "name": "show_label", "arguments": {"label": "R999"}}, {"type": "view_refuse"}]}


def show(name, **arguments):
    body = "".join(f"<parameter={k}>\n{json.dumps(v) if isinstance(v, list) else v}\n</parameter>\n"
                   for k, v in arguments.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def test_view_scoring_requires_correct_calls_boxes_and_grounded_replies():
    row = view_row()
    perfect = {"board-view#0": show("show_label", label="r1"), "board-view#1": "Showing R1.",
               "board-view#2": show("show_region", box=[110, 110, 310, 310]), "board-view#3": "Done.",
               "board-view#4": show("show_label", label="R999"), "board-view#5": "R999 is not on this sheet."}
    metrics, counts = score([row], perfect)
    assert metrics == {"view.call_acc": 1, "view.region_acc": 1, "view.reply_acc": 1, "view.refuse_acc": 1}
    assert not gate_failures(metrics, counts, PROFILES["view-commands"])
    wrong = {**perfect, "board-view#2": show("show_region", box=[500, 500, 700, 700]),
             "board-view#1": show("show_full_sheet"), "board-view#5": "It is near the top-left."}
    metrics, _ = score([row], wrong)
    assert metrics["view.region_acc"] == 0 and metrics["view.reply_acc"] == 0.5 and metrics["view.refuse_acc"] == 0
    assert grade(row["gold"][0], "<tool_call>broken") == {"view.call_acc": False}
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0


def test_view_gold_is_validated():
    for bad in ({"type": "view", "name": "show_label", "arguments": {}},
                {"type": "view", "name": "show_region", "arguments": {"box": [0, 0, 5, 5]}, "min_iou": 0},
                {"type": "view_reply", "label": " "}):
        with pytest.raises(EvaluationError):
            validate_gold_rows([{"id": "x", "gold": [bad]}])
