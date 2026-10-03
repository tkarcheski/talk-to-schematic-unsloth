"""Viewer commands the model can issue, in one train/serve wire format.

The browser exposes three commands that move the schematic view. They are
offered to the model as OpenAI-style function tools. Training rows, the local
server and the evaluator all use :func:`plain_messages` to turn tool
definitions, assistant ``tool_calls`` and ``tool`` results into ordinary
user/assistant text. The text reproduces the Qwen3.5 chat template's tool
format (``<tool_call><function=...><parameter=...>``), so the tokens a model is
trained on are the tokens it is served, and the existing training, benchmark
and masking code needs no tool-specific path.

Coordinates are integers from 0 to 1000 relative to the full image of the
current page, x then y, ``[x1, y1, x2, y2]``.
"""
from __future__ import annotations

import json
import re
from typing import Any

COORDINATES = "integers 0-1000 relative to the full current page image, [x1, y1, x2, y2]"
LABEL = re.compile(r"[A-Za-z0-9_$+./#-]{1,64}")
MAX_CALLS = 4

VIEW_TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "show_label",
        "description": ("Center the view on a reference designator (such as R1 or U3) or a net label "
                        "(such as SCL) on the current page. The viewer returns where it found the "
                        "label, or not_found."),
        "parameters": {"type": "object", "properties": {
            "label": {"type": "string", "description": "Exact reference designator or net name"}},
            "required": ["label"]}}},
    {"type": "function", "function": {
        "name": "show_region",
        "description": ("Zoom the view onto a rectangle of the current page. Box coordinates are "
                        + COORDINATES + "."),
        "parameters": {"type": "object", "properties": {
            "box": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4,
                    "description": "[x1, y1, x2, y2], " + COORDINATES},
            "label": {"type": "string", "description": "Short caption for the region"}},
            "required": ["box"]}}},
    {"type": "function", "function": {
        "name": "show_full_sheet",
        "description": "Zoom out to show the whole current page.",
        "parameters": {"type": "object", "properties": {}}}},
]
TOOL_NAMES = {tool["function"]["name"] for tool in VIEW_TOOLS}

# Verbatim from the Qwen3.5 chat template tool preamble.
_PREAMBLE = (
    "# Tools\n\nYou have access to the following functions:\n\n<tools>{tools}\n</tools>"
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block must be "
    "nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the function "
    "call, but NOT after\n- If there is no function call available, answer the question like normal with "
    "your current knowledge and do not tell the user about function calls\n</IMPORTANT>"
)
VIEWER_INSTRUCTIONS = (
    "You control the schematic viewer beside this conversation. When the user asks to see, find, show or "
    "zoom to something, call a viewer function first, then answer briefly from its result. Use show_label "
    "for a named part or net, show_region for an area you can see, and show_full_sheet to zoom out. If "
    "show_label returns not_found, say the label is not on this page; do not guess a location."
)


class ToolFormatError(ValueError):
    """A tool definition, call or result does not follow the viewer protocol."""


def _block_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(isinstance(b, dict) and b.get("type") == "text"
                                         and isinstance(b.get("text"), str) for b in content):
        return "".join(b["text"] for b in content)
    raise ToolFormatError("tool and assistant tool-call content must be text")


def validate_arguments(name: str, arguments: Any) -> dict[str, Any]:
    """Return normalized arguments or raise; never silently clamp a box."""
    if name not in TOOL_NAMES:
        raise ToolFormatError(f"unknown viewer function {name!r}")
    if not isinstance(arguments, dict):
        raise ToolFormatError("function arguments must be an object")
    if name == "show_full_sheet":
        if arguments:
            raise ToolFormatError("show_full_sheet takes no arguments")
        return {}
    if name == "show_label":
        label = arguments.get("label")
        if set(arguments) != {"label"} or not isinstance(label, str) or not LABEL.fullmatch(label.strip()):
            raise ToolFormatError("show_label needs one label of 1-64 schematic identifier characters")
        return {"label": label.strip()}
    if set(arguments) - {"box", "label"} or "box" not in arguments:
        raise ToolFormatError("show_region needs box and an optional label")
    box = arguments["box"]
    if (not isinstance(box, list) or len(box) != 4
            or any(type(v) is not int or not 0 <= v <= 1000 for v in box)
            or box[0] >= box[2] or box[1] >= box[3]):
        raise ToolFormatError("show_region box must be four increasing integers within 0-1000")
    result: dict[str, Any] = {"box": box}
    if "label" in arguments:
        label = arguments["label"]
        if not isinstance(label, str) or not 1 <= len(label.strip()) <= 80:
            raise ToolFormatError("show_region label must be 1-80 characters")
        result["label"] = label.strip()
    return result


def validate_tools(tools: Any) -> list[dict[str, Any]]:
    """Only the server's own viewer functions may be offered to the model."""
    if tools != VIEW_TOOLS:
        raise ToolFormatError("tools must be exactly the published viewer functions")
    return VIEW_TOOLS


def _render_call(call: Any) -> str:
    if not isinstance(call, dict) or call.get("type", "function") != "function":
        raise ToolFormatError("tool calls must be function calls")
    function = call.get("function")
    if not isinstance(function, dict):
        raise ToolFormatError("tool call needs a function object")
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ToolFormatError("tool call arguments must be JSON") from exc
    arguments = validate_arguments(function.get("name"), arguments)
    text = f"<tool_call>\n<function={function['name']}>\n"
    for key, value in arguments.items():
        rendered = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value)
        text += f"<parameter={key}>\n{rendered}\n</parameter>\n"
    return text + "</function>\n</tool_call>"


def plain_messages(messages: Any, tools: Any = None) -> list[dict[str, Any]]:
    """Render tool use as text messages exactly as the Qwen3.5 template would.

    Without tools, messages are returned unchanged (as new lists). With tools,
    the preamble is prepended to the system message, assistant ``tool_calls``
    become ``<tool_call>`` text and consecutive ``tool`` results become one user
    message of ``<tool_response>`` blocks.
    """
    if not isinstance(messages, list):
        raise ToolFormatError("messages must be a list")
    uses_tools = any(isinstance(m, dict) and (m.get("role") == "tool" or "tool_calls" in m) for m in messages)
    if tools is None:
        if uses_tools:
            raise ToolFormatError("tool calls and results require the viewer tool definitions")
        return [dict(m) if isinstance(m, dict) else m for m in messages]
    validate_tools(tools)
    preamble = _PREAMBLE.format(tools="".join("\n" + json.dumps(t, ensure_ascii=False) for t in tools))
    output: list[dict[str, Any]] = []
    rest = list(messages)
    if rest and isinstance(rest[0], dict) and rest[0].get("role") == "system":
        system = _block_text(rest.pop(0).get("content")).strip()
        output.append({"role": "system", "content": preamble + ("\n\n" + system if system else "")})
    else:
        output.append({"role": "system", "content": preamble})
    pending_calls = 0
    for message in rest:
        if not isinstance(message, dict):
            raise ToolFormatError("messages must be objects")
        role = message.get("role")
        if role == "tool":
            if set(message) - {"role", "content", "tool_call_id", "name"}:
                raise ToolFormatError("tool results allow only role, content, tool_call_id and name")
            if pending_calls < 1:
                raise ToolFormatError("each tool result must answer an earlier tool call")
            pending_calls -= 1
            block = "<tool_response>\n" + _block_text(message.get("content")) + "\n</tool_response>"
            previous = output[-1]
            if previous["role"] == "user" and isinstance(previous["content"], str) \
                    and previous["content"].startswith("<tool_response>"):
                previous["content"] += "\n" + block
            else:
                output.append({"role": "user", "content": block})
            continue
        if pending_calls:
            raise ToolFormatError("every tool call needs a tool result before the conversation continues")
        if role == "assistant" and message.get("tool_calls"):
            if set(message) - {"role", "content", "tool_calls"}:
                raise ToolFormatError("assistant tool-call messages allow only role, content and tool_calls")
            calls = message["tool_calls"]
            if not isinstance(calls, list) or not 1 <= len(calls) <= MAX_CALLS:
                raise ToolFormatError(f"assistant messages may make 1-{MAX_CALLS} tool calls")
            content = message.get("content") or ""
            content = _block_text(content).strip() if content else ""
            text = content + ("\n\n" if content else "") + "\n".join(_render_call(c) for c in calls)
            output.append({"role": "assistant", "content": text})
            pending_calls = len(calls)
            continue
        output.append({key: value for key, value in message.items() if key != "tool_calls"})
    return output


_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_FUNCTION = re.compile(r"<function=([A-Za-z_][A-Za-z0-9_]*)>\s*(.*?)\s*</function>", re.S)
_PARAMETER = re.compile(r"<parameter=([A-Za-z_][A-Za-z0-9_]*)>\n?(.*?)\n?</parameter>", re.S)


def _parameter_value(name: str, raw: str) -> Any:
    raw = raw.strip()
    if name == "box":
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ToolFormatError("box must be a JSON list") from exc
    return raw


def parse_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Split generated text into (visible content, validated calls).

    Text after the last call is not allowed by the template contract; any
    malformed or unknown call raises rather than being dropped.
    """
    if not isinstance(text, str):
        raise ToolFormatError("generated text must be a string")
    if "<tool_call>" not in text and "</tool_call>" not in text:
        return text.strip(), []
    blocks = list(_CALL.finditer(text))
    if not blocks or text.count("<tool_call>") != len(blocks) or text.count("</tool_call>") != len(blocks):
        raise ToolFormatError("unbalanced tool call markup")
    if len(blocks) > MAX_CALLS:
        raise ToolFormatError(f"at most {MAX_CALLS} viewer calls per answer")
    if text[blocks[-1].end():].strip():
        raise ToolFormatError("text after a tool call is not allowed")
    calls = []
    for index, block in enumerate(blocks):
        between = text[blocks[index - 1].end():block.start()] if index else ""
        if between.strip():
            raise ToolFormatError("text between tool calls is not allowed")
        function = _FUNCTION.fullmatch(block[1])
        if function is None:
            raise ToolFormatError("tool call must contain one <function=...> block")
        body = function[2]
        parameters = list(_PARAMETER.finditer(body))
        if _PARAMETER.sub("", body).strip():
            raise ToolFormatError("unexpected text inside function call")
        arguments = {}
        for parameter in parameters:
            if parameter[1] in arguments:
                raise ToolFormatError("duplicate function parameter")
            arguments[parameter[1]] = _parameter_value(parameter[1], parameter[2])
        calls.append({"name": function[1], "arguments": validate_arguments(function[1], arguments)})
    return text[:blocks[0].start()].strip(), calls


def openai_tool_calls(calls: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    return [{"id": f"{prefix}-{index}", "type": "function",
             "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])}}
            for index, call in enumerate(calls)]


def box_iou(first: list[int], second: list[int]) -> float:
    width = min(first[2], second[2]) - max(first[0], second[0])
    height = min(first[3], second[3]) - max(first[1], second[1])
    if width <= 0 or height <= 0:
        return 0.0
    intersection = width * height
    area = lambda box: (box[2] - box[0]) * (box[3] - box[1])  # noqa: E731
    return intersection / (area(first) + area(second) - intersection)
