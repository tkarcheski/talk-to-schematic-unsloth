"""Serve the browser UI with a scripted stand-in for the model (no GPU).

This exists to develop and screenshot the viewer. Health reports the model as
``ui-preview-scripted``; the replies are fixed rules, not model output, and
must never be presented as model results.

  uv run --locked python scripts/ui_preview.py --examples data/sparkfun --examples data/real
"""
from __future__ import annotations

import argparse
import json
import re
from types import SimpleNamespace

from schematic_model.deployment import DeploymentConfig, make_server
from schematic_model.view_tools import plain_messages
from schematic_model.web_ui import ExampleCatalog

NAME = "ui-preview-scripted"


def _text(message) -> str:
    content = message["content"]
    if isinstance(content, str):
        return content
    return " ".join(block.get("text", "") for block in content if block.get("type") == "text")


def _call(name: str, **arguments) -> str:
    body = "".join(f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in arguments.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def scripted(messages, tools=None) -> str:
    last = _text(plain_messages(messages, tools)[-1])
    if last.startswith("<tool_response>"):
        result = json.loads(last.split("<tool_response>\n", 1)[1].rsplit("\n</tool_response>", 1)[0])
        if result.get("status") == "not_found":
            return f"{result['label']} is not on this page. (Scripted UI preview, not a model.)"
        if result.get("view") == "full_sheet":
            return "Showing the full sheet. (Scripted UI preview, not a model.)"
        return f"Showing {result.get('label', 'that region')}. (Scripted UI preview, not a model.)"
    if tools and re.search(r"\b(zoom (back )?out|whole sheet|full sheet)\b", last, re.I):
        return _call("show_full_sheet")
    target = re.search(r"\b(?:show(?: me)?|where is|find|zoom to(?: the)?|go to)\s+([A-Za-z0-9_$+#/-]+)", last, re.I)
    if tools and target:
        return _call("show_label", label=target[1].rstrip(".?!"))
    return ("No model is loaded in this UI preview, so there is no real answer. "
            "Try “Show me U1” or “Zoom back out” to see the viewer commands.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--examples", action="append", required=True)
    parser.add_argument("--port", type=int, default=8891)
    args = parser.parse_args(argv)
    config = DeploymentConfig(model=".", served_model_name=NAME)
    engine = SimpleNamespace(config=config, model=object(), fingerprint="ui-preview")
    engine.complete = lambda model, messages, tools=None, **_: {
        "output": scripted(messages, tools), "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
    with make_server(engine, args.port, examples=ExampleCatalog(args.examples)) as server:
        print(json.dumps({"url": f"http://127.0.0.1:{server.server_port}/", "model": NAME}), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
