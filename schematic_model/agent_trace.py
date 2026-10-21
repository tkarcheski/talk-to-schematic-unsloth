"""Session-only trace views; never include authentication headers or image bytes."""
from __future__ import annotations

import copy
import secrets


def message_trace(messages):
    """Keep message structure visible while bounding text and removing image data."""
    output = []
    remaining = 128_000
    for message in messages:
        item = {key: copy.deepcopy(message[key]) for key in ("role", "tool_call_id", "tool_calls") if key in message}
        content = message.get("content")
        if isinstance(content, list):
            blocks = []
            for block in content:
                if block.get("type") == "image_url":
                    value = block.get("image_url", {}).get("url", "")
                    blocks.append({"type": "image", "media_type": value.split(";", 1)[0].removeprefix("data:"),
                                   "encoded_characters": len(value), "payload": "omitted from trace"})
                elif block.get("type") == "text":
                    value = block.get("text", "")
                    blocks.append({"type": "text", "text": value[:remaining], "truncated": len(value) > remaining})
                    remaining = max(0, remaining - len(value))
            item["content"] = blocks
        elif isinstance(content, str):
            item["content"] = content[:remaining]
            if len(content) > remaining:
                item["truncated"] = True
            remaining = max(0, remaining - len(content))
        else:
            item["content"] = content
        output.append(item)
    return output


def trace_event(state, kind, data):
    state.setdefault("trace", []).append({"id": secrets.token_hex(8), "kind": kind, "data": data})
