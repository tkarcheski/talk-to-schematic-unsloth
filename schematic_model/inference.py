"""Bounded OpenAI-compatible inference and portable image conversations."""

import base64
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from PIL import Image

from evaluate import strip

SYSTEM = (
    "You are a schematic assistant. Use only the supplied schematic and reference evidence. "
    "Cite component designators and pins. Distinguish observed connections from assumptions. "
    "Say when the evidence does not contain the requested detail."
)


class InferenceError(RuntimeError):
    """A request did not produce a complete usable answer."""


def resolve_image(value, dataset):
    path = Path(value)
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = [Path(dataset).resolve().parent / path,
                      Path(dataset).resolve().parent.parent / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(f"Image does not exist relative to dataset: {value}")


def image_content(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("Image must be an existing regular file smaller than 32 MiB")
    with Image.open(path) as picture:
        mime = Image.MIME.get(picture.format)
        picture.verify()
    if mime not in {"image/png", "image/jpeg", "image/webp"}:
        raise ValueError("Supported image formats: PNG, JPEG, WebP")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


def convert_messages(messages, dataset):
    converted = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if not isinstance(content, list):
            raise ValueError("Message content must be text or a list of content blocks")
        blocks = []
        for block in content:
            if block.get("type") == "image":
                blocks.append(image_content(resolve_image(block["image"], dataset)))
            elif block.get("type") == "text" and isinstance(block.get("text"), str):
                text = strip(block["text"]) if message["role"] == "assistant" else block["text"]
                blocks.append({"type": "text", "text": text})
            else:
                raise ValueError("Unsupported message content block")
        converted.append({"role": message["role"], "content": blocks})
    return converted


class Client:
    def __init__(self, base_url="http://localhost:8888/v1", *, timeout=180, api_key=None):
        url = urlsplit(base_url)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
            raise ValueError("base URL must be HTTP(S), without embedded credentials")
        if url.query or url.fragment or timeout <= 0:
            raise ValueError("base URL cannot have query/fragment; timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key if api_key is not None else os.environ.get("UNSLOTH_API_KEY", "")

    def complete(self, model, messages, *, max_tokens=1024, temperature=0.0):
        if not model or max_tokens < 1:
            raise ValueError("model and positive max_tokens are required")
        payload = {"model": model, "messages": messages, "temperature": temperature,
                   "max_tokens": max_tokens, "stream": False}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(self.base_url + "/chat/completions", json.dumps(payload).encode(), headers)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read(16 * 1024 * 1024 + 1)
        except HTTPError as exc:
            raise InferenceError(f"Inference server returned HTTP {exc.code}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise InferenceError("Inference server unavailable or timed out") from exc
        if len(body) > 16 * 1024 * 1024:
            raise InferenceError("Inference response exceeded 16 MiB")
        try:
            result = json.loads(body)
            choice = result["choices"][0]
            content = choice["message"]["content"]
            reason = choice["finish_reason"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise InferenceError("Malformed inference response") from exc
        if reason != "stop":
            raise InferenceError(f"Inference did not complete normally (finish_reason={reason!r})")
        if not isinstance(content, str) or not strip(content):
            raise InferenceError("Inference returned no final answer")
        return {"output": strip(content), "finish_reason": reason,
                "model": result.get("model", model), "usage": result.get("usage", {})}
