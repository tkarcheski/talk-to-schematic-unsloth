"""Bounded OpenAI-compatible inference and portable image conversations."""

import base64
import hashlib
import io
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from PIL import Image

from evaluate import strip

SYSTEM = (
    "You are a schematic assistant. Use only the supplied schematic and reference evidence. "
    "Cite component designators and pins. Distinguish observed connections from assumptions. "
    "Say when the evidence does not contain the requested detail."
)


class InferenceError(RuntimeError):
    """A request did not produce a complete usable answer."""


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        raise HTTPError(request.full_url, code, "Inference redirects are disabled", headers, response)


# An authenticated request must never forward its credentials or schematic
# data to a redirected endpoint, including one on the same origin.
_OPENER = build_opener(_RejectRedirects())
MAX_IMAGE_BYTES = 32 * 1024 * 1024


def resolve_image(value, dataset):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Image reference must be a nonempty local path")
    path = Path(value)
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = [Path(dataset).resolve().parent / path,
                      Path(dataset).resolve().parent.parent / path]
    matches = []
    for candidate in candidates:
        if candidate.is_file() and not any(candidate.samefile(other) for other in matches):
            matches.append(candidate.resolve())
    if len(matches) > 1:
        raise ValueError(f"Ambiguous image path relative to dataset: {value}")
    if matches:
        return matches[0]
    raise ValueError(f"Image does not exist relative to dataset: {value}")


def _read_image(path, *, expected_sha256=None):
    path = Path(path)
    if not path.is_file():
        raise ValueError("Image must be an existing regular file smaller than 32 MiB")
    with path.open("rb") as handle:
        data = handle.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Image must be an existing regular file smaller than 32 MiB")
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError(f"Image bytes changed after prediction preflight: {path}")
    # Verify and encode the same bytes; reopening the path after verification
    # could otherwise transmit different evidence if the file changes.
    with Image.open(io.BytesIO(data)) as picture:
        mime = Image.MIME.get(picture.format)
        picture.verify()
    if mime not in {"image/png", "image/jpeg", "image/webp"}:
        raise ValueError("Supported image formats: PNG, JPEG, WebP")
    return data, mime


def image_content(path, *, expected_sha256=None):
    data, mime = _read_image(path, expected_sha256=expected_sha256)
    encoded = base64.b64encode(data).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


def _content_blocks(message):
    content = message.get("content")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    if not isinstance(content, list) or any(not isinstance(block, dict) for block in content):
        raise ValueError("Message content must be text or a list of content objects")
    return content


def image_fingerprints(messages, dataset):
    """Return verified image-byte hashes keyed by their literal dataset references."""
    hashes = {}
    for message in messages:
        for block in _content_blocks(message):
            if block.get("type") == "image":
                reference = block.get("image")
                path = resolve_image(reference, dataset)
                data, _ = _read_image(path)
                digest = hashlib.sha256(data).hexdigest()
                if reference in hashes and hashes[reference] != digest:
                    raise ValueError(f"Image bytes changed during prediction preflight: {reference}")
                hashes[reference] = digest
    return hashes


def convert_messages(messages, dataset, *, image_hashes=None):
    converted = []
    for message in messages:
        blocks = []
        for block in _content_blocks(message):
            if block.get("type") == "image":
                reference = block.get("image")
                path = resolve_image(reference, dataset)
                if image_hashes is not None and reference not in image_hashes:
                    raise ValueError(f"Image was not bound by prediction preflight: {reference}")
                digest = image_hashes[reference] if image_hashes is not None else None
                blocks.append(image_content(path, expected_sha256=digest))
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
            with _OPENER.open(request, timeout=self.timeout) as response:
                body = response.read(16 * 1024 * 1024 + 1)
        except HTTPError as exc:
            if exc.code in {301, 302, 303, 307, 308}:
                raise InferenceError("Inference server redirected the request; redirects are disabled") from exc
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
