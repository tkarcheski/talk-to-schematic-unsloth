"""Local Unsloth batch inference and a bounded loopback OpenAI-compatible server.

The HTTP interface accepts image data URLs only. It has no model-switching,
filesystem, upload, or remote URL fetch endpoint. Model imports are lazy.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from dataclasses import asdict, dataclass
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
import math
from pathlib import Path
import shutil
from socketserver import ThreadingMixIn
import sys
import tempfile
import threading
import time
from typing import Any
import uuid

from schematic_model.inference import InferenceError, convert_messages
from schematic_model.training import gpu_inventory, installed_versions, sha256, write_manifest

MAX_BODY_BYTES = 32 * 1024 * 1024
MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MAX_MESSAGES = 64
MAX_IMAGES = 4
MAX_TEXT_CHARS = 200_000
MAX_HTTP_CONNECTIONS = 8


@dataclass(frozen=True)
class DeploymentConfig:
    model: str
    served_model_name: str = "schematic"
    max_seq: int = 8192
    max_image_size: int = 1024
    max_tokens: int = 1024
    seed: int = 3407
    four_bit: bool = False
    min_free_vram_gb: float = 10.0

    def validate(self) -> None:
        if not Path(self.model).is_dir():
            raise ValueError("model must be a local safetensors model or adapter directory")
        if not self.served_model_name or len(self.served_model_name) > 128:
            raise ValueError("served model name must contain 1–128 characters")
        if not 1 <= self.max_image_size <= 4096:
            raise ValueError("max image size must be between 1 and 4096")
        if not 1 <= self.max_tokens < self.max_seq <= 32768:
            raise ValueError("require 1 <= max_tokens < max_seq <= 32768")
        if not math.isfinite(self.min_free_vram_gb) or self.min_free_vram_gb < 0:
            raise ValueError("minimum free VRAM must be finite and nonnegative")


def _digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError(f"Model metadata must be an object: {path.name}")
    return value


def model_artifacts(path: Path) -> dict[str, Any]:
    """Hash all top-level weights, processor and tokenizer files once per process."""
    path = path.resolve()
    if not path.is_dir():
        raise ValueError("local model directory does not exist")
    files = [item for item in sorted(path.iterdir()) if item.is_file()
             and item.suffix in {".safetensors", ".json", ".jinja", ".txt"}
             and item.name not in {"training_manifest.json", "trainer_state.json", "deployment_manifest.json"}]
    if not any(item.suffix == ".safetensors" for item in files):
        raise ValueError("local model directory contains no safetensors weights")
    for item in files:
        if item.suffix == ".safetensors" and not item.stat().st_size:
            raise ValueError(f"Empty model weights: {item.name}")
        if item.name.endswith(".safetensors.index.json"):
            mapping = _read_object(item).get("weight_map")
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("Safetensors index requires a nonempty weight_map")
            for tensor, shard in mapping.items():
                if (not isinstance(tensor, str) or not tensor or not isinstance(shard, str)
                        or Path(shard).name != shard or not shard.endswith(".safetensors")
                        or not (path / shard).is_file()):
                    raise ValueError("Safetensors index references an invalid or missing local shard")
    hashes = [{"name": item.name, "bytes": item.stat().st_size, "sha256": sha256(item)} for item in files]
    result = {"path": str(path), "files": hashes, "sha256": _digest_json(hashes)}
    adapter = path / "adapter_config.json"
    if adapter.is_file():
        config = _read_object(adapter)
        reference = config.get("base_model_name_or_path")
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError("adapter must specify a local base_model_name_or_path")
        base = Path(reference)
        if not base.is_dir() or base.resolve() == path or (base / "adapter_config.json").exists():
            raise ValueError("adapter must reference a local base model; create a deployment bundle with --base-model")
        result["base_model"] = model_artifacts(base)
    return result


def prepare_bundle(adapter: str, base_model: str, output: str) -> dict[str, Any]:
    """Copy an adapter and remap its base path without modifying original artifacts."""
    source, base, destination = Path(adapter).resolve(), Path(base_model).resolve(), Path(output).resolve()
    if not (source / "adapter_config.json").is_file() or not (source / "adapter_model.safetensors").is_file():
        raise ValueError("bundle requires adapter_config.json and adapter_model.safetensors")
    if destination.exists():
        raise ValueError("bundle output already exists; choose a new directory")
    if destination.is_relative_to(source) or destination.is_relative_to(base):
        raise ValueError("bundle output must be outside the source adapter and base model")
    base_manifest = model_artifacts(base)
    if "base_model" in base_manifest:
        raise ValueError("base-model must contain full model weights, not another adapter")
    adapter_config = _read_object(source / "adapter_config.json")
    adapter_config["base_model_name_or_path"] = str(base)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.partial-", dir=destination.parent))
    try:
        for path in source.iterdir():
            if path.is_file() and path.suffix in {".safetensors", ".json", ".jinja", ".txt"} and path.name not in {
                "adapter_config.json", "training_manifest.json", "trainer_state.json", "deployment_manifest.json"
            }:
                shutil.copyfile(path, staging / path.name)
        (staging / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2) + "\n", encoding="utf-8")
        bundled = model_artifacts(staging)
        bundled["path"] = str(destination)
        manifest = {"schema_version": 1, "source_adapter": str(source), "base_model": base_manifest,
                    "bundle": bundled}
        write_manifest(staging / "deployment_manifest.json", manifest)
        if destination.exists():
            raise ValueError("bundle output already exists; choose a new directory")
        staging.rename(destination)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def decode_image(value: str, max_side: int):
    """Decode an inline raster with explicit byte/pixel budgets; never fetch a URL."""
    from PIL import Image

    if not isinstance(value, str) or len(value) > MAX_IMAGE_BYTES * 4 // 3 + 128:
        raise ValueError("image data URL exceeds the image size limit")
    mime, separator, encoded = value.partition(";base64,")
    formats = {"data:image/png": "PNG", "data:image/jpeg": "JPEG", "data:image/webp": "WEBP"}
    if not separator or mime not in formats:
        raise ValueError("images must be PNG, JPEG or WebP base64 data URLs")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid image base64") from exc
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("image exceeds the byte limit")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != formats[mime]:
                raise ValueError("image MIME type does not match its contents")
            if image.width * image.height > MAX_IMAGE_PIXELS or image.width < 1 or image.height < 1:
                raise ValueError("image exceeds the pixel limit")
            image.load()
            image = image.convert("RGB")
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            return image
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("invalid raster image") from exc


def normalize_messages(messages: Any, max_side: int) -> list[dict[str, Any]]:
    """Validate OpenAI input without silently dropping messages or content fields."""
    if not isinstance(messages, list) or not 1 <= len(messages) <= MAX_MESSAGES:
        raise ValueError(f"messages must contain 1–{MAX_MESSAGES} entries")
    normalized, image_count, text_length, expected = [], 0, 0, "user"
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError("messages require exactly role and content")
        role = message["role"]
        if index == 0 and role == "system":
            pass
        elif role != expected:
            raise ValueError("messages must alternate user and assistant after an optional system message")
        else:
            expected = "assistant" if role == "user" else "user"
        content = message["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if not isinstance(content, list) or not 1 <= len(content) <= 32:
            raise ValueError("message content must be text or 1–32 blocks")
        blocks = []
        for block in content:
            if not isinstance(block, dict):
                raise ValueError("content block must be an object")
            if block.get("type") == "text" and set(block) == {"type", "text"}:
                if not isinstance(block["text"], str) or not block["text"].strip():
                    raise ValueError("text blocks cannot be empty")
                text_length += len(block["text"])
                if text_length > MAX_TEXT_CHARS:
                    raise ValueError("conversation exceeds the text size limit")
                blocks.append(dict(block))
            elif block.get("type") == "image_url" and role == "user" and set(block) == {"type", "image_url"}:
                info = block["image_url"]
                if not isinstance(info, dict) or set(info) != {"url"}:
                    raise ValueError("image_url must contain only url")
                image_count += 1
                if image_count > MAX_IMAGES:
                    raise ValueError("conversation exceeds the image count limit")
                blocks.append({"type": "image", "image": decode_image(info["url"], max_side)})
            else:
                raise ValueError("unsupported content block; local paths and remote URLs are not accepted")
        normalized.append({"role": role, "content": blocks})
    if normalized[-1]["role"] != "user":
        raise ValueError("conversation must end with a user message")
    return normalized


def validate_request(payload: Any, config: DeploymentConfig) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) - {"model", "messages", "max_tokens", "temperature", "stream"}:
        raise ValueError("unsupported completion request fields")
    if payload.get("model") != config.served_model_name:
        raise ValueError("requested model is not served")
    if payload.get("stream", False) is not False:
        raise ValueError("streaming is not supported")
    max_tokens = payload.get("max_tokens", config.max_tokens)
    if type(max_tokens) is not int or not 1 <= max_tokens <= config.max_tokens:
        raise ValueError("max_tokens exceeds the configured generation limit")
    temperature = payload.get("temperature", 0.0)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ValueError("temperature must be a finite number between 0 and 2")
    return {"messages": normalize_messages(payload.get("messages"), config.max_image_size),
            "max_tokens": max_tokens, "temperature": float(temperature)}


def final_answer(text: str, generated_ids: list[int], eos_ids: set[int]) -> str:
    from evaluate import strip

    if not generated_ids or generated_ids[-1] not in eos_ids:
        raise InferenceError("Generation did not reach an end token; increase the generation budget")
    if text.lower().count("<think>") != text.lower().count("</think>"):
        raise InferenceError("Generation contains an unfinished reasoning block")
    answer = strip(text)
    if not answer or "<think>" in answer.lower() or "</think>" in answer.lower():
        raise InferenceError("Generation did not produce a complete final answer")
    return answer


class LocalModel:
    """Client-compatible lazy engine. One model per process; no implicit switching."""

    def __init__(self, config: DeploymentConfig):
        config.validate()
        self.config = config
        self.provenance = {"schema_version": 1, "backend": "unsloth-library", "settings": asdict(config),
                           "model": model_artifacts(Path(config.model)), "packages": installed_versions()}
        self.fingerprint = _digest_json(self.provenance)
        self.base_url = "local-unsloth://" + self.fingerprint
        self.model = self.processor = self.torch = None

    def load(self) -> None:
        if self.model is not None:
            return
        devices = gpu_inventory()
        if not devices or max(device["free_mib"] for device in devices) < self.config.min_free_vram_gb * 1024:
            raise InferenceError("Insufficient free GPU memory to load the local model")
        from unsloth import FastVisionModel, is_bf16_supported
        import torch

        if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < self.config.min_free_vram_gb * 1024**3:
            raise InferenceError("The selected CUDA device has insufficient available memory")
        model, processor = FastVisionModel.from_pretrained(
            model_name=str(Path(self.config.model).resolve()), max_seq_length=self.config.max_seq,
            load_in_4bit=self.config.four_bit,
            dtype=torch.bfloat16 if is_bf16_supported() else torch.float16,
            local_files_only=True, trust_remote_code=False,
        )
        FastVisionModel.for_inference(model)
        self.model, self.processor, self.torch = model, processor, torch

    def complete(self, model, messages, *, max_tokens=None, temperature=0.0):
        request = validate_request({"model": model, "messages": messages,
                                    "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
                                    "temperature": temperature}, self.config)
        self.load()
        started = time.monotonic()
        inputs = self.processor.apply_chat_template(
            request["messages"], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", enable_thinking=False,
        )
        input_length = inputs["input_ids"].shape[-1]
        if input_length + request["max_tokens"] > self.config.max_seq:
            raise InferenceError("Conversation plus generation budget exceeds the context limit; shorten history or resize images")
        inputs = inputs.to(self.model.device)
        self.torch.manual_seed(self.config.seed)
        options = {"max_new_tokens": request["max_tokens"], "do_sample": request["temperature"] > 0, "use_cache": True}
        if request["temperature"] > 0:
            options["temperature"] = request["temperature"]
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **options)
        token_ids = output[0][input_length:].tolist()
        text = self.processor.decode(token_ids, skip_special_tokens=True)
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.processor.tokenizer.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        answer = final_answer(text, token_ids, eos_ids)
        return {"output": answer, "finish_reason": "stop", "model": self.config.served_model_name,
                "usage": {"prompt_tokens": input_length, "completion_tokens": len(token_ids), "total_tokens": input_length + len(token_ids)},
                "elapsed_seconds": round(time.monotonic() - started, 4), "model_fingerprint": self.fingerprint}


def batch_predict(config: DeploymentConfig, dataset: str, output: str, *, resume=False, history="gold", engine=None):
    """Reuse the same durable prediction loop as HTTP inference, bound to image bytes."""
    from evaluate import load_gold
    from predict import predict

    config.validate()
    dataset_path, output_path = Path(dataset), Path(output)
    rows = load_gold(dataset_path)
    image_inputs = []
    for row in rows:
        messages = convert_messages(row["messages"], dataset_path)
        for message in messages:
            for block in message["content"]:
                if block["type"] == "image_url":
                    image_inputs.append(hashlib.sha256(block["image_url"]["url"].encode()).hexdigest())
    client = engine if engine is not None else LocalModel(config)
    protected = [Path(config.model).resolve()]
    artifact = client.provenance.get("model")
    while isinstance(artifact, dict):
        if isinstance(artifact.get("path"), str):
            protected.append(Path(artifact["path"]).resolve())
        artifact = artifact.get("base_model")
    if any(output_path.resolve().is_relative_to(directory) for directory in protected):
        raise ValueError("Prediction output must be outside model and base-model artifact directories")
    binding = {"deployment": client.provenance, "dataset_sha256": sha256(dataset_path),
               "image_input_hashes": image_inputs, "history": history}
    client.base_url = "local-unsloth://" + _digest_json(binding)
    provenance_path = output_path.with_suffix(output_path.suffix + ".deployment.json")
    if provenance_path.exists():
        if not resume or json.loads(provenance_path.read_text()) != binding:
            raise ValueError("Deployment provenance differs; use a new output file")
    elif resume and output_path.exists():
        raise ValueError("Cannot resume predictions without deployment provenance")
    else:
        if output_path.exists():
            raise ValueError("Prediction output exists; use --resume")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with provenance_path.open("x", encoding="utf-8") as stream:
            json.dump(binding, stream, indent=2)
            stream.write("\n")
    return predict(dataset_path, output_path, config.served_model_name, client,
                   resume=resume, history=history, max_tokens=config.max_tokens)


def make_server(engine, port: int = 8891, *, examples=None) -> HTTPServer:
    """Bound concurrent connections while serializing GPU work; caller owns lifetime."""
    generation_lock = threading.Lock()

    class BoundedServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

        def __init__(self, *args):
            self.connection_slots = threading.BoundedSemaphore(MAX_HTTP_CONNECTIONS)
            super().__init__(*args)

        def process_request(self, request, client_address):
            if not self.connection_slots.acquire(blocking=False):
                try:
                    request.settimeout(1)
                    request.sendall(b"HTTP/1.0 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                except OSError:
                    pass
                finally:
                    self.shutdown_request(request)
                return
            try:
                super().process_request(request, client_address)
            except Exception:
                self.connection_slots.release()
                raise

        def process_request_thread(self, request, client_address):
            try:
                super().process_request_thread(request, client_address)
            finally:
                self.connection_slots.release()

    class Handler(BaseHTTPRequestHandler):
        server_version = "SchematicLocal/1"

        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def log_message(self, format, *args):
            # Never log prompts, image data or paths. Status counters belong to callers.
            pass

        def send_bytes(self, code, body, content_type):
            try:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
                self.send_header("Referrer-Policy", "no-referrer")
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                # A disconnected client cannot consume or poison the next request.
                self.close_connection = True

        def send_json(self, code, value):
            self.send_bytes(code, json.dumps(value, allow_nan=False).encode(), "application/json")

        def valid_host(self):
            expected = {"localhost", "127.0.0.1", f"localhost:{self.server.server_port}", f"127.0.0.1:{self.server.server_port}"}
            return len(self.headers.get_all("Host", [])) == 1 and self.headers.get("Host") in expected

        def do_GET(self):
            if not self.valid_host():
                self.send_json(400, {"error": {"message": "Host must name this loopback server"}})
                return
            if self.path == "/health":
                self.send_json(200, {"status": "ok", "model": engine.config.served_model_name,
                                     "model_fingerprint": engine.fingerprint,
                                     "loaded": getattr(engine, "model", None) is not None,
                                     "busy": generation_lock.locked()})
            elif self.path == "/v1/models":
                self.send_json(200, {"object": "list", "data": [{"id": engine.config.served_model_name, "object": "model", "owned_by": "local"}]})
            else:
                from .web_ui import static_asset

                if self.path == "/api/examples":
                    self.send_json(200, {"examples": examples.index() if examples else []})
                    return
                if self.path.startswith("/api/examples/") and examples is not None:
                    route = self.path.removeprefix("/api/examples/").split("/")
                    try:
                        if len(route) == 1 and examples.details(route[0]) is not None:
                            self.send_json(200, examples.details(route[0]))
                            return
                        if len(route) == 2 and route[1] == "image":
                            content = examples.image(route[0])
                            if content is not None:
                                self.send_bytes(200, content, "image/png")
                                return
                    except (ValueError, OSError):
                        self.send_json(503, {"error": {"message": "Example integrity check failed; restart with verified data"}})
                        return
                asset = static_asset(self.path)
                if asset is not None:
                    self.send_bytes(200, *asset)
                else:
                    self.send_json(404, {"error": {"message": "Unknown endpoint"}})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_json(404, {"error": {"message": "Unknown endpoint"}})
                return
            if not self.valid_host():
                self.send_json(400, {"error": {"message": "Host must name this loopback server"}})
                return
            origins = self.headers.get_all("Origin", [])
            if (len(origins) > 1 or origins and origins[0] != f"http://{self.headers.get('Host')}"
                    or self.headers.get("Transfer-Encoding") is not None):
                self.send_json(400, {"error": {"message": "Cross-origin and chunked requests are not supported"}})
                return
            if (len(self.headers.get_all("Content-Type", [])) != 1
                    or self.headers.get("Content-Type", "").split(";", 1)[0].strip() != "application/json"):
                self.send_json(415, {"error": {"message": "Content-Type must be application/json"}})
                return
            try:
                lengths = self.headers.get_all("Content-Length", [])
                if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdecimal():
                    raise ValueError("exactly one decimal Content-Length is required")
                length = int(lengths[0])
                if not 0 < length <= MAX_BODY_BYTES:
                    self.send_json(413, {"error": {"message": "Request body is missing or exceeds size limit"}})
                    return
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError("incomplete request body")
                payload = json.loads(body, object_pairs_hook=_unique_object)
                request = validate_request(payload, engine.config)
                if not generation_lock.acquire(blocking=False):
                    self.send_json(503, {"error": {"message": "Local generation is busy; retry when idle"}})
                    return
                try:
                    result = engine.complete(payload["model"], payload["messages"],
                                             max_tokens=request["max_tokens"], temperature=request["temperature"])
                finally:
                    generation_lock.release()
                self.send_json(200, {"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion",
                                     "created": int(time.time()), "model": engine.config.served_model_name,
                                     "choices": [{"index": 0, "message": {"role": "assistant", "content": result["output"]}, "finish_reason": "stop"}],
                                     "usage": result["usage"]})
            except (ValueError, TypeError, UnicodeError, RecursionError):
                self.send_json(400, {"error": {"message": "Invalid completion request"}})
            except InferenceError:
                self.send_json(422, {"error": {"message": "No complete answer within the configured context and generation limits"}})
            except (RuntimeError, OSError, TimeoutError):
                self.send_json(503, {"error": {"message": "Local generation is unavailable"}})
            except Exception:
                self.send_json(500, {"error": {"message": "Local generation failed"}})

    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    return BoundedServer(("127.0.0.1", port), Handler)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object field")
        result[key] = value
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("predict", "serve"):
        command = commands.add_parser(name)
        command.add_argument("--model", required=True)
        command.add_argument("--served-model-name", default="schematic")
        command.add_argument("--max-seq", type=int, default=8192)
        command.add_argument("--max-image-size", type=int, default=1024)
        command.add_argument("--max-tokens", type=int, default=1024)
        command.add_argument("--seed", type=int, default=3407)
        command.add_argument("--4bit", dest="four_bit", action="store_true")
        command.add_argument("--min-free-vram-gb", type=float, default=10)
        if name == "predict":
            command.add_argument("--data", required=True)
            command.add_argument("--out", required=True)
            command.add_argument("--resume", action="store_true")
            command.add_argument("--history", choices=("gold", "generated"), default="gold")
        else:
            command.add_argument("--port", type=int, default=8891)
            command.add_argument("--examples", help="verified real corpus directory for the browser example catalog")
    bundle = commands.add_parser("bundle", help="copy an adapter and bind it to a local base model")
    bundle.add_argument("--model", required=True)
    bundle.add_argument("--base-model", required=True)
    bundle.add_argument("--out", required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        if command == "bundle":
            result = prepare_bundle(args["model"], args["base_model"], args["out"])
        elif command == "predict":
            dataset, output, resume, history = (args.pop(key) for key in ("data", "out", "resume", "history"))
            result = batch_predict(DeploymentConfig(**args), dataset, output, resume=resume, history=history)
        else:
            port = args.pop("port")
            examples = args.pop("examples")
            if examples is not None:
                from .web_ui import ExampleCatalog
                examples = ExampleCatalog(examples)
            engine = LocalModel(DeploymentConfig(**args))
            engine.load()
            with make_server(engine, port, examples=examples) as server:
                print(json.dumps({"url": f"http://127.0.0.1:{server.server_port}/v1", "provenance": engine.provenance}), flush=True)
                try:
                    server.serve_forever()
                except KeyboardInterrupt:
                    pass
            return 0
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"deployment: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
