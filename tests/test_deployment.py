"""CPU tests for local generation boundaries, reproducibility, and HTTP transport."""

import base64
from contextlib import nullcontext
import copy
import io
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image
import pytest

from schematic_model import deployment as deploy
from schematic_model.inference import InferenceError


@pytest.fixture
def config(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"test model placeholder")
    (model / "config.json").write_text('{"model_type":"test"}')
    return deploy.DeploymentConfig(model=str(model), max_tokens=16, max_seq=64)


def image_url(color="white"):
    buffer = io.BytesIO()
    Image.new("RGB", (24, 16), color).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def messages():
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": image_url()}},
        {"type": "text", "text": "Read R1 exactly."},
    ]}]


def test_help_and_engine_construction_do_not_import_gpu_libraries(config):
    code = """
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','unsloth','transformers','trl'}:
            raise AssertionError('GPU import: ' + fullname)
sys.meta_path.insert(0, Guard())
from schematic_model.deployment import LocalModel, DeploymentConfig, main
LocalModel(DeploymentConfig(model=sys.argv[1]))
main(['--help'])
"""
    result = subprocess.run([sys.executable, "-B", "-c", code, config.model], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("url", ["/etc/passwd", "file:///etc/passwd", "http://localhost/private", "https://example.com/a.png"])
def test_http_images_cannot_read_local_files_or_fetch_remote_urls(url):
    data = messages()
    data[0]["content"][0]["image_url"]["url"] = url
    with pytest.raises(ValueError, match="base64 data URLs"):
        deploy.normalize_messages(data, 128)


def test_decode_enforces_content_type_pixel_budget_and_preserves_aspect_ratio(monkeypatch):
    picture = deploy.decode_image(image_url(), 12)
    assert picture.size == (12, 8)
    with pytest.raises(ValueError, match="MIME type"):
        deploy.decode_image(image_url().replace("image/png", "image/jpeg"), 12)
    monkeypatch.setattr(deploy, "MAX_IMAGE_PIXELS", 10)
    with pytest.raises(ValueError, match="pixel limit"):
        deploy.decode_image(image_url(), 12)


def test_normalization_never_mutates_input_or_changes_text():
    original = messages()
    before = copy.deepcopy(original)
    result = deploy.normalize_messages(original, 128)
    assert original == before
    assert result[0]["content"][1]["text"] == "Read R1 exactly."
    assert isinstance(result[0]["content"][0]["image"], Image.Image)


@pytest.mark.parametrize("override", [
    {"model": "unloaded"}, {"stream": True}, {"tools": []}, {"max_tokens": 17},
    {"max_tokens": True}, {"temperature": float("nan")}, {"temperature": True},
    {"messages": [{"role": "user", "content": [{"type": "image", "image": "/etc/passwd"}]}]},
])
def test_request_rejects_unsupported_or_unbounded_operations(config, override):
    request = {"model": "schematic", "messages": messages(), **override}
    with pytest.raises(ValueError):
        deploy.validate_request(request, config)


@pytest.mark.parametrize("text,tokens", [
    ("partial answer", [10, 11]), ("<think>unfinished", [10, 2]),
    ("answer prefix <think>unfinished", [10, 2]), ("<think>only reasoning</think>", [10, 2]),
])
def test_unfinished_or_reasoning_only_output_is_never_a_prediction(text, tokens):
    with pytest.raises(InferenceError):
        deploy.final_answer(text, tokens, {2})


def test_only_final_answer_is_returned_after_eos():
    assert deploy.final_answer("<think>private notes</think>R1 is 10k.", [10, 2], {2}) == "R1 is 10k."


def test_model_fingerprint_changes_when_weights_change(config):
    before = deploy.LocalModel(config).fingerprint
    Path(config.model, "model.safetensors").write_bytes(b"different test weights")
    assert deploy.LocalModel(config).fingerprint != before


def test_bundle_copies_adapter_and_remaps_base_without_mutating_original(config, tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    original = '{"base_model_name_or_path":"remote/not-available","r":16}'
    (adapter / "adapter_config.json").write_text(original)
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter weights")
    destination = tmp_path / "bundle"
    result = deploy.prepare_bundle(str(adapter), config.model, str(destination))
    assert (adapter / "adapter_config.json").read_text() == original
    assert (destination / "adapter_model.safetensors").read_bytes() == b"adapter weights"
    assert json.loads((destination / "adapter_config.json").read_text())["base_model_name_or_path"] == config.model
    assert result["bundle"]["base_model"]["path"] == config.model
    with pytest.raises(ValueError, match="already exists"):
        deploy.prepare_bundle(str(adapter), config.model, str(destination))


class Ids(list):
    def __getitem__(self, index):
        result = super().__getitem__(index)
        return Ids(result) if isinstance(index, slice) else result

    def tolist(self):
        return list(self)


class Inputs(dict):
    def to(self, device):
        assert device == "test"
        return self


def fake_loaded_engine(config, input_length=3):
    engine = deploy.LocalModel(config)
    captured = {}

    def apply_chat_template(conversation, **kwargs):
        captured["messages"], captured["template_settings"] = conversation, kwargs
        return Inputs(input_ids=SimpleNamespace(shape=(1, input_length)))

    def generate(**kwargs):
        captured["generation_settings"] = kwargs
        return [Ids([0] * input_length + [10, 2])]

    engine.processor = SimpleNamespace(apply_chat_template=apply_chat_template, decode=lambda *a, **k: "R1 is 10k.")
    engine.model = SimpleNamespace(device="test", generate=generate, generation_config=SimpleNamespace(eos_token_id=2))
    engine.torch = SimpleNamespace(manual_seed=lambda seed: None, inference_mode=nullcontext)
    return engine, captured


def test_generation_disables_reasoning_preserves_messages_and_decodes_only_new_tokens(config):
    engine, captured = fake_loaded_engine(config)
    result = engine.complete("schematic", messages(), max_tokens=16)
    assert result["output"] == "R1 is 10k."
    assert result["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    assert captured["template_settings"]["enable_thinking"] is False
    assert captured["template_settings"]["tokenize"] is True
    assert captured["generation_settings"]["do_sample"] is False
    assert captured["messages"][0]["content"][1]["text"] == "Read R1 exactly."


def test_context_overflow_does_not_generate(config):
    engine, captured = fake_loaded_engine(config, input_length=63)
    with pytest.raises(InferenceError, match="context limit"):
        engine.complete("schematic", messages(), max_tokens=16)
    assert "generation_settings" not in captured


class FakeEngine:
    def __init__(self, config, fail_at=None):
        self.config, self.fail_at, self.calls = config, fail_at, []
        self.fingerprint = "fake-test-fingerprint"
        self.provenance = {"model": "test", "settings": {"max_tokens": config.max_tokens}}

    def complete(self, model, messages, **kwargs):
        self.calls.append(copy.deepcopy(messages))
        if len(self.calls) == self.fail_at:
            raise InferenceError("interrupted for test")
        return {"output": "R1 is 10k.", "finish_reason": "stop", "usage": {"total_tokens": 5}}


def write_dataset(tmp_path):
    Image.new("RGB", (24, 16), "white").save(tmp_path / "sheet.png")
    row = {"id": "sheet-1", "messages": [
        {"role": "user", "content": [{"type": "image", "image": "sheet.png"}, {"type": "text", "text": "Read R1"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "10k"}]},
        {"role": "user", "content": [{"type": "text", "text": "Read R2"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "22k"}]},
    ], "gold": [{"type": "value", "value": "10k"}, {"type": "value", "value": "22k"}]}
    path = tmp_path / "test.jsonl"
    path.write_text(json.dumps(row) + "\n")
    return path


def test_interrupted_batch_resumes_and_image_changes_invalidate_resume(config, tmp_path):
    dataset, output = write_dataset(tmp_path), tmp_path / "predictions.jsonl"
    with pytest.raises(InferenceError):
        deploy.batch_predict(config, str(dataset), str(output), engine=FakeEngine(config, fail_at=2))
    first_line = output.read_text()
    resumed = FakeEngine(config)
    deploy.batch_predict(config, str(dataset), str(output), resume=True, engine=resumed)
    assert len(resumed.calls) == 1
    assert output.read_text().startswith(first_line)
    Image.new("RGB", (24, 16), "black").save(tmp_path / "sheet.png")
    with pytest.raises(ValueError, match="provenance differs"):
        deploy.batch_predict(config, str(dataset), str(output), resume=True, engine=FakeEngine(config))


@pytest.fixture
def server(config):
    engine = FakeEngine(config)
    instance = deploy.make_server(engine, port=0)
    worker = threading.Thread(target=instance.serve_forever, daemon=True)
    worker.start()
    yield instance, engine
    instance.shutdown()
    instance.server_close()
    worker.join(timeout=3)


def request(server, path, payload=None, headers=None):
    instance, _ = server
    body = None if payload is None else json.dumps(payload).encode()
    request = Request(f"http://127.0.0.1:{instance.server_port}" + path, data=body,
                      headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urlopen(request, timeout=3) as response:
            return response.status, json.loads(response.read())
    except HTTPError as error:
        return error.code, json.loads(error.read())


def test_loopback_http_health_models_and_completion(server):
    instance, engine = server
    assert instance.server_address[0] == "127.0.0.1"
    assert request(server, "/health")[1]["status"] == "ok"
    assert request(server, "/v1/models")[1]["data"][0]["id"] == "schematic"
    status, response = request(server, "/v1/chat/completions", {"model": "schematic", "messages": messages()})
    assert status == 200
    assert response["choices"][0]["message"]["content"] == "R1 is 10k."
    assert len(engine.calls) == 1


def test_http_rejects_filesystem_remote_urls_origins_and_unknown_routes(server):
    invalid = messages()
    invalid[0]["content"][0]["image_url"]["url"] = "file:///etc/passwd"
    assert request(server, "/v1/chat/completions", {"model": "schematic", "messages": invalid})[0] == 400
    assert request(server, "/v1/chat/completions", {"model": "schematic", "messages": messages()},
                   headers={"Origin": "https://example.com"})[0] == 400
    assert request(server, "/v1/chat/completions", {"model": "schematic", "messages": messages()},
                   headers={"Host": "example.com"})[0] == 400
    assert request(server, "/../../etc/passwd")[0] == 404
    assert request(server, "/api/inference/load", {})[0] == 404
    assert not server[1].calls


def test_http_rejects_oversized_body_before_generation(server, monkeypatch):
    monkeypatch.setattr(deploy, "MAX_BODY_BYTES", 10)
    assert request(server, "/v1/chat/completions", {"model": "schematic", "messages": messages()})[0] == 413
    assert not server[1].calls


def test_idle_socket_does_not_block_health(server):
    instance, _ = server
    with socket.create_connection(instance.server_address, timeout=2) as idle:
        idle.sendall(b"POST /v1/chat/completions HTTP/1.1\r\n")
        assert request(server, "/health")[0] == 200


def test_connection_limit_returns_busy_and_recovers(server):
    instance, _ = server
    for _ in range(deploy.MAX_HTTP_CONNECTIONS):
        assert instance.connection_slots.acquire(blocking=False)
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(f"http://127.0.0.1:{instance.server_port}/health", timeout=2)
        assert error.value.code == 503
    finally:
        for _ in range(deploy.MAX_HTTP_CONNECTIONS):
            instance.connection_slots.release()
    assert request(server, "/health")[0] == 200


def test_gpu_requests_are_serialized_without_blocking_health(server, monkeypatch):
    instance, engine = server
    started, release = threading.Event(), threading.Event()
    original = engine.complete

    def waiting(*args, **kwargs):
        started.set()
        assert release.wait(3)
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "complete", waiting)
    results = []
    body = {"model": "schematic", "messages": messages()}
    worker = threading.Thread(target=lambda: results.append(request(server, "/v1/chat/completions", body)))
    worker.start()
    try:
        assert started.wait(2)
        assert request(server, "/health")[0] == 200
        status, reply = request(server, "/v1/chat/completions", body)
        assert status == 503 and "busy" in reply["error"]["message"]
    finally:
        release.set()
        worker.join(timeout=3)
    assert results[0][0] == 200
    assert len(engine.calls) == 1


@pytest.mark.parametrize("failure,status", [(RuntimeError("gpu failure"), 503), (InferenceError("incomplete"), 422)])
def test_generation_failure_releases_slot_and_preserves_server(server, monkeypatch, failure, status):
    engine = server[1]
    original = engine.complete

    def broken(*args, **kwargs):
        raise failure

    monkeypatch.setattr(engine, "complete", broken)
    body = {"model": "schematic", "messages": messages()}
    assert request(server, "/v1/chat/completions", body)[0] == status
    monkeypatch.setattr(engine, "complete", original)
    assert request(server, "/health")[0] == 200
    assert request(server, "/v1/chat/completions", body)[0] == 200


@pytest.mark.parametrize("extra", [b"Content-Length: 2\r\n", b"Host: example.com\r\n", b"Transfer-Encoding:\r\n"])
def test_ambiguous_http_framing_is_rejected_without_generation(server, extra):
    instance, engine = server
    with socket.create_connection(instance.server_address, timeout=2) as client:
        client.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                       b"Content-Type: application/json\r\nContent-Length: 2\r\n" + extra + b"\r\n{}")
        assert client.recv(4096).startswith(b"HTTP/1.0 400")
    assert not engine.calls
    assert request(server, "/health")[0] == 200


def make_adapter(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"base_model_name_or_path":"upstream/model"}')
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter weights")
    return adapter


def test_failed_bundle_is_not_published_and_can_be_retried(config, tmp_path, monkeypatch):
    adapter = make_adapter(tmp_path)
    destination = tmp_path / "bundle"
    original = deploy.shutil.copyfile

    def failed_copy(source, target):
        Path(target).write_bytes(b"partial")
        raise OSError("simulated disk failure")

    monkeypatch.setattr(deploy.shutil, "copyfile", failed_copy)
    with pytest.raises(OSError, match="disk failure"):
        deploy.prepare_bundle(str(adapter), config.model, str(destination))
    assert not destination.exists()
    assert not list(tmp_path.glob(".bundle.partial-*"))
    monkeypatch.setattr(deploy.shutil, "copyfile", original)
    result = deploy.prepare_bundle(str(adapter), config.model, str(destination))
    assert result["bundle"]["path"] == str(destination)
    assert result["bundle"] == deploy.model_artifacts(destination)


@pytest.mark.parametrize("parent", ["adapter", "base"])
def test_bundle_cannot_modify_input_artifact_directories(config, tmp_path, parent):
    adapter = make_adapter(tmp_path)
    destination = (adapter if parent == "adapter" else Path(config.model)) / "bundle"
    with pytest.raises(ValueError, match="outside"):
        deploy.prepare_bundle(str(adapter), config.model, str(destination))
    assert not destination.exists()


@pytest.mark.parametrize("mapping", [{}, {"tensor": "missing.safetensors"}, {"tensor": "../model.safetensors"}, {"tensor": 3}])
def test_missing_or_invalid_weight_shards_fail_before_gpu_import(config, mapping):
    Path(config.model, "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    with pytest.raises(ValueError, match="Safetensors index"):
        deploy.LocalModel(config)


@pytest.mark.parametrize("metadata", [[], {}, {"base_model_name_or_path": None}])
def test_invalid_adapter_metadata_is_a_validation_error(config, metadata):
    Path(config.model, "adapter_config.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        deploy.LocalModel(config)


def test_prediction_sidecars_cannot_pollute_model_fingerprint(config, tmp_path):
    dataset = write_dataset(tmp_path)
    output = Path(config.model) / "predictions.jsonl"
    with pytest.raises(ValueError, match="outside model"):
        deploy.batch_predict(config, str(dataset), str(output), engine=FakeEngine(config))
    assert not list(Path(config.model).glob("predictions*"))
