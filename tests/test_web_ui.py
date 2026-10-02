"""Corpus integrity, no-gold catalog, static assets and browser transport routes."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import threading
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image
import pytest

from schematic_model.deployment import DeploymentConfig, make_server
from schematic_model.web_ui import ExampleCatalog, static_asset


@pytest.fixture
def corpus(tmp_path):
    repository = "adafruit/Adafruit-TXB0104-PCB"
    source_dir = tmp_path / "sources" / repository.replace("/", "__")
    source_dir.mkdir(parents=True)
    source = source_dir / "board.sch"
    shutil.copyfile(Path(__file__).parent / "fixtures/real/txb0104.sch", source)
    image = tmp_path / "images" / "sheet.png"
    image.parent.mkdir()
    Image.new("RGB", (24, 16), "white").save(image)
    record = {
        "id": "real-0123456789abcdef", "repository": repository,
        "commit": "1" * 40, "source_path": "board.sch",
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "license": "CC-BY-SA-3.0",
        "images": [{"path": "images/sheet.png", "page": 1, "sha256": hashlib.sha256(image.read_bytes()).hexdigest()}],
    }
    (tmp_path / "manifest.jsonl").write_text(json.dumps(record) + "\n")
    return tmp_path, record


def test_catalog_uses_native_source_facts_without_dataset_gold(corpus):
    directory, _ = corpus
    catalog = ExampleCatalog(directory)
    assert len(catalog.index()) == 1
    detail = catalog.details("real-0123456789abcdef-p1")
    assert detail["source_evidence"]["nets"]["OE"] == ["JP4.1", "R1.1", "U2.8"]
    assert "1" * 40 in detail["source_url"]
    assert detail["license"] == "CC-BY-SA-3.0"
    assert "gold" not in detail and "messages" not in detail
    assert not list(directory.glob("train*"))  # Source catalog does not need labelled conversations.
    assert catalog.image(detail["id"]) == (directory / "images/sheet.png").read_bytes()
    assert catalog.details("../../etc/passwd") is None
    assert catalog.image("../../etc/passwd") is None


@pytest.mark.parametrize("change", ["source", "image", "page", "traversal", "duplicate", "missing"])
def test_catalog_rejects_inconsistent_source_image_and_manifest(corpus, change):
    directory, record = corpus
    if change == "source":
        record["sha256"] = "0" * 64
    elif change == "image":
        record["images"][0]["sha256"] = "0" * 64
    elif change == "page":
        record["images"][0]["page"] = 999
    elif change == "traversal":
        record["images"][0]["path"] = "../../etc/passwd"
    elif change == "duplicate":
        record["images"].append(copy.deepcopy(record["images"][0]))
    elif change == "missing":
        record["images"] = []
    (directory / "manifest.jsonl").write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError):
        ExampleCatalog(directory)


def test_catalog_detects_images_changed_after_startup(corpus):
    directory, _ = corpus
    catalog = ExampleCatalog(directory)
    (directory / "images/sheet.png").write_bytes(b"changed image")
    with pytest.raises(ValueError, match="changed after"):
        catalog.image(catalog.index()[0]["id"])


@pytest.mark.parametrize("update", [{"repository": None}, {"images": None}, {"images": [None]}, {"source_path": 4}])
def test_malformed_catalog_fields_are_validation_errors(corpus, update):
    directory, record = corpus
    record.update(update)
    (directory / "manifest.jsonl").write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError):
        ExampleCatalog(directory)


def test_static_routes_are_packaged_allowlist():
    html, mime = static_asset("/")
    assert mime.startswith("text/html")
    assert b'id="example"' in html and b'id="mode"' in html
    assert b'role="log"' in html and b'role="alert"' in html
    assert b'href="/favicon.svg"' in html
    favicon, favicon_type = static_asset("/favicon.svg")
    assert favicon.startswith(b"<svg") and favicon_type == "image/svg+xml"
    assert static_asset("/assets/app.js")[1].startswith("text/javascript")
    assert static_asset("/assets/style.css")[1].startswith("text/css")
    assert static_asset("/assets/../../deployment.py") is None


@pytest.fixture
def web_server(corpus, tmp_path):
    engine = SimpleNamespace(config=DeploymentConfig(model=str(tmp_path)), model=object(), fingerprint="test")
    calls = []

    def complete(model, messages, **kwargs):
        calls.append(messages)
        return {"output": "CPU transport test response", "usage": {"total_tokens": 4}}

    engine.complete = complete
    server = make_server(engine, port=0, examples=ExampleCatalog(corpus[0]))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield server, calls
    server.shutdown()
    server.server_close()
    worker.join(timeout=3)


def fetch(server, path, payload=None, origin=None):
    url = f"http://127.0.0.1:{server.server_port}"
    body = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if origin is not None:
        headers["Origin"] = url if origin == "same" else origin
    try:
        with urlopen(Request(url + path, data=body, headers=headers), timeout=3) as response:
            return response.status, response.headers, response.read()
    except HTTPError as error:
        return error.code, error.headers, error.read()


def test_browser_catalog_image_security_headers_and_loaded_status(web_server):
    server, _ = web_server
    code, headers, page = fetch(server, "/")
    assert code == 200 and b"Ask the schematic" in page
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert "script-src 'self'" in headers["Content-Security-Policy"]
    code, headers, favicon = fetch(server, "/favicon.svg")
    assert code == 200 and headers["Content-Type"] == "image/svg+xml" and b"<svg" in favicon
    health = json.loads(fetch(server, "/health")[2])
    assert health["loaded"] is True and health["busy"] is False
    index = json.loads(fetch(server, "/api/examples")[2])["examples"]
    detail = json.loads(fetch(server, "/api/examples/" + index[0]["id"])[2])
    assert detail["source_evidence"]["nets"]["OE"] == ["JP4.1", "R1.1", "U2.8"]
    code, headers, image = fetch(server, detail["image_url"])
    assert code == 200 and headers["Content-Type"] == "image/png" and image.startswith(b"\x89PNG")
    for route in ("/api/examples/../../etc/passwd", "/assets/../../deployment.py", "/api/examples/missing/image"):
        assert fetch(server, route)[0] == 404


def test_only_same_origin_browser_calls_reach_model(web_server):
    server, calls = web_server
    payload = {"model": "schematic", "messages": [{"role": "user", "content": "Read R1"}]}
    assert fetch(server, "/v1/chat/completions", payload, "https://example.com")[0] == 400
    assert fetch(server, "/v1/chat/completions", payload, "null")[0] == 400
    assert not calls
    assert fetch(server, "/v1/chat/completions", payload, "same")[0] == 200
    assert len(calls) == 1 and calls[0][0]["content"] == "Read R1"


def test_changed_example_image_fails_closed_without_affecting_model(web_server, corpus):
    server, _ = web_server
    (corpus[0] / "images/sheet.png").write_bytes(b"changed")
    assert fetch(server, "/api/examples/real-0123456789abcdef-p1/image")[0] == 503
    assert fetch(server, "/health")[0] == 200
