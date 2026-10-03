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
    for element in (b'id="library"', b'id="upload"', b'id="stage"', b'id="mode"', b'id="gate-notice"'):
        assert element in html
    assert b'accept="application/pdf' in html
    assert b'role="log"' in html and b'role="alert"' in html
    assert b'href="/favicon.svg"' in html
    favicon, favicon_type = static_asset("/favicon.svg")
    assert favicon.startswith(b"<svg") and favicon_type == "image/svg+xml"
    assert static_asset("/assets/app.js")[1].startswith("text/javascript")
    assert static_asset("/assets/style.css")[1].startswith("text/css")
    assert static_asset("/assets/../../deployment.py") is None
    assert static_asset("/assets/vendor/LICENSE") is None


PDFJS = {
    "pdf.min.mjs": "57456c8e0c81e46be31174b499ef77f2b9f5ee46d04412ba627320a36755d4c2",
    "pdf.worker.min.mjs": "9536359f1b8367850d485731ca1d5e45c159a7b7a0912325e539937aa21ceb18",
}


@pytest.mark.parametrize("name", sorted(PDFJS))
def test_vendored_pdfjs_is_pinned_and_served_as_javascript(name):
    content, mime = static_asset(f"/assets/vendor/{name}")
    assert hashlib.sha256(content).hexdigest() == PDFJS[name]
    assert mime.startswith("text/javascript")
    vendor = Path(__file__).parents[1] / "schematic_model/web/vendor"
    assert "Apache License" in (vendor / "pdfjs-LICENSE").read_text()
    assert PDFJS[name] in (vendor / "README.md").read_text()


def sparkfun_corpus(directory, *, license="CC-BY-SA-4.0", owner="sparkfun", svg_digest=True):
    from schematic_model.render_eagle import source_layout

    repository = f"{owner}/SparkFun_Qwiic_Test"
    source_dir = directory / "sources" / repository.replace("/", "__")
    source_dir.mkdir(parents=True)
    source = source_dir / "board.sch"
    shutil.copyfile(Path(__file__).parent / "fixtures/real/txb0104.sch", source)
    image = directory / "images" / "sheet.png"
    image.parent.mkdir()
    Image.new("RGB", (24, 16), "white").save(image)
    svg, _ = source_layout(source, 1, attribution="SparkFun Electronics | CC BY-SA 4.0 | source notices retained")
    picture = {"path": "images/sheet.png", "page": 1, "sha256": hashlib.sha256(image.read_bytes()).hexdigest()}
    if svg_digest:
        picture["svg_sha256"] = hashlib.sha256(svg.encode()).hexdigest()
    record = {"id": "real-fedcba9876543210", "repository": repository, "commit": "2" * 40,
              "source_path": "board.sch", "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "license": license, "images": [picture]}
    (directory / "manifest.jsonl").write_text(json.dumps(record) + "\n")
    return directory


def test_multi_directory_catalog_lists_sparkfun_first_with_regions(corpus, tmp_path):
    adafruit, _ = corpus
    sparkfun = sparkfun_corpus(tmp_path / "sparkfun")
    catalog = ExampleCatalog([adafruit, sparkfun])
    index = catalog.index()
    assert [entry["publisher"] for entry in index] == ["SparkFun Electronics", "Adafruit Industries"]
    detail = catalog.details("real-fedcba9876543210-p1")
    assert detail["license"] == "CC-BY-SA-4.0" and detail["title"] == "Qwiic Test"
    parts = detail["regions"]["parts"]
    assert "R1" in parts and all(0 <= value <= 1000 for box in parts.values() for value in box)
    assert all(box[0] < box[2] and box[1] < box[3] for box in parts.values())
    prompts = detail["suggested_questions"]
    shown = prompts[0].removeprefix("Show me ").removesuffix(".")
    assert prompts[0].startswith("Show me ") and shown in parts
    assert prompts[-1] == "Zoom back out to the whole sheet."


def test_regions_are_withheld_when_the_render_is_not_reproduced(tmp_path):
    catalog = ExampleCatalog(sparkfun_corpus(tmp_path, svg_digest=False))
    detail = catalog.details("real-fedcba9876543210-p1")
    assert detail["regions"] is None
    assert not any(prompt.startswith("Show me") for prompt in detail["suggested_questions"])
    assert "Zoom back out to the whole sheet." not in detail["suggested_questions"]


@pytest.mark.parametrize("change", [{"license": "CC-BY-SA-3.0"}, {"owner": "someone"}])
def test_catalog_rejects_wrong_license_or_owner_for_a_publisher(tmp_path, change):
    with pytest.raises(ValueError):
        ExampleCatalog(sparkfun_corpus(tmp_path, **change))


@pytest.fixture
def web_server(corpus, tmp_path):
    engine = SimpleNamespace(config=DeploymentConfig(model=str(tmp_path)), model=object(), fingerprint="test")
    calls = []
    engine.outputs = []

    def complete(model, messages, **kwargs):
        calls.append(messages)
        engine.tools_seen = kwargs.get("tools")
        output = engine.outputs.pop(0) if engine.outputs else "CPU transport test response"
        return {"output": output, "usage": {"total_tokens": 4}}

    engine.complete = complete
    server = make_server(engine, port=0, examples=ExampleCatalog(corpus[0]))
    server.test_engine = engine
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


def test_view_tool_round_trip_through_the_browser_transport(web_server):
    from schematic_model.view_tools import VIEW_TOOLS

    server, calls = web_server
    code, _, body = fetch(server, "/api/view-tools")
    tools = json.loads(body)
    assert code == 200 and tools["tools"] == VIEW_TOOLS and "show_label" in tools["instructions"]
    engine = server.test_engine
    engine.outputs = ["<tool_call>\n<function=show_label>\n<parameter=label>\nR1\n</parameter>\n</function>\n</tool_call>"]
    request = {"model": engine.config.served_model_name, "tools": VIEW_TOOLS,
               "messages": [{"role": "system", "content": tools["instructions"]},
                            {"role": "user", "content": "Show me R1."}]}
    code, _, body = fetch(server, "/v1/chat/completions", request, origin="same")
    choice = json.loads(body)["choices"][0]
    assert code == 200 and choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "show_label"
    assert json.loads(call["function"]["arguments"]) == {"label": "R1"}
    assert engine.tools_seen == VIEW_TOOLS
    engine.outputs = ["<tool_call>\n<function=open_url>\n</function>\n</tool_call>"]
    code, _, body = fetch(server, "/v1/chat/completions", request, origin="same")
    assert code == 422  # Malformed viewer commands are never forwarded to the browser.
    changed = {**request, "tools": VIEW_TOOLS[:1]}
    code, _, _ = fetch(server, "/v1/chat/completions", changed, origin="same")
    assert code == 400
