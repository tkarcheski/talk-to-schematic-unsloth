"""Native inspection must protect every file produced by a render request."""

import json
import os
from pathlib import Path
import shutil

from PIL import Image
import pytest

from schematic_model.inspect import main


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.sch"
    fixture = Path(__file__).parent / "fixtures/real/txb0104.sch"
    path.write_bytes(fixture.read_bytes())
    return path


def test_existing_companion_svg_is_preserved_before_any_output(source, tmp_path):
    png, svg, evidence = (tmp_path / name for name in ("drawing.png", "drawing.svg", "facts.json"))
    svg.write_bytes(b"existing drawing")
    assert main([str(source), "--render", str(png), "--out", str(evidence)]) == 2
    assert svg.read_bytes() == b"existing drawing"
    assert not png.exists() and not evidence.exists()


@pytest.mark.parametrize("output", ["drawing.png", "drawing.svg"])
def test_evidence_cannot_alias_either_render_artifact(source, tmp_path, output):
    assert main([str(source), "--render", str(tmp_path / "drawing.png"),
                 "--out", str(tmp_path / output)]) == 2
    assert not (tmp_path / "drawing.png").exists()
    assert not (tmp_path / "drawing.svg").exists()


def test_source_and_hardlink_cannot_be_overwritten(source, tmp_path):
    original = source.read_bytes()
    assert main([str(source), "--out", str(source)]) == 2
    os.link(source, tmp_path / "drawing.svg")
    assert main([str(source), "--render", str(tmp_path / "drawing.png")]) == 2
    assert source.read_bytes() == original
    assert (tmp_path / "drawing.svg").read_bytes() == original


def test_dangling_companion_symlink_is_not_followed(source, tmp_path):
    svg = tmp_path / "drawing.svg"
    svg.symlink_to(tmp_path / "missing-target")
    assert main([str(source), "--render", str(tmp_path / "drawing.png")]) == 2
    assert svg.is_symlink()
    assert not (tmp_path / "missing-target").exists()


def test_render_requires_png_destination(source, tmp_path):
    assert main([str(source), "--render", str(tmp_path / "drawing.svg")]) == 2
    assert not (tmp_path / "drawing.svg").exists()


@pytest.mark.parametrize("content", [
    b'<?xml version="1.0"?><eagle><drawing>',
    b'<?xml version="1.0" encoding="unsupported_encoding"?><eagle/>',
])
def test_malformed_xml_is_reported_without_traceback_or_outputs(tmp_path, capsys, content):
    source = tmp_path / "broken.sch"
    source.write_bytes(content)
    assert main([str(source), "--render", str(tmp_path / "drawing.png"),
                 "--out", str(tmp_path / "facts.json")]) == 2
    captured = capsys.readouterr()
    assert "Malformed EAGLE XML" in captured.err
    assert "Traceback" not in captured.err and captured.out == ""
    assert {path.name for path in tmp_path.iterdir()} == {"broken.sch"}


def test_failed_renderer_publishes_no_partial_files(source, tmp_path, monkeypatch):
    def fail_after_writing(source, page, output):
        output.write_bytes(b"partial PNG")
        output.with_suffix(".svg").write_text("partial SVG")
        raise ValueError("rasterization failed")

    monkeypatch.setattr("schematic_model.render_eagle.render_eagle", fail_after_writing)
    assert main([str(source), "--render", str(tmp_path / "drawing.png"),
                 "--out", str(tmp_path / "facts.json")]) == 2
    assert {path.name for path in tmp_path.iterdir()} == {"source.sch"}


def test_output_created_during_render_is_preserved_and_earlier_publication_rolled_back(
        source, tmp_path, monkeypatch):
    def concurrent_writer(source, page, output):
        output.write_bytes(b"new PNG")
        output.with_suffix(".svg").write_bytes(b"new SVG")
        (tmp_path / "drawing.svg").write_bytes(b"another writer owns this")

    monkeypatch.setattr("schematic_model.render_eagle.render_eagle", concurrent_writer)
    assert main([str(source), "--render", str(tmp_path / "drawing.png"),
                 "--out", str(tmp_path / "facts.json")]) == 2
    assert (tmp_path / "drawing.svg").read_bytes() == b"another writer owns this"
    assert not (tmp_path / "drawing.png").exists()
    assert not (tmp_path / "facts.json").exists()
    assert not list(tmp_path.glob(".schematic-inspect-*"))


@pytest.mark.skipif(shutil.which("rsvg-convert") is None, reason="rsvg-convert is required")
def test_real_source_renders_both_images_and_evidence_in_distinct_directories(source, tmp_path):
    png, evidence = tmp_path / "images/drawing.png", tmp_path / "facts/facts.json"
    assert main([str(source), "--render", str(png), "--out", str(evidence)]) == 0
    with Image.open(png) as image:
        assert image.format == "PNG"
        image.verify()
    assert "<svg" in png.with_suffix(".svg").read_text()
    assert json.loads(evidence.read_text())["sheets"]
    assert not list(tmp_path.rglob(".schematic-inspect-*"))
