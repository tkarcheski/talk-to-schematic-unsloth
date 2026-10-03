"""Conservative geometry flags on exact source coordinates, never pixel claims."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import pytest

from schematic_model.render_eagle import _Drawing
from scripts.audit_readability import audit, close_anchor, source_name_value_labels, target_refdes, text_record


def draw(value, x, y, *, size=1.778, rotation="R0"):
    drawing = _Drawing()
    drawing.text(value, x, y, size=size, rotation=rotation)
    return text_record(ET.fromstring(drawing.elements[0]))


def test_exact_tft_source_overlap_is_flagged_without_moving_labels():
    fixture = json.loads((Path(__file__).parent / "fixtures/real/tfttouchshield-overlap.json").read_text())
    assert "048abde4fd0ae02c8872e1938297e66b5d4e8784" in fixture["source_url"]
    labels = {}
    for part in fixture["instances"]:
        instance = ET.fromstring(part["xml"])
        for attribute in instance.findall("attribute"):
            drawing = _Drawing()
            drawing.primitive(attribute, substitutions={"NAME": part["refdes"], "VALUE": part["value"]})
            labels[part["refdes"], attribute.get("name")] = text_record(ET.fromstring(drawing.elements[0]))
    first, second = labels["R1", "VALUE"], labels["R2", "NAME"]
    assert (first["x_mm"], first["y_mm"]) == (171.45, -70.358)
    assert (second["x_mm"], second["y_mm"]) == (171.45, -70.0786)
    finding = close_anchor(first, second)
    assert finding["perpendicular_gap_mm"] == 0.2794
    assert finding["along_baseline_delta_mm"] == 0
    assert finding["threshold_mm"] == 0.4445
    assert close_anchor(labels["R1", "NAME"], labels["R1", "VALUE"]) is None


def test_flag_rotates_with_text_and_does_not_invent_width_based_collisions():
    assert close_anchor(draw("R1", 0, 0, rotation="R90"), draw("22", .1, 0, rotation="R90"))
    assert close_anchor(draw("R1", 0, 0), draw("22", 1, 0)) is None
    assert close_anchor(draw("R1", 0, 0), draw("22", 0, 2)) is None
    assert close_anchor(draw("R1", 0, 0), draw("R1", 0, 0)) is None
    assert close_anchor(draw("R1", 0, 0), draw("line one\nline two", 0, 0)) is None


def test_different_alignment_is_explicitly_outside_the_narrow_check():
    first, second = draw("R1", 0, 0), draw("22", 0, 0)
    second["horizontal_anchor"] = "end"
    assert close_anchor(first, second) is None


def test_target_identity_must_be_known_not_guessed_from_value():
    assert target_refdes("Read the value shown next to R1.", {"type": "value", "value": "22"}) == "R1"
    assert target_refdes("And what value is marked for R2?", {}) == "R2"
    assert target_refdes("custom prompt", {"refdes": "U$5"}) == "U$5"
    with pytest.raises(ValueError, match="Unknown value question"):
        target_refdes("Tell me about these resistors", {"value": "22"})


def test_native_labels_keep_source_alignment_and_instance_coordinates():
    source = Path(__file__).parent / "fixtures/real/txb0104.sch"
    labels = source_name_value_labels(source, 1)
    assert {label["kind"] for label in labels["R1"]} == {"NAME", "VALUE"}
    for label in labels["R1"]:
        assert "alignment" in label["source"] and len(label["source"]["instance_transform"]) == 4
        assert label["rendered"]["size_mm"] > 0


def test_audit_preserves_inputs_and_does_not_label_unflagged_targets_readable(tmp_path):
    fixture = Path(__file__).parent / "fixtures/real/txb0104.sch"
    source = tmp_path / "sources/adafruit__fixture/board.sch"
    source.parent.mkdir(parents=True)
    shutil.copyfile(fixture, source)
    image = tmp_path / "image.png"
    image.write_bytes(b"frozen image bytes checked by hash; this audit does not decode pixels")
    data = tmp_path / "vision"
    data.mkdir()
    row = {"id": "fixture", "messages": [{"role": "user", "content": [
        {"type": "image", "image": "../image.png"}, {"type": "text", "text": "Read the value shown next to R1."}]}],
        "gold": [{"type": "value", "value": "10K"}], "provenance": {
            "repository": "adafruit/fixture", "source_path": "board.sch", "page": 1,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "source_url": "https://example.invalid/pinned-source",
            "evidence_mode": "image_only_values_and_refusals",
            "images": [{"page": 1, "sha256": hashlib.sha256(image.read_bytes()).hexdigest()}]}}
    for split in ("train", "val", "test"):
        (data / f"{split}.jsonl").write_text(json.dumps(row) + "\n" if split == "train" else "")
    before = {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    result = audit(data, tmp_path)
    assert result["counts"]["targets"] == 1
    assert result["targets"][0]["status"] == "unreviewed_no_axis_collision_flag"
    assert "font-metric" in result["method"]["limitations"]
    assert before == {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    bad = copy.deepcopy(row)
    bad["provenance"]["sha256"] = "0" * 64
    (data / "train.jsonl").write_text(json.dumps(bad) + "\n")
    with pytest.raises(ValueError, match="recorded hash"):
        audit(data, tmp_path)
