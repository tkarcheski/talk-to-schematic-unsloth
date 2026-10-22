"""CPU checks for private raster review artifacts, not visual-quality scores."""
import copy
import hashlib
import json
import os
import stat
from pathlib import Path

from PIL import Image
import pytest

from schematic_model import visual_review


@pytest.fixture
def review_source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    image = tmp_path / "source.png"
    Image.new("RGBA", (400, 200), (100, 80, 50, 200)).save(image)
    row = {
        "id": "example", "design_id": "design", "family_id": "family", "split": "val",
        "provenance": {
            "repository": "example/circuits", "source_path": "board.sch", "sha256": "a" * 64,
            "license": "CC-BY-SA-3.0", "page": 1, "evidence_mode": "image_only_values_and_refusals",
            "images": [{"page": 1, "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                        "path": str(image)}],
            "private_unneeded_field": str(tmp_path),
        },
        "messages": [{"role": "user", "content": [
            {"type": "image", "image": str(image)}, {"type": "text", "text": "Read R1."},
        ]}, {"role": "assistant", "content": [{"type": "text", "text": "R1 is marked 10k."}]}],
    }
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(row) + "\n")
    return source, tmp_path / "outputs" / "review", row


def test_pack_preserves_hashes_split_and_provenance_without_paths(review_source):
    source, output, row = review_source
    original = source.read_bytes()
    report = visual_review.build(source, output, max_side=100)
    manifest = (output / "manifest.json").read_text()
    assert str(source.parent) not in manifest
    assert "private_unneeded_field" not in manifest
    assert source.read_bytes() == original
    record = report["rows"][0]
    assert record["review"] == {"status": "unreviewed", "readability": None, "notes": None}
    assert record["split"] == "val" and record["family_id"] == "family"
    assert record["provenance"]["license"] == row["provenance"]["license"]
    assert record["source_image_sha256"] == row["provenance"]["images"][0]["sha256"]
    assert record["source_dimensions"] == [400, 200]
    with Image.open(output / record["review_image"]) as image:
        assert image.size == (100, 50) and image.mode == "RGB"
    assert record["review_image_sha256"] == hashlib.sha256((output / record["review_image"]).read_bytes()).hexdigest()
    assert "not exact processor parity" in report["scope"]
    assert json.loads(manifest) == report


def test_does_not_upscale_and_limits_rows(review_source):
    source, output, row = review_source
    second = copy.deepcopy(row)
    second["id"] = "second"
    source.write_text(json.dumps(row) + "\n" + json.dumps(second) + "\n")
    report = visual_review.build(source, output, max_side=1024, max_rows=1)
    assert report["rows"][0]["review_dimensions"] == [400, 200]
    assert report["selection"]["selected_rows"] == 1


def test_hash_mismatch_removes_staging_and_leaves_sources(review_source):
    source, output, row = review_source
    second = copy.deepcopy(row)
    second["id"] = "bad-image"
    second["provenance"]["images"][0]["sha256"] = "b" * 64
    source.write_text(json.dumps(row) + "\n" + json.dumps(second))
    with pytest.raises(ValueError, match="provenance hash"):
        visual_review.build(source, output, max_side=100)
    assert not output.exists()
    assert not list(output.parent.glob(".visual-review-*"))
    assert source.exists()


@pytest.mark.parametrize("change,error", [
    ({"evidence_mode": "native"}, "image-only"),
    ({"source_path": "/private/board.sch"}, "absolute path"),
    ({"source_path": "../board.sch"}, "repository-relative"),
    ({"images": []}, "exactly one image"),
    ({"page": 0}, "positive page"),
    ({"license": ""}, "nonempty"),
])
def test_rejects_unsupported_or_incomplete_provenance(review_source, change, error):
    source, output, row = review_source
    row["provenance"].update(change)
    source.write_text(json.dumps(row))
    with pytest.raises(ValueError, match=error):
        visual_review.build(source, output, max_side=100)
    assert not output.exists()


def test_refuses_existing_output_and_output_outside_ignored_directory(review_source):
    source, output, _ = review_source
    output.mkdir(parents=True)
    sentinel = output / "keep.txt"
    sentinel.write_text("existing work")
    with pytest.raises(ValueError, match="Output exists"):
        visual_review.build(source, output, max_side=100)
    assert sentinel.read_text() == "existing work"
    with pytest.raises(ValueError, match="ignored outputs"):
        visual_review.build(source, source.parent / "public-review", max_side=100)


@pytest.mark.parametrize("settings", [{"max_side": 0}, {"max_side": 4097}, {"max_side": 100, "max_rows": 0}])
def test_rejects_invalid_budgets_before_writing(review_source, settings):
    source, output, _ = review_source
    with pytest.raises(ValueError):
        visual_review.build(source, output, **settings)
    assert not output.parent.exists()


def test_selected_family_split_leakage_is_rejected(review_source):
    source, output, row = review_source
    second = copy.deepcopy(row)
    second.update(id="second", split="train")
    source.write_text(json.dumps(row) + "\n" + json.dumps(second))
    with pytest.raises(ValueError, match="crosses source splits"):
        visual_review.build(source, output, max_side=100)
    assert not output.exists()


def test_decode_uses_hashed_bytes_even_if_source_changes(review_source, monkeypatch):
    source, output, _ = review_source
    original_read = Path.read_bytes
    image_path = source.parent / "source.png"

    def swap_after_read(path):
        raw = original_read(path)
        if path == image_path:
            Image.new("RGB", (20, 20)).save(image_path)
        return raw

    monkeypatch.setattr(Path, "read_bytes", swap_after_read)
    report = visual_review.build(source, output, max_side=100)
    assert report["rows"][0]["source_dimensions"] == [400, 200]


def test_cli_reports_only_pack_status(review_source, capsys):
    source, output, _ = review_source
    assert visual_review.main(["--source", str(source), "--out", str(output), "--max-side", "100"]) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "unreviewed", "selected_rows": 1}


def test_private_permissions_even_with_permissive_umask(review_source):
    source, output, _ = review_source
    previous = os.umask(0)
    try:
        visual_review.build(source, output, max_side=100)
    finally:
        os.umask(previous)
    for path in [output, *output.rglob("*")]:
        assert stat.S_IMODE(path.stat().st_mode) == (0o700 if path.is_dir() else 0o600)


def test_unselected_rows_cannot_hide_family_split_leakage(review_source):
    source, output, row = review_source
    second = copy.deepcopy(row)
    second.update(id="second", split="train")
    source.write_text(json.dumps(row) + "\n" + json.dumps(second))
    with pytest.raises(ValueError, match="crosses source splits"):
        visual_review.build(source, output, max_side=100, max_rows=1)
    assert not output.exists()


def test_output_root_is_runtime_working_directory(review_source, monkeypatch):
    source, output, _ = review_source
    other = source.parent / "another-project"
    other.mkdir()
    monkeypatch.chdir(other)
    with pytest.raises(ValueError, match="ignored outputs"):
        visual_review.build(source, output, max_side=100)
    new_output = other / "outputs" / "review"
    visual_review.build(source, new_output, max_side=100)
    assert (new_output / "manifest.json").is_file()
