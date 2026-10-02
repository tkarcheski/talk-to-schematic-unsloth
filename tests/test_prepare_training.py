import json
from pathlib import Path

from PIL import Image
import pytest

from scripts.prepare_training import prepare
from schematic_model.training import preflight, TrainingConfig


def sources(tmp_path):
    evidence, vision = tmp_path / "evidence", tmp_path / "vision"
    for view, directory in (("evidence", evidence), ("vision", vision)):
        directory.mkdir()
        for split, color in (("train", "red"), ("val", "green"), ("test", "blue")):
            Image.new("RGB", (12, 12), color).save(directory / f"{split}.png")
            row = {"id": f"{split}-{view}", "design_id": split, "family_id": f"family-{split}", "split": split,
                   "messages": [
                       {"role": "user", "content": [{"type": "image", "image": f"{split}.png"},
                                                       {"type": "text", "text": "What is R1?"}]},
                       {"role": "assistant", "content": [{"type": "text", "text": "R1 is 10k"}]},
                   ], "gold": [{"type": "value", "value": "10k"}]}
            (directory / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
    return evidence, vision


def test_mixed_snapshot_retains_every_row_and_keeps_holdouts_separate(tmp_path):
    evidence, vision = sources(tmp_path)
    output = tmp_path / "snapshot"
    manifest = prepare(str(evidence), str(vision), str(output))
    assert manifest["families"] == {split: [f"family-{split}"] for split in ("train", "val", "test")}
    train = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert {row["id"] for row in train} == {"train-evidence", "train-vision"}
    assert len(manifest["images"]) == 3
    assert preflight(TrainingConfig(data=str(output)))["datasets"]["train"]["rows"] == 2
    for view in ("evidence", "vision"):
        row = json.loads((output / view / "test.jsonl").read_text())
        image = row["messages"][0]["content"][0]["image"]
        assert (output / view / image).is_file()


def test_snapshot_is_reproducible_and_source_mutation_cannot_change_it(tmp_path):
    evidence, vision = sources(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    one = prepare(str(evidence), str(vision), str(first))
    two = prepare(str(evidence), str(vision), str(second))
    assert one["snapshot_sha256"] == two["snapshot_sha256"]
    before = [path.read_bytes() for path in sorted((first / "images").iterdir())]
    (evidence / "train.png").write_bytes(b"later mutation")
    assert before == [path.read_bytes() for path in sorted((first / "images").iterdir())]
    with pytest.raises(ValueError, match="already exists"):
        prepare(str(evidence), str(vision), str(first))


def test_family_leakage_fails_before_writing_snapshot(tmp_path):
    evidence, vision = sources(tmp_path)
    path = vision / "test.jsonl"
    row = json.loads(path.read_text())
    row["family_id"] = "family-train"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="family leakage"):
        prepare(str(evidence), str(vision), str(tmp_path / "snapshot"))
    assert not (tmp_path / "snapshot").exists()


def test_heldout_views_must_have_matching_board_families(tmp_path):
    evidence, vision = sources(tmp_path)
    path = vision / "test.jsonl"
    row = json.loads(path.read_text())
    row["family_id"] = "unrelated-test-family"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="test families differ"):
        prepare(str(evidence), str(vision), str(tmp_path / "snapshot"))
