"""CPU contracts for training; actual GPU smoke results live with run artifacts."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

from schematic_model import training


def row(key, image):
    return {
        "id": key,
        "design_id": key,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "Read the drawing."}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What is visible?"},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "A test drawing."}]},
        ],
    }


@pytest.fixture
def dataset(tmp_path):
    data = tmp_path / "dataset"
    (data / "images").mkdir(parents=True)
    for split, color in (("train", "white"), ("val", "black")):
        image = f"images/{split}.png"
        Image.new("RGB", (32, 24), color).save(data / image)
        (data / f"{split}.jsonl").write_text(json.dumps(row(split, image)) + "\n")
    return training.TrainingConfig(data=str(data), out=str(tmp_path / "output"))


def test_import_and_help_never_import_gpu_libraries():
    script = """
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'unsloth', 'transformers', 'trl'}:
            raise AssertionError('GPU import: ' + fullname)
sys.meta_path.insert(0, Guard())
import train_unsloth
from schematic_model.training import main
try:
    main(['--help'])
except SystemExit as result:
    assert result.code == 0
"""
    result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_dry_run_reports_hashes_and_creates_no_output(dataset, capsys):
    assert training.main(["--data", dataset.data, "--out", dataset.out, "--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "preflight"
    assert report["datasets"]["train"]["rows"] == 1
    assert all(len(image["sha256"]) == 64 for image in report["images"])
    assert report["model_revision"] == training.DEFAULT_REVISION
    assert not Path(dataset.out).exists()


def test_image_paths_are_independent_of_working_directory(dataset, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path.parent)
    report = training.preflight(dataset)
    assert len(report["images"]) == 2
    legacy = Path(dataset.data, "train.jsonl")
    legacy.write_text(json.dumps(row("train", "dataset/images/train.png")))
    assert training.read_rows(legacy)[0]["messages"][1]["content"][0]["image"] == str(Path(dataset.data, "images/train.png"))


def test_split_leakage_is_rejected_even_with_distinct_conversation_ids(dataset):
    validation = row("another-chat", "images/val.png")
    validation["design_id"] = "train"
    Path(dataset.data, "val.jsonl").write_text(json.dumps(validation))
    with pytest.raises(ValueError, match="leakage in design_id"):
        training.preflight(dataset)


def test_duplicate_image_content_is_rejected_across_splits(dataset):
    Path(dataset.data, "images/val.png").write_bytes(Path(dataset.data, "images/train.png").read_bytes())
    with pytest.raises(ValueError, match="identical image bytes"):
        training.preflight(dataset)


@pytest.mark.parametrize("bad_image", ["missing.png", "https://example.com/image.png", "data:image/png;base64,abc"])
def test_missing_and_remote_images_fail_before_model_loading(dataset, bad_image):
    Path(dataset.data, "train.jsonl").write_text(json.dumps(row("train", bad_image)))
    with pytest.raises(ValueError, match="train.jsonl:1"):
        training.preflight(dataset)


def test_invalid_image_is_not_accepted_as_training_data(dataset):
    Path(dataset.data, "images/train.png").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="invalid training image"):
        training.preflight(dataset)


def test_malformed_conversations_report_location(dataset):
    malformed = row("train", "images/train.png")
    malformed["messages"][-1]["role"] = "user"
    Path(dataset.data, "train.jsonl").write_text("\n" + json.dumps(malformed))
    with pytest.raises(ValueError, match="train.jsonl:2: expected assistant"):
        training.preflight(dataset)


@pytest.mark.parametrize("change", [
    {"max_steps": 0}, {"max_train_rows": 0}, {"max_seq": 0}, {"lr": float("nan")},
    {"epochs": float("inf")}, {"model": "Qwen/model-GGUF"}, {"resume": "/nonexistent/checkpoint"},
])
def test_invalid_run_configuration_is_rejected(dataset, change):
    with pytest.raises(ValueError):
        training.preflight(replace(dataset, **change))


def test_low_memory_refuses_before_gpu_import_or_output_creation(dataset, monkeypatch):
    monkeypatch.setattr(training, "gpu_inventory", lambda: [{"free_mib": 1024}])
    with pytest.raises(RuntimeError, match="GiB free VRAM"):
        training.run_training(dataset)
    assert not Path(dataset.out).exists()


def test_existing_output_is_preserved(dataset):
    output = Path(dataset.out)
    output.mkdir()
    sentinel = output / "adapter.safetensors"
    sentinel.write_text("existing weights")
    with pytest.raises(ValueError, match="not empty"):
        training.run_training(dataset)
    assert sentinel.read_text() == "existing weights"


class Labels:
    def __init__(self, supervised):
        self.supervised = supervised

    def __ne__(self, other):
        return self

    def any(self, dim):
        return self

    def all(self):
        return self

    def item(self):
        return self.supervised


class NativeCollator:
    def __init__(self, length, supervised=True):
        self.length, self.supervised = length, supervised

    def __call__(self, examples):
        assert isinstance(examples[0]["messages"][1]["content"][0]["image"], Image.Image)
        return {"input_ids": SimpleNamespace(shape=(1, self.length)), "labels": Labels(self.supervised)}


def test_collator_rejects_overlength_instead_of_truncating_images(dataset):
    examples = training.read_rows(Path(dataset.data, "train.jsonl"))
    native = NativeCollator(5000)
    checked = training.CheckedVisionCollator(native, max_seq=4096)
    assert native.truncation is False
    with pytest.raises(ValueError, match="5000 tokens"):
        checked(examples)
    assert isinstance(examples[0]["messages"][1]["content"][0]["image"], str)


def test_collator_rejects_an_all_masked_training_example(dataset):
    checked = training.CheckedVisionCollator(NativeCollator(10, supervised=False), max_seq=4096)
    with pytest.raises(ValueError, match="no supervised assistant tokens"):
        checked(training.read_rows(Path(dataset.data, "train.jsonl")))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_numerical_failure_cannot_be_published_as_completed(value):
    with pytest.raises(RuntimeError, match="nonfinite train_loss"):
        training.validate_metrics({"train_loss": value})
