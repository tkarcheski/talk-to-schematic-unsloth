"""CPU contracts for training; actual GPU smoke results live with run artifacts."""

from dataclasses import replace
import builtins
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import signal
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

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


def test_board_family_leakage_is_rejected_with_distinct_source_ids(dataset):
    for split in ("train", "val"):
        example = row(split, f"images/{split}.png")
        example.update(family_id="same-board-family", source_id=f"different-{split}-revision")
        Path(dataset.data, f"{split}.jsonl").write_text(json.dumps(example))
    with pytest.raises(ValueError, match="leakage in family_id"):
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


@pytest.mark.parametrize("split", ["train", "val"])
def test_training_reread_rejects_changed_jsonl_after_preflight(dataset, split):
    manifest = training.preflight(dataset)
    path = Path(dataset.data, f"{split}.jsonl")
    changed = json.loads(path.read_text())
    changed["messages"][-1]["content"][0]["text"] = "A different training target."
    path.write_text(json.dumps(changed) + "\n")
    hashes = {name: record["sha256"] for name, record in manifest["datasets"].items()}
    with pytest.raises(ValueError, match=f"dataset changed since preflight: .*{split}.jsonl"):
        training._read_data(dataset, expected_hashes=hashes)


def test_training_reread_parses_verified_jsonl_bytes_if_path_changes_after_read(dataset):
    manifest = training.preflight(dataset)
    hashes = {name: record["sha256"] for name, record in manifest["datasets"].items()}
    source = Path(dataset.data, "train.jsonl").resolve()
    read_bytes = Path.read_bytes

    def mutate_after_read(path):
        content = read_bytes(path)
        if path == source:
            changed = json.loads(content)
            changed["messages"][-1]["content"][0]["text"] = "Mutated after read."
            path.write_text(json.dumps(changed) + "\n")
        return content

    with patch.object(Path, "read_bytes", mutate_after_read):
        train, _ = training._read_data(dataset, expected_hashes=hashes)
    assert train[0]["messages"][-1]["content"][0]["text"] == "A test drawing."
    assert json.loads(source.read_text())["messages"][-1]["content"][0]["text"] == "Mutated after read."


def test_preflight_rejects_jsonl_changed_between_hash_and_parse(dataset):
    source = Path(dataset.data, "train.jsonl")
    hash_file = training.sha256

    def mutate_after_hash(path):
        digest = hash_file(path)
        if path == source:
            changed = json.loads(path.read_text())
            changed["messages"][-1]["content"][0]["text"] = "Changed during preflight."
            path.write_text(json.dumps(changed) + "\n")
        return digest

    with patch.object(training, "sha256", side_effect=mutate_after_hash):
        with pytest.raises(ValueError, match="dataset changed since preflight"):
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


def test_sigterm_records_interruption_and_cli_exits_143_without_gpu_imports(dataset):
    script = """
import builtins, os, signal, sys
from schematic_model import training

def previous_handler(signum, frame):
    raise AssertionError('previous handler must not run during training')

signal.signal(signal.SIGTERM, previous_handler)
training.gpu_inventory = lambda: [{'free_mib': 16384}]
original_import = builtins.__import__
def cpu_only_import(name, *args, **kwargs):
    if name == 'unsloth':
        os.kill(os.getpid(), signal.SIGTERM)
        raise AssertionError('SIGTERM did not unwind training')
    if name.split('.')[0] in {'torch', 'transformers', 'trl'}:
        raise AssertionError('unexpected GPU import: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = cpu_only_import
code = training.main(['--data', sys.argv[1], '--out', sys.argv[2]])
assert signal.getsignal(signal.SIGTERM) is previous_handler
raise SystemExit(code)
"""
    result = subprocess.run([sys.executable, "-B", "-c", script, dataset.data, dataset.out],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 143, result.stderr
    assert "interrupted by SIGTERM" in result.stderr and "Traceback" not in result.stderr
    manifest = json.loads(Path(dataset.out, "training_manifest.json").read_text())
    assert manifest["status"] == "interrupted"
    assert manifest["signal"] == "SIGTERM" and manifest["signal_number"] == 15
    assert manifest["elapsed_seconds"] >= 0
    assert "training_metrics" not in manifest


def test_sigterm_guard_restores_handler_after_normal_exit():
    previous = signal.getsignal(signal.SIGTERM)
    with training._interrupt_on_sigterm():
        assert signal.getsignal(signal.SIGTERM) != previous
    assert signal.getsignal(signal.SIGTERM) == previous


def test_worker_thread_does_not_try_to_register_process_signal_handler():
    previous = signal.getsignal(signal.SIGTERM)

    def worker():
        with training._interrupt_on_sigterm():
            return signal.getsignal(signal.SIGTERM)

    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(worker).result(timeout=5) == previous
    assert signal.getsignal(signal.SIGTERM) == previous


def test_training_failure_stays_failed_and_restores_signal_handler(dataset, monkeypatch):
    previous = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(training, "gpu_inventory", lambda: [{"free_mib": 16384}])
    original_import = builtins.__import__

    def unavailable_backend(name, *args, **kwargs):
        if name == "unsloth":
            raise ImportError("CPU-only simulated missing backend")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable_backend)
    assert training.main(["--data", dataset.data, "--out", dataset.out]) == 2
    assert signal.getsignal(signal.SIGTERM) == previous
    manifest = json.loads(Path(dataset.out, "training_manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["error_type"] == "ImportError"
    assert "signal" not in manifest and "signal_number" not in manifest


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


def image_hashes(dataset):
    return {image["path"]: image["sha256"] for image in training.preflight(dataset)["images"]}


def test_collator_rejects_overlength_instead_of_truncating_images(dataset):
    examples = training.read_rows(Path(dataset.data, "train.jsonl"))
    native = NativeCollator(5000)
    checked = training.CheckedVisionCollator(native, max_seq=4096, image_hashes=image_hashes(dataset))
    assert native.truncation is False
    with pytest.raises(ValueError, match="5000 tokens"):
        checked(examples)
    assert isinstance(examples[0]["messages"][1]["content"][0]["image"], str)


def test_collator_rejects_an_all_masked_training_example(dataset):
    checked = training.CheckedVisionCollator(NativeCollator(10, supervised=False), max_seq=4096,
                                             image_hashes=image_hashes(dataset))
    with pytest.raises(ValueError, match="no supervised assistant tokens"):
        checked(training.read_rows(Path(dataset.data, "train.jsonl")))


def test_collator_rejects_image_changed_after_preflight(dataset):
    expected = image_hashes(dataset)
    examples = training.read_rows(Path(dataset.data, "train.jsonl"))
    Image.new("RGB", (32, 24), "red").save(Path(dataset.data, "images/train.png"))
    native = Mock()
    checked = training.CheckedVisionCollator(native, max_seq=4096, image_hashes=expected)
    with pytest.raises(ValueError, match="changed since preflight"):
        checked(examples)
    native.assert_not_called()


def test_collator_rejects_an_image_not_recorded_at_preflight(dataset):
    native = Mock()
    checked = training.CheckedVisionCollator(native, max_seq=4096, image_hashes={})
    with pytest.raises(ValueError, match="not recorded at preflight"):
        checked(training.read_rows(Path(dataset.data, "train.jsonl")))
    native.assert_not_called()


def test_collator_decodes_the_verified_bytes_when_path_changes_after_read(dataset):
    expected = image_hashes(dataset)
    examples = training.read_rows(Path(dataset.data, "train.jsonl"))
    source = Path(dataset.data, "images/train.png").resolve()
    read_bytes = Path.read_bytes

    def mutate_after_read(path):
        data = read_bytes(path)
        if path == source:
            Image.new("RGB", (32, 24), "red").save(source)
        return data

    pixels = []

    def native(examples):
        pixels.append(examples[0]["messages"][1]["content"][0]["image"].getpixel((0, 0)))
        return {"input_ids": SimpleNamespace(shape=(1, 10)), "labels": Labels(True)}

    checked = training.CheckedVisionCollator(native, max_seq=4096, image_hashes=expected)
    with patch.object(Path, "read_bytes", mutate_after_read):
        checked(examples)
    assert pixels == [(255, 255, 255)]
    with Image.open(source) as now:
        assert now.getpixel((0, 0)) == (255, 0, 0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_numerical_failure_cannot_be_published_as_completed(value):
    with pytest.raises(RuntimeError, match="nonfinite train_loss"):
        training.validate_metrics({"train_loss": value})
