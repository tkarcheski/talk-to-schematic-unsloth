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
    {"placement": "unknown"},
])
def test_invalid_run_configuration_is_rejected(dataset, change):
    with pytest.raises(ValueError):
        training.preflight(replace(dataset, **change))


@pytest.fixture
def resume_checkpoint(tmp_path):
    """Structural checkpoint fixture; tests never deserialize these binary stubs."""
    checkpoint = tmp_path / "checkpoint-300"
    checkpoint.mkdir()
    for name in ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt",
                 "rng_state.pth", "training_args.bin"):
        (checkpoint / name).write_bytes(b"nonempty binary fixture")
    (checkpoint / "adapter_config.json").write_text('{"peft_type": "LORA", "r": 16}')
    (checkpoint / "trainer_state.json").write_text('{"global_step": 300}')
    (checkpoint.parent / "training_manifest.json").write_text(json.dumps({"config": {}}))
    return checkpoint


def assert_resume_rejected_without_side_effects(dataset, checkpoint, monkeypatch, error):
    output = Path(dataset.out)
    output.mkdir()
    previous = b'{"status": "interrupted", "previous_run": true}\n'
    manifest = output / "training_manifest.json"
    manifest.write_bytes(previous)
    gpu = Mock(side_effect=AssertionError("resume validation must precede GPU inspection"))
    monkeypatch.setattr(training, "gpu_inventory", gpu)
    with pytest.raises(ValueError, match=error):
        training.run_training(replace(dataset, resume=str(checkpoint)))
    gpu.assert_not_called()
    assert manifest.read_bytes() == previous
    assert {path.name for path in output.iterdir()} == {"training_manifest.json"}


@pytest.mark.parametrize("name", [
    "adapter_model.safetensors", "adapter_config.json", "trainer_state.json",
    "optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin",
])
@pytest.mark.parametrize("problem", ["missing", "empty", "directory"])
def test_incomplete_resume_preserves_prior_manifest_before_gpu(
        dataset, resume_checkpoint, monkeypatch, name, problem):
    artifact = resume_checkpoint / name
    if problem == "empty":
        artifact.write_bytes(b"")
    else:
        artifact.unlink()
        if problem == "directory":
            artifact.mkdir()
    assert_resume_rejected_without_side_effects(dataset, resume_checkpoint, monkeypatch, "nonempty regular file")


@pytest.mark.parametrize("content", [
    b"not JSON", b"\xff", b"[]", b"null", b"{}", b'{"global_step": true}',
    b'{"global_step": 0}', b'{"global_step": -1}', b'{"global_step": 300.0}',
    b'{"global_step": "300"}', b'{"global_step": 299}',
    b'{"global_step": NaN}', b'{"global_step": Infinity}',
    b'{"global_step": 301, "global_step": 300}',
])
def test_invalid_resume_state_preserves_prior_manifest_before_gpu(
        dataset, resume_checkpoint, monkeypatch, content):
    (resume_checkpoint / "trainer_state.json").write_bytes(content)
    assert_resume_rejected_without_side_effects(dataset, resume_checkpoint, monkeypatch, "Resume checkpoint")


@pytest.mark.parametrize("name", ["checkpoint-0", "checkpoint-latest", "arbitrary-directory"])
def test_resume_directory_must_identify_its_positive_checkpoint_step(
        dataset, resume_checkpoint, monkeypatch, name):
    renamed = resume_checkpoint.with_name(name)
    resume_checkpoint.rename(renamed)
    assert_resume_rejected_without_side_effects(dataset, renamed, monkeypatch, "checkpoint-N")


def recorded_resume(dataset, resume_checkpoint):
    output = Path(dataset.out)
    output.mkdir()
    checkpoint = output / resume_checkpoint.name
    resume_checkpoint.rename(checkpoint)
    manifest = output / "training_manifest.json"
    manifest.write_text(json.dumps(training.preflight(dataset)))
    return replace(dataset, resume=str(checkpoint)), manifest


def test_complete_bound_resume_passes_cpu_preflight_without_writes(dataset, resume_checkpoint, monkeypatch):
    config, manifest = recorded_resume(dataset, resume_checkpoint)
    before = manifest.read_bytes()
    gpu = Mock(side_effect=AssertionError("preflight must stay on CPU"))
    monkeypatch.setattr(training, "gpu_inventory", gpu)
    report = training.preflight(config)
    assert report["config"]["resume"] == config.resume
    assert manifest.read_bytes() == before
    gpu.assert_not_called()


@pytest.mark.parametrize("placement", ["auto", "single_gpu"])
def test_resume_legacy_placement_preserves_default_policy(dataset, resume_checkpoint, placement):
    config, manifest = recorded_resume(dataset, resume_checkpoint)
    previous = json.loads(manifest.read_text())
    del previous["config"]["placement"]
    manifest.write_text(json.dumps(previous))
    before = manifest.read_bytes()
    config = replace(config, placement=placement)
    if placement == "auto":
        assert training.preflight(config)["config"]["placement"] == "auto"
    else:
        with pytest.raises(ValueError, match="Resume recipe differs"):
            training.preflight(config)
    assert manifest.read_bytes() == before


@pytest.mark.parametrize("change", ["recipe", "placement", "dataset", "image", "model", "tokenizer", "chat_template", "missing_binding", "foreign_checkpoint"])
def test_resume_changed_binding_preserves_manifest_before_gpu(dataset, resume_checkpoint, monkeypatch, change):
    model = Path(dataset.data).parent / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"original weights")
    (model / "tokenizer.model").write_bytes(b"original tokenizer")
    (model / "chat_templates").mkdir()
    template = model / "chat_templates/default.jinja"
    template.write_text("original chat template")
    config, manifest = recorded_resume(replace(dataset, model=str(model)), resume_checkpoint)
    if change == "recipe":
        config = replace(config, lr=config.lr / 2)
    elif change == "placement":
        config = replace(config, placement="single_gpu")
    elif change == "dataset":
        path = Path(config.data) / "train.jsonl"
        path.write_text(path.read_text() + "\n")
    elif change == "image":
        Image.new("RGB", (32, 24), "red").save(Path(config.data) / "images/train.png")
    elif change == "model":
        (model / "model.safetensors").write_bytes(b"different weights")
    elif change == "tokenizer":
        (model / "tokenizer.model").write_bytes(b"changed tokenizer")
    elif change == "chat_template":
        template.write_text("changed chat template")
    elif change == "missing_binding":
        value = json.loads(manifest.read_text())
        del value["model_binding"]
        manifest.write_text(json.dumps(value))
    else:
        foreign = Path(config.out).parent / "checkpoint-300"
        Path(config.resume).rename(foreign)
        config = replace(config, resume=str(foreign))
    before = manifest.read_bytes()
    gpu = Mock(side_effect=AssertionError("validation must precede GPU inspection"))
    monkeypatch.setattr(training, "gpu_inventory", gpu)
    with pytest.raises(ValueError):
        training.run_training(config)
    assert manifest.read_bytes() == before
    gpu.assert_not_called()


def test_model_binding_detects_changed_adapter_base(tmp_path):
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    (base / "model.safetensors").write_bytes(b"base")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": str(base)}))
    before = training.model_binding(str(adapter))
    (base / "model.safetensors").write_bytes(b"changed")
    assert training.model_binding(str(adapter)) != before


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
        del dim
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


@pytest.mark.parametrize("change", [
    {"eval_steps": 0},
    {"load_best_model_at_end": True},
    {"load_best_model_at_end": True, "eval_steps": 25, "save_steps": 50},
    {"load_best_model_at_end": True, "eval_steps": 50, "max_steps": 10},
    {"early_stopping_patience": 1},
    {"early_stopping_patience": 0, "load_best_model_at_end": True, "eval_steps": 50},
])
def test_invalid_loss_evaluation_recipe_fails_before_gpu(dataset, change, monkeypatch):
    gpu = Mock(side_effect=AssertionError("GPU must not be queried"))
    monkeypatch.setattr(training, "gpu_inventory", gpu)
    with pytest.raises(ValueError):
        training.run_training(replace(dataset, **change))
    gpu.assert_not_called()
    assert not Path(dataset.out).exists()


def test_loss_evaluation_is_opt_in_and_dry_run_records_recipe(dataset, capsys):
    assert training.evaluation_arguments(dataset) == {"eval_strategy": "no"}
    assert training.main([
        "--data", dataset.data, "--out", dataset.out, "--dry-run",
        "--eval-steps", "10", "--save-steps", "10", "--max-steps", "10", "--load-best-model-at-end",
        "--early-stopping-patience", "3",
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["config"]["eval_steps"] == 10
    assert report["config"]["load_best_model_at_end"] is True
    assert report["config"]["early_stopping_patience"] == 3
    assert not Path(dataset.out).exists()


def test_loss_history_rejects_nonfinite_or_missing_best_checkpoint(dataset):
    config = replace(dataset, eval_steps=50, load_best_model_at_end=True)
    state = SimpleNamespace(log_history=[{"step": 50, "eval_loss": float("nan")}])
    with pytest.raises(ValueError, match="nonfinite"):
        training.evaluation_summary(config, state)
    state.log_history = [{"step": 50, "eval_loss": 0.5}]
    with pytest.raises(ValueError, match="without a valid evaluated checkpoint"):
        training.evaluation_summary(config, state)
    assert training.evaluation_summary(config, state, require_best=False)["best_checkpoint"] is None


@pytest.mark.parametrize("placement,expected_device_map", [("auto", None), ("single_gpu", {"": 0})])
def test_mocked_training_wires_loss_selection_callback_and_records_history(dataset, monkeypatch, placement, expected_device_map):
    """Exercise the training integration without importing GPU/Transformers packages."""
    config = replace(dataset, max_steps=100, eval_steps=50, load_best_model_at_end=True, early_stopping_patience=2, placement=placement)
    model = SimpleNamespace(config=SimpleNamespace(_commit_hash="test-revision"), save_pretrained=Mock())
    processor = SimpleNamespace(chat_template="fixture", save_pretrained=Mock())
    state = SimpleNamespace(
        global_step=100, best_model_checkpoint="checkpoint-50", best_metric=0.3,
        log_history=[{"step": 50, "eval_loss": 0.3}, {"step": 100, "eval_loss": 0.4}],
    )
    trainer = SimpleNamespace(state=state, train=Mock(return_value=SimpleNamespace(metrics={"train_loss": 0.2})),
                              save_state=Mock())
    trainer_factory = Mock(return_value=trainer)
    early_stop = Mock()
    cuda = SimpleNamespace(
        is_available=lambda: True, mem_get_info=lambda: (20 * 1024**3, 24 * 1024**3),
        reset_peak_memory_stats=lambda: None, get_device_name=lambda: "test GPU",
        max_memory_allocated=lambda: 1, max_memory_reserved=lambda: 2,
    )
    modules = {
        "unsloth": SimpleNamespace(FastVisionModel=SimpleNamespace(
            from_pretrained=Mock(return_value=(model, processor)), get_peft_model=lambda model, **kw: model,
            for_training=lambda model: None), is_bf16_supported=lambda: True),
        "unsloth.trainer": SimpleNamespace(UnslothVisionDataCollator=lambda *a, **kw: NativeCollator(10)),
        "torch": SimpleNamespace(cuda=cuda, bfloat16="bf16", float16="fp16"),
        "trl": SimpleNamespace(SFTConfig=lambda **kw: SimpleNamespace(**kw), SFTTrainer=trainer_factory),
        "transformers": SimpleNamespace(TrainerCallback=object, EarlyStoppingCallback=early_stop),
    }
    for key, value in modules.items():
        monkeypatch.setitem(sys.modules, key, value)
    monkeypatch.delenv("SCHEMATIC_DEADLINE", raising=False)  # cspell:ignore delenv
    monkeypatch.setattr(training, "gpu_inventory", lambda: [{"free_mib": 20480}])
    report = training.run_training(config)
    assert modules["unsloth"].FastVisionModel.from_pretrained.call_args.kwargs.get("device_map") == expected_device_map
    kwargs = trainer_factory.call_args.kwargs
    assert kwargs["args"].eval_strategy == "steps"
    assert kwargs["args"].eval_steps == kwargs["args"].save_steps == 50
    assert kwargs["args"].load_best_model_at_end is True
    assert kwargs["args"].metric_for_best_model == "eval_loss"
    assert kwargs["args"].greater_is_better is False
    assert kwargs["args"].prediction_loss_only is True
    assert kwargs["args"].restore_callback_states_from_checkpoint is True
    early_stop.assert_called_once_with(early_stopping_patience=2)
    assert kwargs["callbacks"] == [early_stop.return_value]
    assert len(kwargs["eval_dataset"]) == 1
    assert report["status"] == "completed"
    assert report["validation"]["best_checkpoint"] == "checkpoint-50"
    assert report["validation"]["best_metric"] == 0.3
    assert report["validation"]["observations"] == state.log_history
    assert "quality is not measured" in report["validation"]["scope"]
    model.save_pretrained.assert_called_once()


@pytest.mark.parametrize("recorded", [
    {"eval_steps": 25},
    {"load_best_model_at_end": True},
    {"early_stopping_patience": 3},
])
def test_plain_resume_rejects_changed_evaluation_recipe(dataset, resume_checkpoint, monkeypatch, recorded):
    (resume_checkpoint.parent / "training_manifest.json").write_text(json.dumps({"config": recorded}))
    gpu = Mock(side_effect=AssertionError("GPU must not be queried"))
    monkeypatch.setattr(training, "gpu_inventory", gpu)
    with pytest.raises(ValueError, match="Resume evaluation recipe differs"):
        training.run_training(replace(dataset, resume=str(resume_checkpoint)))
    gpu.assert_not_called()
    assert not Path(dataset.out).exists()


def test_plain_resume_accepts_identical_evaluation_recipe(dataset, resume_checkpoint):
    recipe = {"eval_steps": 50, "load_best_model_at_end": True, "early_stopping_patience": 3}
    config, manifest = recorded_resume(replace(dataset, max_steps=400, **recipe), resume_checkpoint)
    before = manifest.read_bytes()
    report = training.preflight(config)
    assert report["config"]["early_stopping_patience"] == 3
    assert manifest.read_bytes() == before


def test_plain_resume_requires_manifest_before_gpu(dataset, resume_checkpoint, monkeypatch):
    (resume_checkpoint.parent / "training_manifest.json").unlink()
    gpu = Mock(side_effect=AssertionError("GPU must not be queried"))
    monkeypatch.setattr(training, "gpu_inventory", gpu)
    with pytest.raises(ValueError, match="Resume requires a readable training_manifest"):
        training.run_training(replace(dataset, resume=str(resume_checkpoint)))
    gpu.assert_not_called()
    assert not Path(dataset.out).exists()


@pytest.mark.parametrize("rows,epochs,limit,explicit,updates,total", [
    (1, 2.0, None, -1, 1, 2),
    (4, 1.0, None, -1, 1, 1),
    (5, 1.0, None, -1, 2, 2),
    (5, 0.5, None, -1, 2, 1),
    (5, 0.51, None, -1, 2, 2),
    (5, 1.01, None, -1, 2, 3),
    (5, 1.01, 4, -1, 1, 2),
    (5, 1.0, 99, -1, 2, 2),
    (1, 0.01, None, 50, 1, 50),
])
def test_optimizer_step_budget_matches_single_gpu_remainder_rounding(
        dataset, rows, epochs, limit, explicit, updates, total):
    config = replace(dataset, epochs=epochs, max_train_rows=limit, max_steps=explicit)
    budget = training.training_step_budget(config, rows)
    assert budget["optimizer_steps_per_epoch"] == updates
    assert budget["max_optimizer_steps"] == total
    assert budget["selected_rows"] == min(rows, limit or rows)


@pytest.mark.parametrize("row_count,epochs,limit,eval_steps", [
    (1, 2.0, None, 3),
    (5, 1.0, 4, 2),
    (5, 0.5, None, 2),
])
def test_epoch_best_selection_refuses_unreachable_evaluation_before_gpu(
        dataset, monkeypatch, row_count, epochs, limit, eval_steps):
    source = Path(dataset.data, "train.jsonl")
    original = json.loads(source.read_text())
    rows = [dict(original, id=f"training-{index}") for index in range(row_count)]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    gpu = Mock(side_effect=AssertionError("GPU must not be queried"))
    monkeypatch.setattr(training, "gpu_inventory", gpu)
    config = replace(dataset, epochs=epochs, max_train_rows=limit, eval_steps=eval_steps,
                     save_steps=eval_steps, load_best_model_at_end=True)
    with pytest.raises(ValueError, match="cannot reach the first evaluation"):
        training.run_training(config)
    gpu.assert_not_called()
    assert not Path(dataset.out).exists()


def test_epoch_best_selection_accepts_evaluation_at_exact_final_step(dataset):
    config = replace(dataset, epochs=2, eval_steps=2, save_steps=2, load_best_model_at_end=True)
    report = training.preflight(config)
    assert report["training_step_budget"]["max_optimizer_steps"] == 2
    assert not Path(dataset.out).exists()


@pytest.mark.parametrize("max_steps", [299, 300])
def test_resume_exhausted_explicit_budget_rejected_before_gpu(
        dataset, resume_checkpoint, monkeypatch, max_steps):
    assert_resume_rejected_without_side_effects(
        replace(dataset, max_steps=max_steps), resume_checkpoint, monkeypatch,
        "requested budget is exhausted")


@pytest.mark.parametrize("max_steps", [301, -1])
def test_resume_remaining_explicit_budget_and_epoch_mode_preflight(
        dataset, resume_checkpoint, monkeypatch, max_steps):
    gpu = Mock(side_effect=AssertionError("preflight must stay on CPU"))
    monkeypatch.setattr(training, "gpu_inventory", gpu)
    report = training.preflight(replace(dataset, max_steps=max_steps, epochs=301, resume=str(resume_checkpoint)))
    assert report["config"]["max_steps"] == max_steps
    gpu.assert_not_called()
    assert not Path(dataset.out).exists()


@pytest.mark.parametrize("row_count,epochs,limit", [
    (1, 299, None),
    (1, 300, None),
    (5, 149.6, None),
    (5, 300, 4),
])
def test_resume_exhausted_epoch_budget_rejected_before_gpu(
        dataset, resume_checkpoint, monkeypatch, row_count, epochs, limit):
    source = Path(dataset.data, "train.jsonl")
    original = json.loads(source.read_text())
    source.write_text("".join(json.dumps(dict(original, id=f"training-{index}")) + "\n"
                              for index in range(row_count)))
    assert_resume_rejected_without_side_effects(
        replace(dataset, epochs=epochs, max_train_rows=limit), resume_checkpoint, monkeypatch,
        "effective epoch step budget; requested budget is exhausted")


@pytest.mark.parametrize("row_count,epochs,limit", [
    (5, 150.01, None),
    (5, 300.01, 4),
])
def test_resume_fractional_epoch_budget_with_one_remaining_step(
        dataset, resume_checkpoint, monkeypatch, row_count, epochs, limit):
    source = Path(dataset.data, "train.jsonl")
    original = json.loads(source.read_text())
    source.write_text("".join(json.dumps(dict(original, id=f"training-{index}")) + "\n"
                              for index in range(row_count)))
    gpu = Mock(side_effect=AssertionError("preflight must stay on CPU"))
    monkeypatch.setattr(training, "gpu_inventory", gpu)
    report = training.preflight(replace(dataset, epochs=epochs, max_train_rows=limit,
                                        resume=str(resume_checkpoint)))
    assert report["training_step_budget"]["max_optimizer_steps"] == 301
    gpu.assert_not_called()
    assert not Path(dataset.out).exists()
def test_model_binding_rejects_missing_indexed_shard(tmp_path):
    (tmp_path / "model-1.safetensors").write_bytes(b"fixture")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"weight": "model-2.safetensors"}}))
    with pytest.raises(ValueError, match="missing or invalid shard"):
        training.model_binding(str(tmp_path))
