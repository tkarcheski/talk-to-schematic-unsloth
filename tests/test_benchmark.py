import copy
from dataclasses import replace
import json
import os
from pathlib import Path

from PIL import Image
import pytest

from schematic_model.benchmark import run_benchmark
from schematic_model.deployment import DeploymentConfig
from schematic_model.inference import InferenceError


@pytest.fixture
def case(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    Image.new("RGB", (16, 16), "white").save(tmp_path / "sheet.png")
    messages = []
    for index in range(3):
        content = [{"type": "text", "text": f"What is R{index + 1}?"}]
        if index == 0:
            content.insert(0, {"type": "image", "image": "sheet.png"})
        messages += [{"role": "user", "content": content},
                     {"role": "assistant", "content": [{"type": "text", "text": "10k"}]}]
    row = {"id": "sheet", "messages": messages, "gold": [{"type": "value", "value": "10k"}] * 3}
    dataset = tmp_path / "test.jsonl"
    dataset.write_text(json.dumps(row) + "\n")
    return DeploymentConfig(model=str(model), max_tokens=1024), dataset, tmp_path / "out.jsonl"


class Engine:
    provenance = {"model": "fixed-test-model", "settings": {"max_tokens": 1024}}

    def __init__(self, fail_at=None, crash_at=None):
        self.calls = []
        self.fail_at, self.crash_at = fail_at, crash_at

    def complete(self, model, messages, **kwargs):
        self.calls.append(copy.deepcopy(messages))
        if len(self.calls) == self.crash_at:
            raise RuntimeError("hardware failure")
        if len(self.calls) == self.fail_at:
            raise InferenceError("Generation did not reach an end token")
        return {"output": "10k", "finish_reason": "stop"}


def test_gold_history_keeps_model_failure_in_denominator_and_continues(case):
    config, dataset, output = case
    engine = Engine(fail_at=2)
    report = run_benchmark(config, str(dataset), str(output), profile="real-vision", engine=engine)
    assert report["turns"] == report["attempted_turns"] == 3
    assert report["failed_turns"] == 1
    assert report["counts"]["value.acc"] == 3
    assert report["metrics"]["value.acc"] == pytest.approx(2 / 3)
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert records[1]["status"] == "failure" and records[1]["output"] == ""
    assert records[1]["error"]["code"] == "generation_failed"
    assert engine.calls[2][3]["content"][0]["text"] == "10k"
    assert report["release_passed"] is False


def test_generated_history_blocks_dependents_without_gold_fallback(case):
    config, dataset, output = case
    engine = Engine(fail_at=1)
    report = run_benchmark(config, str(dataset), str(output), profile="real-vision", history="generated", engine=engine)
    assert len(engine.calls) == report["attempted_turns"] == 1
    assert report["failed_turns"] == 3
    assert report["metrics"]["value.acc"] == 0
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert records[1]["error"]["code"] == "dependency_failed"
    assert records[2]["error"]["caused_by"] == "sheet#0"


def test_infrastructure_failure_stops_and_resume_preserves_attempted_results(case):
    config, dataset, output = case
    with pytest.raises(RuntimeError, match="hardware failure"):
        run_benchmark(config, str(dataset), str(output), profile="real-vision", engine=Engine(crash_at=2))
    first = output.read_text()
    engine = Engine()
    report = run_benchmark(config, str(dataset), str(output), profile="real-vision", engine=engine, resume=True)
    assert len(engine.calls) == 2
    assert output.read_text().startswith(first)
    assert report["successful_turns"] == 3


def test_failure_records_are_not_retried_during_resume(case):
    config, dataset, output = case
    run_benchmark(config, str(dataset), str(output), profile="real-vision", engine=Engine(fail_at=1), history="generated")
    engine = Engine()
    report = run_benchmark(config, str(dataset), str(output), profile="real-vision", engine=engine, history="generated", resume=True)
    assert not engine.calls
    assert report["failed_turns"] == 3


def test_resume_rejects_changed_images_or_generation_settings(case):
    config, dataset, output = case
    run_benchmark(config, str(dataset), str(output), profile="real-vision", engine=Engine())
    with pytest.raises(ValueError, match="provenance differs"):
        run_benchmark(replace(config, max_tokens=512), str(dataset), str(output), profile="real-vision", engine=Engine(), resume=True)
    Image.new("RGB", (16, 16), "black").save(dataset.parent / "sheet.png")
    with pytest.raises(ValueError, match="provenance differs"):
        run_benchmark(config, str(dataset), str(output), profile="real-vision", engine=Engine(), resume=True)


def test_incomplete_success_response_never_saves_partial_text(case):
    config, dataset, output = case
    engine = Engine()
    engine.complete = lambda *args, **kwargs: {"output": "10k plus unfinished text", "finish_reason": "length"}
    report = run_benchmark(config, str(dataset), str(output), profile="real-vision", engine=engine)
    assert report["failed_turns"] == 3
    assert report["metrics"]["value.acc"] == 0
    assert "unfinished text" not in output.read_text()


def test_benchmark_cannot_overwrite_gold(case):
    config, dataset, _ = case
    original = dataset.read_bytes()
    with pytest.raises(ValueError, match="aliases protected"):
        run_benchmark(config, str(dataset), str(dataset), profile="real-vision", engine=Engine())
    assert dataset.read_bytes() == original


def test_completed_failure_rows_are_compatible_with_existing_grader(case):
    from evaluate import load, load_gold, score

    config, dataset, output = case
    report = run_benchmark(config, str(dataset), str(output), profile="real-vision", engine=Engine(fail_at=2))
    metrics, counts = score(load_gold(dataset), load(output))
    assert metrics == report["metrics"]
    assert counts == report["counts"]


@pytest.mark.parametrize("destination", ["output", "metadata", "report"])
@pytest.mark.parametrize("source", ["dataset", "image", "model"])
def test_every_output_rejects_hardlink_aliases_of_all_evidence(case, destination, source):
    config, dataset, output = case
    model = Path(config.model) / "model.safetensors"
    model.write_bytes(b"saved model")
    sources = {"dataset": dataset, "image": dataset.parent / "sheet.png", "model": model}
    destinations = {"output": output, "metadata": output.with_suffix(".jsonl.meta.json"),
                    "report": output.with_suffix(".jsonl.report.json")}
    original = sources[source].read_bytes()
    os.link(sources[source], destinations[destination])
    with pytest.raises(ValueError, match="aliases protected"):
        run_benchmark(config, str(dataset), str(output), profile="real-vision", engine=Engine())
    assert sources[source].read_bytes() == original
    assert destinations[destination].read_bytes() == original


def test_changed_dataset_failure_invalidates_old_report(case):
    config, dataset, output = case
    run_benchmark(config, str(dataset), str(output), profile="real-vision", engine=Engine())
    report = output.with_suffix(".jsonl.report.json")
    assert report.exists()
    dataset.write_text("not valid json\n")
    with pytest.raises(ValueError):
        run_benchmark(config, str(dataset), str(output), profile="real-vision", engine=Engine(), resume=True)
    assert not report.exists()


def test_infrastructure_failure_invalidates_old_report(case):
    config, dataset, output = case
    run_benchmark(config, str(dataset), str(output), profile="real-vision", engine=Engine())
    report = output.with_suffix(".jsonl.report.json")
    output.write_text("")
    with pytest.raises(RuntimeError):
        run_benchmark(config, str(dataset), str(output), profile="real-vision", engine=Engine(crash_at=1), resume=True)
    assert not report.exists()


def test_model_startup_failure_is_infrastructure_not_failed_answers(case):
    config, dataset, output = case
    engine = Engine()

    def failed_load():
        raise InferenceError("Insufficient free GPU memory to load the local model")

    engine.load = failed_load
    with pytest.raises(InferenceError, match="Insufficient free GPU"):
        run_benchmark(config, str(dataset), str(output), profile="real-vision", engine=engine)
    assert not output.exists()
    assert not output.with_suffix(".jsonl.report.json").exists()
    assert not engine.calls


def test_completed_resume_does_not_reload_model(case):
    config, dataset, output = case
    run_benchmark(config, str(dataset), str(output), profile="real-vision", engine=Engine())
    engine = Engine()

    def failed_load():
        raise AssertionError("Completed resume must not load the model")

    engine.load = failed_load
    report = run_benchmark(config, str(dataset), str(output), profile="real-vision", engine=engine, resume=True)
    assert report["successful_turns"] == 3
    assert not engine.calls


def test_atomic_writer_cannot_overwrite_image_at_fixed_temporary_name(case):
    config, dataset, output = case
    row = json.loads(dataset.read_text())
    collision = output.with_suffix(".jsonl.report.tmp")
    collision.write_bytes((dataset.parent / "sheet.png").read_bytes())
    row["messages"][0]["content"][0]["image"] = collision.name
    dataset.write_text(json.dumps(row) + "\n")
    original = collision.read_bytes()
    report = run_benchmark(config, str(dataset), str(output), profile="real-vision", engine=Engine())
    assert report["successful_turns"] == 3
    assert collision.read_bytes() == original
