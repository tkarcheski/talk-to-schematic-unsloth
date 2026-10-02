from dataclasses import asdict, replace
import fcntl
import json
import os
from pathlib import Path

from PIL import Image
import pytest

from schematic_model.deployment import model_artifacts
from schematic_model.inference import InferenceError
from schematic_model.training import sha256
from scripts.run_experiments import ExperimentConfig, _digest, build_plan, run_experiments


@pytest.fixture
def case(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    image = snapshot / "sheet.png"
    Image.new("RGB", (16, 16), "white").save(image)
    datasets = []
    for view in ("evidence", "vision"):
        (snapshot / view).mkdir()
        for split in ("val", "test"):
            rows = [{"id": f"{view}-{split}-{index}", "messages": [
                {"role": "user", "content": [{"type": "image", "image": "../sheet.png"},
                                               {"type": "text", "text": "What is R1?"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "10k"}]}],
                "gold": [{"type": "value", "value": "10k"}]} for index in range(3)]
            path = snapshot / view / f"{split}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            datasets.append({"path": str(path.relative_to(snapshot)), "sha256": sha256(path), "rows": 3})
    manifest = {"schema_version": 1, "datasets": datasets,
                "images": [{"path": "sheet.png", "sha256": sha256(image)}]}
    manifest["snapshot_sha256"] = _digest(manifest)
    (snapshot / "manifest.json").write_text(json.dumps(manifest))
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"test weights")
    return ExperimentConfig(snapshot=str(snapshot), model=str(model), out=str(tmp_path / "run"))


class Engines:
    def __init__(self, fail_call=None, incomplete_call=None):
        self.calls = 0
        self.created = []
        self.fail_call = fail_call
        self.incomplete_call = incomplete_call

    def __call__(self, config):
        self.created.append(config)
        owner = self

        class Engine:
            provenance = {"settings": asdict(config), "model": model_artifacts(Path(config.model))}

            def complete(self, *args, **kwargs):
                owner.calls += 1
                if owner.calls == owner.fail_call:
                    raise RuntimeError("hardware failure")
                if owner.calls == owner.incomplete_call:
                    raise InferenceError("No end token")
                return {"output": "10k", "finish_reason": "stop"}

        return Engine()


def test_full_queue_reuses_one_engine_and_preserves_explicit_jobs(case):
    engines = Engines(incomplete_call=2)
    state = run_experiments(case, engine_factory=engines)
    assert state["status"] == "completed"
    assert state["plan"]["grader"] == "keyword-refusal-v1"
    assert len(state["jobs"]) == 6
    assert len(engines.created) == 1
    assert engines.calls == 18
    assert sum(job["failed_turns"] for job in state["jobs"].values()) == 1
    assert all(job["status"] == "completed" for job in state["jobs"].values())


def test_completed_resume_verifies_all_artifacts_and_does_not_load_engine(case):
    run_experiments(case, engine_factory=Engines())
    engines = Engines()
    state = run_experiments(case, resume=True, engine_factory=engines)
    assert state["status"] == "completed"
    assert not engines.created


def test_interrupted_job_resumes_saved_turns_and_skips_completed_jobs(case):
    with pytest.raises(RuntimeError, match="hardware failure"):
        run_experiments(case, engine_factory=Engines(fail_call=5))
    state = json.loads((Path(case.out) / "progress.json").read_text())
    assert state["status"] == "failed"
    assert [job["status"] for job in state["jobs"].values()].count("completed") == 1
    engines = Engines()
    state = run_experiments(case, resume=True, engine_factory=engines)
    assert state["status"] == "completed"
    assert engines.calls == 14  # Four durable successful turns are reused.


@pytest.mark.parametrize("suffix", ["", ".meta.json", ".report.json"])
def test_completed_artifact_tampering_rejects_resume_before_engine(case, suffix):
    state = run_experiments(case, engine_factory=Engines())
    job = state["plan"]["jobs"][0]
    path = Path(job["output"] + suffix)
    path.write_text(path.read_text() + " ")
    engines = Engines()
    with pytest.raises(ValueError, match="artifact changed"):
        run_experiments(case, resume=True, engine_factory=engines)
    assert not engines.created


@pytest.mark.parametrize("changed", ["image", "dataset", "manifest", "model", "settings"])
def test_changed_source_or_configuration_refuses_resume(case, changed):
    run_experiments(case, engine_factory=Engines())
    if changed == "settings":
        case = replace(case, max_tokens=512)
    else:
        path = {"image": Path(case.snapshot) / "sheet.png",
                "dataset": Path(case.snapshot) / "vision/val.jsonl",
                "manifest": Path(case.snapshot) / "manifest.json",
                "model": Path(case.model) / "model.safetensors"}[changed]
        path.write_bytes(path.read_bytes() + b" ")
    engines = Engines()
    with pytest.raises(ValueError):
        run_experiments(case, resume=True, engine_factory=engines)
    assert not engines.created


def test_dry_run_verifies_sources_without_writes_or_gpu(case):
    engines = Engines()
    plan = run_experiments(case, dry_run=True, engine_factory=engines)
    assert len(plan["jobs"]) == 6
    assert not Path(case.out).exists()
    assert not engines.created


def test_resolution_ablation_has_validation_only_explicit_sizes(case):
    case = replace(case, views=("vision",), splits=("val",), generated_smoke=0,
                   image_sizes=(1536, 2048))
    engines = Engines()
    state = run_experiments(case, engine_factory=engines)
    assert len(state["jobs"]) == 2
    assert [config.max_image_size for config in engines.created] == [1536, 2048]
    assert all("test" not in job["id"] for job in state["plan"]["jobs"])
    assert engines.calls == 6


def test_base_model_queue_is_explicit_and_separate(case):
    state = run_experiments(replace(case, base=case.model), dry_run=True)
    assert len(state["jobs"]) == 12
    assert {job["model"] for job in state["jobs"]} == {"candidate", "base"}


def test_fresh_run_refuses_nonempty_output(case):
    Path(case.out).mkdir()
    (Path(case.out) / "unrelated.txt").write_text("keep")
    with pytest.raises(ValueError, match="must be empty"):
        run_experiments(case, engine_factory=Engines())
    assert (Path(case.out) / "unrelated.txt").read_text() == "keep"


def test_output_hardlink_cannot_overwrite_snapshot(case):
    Path(case.out).mkdir()
    source = Path(case.snapshot) / "vision/val.jsonl"
    original = source.read_bytes()
    os.link(source, Path(case.out) / "progress.json")
    with pytest.raises(ValueError, match="aliases protected"):
        run_experiments(case, resume=True, engine_factory=Engines())
    assert source.read_bytes() == original


def test_queue_lock_prevents_concurrent_writer(case):
    Path(case.out).mkdir()
    with (Path(case.out) / ".queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Resume gets past the explicit manifest existence check, then must stop at the lock.
        (Path(case.out) / "progress.json").write_text("{}")
        with pytest.raises(ValueError, match="Another process"):
            run_experiments(case, resume=True, engine_factory=Engines())


def test_model_change_between_preflight_and_engine_is_rejected(case):
    engines = Engines()

    def changed(config):
        (Path(config.model) / "model.safetensors").write_bytes(b"different")
        return engines(config)

    with pytest.raises(ValueError, match="Model artifacts changed"):
        run_experiments(case, engine_factory=changed)
    assert engines.calls == 0


@pytest.mark.parametrize("changed", ["dataset", "image", "manifest"])
def test_source_change_during_engine_construction_stops_before_generation(case, changed):
    case = replace(case, views=("vision",), splits=("val",), generated_smoke=0)
    engines = Engines()

    def changed_engine(config):
        path = Path(case.snapshot) / {"dataset": "vision/val.jsonl", "image": "sheet.png",
                                      "manifest": "manifest.json"}[changed]
        path.write_bytes(path.read_bytes() + b" ")
        return engines(config)

    with pytest.raises(ValueError, match="source changed after preflight"):
        run_experiments(case, engine_factory=changed_engine)
    state = json.loads((Path(case.out) / "progress.json").read_text())
    assert state["status"] == "failed"
    assert next(iter(state["jobs"].values()))["status"] == "failed"
    assert engines.calls == 0
    assert not list(Path(case.out).glob("*.jsonl"))


def test_completed_report_must_match_plan_before_job_is_completed(case, monkeypatch):
    import scripts.run_experiments as runner

    case = replace(case, views=("vision",), splits=("val",), generated_smoke=0)
    actual_benchmark = runner.run_benchmark

    def mismatched_report(config, data, output, **kwargs):
        result = actual_benchmark(config, data, output, **kwargs)
        metadata_path = Path(output + ".meta.json")
        metadata = json.loads(metadata_path.read_text())
        metadata["dataset_sha256"] = "0" * 64
        metadata_path.write_text(json.dumps(metadata))
        result["provenance"] = metadata
        Path(output + ".report.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr(runner, "run_benchmark", mismatched_report)
    with pytest.raises(ValueError, match="provenance disagrees"):
        run_experiments(case, engine_factory=Engines())
    state = json.loads((Path(case.out) / "progress.json").read_text())
    assert state["status"] == "failed"
    assert next(iter(state["jobs"].values()))["status"] == "failed"
    # Keep the attempted answers and mismatched metadata as failure evidence.
    output = Path(state["plan"]["jobs"][0]["output"])
    assert len(output.read_text().splitlines()) == 3
    assert json.loads(Path(str(output) + ".meta.json").read_text())["dataset_sha256"] == "0" * 64


def test_smoke_selection_and_hash_are_repeatable(case):
    first, _, generated = build_plan(case)
    second, _, again = build_plan(case)
    assert first == second and generated == again
    assert len(generated) == 2
    assert all(len(content.splitlines()) == 3 for content in generated.values())


def test_validation_only_requires_smoke_to_be_explicitly_disabled(case):
    with pytest.raises(ValueError, match="explicit test split"):
        build_plan(replace(case, splits=("val",)))


def test_smoke_staging_failure_is_durable_and_can_resume_before_any_job(case, monkeypatch):
    import scripts.run_experiments as runner

    def failed_write(*args):
        raise OSError("disk full")

    with monkeypatch.context() as context:
        context.setattr(runner, "_write_smoke", failed_write)
        with pytest.raises(OSError, match="disk full"):
            run_experiments(case, engine_factory=Engines())
    state = json.loads((Path(case.out) / "progress.json").read_text())
    assert state["status"] == "failed"
    assert state["error"]["type"] == "OSError"
    assert all(job["status"] == "pending" for job in state["jobs"].values())
    resumed = run_experiments(case, resume=True, engine_factory=Engines())
    assert resumed["status"] == "completed"
    assert "error" not in resumed


def test_missing_smoke_after_jobs_completed_is_not_silently_rebuilt(case):
    run_experiments(case, engine_factory=Engines())
    (Path(case.out) / "generated-vision.jsonl").unlink()
    with pytest.raises(ValueError, match="smoke dataset is missing"):
        run_experiments(case, resume=True, engine_factory=Engines())
