"""CPU-only qualification evidence; model files are inert hashing fixtures."""
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import stat

from PIL import Image
import pytest

from schematic_model import checkpoint_qualification as qualification
from schematic_model.deployment import DeploymentConfig, model_artifacts
from schematic_model.inference import image_fingerprints


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_evidence(dataset, path, model, outputs, *, history="gold"):
    row = json.loads(dataset.read_text())
    records = []
    blocked = None
    for turn, output in enumerate(outputs):
        key = f"fixture#{turn}"
        if blocked:
            record = {"key": key, "status": "failure", "attempted": False, "output": "",
                      "finish_reason": "dependency_failed", "error": {"code": "dependency_failed", "caused_by": blocked}}
        elif output is None:
            record = {"key": key, "status": "failure", "attempted": True, "output": "",
                      "finish_reason": "error", "error": {"code": "generation_failed"}}
            if history == "generated":
                blocked = key
        else:
            record = {"key": key, "status": "success", "attempted": True, "output": output, "finish_reason": "stop"}
        records.append(record)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    metadata = {
        "schema_version": 1, "runner": "failure-accounted-benchmark", "dataset_sha256": digest(dataset),
        "images_sha256": image_fingerprints(row["messages"], dataset), "profile": "real-vision", "history": history,
        "max_tokens": 1024, "temperature": 0.0,
        "failure_policy": "empty-answer; generated history blocks dependent turns",
        "deployment": {"schema_version": 1, "backend": "unsloth-library", "model": model_artifacts(model),
                       "settings": asdict(DeploymentConfig(model=str(model))),
                       "packages": {"unsloth": "fixture", "transformers": "fixture", "torch": "fixture"}},
    }
    path.with_suffix(".jsonl.meta.json").write_text(json.dumps(metadata))
    path.with_suffix(".jsonl.report.json").write_text(json.dumps({
        "schema_version": 1, "provenance": metadata, "predictions_sha256": digest(path),
        "metrics": {"refuse.acc": 1.0}, "release_passed": True,
    }))
    return records


def rebind_predictions(path):
    report_path = path.with_suffix(".jsonl.report.json")
    report = json.loads(report_path.read_text())
    report["predictions_sha256"] = digest(path)
    report_path.write_text(json.dumps(report))


def rebind_meta(path, metadata):
    path.with_suffix(".jsonl.meta.json").write_text(json.dumps(metadata))
    report_path = path.with_suffix(".jsonl.report.json")
    report = json.loads(report_path.read_text())
    report["provenance"] = metadata
    report_path.write_text(json.dumps(report))


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = tmp_path / "base"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"inert base weight fixture")
    (base / "config.json").write_text('{}')
    models = []
    for name in ("candidate", "baseline"):
        model = tmp_path / name
        model.mkdir()
        (model / "adapter_model.safetensors").write_bytes(name.encode())
        (model / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": str(base)}))
        models.append(model)
    Image.new("RGB", (16, 16), "white").save(tmp_path / "sheet.png")
    questions = ["Read the value shown next to R1.", "And what value is marked for R2?",
                 "What value is R999, and what does it connect to?",
                 "What measured voltage was observed at IC1.6 during a powered bench test?"]
    answers = ["R1 is marked 10k.", "R2 is marked 20k.", "There is no component R999 in the schematic.",
               "No bench measurement is supplied in this schematic or source evidence. I cannot infer a measured voltage."]
    messages = []
    for index, (question, answer) in enumerate(zip(questions, answers)):
        content = [{"type": "text", "text": question}]
        if index == 0:
            content.insert(0, {"type": "image", "image": "sheet.png"})
        messages += [{"role": "user", "content": content},
                     {"role": "assistant", "content": [{"type": "text", "text": answer}]}]
    dataset = tmp_path / "val.jsonl"
    dataset.write_text(json.dumps({"id": "fixture", "family_id": "public-family", "split": "val",
                                  "provenance": {"evidence_mode": "image_only_values_and_refusals"},
                                  "messages": messages, "gold": [{"type": "value", "value": "10k"},
                                  {"type": "value", "value": "20k"}, {"type": "refuse"}, {"type": "refuse"}]}))
    paths = [tmp_path / "candidate.jsonl", tmp_path / "baseline.jsonl"]
    for path, model in zip(paths, models):
        write_evidence(dataset, path, model, answers)
    return dataset, paths, models, answers


def qualify(evidence, **kwargs):
    dataset, paths, models, _ = evidence
    return qualification.qualify(dataset, paths[0], models[0], baseline_predictions=paths[1], baseline_model=models[1], **kwargs)


def test_pass_recomputes_metrics_and_keeps_historical_reports(evidence):
    _, paths, _, _ = evidence
    historical = paths[0].with_suffix(".jsonl.report.json").read_bytes()
    report = qualify(evidence)
    assert report["qualified"] is True
    assert report["refusal_scorer"] == "contextual-refusal-v2"
    assert report["candidate_metrics"]["followup.value_literal_accuracy"] == {"correct": 1, "total": 1, "rate": 1.0}
    assert report["candidate_metrics"]["completion.rate"]["total"] == 4
    assert paths[0].with_suffix(".jsonl.report.json").read_bytes() == historical
    assert "not semantic engineering certification" in report["scope"]


def test_literal_followup_and_completion_are_separate(evidence):
    dataset, paths, models, answers = evidence
    write_evidence(dataset, paths[0], models[0], [answers[0], "R2 is marked 200k.", *answers[2:]])
    report = qualify(evidence)
    assert not report["qualified"]
    assert report["candidate_metrics"]["completion.rate"]["rate"] == 1
    assert report["candidate_metrics"]["followup.value_literal_accuracy"]["rate"] == 0
    assert {failure["reason"] for failure in report["failures"]} == {"below_floor", "baseline_regression"}


def test_failed_and_dependency_blocked_turns_remain_in_denominator(evidence):
    dataset, paths, models, answers = evidence
    write_evidence(dataset, paths[0], models[0], [None, *answers[1:]], history="generated")
    write_evidence(dataset, paths[1], models[1], answers, history="generated")
    report = qualify(evidence)
    assert report["candidate_metrics"]["value.literal_accuracy"] == {"correct": 0, "total": 2, "rate": 0}
    assert report["candidate_metrics"]["followup.completion_rate"] == {"correct": 0, "total": 3, "rate": 0}
    assert report["candidate_metrics"]["refusal.accuracy"]["rate"] == 0
    assert not report["qualified"]


@pytest.mark.parametrize("target", ["prediction", "dataset", "image", "adapter", "base", "metadata", "missing_report"])
def test_tampered_or_missing_evidence_refuses_qualification(evidence, target):
    dataset, paths, models, _ = evidence
    if target == "prediction":
        paths[0].write_text(paths[0].read_text().replace("20k", "10k"))
    elif target == "dataset":
        dataset.write_text(dataset.read_text() + "\n")
    elif target == "image":
        Image.new("RGB", (16, 16), "black").save(dataset.parent / "sheet.png")
    elif target == "adapter":
        (models[0] / "adapter_model.safetensors").write_bytes(b"changed")
    elif target == "base":
        (dataset.parent / "base/model.safetensors").write_bytes(b"changed")
    elif target == "metadata":
        metadata = json.loads(paths[0].with_suffix(".jsonl.meta.json").read_text())
        metadata["history"] = "generated"
        paths[0].with_suffix(".jsonl.meta.json").write_text(json.dumps(metadata))
    else:
        paths[0].with_suffix(".jsonl.report.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        qualify(evidence)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unfinished", "false_attempt", "fake_block"])
def test_rebound_but_malformed_predictions_fail(evidence, mutation):
    _, paths, _, _ = evidence
    records = [json.loads(line) for line in paths[0].read_text().splitlines()]
    if mutation == "missing":
        records.pop()
    elif mutation == "duplicate":
        records.append(records[0])
    elif mutation == "unfinished":
        records[0]["finish_reason"] = "length"
    elif mutation == "false_attempt":
        records[0]["attempted"] = False
    else:
        records[0].update(status="failure", attempted=False, output="", finish_reason="dependency_failed",
                          error={"code": "dependency_failed", "caused_by": "nonexistent"})
    paths[0].write_text("".join(json.dumps(record) + "\n" for record in records))
    rebind_predictions(paths[0])
    with pytest.raises(ValueError):
        qualify(evidence)


@pytest.mark.parametrize("setting,value", [("history", "generated"), ("seed", 42), ("max_image_size", 2048), ("four_bit", True)])
def test_comparison_requires_identical_execution_policy(evidence, setting, value):
    _, paths, _, _ = evidence
    metadata = json.loads(paths[0].with_suffix(".jsonl.meta.json").read_text())
    if setting == "history":
        metadata[setting] = value
    else:
        metadata["deployment"]["settings"][setting] = value
    rebind_meta(paths[0], metadata)
    with pytest.raises(ValueError, match="identical inference settings"):
        qualify(evidence)


def test_refusal_version_is_explicit_and_semantic_limit_remains(evidence):
    dataset, paths, models, answers = evidence
    write_evidence(dataset, paths[0], models[0], [*answers[:2], "R999 is not shown in the schematic. It is 10k.", answers[3]])
    assert qualify(evidence, refusal_scorer="v1")["qualified"]
    assert not qualify(evidence, refusal_scorer="v2")["qualified"]


def test_test_split_cannot_select_checkpoints(evidence):
    dataset, _, _, _ = evidence
    row = json.loads(dataset.read_text())
    row["split"] = "test"
    dataset.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="never test rows"):
        qualify(evidence)


def test_missing_followup_values_refuses_gate(evidence):
    dataset, paths, models, answers = evidence
    row = json.loads(dataset.read_text())
    row["gold"].pop(1)
    del row["messages"][2:4]
    dataset.write_text(json.dumps(row))
    for path, model in zip(paths, models):
        write_evidence(dataset, path, model, [answers[0], *answers[2:]])
    with pytest.raises(ValueError, match="follow-up value/completion evidence"):
        qualify(evidence)


def test_preservation_retains_qualified_weights_privately_without_base_or_default_changes(evidence):
    dataset, _, models, _ = evidence
    report = qualify(evidence)
    destination = dataset.parent / "outputs/qualified-one"
    before = (models[0] / "adapter_model.safetensors").read_bytes()
    previous = os.umask(0)
    try:
        saved = qualification.preserve(report, models[0], destination)
    finally:
        os.umask(previous)
    assert saved["base_copied"] is False
    assert (destination / "adapter/adapter_model.safetensors").read_bytes() == before
    assert (models[0] / "adapter_model.safetensors").read_bytes() == before
    for path in [destination, *destination.rglob("*")]:
        assert stat.S_IMODE(path.stat().st_mode) == (0o700 if path.is_dir() else 0o600)
    with pytest.raises(ValueError, match="already exists"):
        qualification.preserve(report, models[0], destination)


def test_preservation_rejects_late_model_change_or_unqualified_report(evidence):
    dataset, _, models, _ = evidence
    report = qualify(evidence)
    (models[0] / "adapter_model.safetensors").write_bytes(b"later checkpoint")
    with pytest.raises(ValueError, match="changed after qualification"):
        qualification.preserve(report, models[0], dataset.parent / "outputs/new")
    report["qualified"] = False
    with pytest.raises(ValueError, match="qualified adapter"):
        qualification.preserve(report, models[0], dataset.parent / "outputs/new")


@pytest.mark.parametrize("field,value", [("max_seq", "8192"), ("model", None), ("seed", True)])
def test_invalid_settings_are_rejected_cleanly(evidence, field, value):
    _, paths, _, _ = evidence
    metadata = json.loads(paths[0].with_suffix(".jsonl.meta.json").read_text())
    metadata["deployment"]["settings"][field] = value
    rebind_meta(paths[0], metadata)
    with pytest.raises(ValueError, match="Invalid inference settings"):
        qualify(evidence)


def test_missing_inference_versions_are_not_comparable(evidence):
    _, paths, _, _ = evidence
    for path in paths:
        metadata = json.loads(path.with_suffix(".jsonl.meta.json").read_text())
        metadata["deployment"]["packages"] = {}
        rebind_meta(path, metadata)
    with pytest.raises(ValueError, match="Missing inference package versions"):
        qualify(evidence)


def test_duplicate_json_fields_cannot_hide_prediction_status(evidence):
    _, paths, _, _ = evidence
    paths[0].write_text(paths[0].read_text().replace('"status": "success"', '"status": "failure", "status": "success"', 1))
    rebind_predictions(paths[0])
    with pytest.raises(ValueError, match="Duplicate JSON field"):
        qualify(evidence)


def test_generated_history_rejects_wrong_dependency_parent(evidence):
    dataset, paths, models, answers = evidence
    write_evidence(dataset, paths[0], models[0], [None, *answers[1:]], history="generated")
    write_evidence(dataset, paths[1], models[1], answers, history="generated")
    records = [json.loads(line) for line in paths[0].read_text().splitlines()]
    records[2]["error"]["caused_by"] = "fixture#1"
    paths[0].write_text("".join(json.dumps(record) + "\n" for record in records))
    rebind_predictions(paths[0])
    with pytest.raises(ValueError, match="dependency evidence is inconsistent"):
        qualify(evidence)


def test_preservation_retains_license_notices(evidence):
    dataset, _, models, _ = evidence
    # LICENSE has no model-artifact suffix, but must still survive retention.
    (models[0] / "LICENSE").write_text("Example licensed fixture notice")
    report = qualify(evidence)
    destination = dataset.parent / "outputs/qualified-notice"
    qualification.preserve(report, models[0], destination)
    assert (destination / "adapter/LICENSE").read_text() == "Example licensed fixture notice"
    assert json.loads((destination / "notices.json").read_text())[0]["sha256"] == digest(models[0] / "LICENSE")


def test_cli_failed_candidate_writes_private_report_but_no_snapshot(evidence, capsys):
    dataset, paths, models, answers = evidence
    write_evidence(dataset, paths[0], models[0], [answers[0], None, *answers[2:]])
    destination = dataset.parent / "outputs/qualifications/candidate.json"
    snapshot = dataset.parent / "outputs/should-not-exist"
    previous = os.umask(0)
    try:
        code = qualification.main([
            "--dataset", str(dataset), "--predictions", str(paths[0]), "--model", str(models[0]),
            "--baseline-predictions", str(paths[1]), "--baseline-model", str(models[1]),
            "--out", str(destination), "--preserve", str(snapshot),
        ])
    finally:
        os.umask(previous)
    assert code == 1 and not json.loads(capsys.readouterr().out)["qualified"]
    assert not snapshot.exists()
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert stat.S_IMODE(destination.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(destination.parent.parent.stat().st_mode) == 0o700


def test_evidence_assisted_rows_cannot_claim_image_only_qualification(evidence):
    dataset, _, _, _ = evidence
    row = json.loads(dataset.read_text())
    row["provenance"]["evidence_mode"] = "native_evidence"
    dataset.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="declare image-only evidence"):
        qualify(evidence)


def test_known_legacy_agent_tools_omission_compares_without_rewriting_metadata(evidence):
    _, paths, _, _ = evidence
    metadata = json.loads(paths[1].with_suffix('.jsonl.meta.json').read_text())
    del metadata['deployment']['settings']['agent_tools']
    rebind_meta(paths[1], metadata)
    original = paths[1].with_suffix('.jsonl.meta.json').read_bytes()
    report = qualify(evidence)
    assert report['qualified']
    assert report['baseline']['settings_compatibility'] == 'legacy-agent-tools-disabled'
    assert report['baseline']['comparison']['settings']['agent_tools'] is False
    assert paths[1].with_suffix('.jsonl.meta.json').read_bytes() == original


@pytest.mark.parametrize('missing_turn', [2, 3])
def test_each_refusal_kind_requires_evidence(evidence, missing_turn):
    dataset, paths, models, answers = evidence
    row = json.loads(dataset.read_text())
    del row['gold'][missing_turn]
    del row['messages'][2 * missing_turn:2 * missing_turn + 2]
    dataset.write_text(json.dumps(row))
    kept = [answer for turn, answer in enumerate(answers) if turn != missing_turn]
    for path, model in zip(paths, models):
        write_evidence(dataset, path, model, kept)
    with pytest.raises(ValueError, match='missing-component/measurement refusal'):
        qualify(evidence)


def test_measurement_refusal_cannot_hide_in_aggregate(evidence):
    dataset, paths, models, answers = evidence
    write_evidence(dataset, paths[0], models[0], [*answers[:3], 'A measured voltage of 3.3 V was observed.'])
    report = qualify(evidence)
    assert report['candidate_metrics']['refusal.missing_component_accuracy']['rate'] == 1
    assert report['candidate_metrics']['refusal.measurement_accuracy']['rate'] == 0
    assert any(failure['metric'] == 'refusal.measurement_accuracy' for failure in report['failures'])


def test_unknown_refusal_question_cannot_receive_a_qualified_label(evidence):
    dataset, paths, models, answers = evidence
    row = json.loads(dataset.read_text())
    row['messages'][-2]['content'][0]['text'] = 'Is this design safe?'
    dataset.write_text(json.dumps(row))
    for path, model in zip(paths, models):
        write_evidence(dataset, path, model, answers)
    with pytest.raises(ValueError, match='supported source-scoped refusal questions'):
        qualify(evidence)


def test_report_and_preservation_cannot_mutate_evaluated_base(evidence):
    dataset, paths, models, answers = evidence
    base = dataset.parent / 'outputs/base'
    base.parent.mkdir()
    (dataset.parent / 'base').rename(base)
    for model, path in zip(models, paths):
        (model / 'adapter_config.json').write_text(json.dumps({'base_model_name_or_path': str(base)}))
        write_evidence(dataset, path, model, answers)
    report = qualify(evidence)
    fingerprint = model_artifacts(base)
    with pytest.raises(ValueError, match='outside evaluated adapter and base-model'):
        qualification.preserve(report, models[0], base / 'snapshot')
    with pytest.raises(ValueError, match='outside evaluated adapter and base-model'):
        qualification.main(['--dataset', str(dataset), '--predictions', str(paths[0]), '--model', str(models[0]),
                            '--baseline-predictions', str(paths[1]), '--baseline-model', str(models[1]),
                            '--out', str(base / 'qualification.json')])
    assert model_artifacts(base) == fingerprint
    assert not (base / 'qualification.json').exists()
    assert not (base / 'snapshot').exists()
