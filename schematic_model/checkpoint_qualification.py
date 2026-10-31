"""CPU qualification of fixed validation predictions, never model inference.

This is a literal-answer/completion regression gate, not an engineering review.
Historical benchmark reports remain unchanged; refusal scoring is versioned here.
"""
# cspell:words WRONLY CREAT fchmod
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

from evaluate import (_invalid_constant, _unique_object, REFUSAL_SCORERS, grade, load_gold,
                      contextual_refusal, question_text, score, strip, validate_gold_rows)
from schematic_model.deployment import model_artifacts
from schematic_model.inference import image_fingerprints

FLOORS = {"value.literal_accuracy": 0.95, "refusal.accuracy": 0.98,
          "refusal.missing_component_accuracy": 0.98, "refusal.measurement_accuracy": 0.98,
          "completion.rate": 1.0, "followup.value_literal_accuracy": 0.95,
          "followup.completion_rate": 1.0, "completion.clean_rate": 1.0,
          "followup.clean_completion_rate": 1.0}
CLEAN_ANSWER_SCORER = "standalone-dialogue-roles-v1"
ROLE_LINE = re.compile(r"^[ \t]*(?:user|assistant|system)[ \t]*:?[ \t\r]*$", re.MULTILINE | re.IGNORECASE)


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw):
    value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    if not isinstance(value, dict):
        raise ValueError("Evidence JSON must contain an object")
    return value


def _records(raw, rows, history):
    records = {}
    for line in raw.splitlines():
        if not line.strip():
            raise ValueError("Prediction rows must not be blank")
        item = _json(line)
        key, output = item.get("key"), item.get("output")
        if not isinstance(key, str) or not isinstance(output, str) or key in records:
            raise ValueError("Invalid or duplicate prediction key/output")
        if type(item.get("attempted")) is not bool:
            raise ValueError("Prediction attempted must be a boolean")
        if item.get("status") == "success":
            if (not item["attempted"] or not output or strip(output) != output
                    or item.get("finish_reason") != "stop" or item.get("error") is not None):
                raise ValueError("Successful predictions must be complete attempted final answers")
        elif item.get("status") == "failure":
            if output != "" or not isinstance(item.get("error"), dict):
                raise ValueError("Failed predictions require an empty answer and structured error")
        else:
            raise ValueError("Unknown prediction status")
        records[key] = item
    if set(records) != validate_gold_rows(rows):
        raise ValueError("Predictions must cover every validation turn exactly")
    for row in rows:
        blocked = None
        for turn in range(len(row["gold"])):
            key = f"{row['id']}#{turn}"
            item = records[key]
            if blocked:
                if (item["status"] != "failure" or item["attempted"]
                        or item.get("finish_reason") != "dependency_failed"
                        or item["error"].get("code") != "dependency_failed"
                        or item["error"].get("caused_by") != blocked):
                    raise ValueError("Generated-history dependency evidence is inconsistent")
            elif item["status"] == "failure":
                if (not item["attempted"] or item.get("finish_reason") != "error"
                        or item["error"].get("code") != "generation_failed"):
                    raise ValueError("An unblocked failed turn must record an attempted generation failure")
                if history == "generated":
                    blocked = key
    return records


def _evidence(dataset, predictions, model, rows, image_hashes):
    metadata_path = predictions.with_suffix(predictions.suffix + ".meta.json")
    report_path = predictions.with_suffix(predictions.suffix + ".report.json")
    raw = predictions.read_bytes()
    metadata_raw, report_raw = metadata_path.read_bytes(), report_path.read_bytes()
    metadata, report = _json(metadata_raw), _json(report_raw)
    if (type(metadata.get("schema_version")) is not int or metadata.get("schema_version") != 1 or metadata.get("runner") != "failure-accounted-benchmark"
            or metadata.get("history") not in ("gold", "generated") or metadata.get("profile") != "real-vision"
            or type(metadata.get("temperature")) not in (int, float) or metadata.get("temperature") != 0.0
            or metadata.get("failure_policy") != "empty-answer; generated history blocks dependent turns"):
        raise ValueError("Unsupported benchmark binding or policy")
    if metadata.get("dataset_sha256") != _digest(dataset.read_bytes()) or metadata.get("images_sha256") != image_hashes:
        raise ValueError("Dataset or image fingerprints differ from the benchmark binding")
    if type(report.get("schema_version")) is not int or report.get("schema_version") != 1 or report.get("provenance") != metadata or report.get("predictions_sha256") != _digest(raw):
        raise ValueError("Benchmark report does not bind the exact predictions and metadata")
    deployment = metadata.get("deployment")
    if (not isinstance(deployment, dict) or type(deployment.get("schema_version")) is not int
            or deployment.get("schema_version") != 1
            or deployment.get("backend") != "unsloth-library" or not isinstance(deployment.get("packages"), dict)):
        raise ValueError("Missing verified local deployment provenance")
    if any(not isinstance(deployment["packages"].get(name), str) or not deployment["packages"][name]
           for name in ("unsloth", "transformers", "torch")):
        raise ValueError("Missing inference package versions")
    fingerprint = model_artifacts(model)
    if deployment.get("model") != fingerprint:
        raise ValueError("Model or base artifact fingerprint differs from evaluated artifacts")
    settings = deployment.get("settings")
    required = {"model", "served_model_name", "max_seq", "max_image_size", "max_tokens", "seed",
                "four_bit", "min_free_vram_gb", "agent_tools"}
    request_bounds = {"max_images", "max_text_chars", "max_messages"}
    if not isinstance(settings, dict) or set(settings) - request_bounds not in (required, required - {"agent_tools"}):
        raise ValueError("Missing or unknown inference settings")
    if any(type(settings.get(key, 1)) is not int or settings.get(key, 1) < 1 for key in request_bounds):
        raise ValueError("Invalid inference settings for image-only qualification")
    # The schema-v1 benchmark predates the optional agent-tools mode. Preserve
    # its raw metadata hash, normalizing only this known historical omission.
    legacy_settings = "agent_tools" not in settings
    settings = {"agent_tools": False, **settings}
    if (not isinstance(settings["model"], str) or not settings["model"]
            or not isinstance(settings["served_model_name"], str) or not settings["served_model_name"]
            or type(settings["max_seq"]) is not int
            or type(settings["min_free_vram_gb"]) not in (int, float)
            or not math.isfinite(settings["min_free_vram_gb"]) or settings["min_free_vram_gb"] < 0
            or Path(settings["model"]).resolve() != model.resolve()
            or type(metadata.get("max_tokens")) is not int or metadata["max_tokens"] != settings["max_tokens"]
            or type(settings["max_tokens"]) is not int or not 1 <= settings["max_tokens"] < settings["max_seq"] <= 32768
            or type(settings["max_image_size"]) is not int or not 1 <= settings["max_image_size"] <= 4096
            or type(settings["seed"]) is not int or type(settings["four_bit"]) is not bool
            or settings["agent_tools"] is not False):
        raise ValueError("Invalid inference settings for image-only qualification")
    records = _records(raw, rows, metadata["history"])
    comparison = {key: metadata[key] for key in ("dataset_sha256", "images_sha256", "profile", "history",
                                                "max_tokens", "temperature", "failure_policy")}
    comparison["settings"] = {key: value for key, value in settings.items()
                              if key not in {"model", "served_model_name", "min_free_vram_gb"} | request_bounds}
    comparison["packages"] = deployment["packages"]
    return {"predictions_sha256": _digest(raw), "metadata_sha256": _digest(metadata_raw),
            "benchmark_report_sha256": _digest(report_raw), "model": fingerprint,
            "settings_compatibility": "legacy-agent-tools-disabled" if legacy_settings else "explicit-agent-tools-disabled",
            "comparison": comparison}, records


def _measure(rows, records, refusal_scorer):
    outputs = {key: item["output"] for key, item in records.items()}
    metrics, counts = score(rows, outputs, refusal_scorer=refusal_scorer)
    clean = {key: item["status"] == "success" and ROLE_LINE.search(item["output"]) is None
             for key, item in records.items()}
    followup_values, followup_complete, followup_clean = [], [], []
    refusals: dict[str, list[bool]] = {"missing_component": [], "measurement": []}
    for row in rows:
        for turn, label in enumerate(row["gold"]):
            key = f"{row['id']}#{turn}"
            if label["type"] == "refuse":
                question = question_text(row, turn)
                kind = contextual_refusal(question, "").get("kind")
                if kind not in refusals:
                    raise ValueError("Qualification requires supported source-scoped refusal questions")
                refusals[kind].append(bool(grade(label, outputs[key], question=question,
                                                refusal_scorer=refusal_scorer)["refuse.acc"]))
            if turn == 0:
                continue
            followup_complete.append(records[key]["status"] == "success")
            followup_clean.append(clean[key])
            if label["type"] == "value":
                followup_values.append(grade(label, outputs[key])["value.acc"])
    numerator = {
        "value.literal_accuracy": round(metrics.get("value.acc", 0) * counts.get("value.acc", 0)),
        "refusal.accuracy": round(metrics.get("refuse.acc", 0) * counts.get("refuse.acc", 0)),
        "completion.rate": sum(record["status"] == "success" for record in records.values()),
        "completion.clean_rate": sum(clean.values()),
        "followup.value_literal_accuracy": sum(followup_values),
        "followup.completion_rate": sum(followup_complete),
        "followup.clean_completion_rate": sum(followup_clean),
    }
    denominators = {"value.literal_accuracy": counts.get("value.acc", 0),
                    "refusal.accuracy": counts.get("refuse.acc", 0), "completion.rate": len(records),
                    "completion.clean_rate": len(records),
                    "followup.value_literal_accuracy": len(followup_values),
                    "followup.completion_rate": len(followup_complete),
                    "followup.clean_completion_rate": len(followup_clean)}
    for kind, values in refusals.items():
        numerator[f"refusal.{kind}_accuracy"] = sum(values)
        denominators[f"refusal.{kind}_accuracy"] = len(values)
    if any(count == 0 for count in denominators.values()):
        raise ValueError("Qualification requires value, missing-component/measurement refusal, and follow-up value/completion evidence")
    return {name: {"correct": int(numerator[name]), "total": total, "rate": numerator[name] / total}
            for name, total in denominators.items()}


def qualify(dataset: Path, predictions: Path, model: Path, *, baseline_predictions: Path,
            baseline_model: Path, refusal_scorer="v2") -> dict:
    if refusal_scorer not in REFUSAL_SCORERS:
        raise ValueError("Unknown refusal scorer version")
    dataset_raw = dataset.read_bytes()
    rows = load_gold(dataset)
    if dataset.read_bytes() != dataset_raw:
        raise ValueError("Dataset changed while loading")
    for row in rows:
        if row.get("split") != "val" or not isinstance(row.get("family_id"), str) or not row["family_id"].strip():
            raise ValueError("Checkpoint selection requires family-identified validation rows, never test rows")
        if (not isinstance(row.get("provenance"), dict)
                or row["provenance"].get("evidence_mode") != "image_only_values_and_refusals"):
            raise ValueError("Validation rows must declare image-only evidence")
        if any(label["type"] not in {"value", "refuse"} for label in row["gold"]):
            raise ValueError("This qualification gate supports image-only values and refusals")
    images: dict[str, str] = {}
    for row in rows:
        row_images = image_fingerprints(row["messages"], dataset)
        if not row_images:
            raise ValueError("Every validation row requires image evidence")
        for reference, digest in row_images.items():
            if reference in images and images[reference] != digest:
                raise ValueError("Image changed during qualification")
            images[reference] = digest
    candidate, candidate_records = _evidence(dataset, predictions, model, rows, images)
    baseline, baseline_records = _evidence(dataset, baseline_predictions, baseline_model, rows, images)
    if candidate["comparison"] != baseline["comparison"]:
        raise ValueError("Candidate and baseline require identical inference settings, history, packages and dataset")
    candidate_metrics = _measure(rows, candidate_records, refusal_scorer)
    baseline_metrics = _measure(rows, baseline_records, refusal_scorer)
    failures = []
    for name, floor in FLOORS.items():
        value = candidate_metrics[name]["rate"]
        if value < floor:
            failures.append({"metric": name, "reason": "below_floor", "required": floor, "actual": value})
        if value < baseline_metrics[name]["rate"]:
            failures.append({"metric": name, "reason": "baseline_regression",
                             "required": baseline_metrics[name]["rate"], "actual": value})
    if dataset.read_bytes() != dataset_raw:
        raise ValueError("Dataset changed during qualification")
    for row in rows:
        if any(images[reference] != digest for reference, digest in image_fingerprints(row["messages"], dataset).items()):
            raise ValueError("Image changed during qualification")
    return {
        "schema_version": 1, "qualified": not failures, "refusal_scorer": REFUSAL_SCORERS[refusal_scorer],
        "clean_answer_scorer": CLEAN_ANSWER_SCORER,
        "clean_answer_scope": "Completed final answers without standalone user/assistant/system role lines "
                              "(case-insensitive, optional colon). A limited format check, not semantic correctness.",
        "role_spill_keys": {"candidate": sorted(key for key, item in candidate_records.items() if ROLE_LINE.search(item["output"])),
                            "baseline": sorted(key for key, item in baseline_records.items() if ROLE_LINE.search(item["output"]))},
        "dataset_sha256": _digest(dataset_raw), "candidate": candidate, "baseline": baseline,
        "candidate_metrics": candidate_metrics, "baseline_metrics": baseline_metrics,
        "floors": FLOORS, "failures": failures,
        "scope": "Generated-answer literal/completion regression qualification only. Failed and blocked turns "
                 "remain in denominators. Refusal scoring is a versioned deterministic grammar, not semantic "
                 "engineering certification. No defaults changed; historical reports remain unchanged.",
    }


def _private_parent(path: Path):
    """Create only missing private directories; do not chmod existing user paths."""
    missing = []
    current = path.absolute()
    while not current.exists():
        missing.append(current)
        current = current.parent
    if current.is_symlink() or not current.is_dir():
        raise ValueError("Private output parent must be a real directory")
    for directory in reversed(missing):
        directory.mkdir(mode=0o700)
        directory.chmod(0o700)


def _private_write(path, raw):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _outside_evaluated_models(report, destination):
    """Reports and snapshots must not mutate either evaluated model or its base."""
    def roots(fingerprint):
        yield Path(fingerprint["path"]).resolve()
        if "base_model" in fingerprint:
            yield from roots(fingerprint["base_model"])
    for role in ("candidate", "baseline"):
        if any(destination.resolve().is_relative_to(root) for root in roots(report[role]["model"])):
            raise ValueError("Output must remain outside evaluated adapter and base-model directories")


def preserve(report: dict, model: Path, destination: Path) -> dict:
    """Retain a qualified adapter before checkpoint rotation; never copy its base."""
    if not report.get("qualified") or not (model / "adapter_config.json").is_file():
        raise ValueError("Preservation requires a qualified adapter")
    if destination.exists() or destination.is_symlink():
        raise ValueError("Preservation output already exists")
    if not destination.resolve().is_relative_to(Path.cwd().resolve() / "outputs"):
        raise ValueError("Preserved snapshots must be private under ignored outputs")
    _outside_evaluated_models(report, destination)
    expected = report["candidate"]["model"]
    if model_artifacts(model) != expected:
        raise ValueError("Adapter changed after qualification")
    _private_parent(destination.parent)
    with tempfile.TemporaryDirectory(prefix=".qualified-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "snapshot"
        staging.mkdir(mode=0o700)
        staging.chmod(0o700)
        adapter = staging / "adapter"
        adapter.mkdir(mode=0o700)
        adapter.chmod(0o700)
        for artifact in expected["files"]:
            source = model / artifact["name"]
            descriptor = os.open(adapter / source.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as target, source.open("rb") as stream:
                os.fchmod(target.fileno(), 0o600)
                shutil.copyfileobj(stream, target)
                target.flush()
                os.fsync(target.fileno())
        notices = []
        copied = {artifact["name"] for artifact in expected["files"]}
        for source in sorted(model.iterdir()):
            if (source.is_file() and source.name not in copied
                    and source.name.upper().startswith(("LICENSE", "COPYING", "NOTICE", "README"))):
                if source.stat().st_size > 1024 * 1024:
                    raise ValueError("License or provenance notice exceeds the snapshot notice budget")
                raw = source.read_bytes()
                _private_write(adapter / source.name, raw)
                notices.append({"name": source.name, "sha256": _digest(raw)})
        actual = model_artifacts(adapter)
        if actual["sha256"] != expected["sha256"] or actual.get("base_model") != expected.get("base_model"):
            raise ValueError("Snapshot differs from qualified artifacts")
        _private_write(staging / "qualification.json", (json.dumps(report, indent=2) + "\n").encode())
        _private_write(staging / "notices.json", (json.dumps(notices, indent=2) + "\n").encode())
        if destination.exists() or destination.is_symlink():
            raise ValueError("Preservation output appeared during staging")
        staging.rename(destination)
    return {"adapter_sha256": expected["sha256"], "base_copied": False,
            "scope": "Adapter retained; its original local base reference must remain available and unchanged."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--refusal-scorer", choices=REFUSAL_SCORERS, default="v2")
    parser.add_argument("--out", type=Path, required=True, help="new private report under outputs/")
    parser.add_argument("--preserve", type=Path, help="new private qualified-adapter snapshot under outputs/")
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink() or not args.out.resolve().is_relative_to(Path.cwd().resolve() / "outputs"):
        parser.error("Report must be a new private path under outputs/")
    if any(args.out.resolve().is_relative_to(model.resolve()) for model in (args.model, args.baseline_model)):
        parser.error("Report must remain outside model artifact directories")
    report = qualify(args.dataset, args.predictions, args.model, baseline_predictions=args.baseline_predictions,
                     baseline_model=args.baseline_model, refusal_scorer=args.refusal_scorer)
    _outside_evaluated_models(report, args.out)
    if args.preserve is not None and report["qualified"]:
        report["preserved"] = preserve(report, args.model, args.preserve)
    _private_parent(args.out.parent)
    _private_write(args.out, (json.dumps(report, indent=2) + "\n").encode())
    print(json.dumps({"qualified": report["qualified"], "refusal_scorer": report["refusal_scorer"],
                      "failed_metrics": sorted({failure["metric"] for failure in report["failures"]})}))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
