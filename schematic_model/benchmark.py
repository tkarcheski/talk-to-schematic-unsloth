"""Account for every benchmark turn, including failed model completions.

This offline runner does not relax the deployment API's complete-answer rule.
Incomplete generations have empty answers and explicit errors. Under generated
history, one failed answer blocks later turns in that conversation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

from evaluate import PROFILES, _write_report, gate_failures, load, load_gold, score, strip, validate_gold_rows
from schematic_model.deployment import DeploymentConfig, LocalModel
from schematic_model.inference import InferenceError, convert_messages, image_fingerprints, resolve_image
from schematic_model.training import sha256


def _load_records(path: Path) -> dict:
    if not path.exists() or not path.stat().st_size:
        return {}
    load(path)  # Shared strict JSONL, duplicate-key and output validation.
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("status") not in {"success", "failure"} or type(record.get("attempted")) is not bool:
            raise ValueError("Resume input is not a benchmark status record")
        if record["status"] == "failure":
            if record["output"] != "" or not isinstance(record.get("error"), dict):
                raise ValueError("Failed benchmark rows must contain an empty output and structured error")
        elif not record["attempted"] or not strip(record["output"]) or record.get("finish_reason") != "stop":
            raise ValueError("Successful benchmark rows must be complete, attempted answers")
        records[record["key"]] = record
    return records


def summarize(rows: list[dict], records: dict, profile: str) -> dict:
    expected = validate_gold_rows(rows)
    if set(records) != expected:
        raise ValueError("Cannot publish a benchmark report with incomplete turn coverage")
    metrics, counts = score(rows, {key: record["output"] for key, record in records.items()})
    failures = [{"key": key, "attempted": record["attempted"], "error": record["error"]}
                for key, record in records.items() if record["status"] == "failure"]
    failed_gates = gate_failures(metrics, counts, PROFILES[profile])
    if failures:
        failed_gates.append("generation.failures")
    return {"profile": profile, "metrics": metrics, "counts": counts,
            "conversations": len(rows), "turns": len(records),
            "attempted_turns": sum(record["attempted"] for record in records.values()),
            "successful_turns": len(records) - len(failures), "failed_turns": len(failures),
            "failure_rate": len(failures) / len(records),
            "failure_reasons": dict(Counter(failure["error"]["code"] for failure in failures)),
            "failures": failures, "failed_gates": failed_gates, "release_passed": not failed_gates,
            "limitations": "Deterministic format/keyword scoring. Failed and dependent turns remain in metric denominators. No partial generation is accepted as an answer."}


def _aliases(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve() or (
        left.exists() and right.exists() and left.samefile(right)
    )


def _check_destinations(destinations: list[Path], inputs: list[Path]) -> None:
    for index, destination in enumerate(destinations):
        if any(_aliases(destination, source) for source in inputs + destinations[:index]):
            raise ValueError("Benchmark output, metadata or report aliases protected evidence")


def _model_paths(directory: Path, seen=None) -> list[Path]:
    seen = set() if seen is None else seen
    directory = directory.resolve()
    if directory in seen or not directory.is_dir():
        return []
    seen.add(directory)
    paths = [path for path in directory.iterdir() if path.is_file()]
    adapter = directory / "adapter_config.json"
    if adapter.is_file():
        try:
            base = json.loads(adapter.read_text()).get("base_model_name_or_path")
            if isinstance(base, str) and base:
                paths.extend(_model_paths(Path(base), seen))
        except (ValueError, OSError):
            pass  # Model loading will report the invalid configuration.
    return paths


def run_benchmark(config: DeploymentConfig, dataset: str, output: str, *, profile: str,
                  history: str = "gold", resume: bool = False, engine=None) -> dict:
    output_path, dataset_path = Path(output).resolve(), Path(dataset).resolve()
    metadata = output_path.with_suffix(output_path.suffix + ".meta.json")
    report_path = output_path.with_suffix(output_path.suffix + ".report.json")
    destinations = [output_path, metadata, report_path]
    protected = [dataset_path, *_model_paths(Path(config.model))]
    # An earlier binding still identifies images if the current dataset became malformed.
    if metadata.is_file():
        try:
            previous = json.loads(metadata.read_text())
            for reference in previous.get("images_sha256", {}):
                try:
                    protected.append(resolve_image(reference, dataset_path))
                except (ValueError, OSError):
                    pass
        except (ValueError, OSError, AttributeError, TypeError):
            pass
    _check_destinations(destinations, protected)
    try:
        return _run_benchmark(config, dataset, output, profile=profile, history=history,
                              resume=resume, engine=engine, protected=protected)
    except Exception:
        # A stale passing report must not survive a failed rerun. Check aliases again
        # because files may have changed since preflight; never replace evidence.
        _check_destinations(destinations, protected)
        report_path.unlink(missing_ok=True)
        raise


def _run_benchmark(config: DeploymentConfig, dataset: str, output: str, *, profile: str,
                   history: str, resume: bool, engine, protected: list[Path]) -> dict:
    config.validate()
    if profile not in PROFILES or history not in {"gold", "generated"}:
        raise ValueError("Unknown benchmark profile or history policy")
    dataset_path, output_path = Path(dataset).resolve(), Path(output).resolve()
    if dataset_path == output_path:
        raise ValueError("Benchmark output cannot replace the dataset")
    dataset_hash = sha256(dataset_path)
    rows = load_gold(dataset_path)
    if sha256(dataset_path) != dataset_hash:
        raise ValueError("Dataset changed during benchmark preflight")
    expected = validate_gold_rows(rows)
    images = {}
    for row in rows:
        for reference, digest in image_fingerprints(row["messages"], dataset_path).items():
            if reference in images and images[reference] != digest:
                raise ValueError("Image changed during benchmark preflight")
            images[reference] = digest
            protected.append(resolve_image(reference, dataset_path))
        convert_messages(row["messages"], dataset_path, image_hashes=images)
    client = engine if engine is not None else LocalModel(config)
    binding = {"schema_version": 1, "runner": "failure-accounted-benchmark",
               "deployment": client.provenance, "dataset_sha256": dataset_hash,
               "images_sha256": images, "profile": profile, "history": history,
               "max_tokens": config.max_tokens, "temperature": 0.0,
               "failure_policy": "empty-answer; generated history blocks dependent turns"}
    metadata = output_path.with_suffix(output_path.suffix + ".meta.json")
    report_path = output_path.with_suffix(output_path.suffix + ".report.json")
    _check_destinations([output_path, metadata, report_path], protected)
    if output_path.exists() or metadata.exists():
        if not resume or not metadata.is_file() or json.loads(metadata.read_text()) != binding:
            raise ValueError("Resume provenance differs; use a fresh benchmark output")
    records = _load_records(output_path)
    if set(records) - expected:
        raise ValueError("Existing benchmark contains keys outside this dataset")
    # Resource/model initialization errors are infrastructure failures, never
    # synthetic failed answers for every turn. Fully completed resumes need no GPU.
    if set(records) != expected:
        load_engine = getattr(client, "load", None)
        if callable(load_engine):
            load_engine()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not metadata.exists():
        _write_report(metadata, binding, protected)
    started = time.monotonic()
    with output_path.open("a" if resume else "x", encoding="utf-8") as stream:
        for row in rows:
            if sha256(dataset_path) != dataset_hash:
                raise ValueError("Dataset changed after benchmark preflight")
            messages = convert_messages(row["messages"], dataset_path, image_hashes=images)
            conversation, turn, blocked_by = [], 0, None
            for message in messages:
                if message["role"] != "assistant":
                    conversation.append(message)
                    continue
                key = f"{row['id']}#{turn}"
                if key in records:
                    record = records[key]
                    if blocked_by and (record["status"] != "failure" or record["attempted"]):
                        raise ValueError("Generated-history resume contradicts an earlier failed turn")
                elif blocked_by:
                    record = {"key": key, "status": "failure", "attempted": False, "output": "",
                              "finish_reason": "dependency_failed",
                              "error": {"code": "dependency_failed", "caused_by": blocked_by,
                                        "message": "A prior generated answer failed; no gold fallback was inserted"}}
                else:
                    turn_started = time.monotonic()
                    try:
                        result = client.complete(config.served_model_name, conversation,
                                                 max_tokens=config.max_tokens, temperature=0.0)
                        if result.get("finish_reason") != "stop" or not isinstance(result.get("output"), str) or not strip(result["output"]):
                            raise InferenceError("Model returned no complete final answer")
                        record = {**result, "key": key, "status": "success", "attempted": True,
                                  "output": strip(result["output"])}
                    except InferenceError as exc:
                        record = {"key": key, "status": "failure", "attempted": True, "output": "",
                                  "finish_reason": "error", "elapsed_seconds": round(time.monotonic() - turn_started, 4),
                                  "error": {"code": "generation_failed", "type": type(exc).__name__, "message": str(exc)}}
                if key not in records:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    records[key] = record
                    print(json.dumps({"event": "turn", "key": key, "status": record["status"],
                                      "completed": len(records), "total": len(expected)}), flush=True)
                if history == "gold":
                    conversation.append(message)
                elif record["status"] == "success":
                    conversation.append({"role": "assistant", "content": record["output"]})
                else:
                    blocked_by = blocked_by or key
                turn += 1
    report = {"schema_version": 1, **summarize(rows, records, profile),
              "provenance": binding, "predictions_sha256": sha256(output_path),
              "elapsed_seconds_this_invocation": round(time.monotonic() - started, 3)}
    _write_report(report_path, report, protected)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--history", choices=("gold", "generated"), default="gold")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--served-model-name", default="schematic")
    parser.add_argument("--max-seq", type=int, default=8192)
    parser.add_argument("--max-image-size", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--4bit", dest="four_bit", action="store_true")
    parser.add_argument("--min-free-vram-gb", type=float, default=10)
    args = vars(parser.parse_args(argv))
    dataset, output, profile, history, resume = (args.pop(key) for key in ("data", "out", "profile", "history", "resume"))
    try:
        report = run_benchmark(DeploymentConfig(**args), dataset, output, profile=profile, history=history, resume=resume)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"benchmark infrastructure/data failure: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    # Completed benchmarks may document a poor model; infrastructure failures alone return 2.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
