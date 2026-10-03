"""Run a fixed benchmark queue without choosing models or changing settings.

From the repository root: python -m scripts.run_experiments --help
Resolution-only example: --views vision --splits val --image-sizes 1536 2048
--generated-smoke 0. All model imports and GPU work remain lazy.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

from evaluate import _write_report, load_gold
from schematic_model.benchmark import _check_destinations, _model_paths, run_benchmark
from schematic_model.deployment import DeploymentConfig, LocalModel, model_artifacts
from schematic_model.inference import resolve_image
from schematic_model.training import installed_versions, sha256


@dataclass(frozen=True)
class ExperimentConfig:
    snapshot: str
    model: str
    out: str
    base: str | None = None
    views: tuple[str, ...] = ("evidence", "vision")
    splits: tuple[str, ...] = ("val", "test")
    image_sizes: tuple[int, ...] = (1024,)
    generated_smoke: int = 3
    max_seq: int = 8192
    max_tokens: int = 1024
    seed: int = 3407
    four_bit: bool = False
    min_free_vram_gb: float = 10.0


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _source(root: Path, reference: str) -> Path:
    if not isinstance(reference, str) or not reference or Path(reference).is_absolute():
        raise ValueError("Snapshot artifacts must have relative paths")
    path = (root / reference).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Snapshot artifact is missing or escapes its directory")
    return path


def _artifacts(output: Path) -> list[Path]:
    return [output, output.with_suffix(output.suffix + ".meta.json"),
            output.with_suffix(output.suffix + ".report.json")]


def build_plan(config: ExperimentConfig) -> tuple[dict, list[Path], dict[Path, str]]:
    """Verify every frozen source before creating any experiment output."""
    for values, allowed in ((config.views, {"evidence", "vision"}), (config.splits, {"val", "test"})):
        if not values or len(values) != len(set(values)) or not set(values) <= allowed:
            raise ValueError("Views and splits must be nonempty, unique supported values")
    if (not config.image_sizes or len(config.image_sizes) != len(set(config.image_sizes))
            or config.generated_smoke < 0 or (config.generated_smoke and "test" not in config.splits)):
        raise ValueError("Image sizes must be unique; generated smoke requires an explicit test split")
    manifest_path = Path(config.snapshot).resolve()
    if manifest_path.is_dir():
        manifest_path /= "manifest.json"
    snapshot = manifest_path.parent
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or not isinstance(manifest.get("datasets"), list) or not manifest["datasets"]
            or not isinstance(manifest.get("images"), list) or not manifest["images"]):
        raise ValueError("Snapshot manifest requires schema 1 dataset and image inventories")
    expected = manifest.get("snapshot_sha256")
    if expected != _digest({key: value for key, value in manifest.items() if key != "snapshot_sha256"}):
        raise ValueError("Snapshot manifest hash does not match its contents")
    protected, sources = [manifest_path], {}
    for record in manifest["datasets"] + manifest["images"]:
        if not isinstance(record, dict) or not isinstance(record.get("sha256"), str):
            raise ValueError("Invalid snapshot artifact record")
        path = _source(snapshot, record.get("path"))
        if path in sources or sha256(path) != record["sha256"]:
            raise ValueError("Snapshot contains duplicate or changed source artifacts")
        sources[path] = record["sha256"]
        protected.append(path)
    out = Path(config.out).resolve()
    models = {"candidate": str(Path(config.model).resolve())}
    if config.base:
        models["base"] = str(Path(config.base).resolve())
    if any(out == directory or out.is_relative_to(directory)
           for directory in [snapshot, *(Path(model) for model in models.values())]):
        raise ValueError("Experiment output must be outside snapshot and model directories")
    model_bindings = {name: model_artifacts(Path(path)) for name, path in models.items()}
    for path in models.values():
        protected.extend(_model_paths(Path(path)))
    datasets, generated = {}, {}
    image_paths = {_source(snapshot, record["path"]) for record in manifest["images"]}
    dataset_rows = {_source(snapshot, record["path"]): record.get("rows") for record in manifest["datasets"]}
    for view in config.views:
        for split in config.splits:
            source = _source(snapshot, f"{view}/{split}.jsonl")
            if source not in sources:
                raise ValueError("Selected dataset is not bound by the frozen manifest")
            rows = load_gold(source)
            if sha256(source) != sources[source] or len(rows) != dataset_rows.get(source):
                raise ValueError("Dataset changed while preparing the queue")
            for row in rows:
                for message in row["messages"]:
                    for block in message["content"]:
                        if block["type"] == "image":
                            image = resolve_image(block["image"], source)
                            if image not in image_paths:
                                raise ValueError("Dataset image is absent from the frozen manifest")
            datasets[view, split, "gold"] = (source, sources[source])
            if split == "test" and config.generated_smoke:
                count = config.generated_smoke
                if count > len(rows):
                    raise ValueError("Generated smoke count exceeds available test conversations")
                indices = [0] if count == 1 else [round(i * (len(rows) - 1) / (count - 1)) for i in range(count)]
                selected = [copy.deepcopy(rows[index]) for index in indices]
                for row in selected:
                    for message in row["messages"]:
                        for block in message["content"]:
                            if block["type"] == "image":
                                block["image"] = str(resolve_image(block["image"], source))
                content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected)
                path = out / f"generated-{view}.jsonl"
                generated[path] = content
                datasets[view, "test-smoke", "generated"] = (path, hashlib.sha256(content.encode()).hexdigest())
    jobs = []
    for name, model in models.items():
        for size in config.image_sizes:
            deployment = DeploymentConfig(model=model, max_seq=config.max_seq, max_image_size=size,
                                          max_tokens=config.max_tokens, seed=config.seed,
                                          four_bit=config.four_bit, min_free_vram_gb=config.min_free_vram_gb)
            deployment.validate()
            for (view, split, history), (source, digest) in datasets.items():
                job_id = f"{name}-{size}-{view}-{split}-{history}"
                jobs.append({"id": job_id, "model": name, "settings": asdict(deployment),
                             "data": str(source), "dataset_sha256": digest, "history": history,
                             "profile": "real-grounding" if view == "evidence" else "real-vision",
                             "output": str(out / (job_id + ".jsonl"))})
    destinations = [out / "progress.json", out / ".queue.lock", *generated]
    for job in jobs:
        destinations.extend(_artifacts(Path(job["output"])))
    _check_destinations(destinations, protected)
    plan = {"schema_version": 1, "grader": "keyword-refusal-v1", "snapshot": str(manifest_path),
            "snapshot_sha256": expected, "manifest_file_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "sources_sha256": {str(path): digest for path, digest in sources.items()},
            "models": model_bindings, "packages": installed_versions(), "jobs": jobs,
            "generated_smoke_selection": "equally spaced source-order indices, endpoints included",
            "generated_smoke_count": config.generated_smoke}
    plan["plan_sha256"] = _digest(plan)
    return plan, protected, generated


def _write_smoke(path: Path, content: str) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _verify_sources(plan: dict, job: dict) -> None:
    if (sha256(Path(job["data"])) != job["dataset_sha256"]
            or any(sha256(Path(path)) != digest for path, digest in plan["sources_sha256"].items())
            or sha256(Path(plan["snapshot"])) != plan["manifest_file_sha256"]):
        raise ValueError("Queued snapshot source changed after preflight")


def _verify_completed(job: dict, record: dict, model: dict) -> dict:
    paths = _artifacts(Path(job["output"]))
    hashes = record.get("artifacts_sha256")
    if not isinstance(hashes, dict) or set(hashes) != {str(path) for path in paths}:
        raise ValueError("Completed job lacks its complete artifact hash binding")
    for path in paths:
        if not path.is_file() or sha256(path) != hashes[str(path)]:
            raise ValueError(f"Completed job artifact changed: {path.name}")
    report, metadata = json.loads(paths[2].read_text()), json.loads(paths[1].read_text())
    if (report.get("predictions_sha256") != hashes[str(paths[0])]
            or report.get("provenance") != metadata
            or metadata.get("dataset_sha256") != job["dataset_sha256"]
            or metadata.get("profile") != job["profile"] or metadata.get("history") != job["history"]
            or metadata.get("deployment", {}).get("settings") != job["settings"]
            or metadata.get("deployment", {}).get("model") != model):
        raise ValueError("Completed job provenance disagrees with its queue binding")
    return report


def run_experiments(config: ExperimentConfig, *, resume=False, dry_run=False, engine_factory=LocalModel) -> dict:
    plan, protected, generated = build_plan(config)
    if dry_run:
        return plan
    out = Path(config.out).resolve()
    state_path = out / "progress.json"
    if not resume and out.exists() and any(out.iterdir()):
        raise ValueError("Fresh experiment output must be empty; use --resume for an existing queue")
    if resume and not state_path.is_file():
        raise ValueError("Resume requires an existing progress manifest")
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another process owns this experiment queue") from exc
        if resume:
            state = json.loads(state_path.read_text())
            if state.get("plan") != plan or set(state.get("jobs", {})) != {job["id"] for job in plan["jobs"]}:
                raise ValueError("Resume model, source, packages or settings differ from the saved queue")
            for job in plan["jobs"]:
                record = state["jobs"][job["id"]]
                if record.get("status") not in {"pending", "running", "failed", "completed"}:
                    raise ValueError("Invalid queue job status")
                if record["status"] == "completed":
                    _verify_completed(job, record, plan["models"][job["model"]])
        else:
            if any(path.name != ".queue.lock" for path in out.iterdir()):
                raise ValueError("Fresh queue output changed while acquiring its lock")
            state = {"schema_version": 1, "plan": plan, "status": "pending",
                     "jobs": {job["id"]: {"status": "pending"} for job in plan["jobs"]}}
        _write_report(state_path, state, protected)
        try:
            for path, content in generated.items():
                if path.exists() and path.read_text() != content:
                    raise ValueError("Generated smoke dataset changed")
                if not path.exists():
                    if resume and any(record["status"] != "pending" for record in state["jobs"].values()):
                        raise ValueError("Generated smoke dataset is missing")
                    _write_smoke(path, content)
        except Exception as exc:
            state.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
            _write_report(state_path, state, protected)
            raise
        engine, active_settings = None, None
        try:
            for job in plan["jobs"]:
                record = state["jobs"][job["id"]]
                if record["status"] == "completed":
                    continue
                _verify_sources(plan, job)
                if active_settings != job["settings"]:
                    torch = getattr(engine, "torch", None)
                    engine = None
                    gc.collect()
                    if torch is not None:
                        torch.cuda.empty_cache()
                    engine = engine_factory(DeploymentConfig(**job["settings"]))
                    if engine.provenance.get("model") != plan["models"][job["model"]]:
                        raise ValueError("Model artifacts changed after queue preflight")
                    active_settings = job["settings"]
                _verify_sources(plan, job)
                state["status"] = record["status"] = "running"
                record["started_at_unix"] = time.time()
                _write_report(state_path, state, protected)
                print(json.dumps({"event": "job_start", "id": job["id"]}), flush=True)
                output = Path(job["output"])
                run_benchmark(DeploymentConfig(**job["settings"]), job["data"], str(output),
                              profile=job["profile"], history=job["history"], engine=engine,
                              resume=output.exists() or _artifacts(output)[1].exists())
                hashes = {str(path): sha256(path) for path in _artifacts(output)}
                report = _verify_completed(job, {"artifacts_sha256": hashes}, plan["models"][job["model"]])
                _verify_sources(plan, job)
                record.update(status="completed", finished_at_unix=time.time(),
                              artifacts_sha256=hashes,
                              metrics=report["metrics"], counts=report["counts"],
                              failed_turns=report["failed_turns"], turns=report["turns"],
                              release_passed=report["release_passed"])
                record.pop("error", None)
                _write_report(state_path, state, protected)
                print(json.dumps({"event": "job_complete", "id": job["id"], **record}), flush=True)
            state["status"] = "completed"
            state.pop("error", None)
            _write_report(state_path, state, protected)
            return state
        except Exception as exc:
            state["status"] = "failed"
            if "record" in locals() and record.get("status") != "completed":
                record.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
            _write_report(state_path, state, protected)
            raise
        finally:
            torch = getattr(engine, "torch", None)
            engine = None
            gc.collect()
            if torch is not None:
                torch.cuda.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, help="Frozen snapshot directory or manifest.json")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base")
    parser.add_argument("--out", required=True)
    parser.add_argument("--views", nargs="+", choices=("evidence", "vision"), default=["evidence", "vision"])
    parser.add_argument("--splits", nargs="+", choices=("val", "test"), default=["val", "test"])
    parser.add_argument("--image-sizes", nargs="+", type=int, default=[1024])
    parser.add_argument("--generated-smoke", type=int, default=3)
    parser.add_argument("--max-seq", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--4bit", dest="four_bit", action="store_true")
    parser.add_argument("--min-free-vram-gb", type=float, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Verify sources and print queue without GPU work or writes")
    args = vars(parser.parse_args(argv))
    resume, dry_run = args.pop("resume"), args.pop("dry_run")
    try:
        result = run_experiments(ExperimentConfig(**args), resume=resume, dry_run=dry_run)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"experiment queue failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
