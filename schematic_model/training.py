"""Unsloth vision LoRA training with CPU preflight and reproducible manifests.

Importing this module, --help, and --dry-run never import GPU libraries. Use an
isolated training environment or the installed Studio Python for actual runs.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import inspect
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Any

DEFAULT_MODEL = "unsloth/Qwen3.5-4B"
DEFAULT_REVISION = "3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636"


class _TerminationRequested(BaseException):
    """Unwind training on SIGTERM so its manifest can record the interruption."""

    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(f"Received {signal.Signals(signum).name}")


@contextmanager
def _interrupt_on_sigterm():
    """Guard CLI/main-thread training; Python cannot register signals in a worker."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def interrupt(signum, _frame):
        raise _TerminationRequested(signum)

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


@dataclass(frozen=True)
class TrainingConfig:
    data: str = "data"
    out: str = "outputs/schematic-lora"
    model: str = DEFAULT_MODEL
    revision: str | None = None
    four_bit: bool = False
    placement: str = "auto"
    offload_layers: int = 0
    offload_prefetch: int = 1
    max_seq: int = 4096
    max_image_size: int = 1024
    epochs: float = 2.0
    max_steps: int = -1
    max_train_rows: int | None = None
    lr: float = 1e-4
    rank: int = 16
    gradient_accumulation: int = 4
    save_steps: int = 50
    eval_steps: int | None = None
    load_best_model_at_end: bool = False
    early_stopping_patience: int | None = None
    last_response_only: bool = False
    seed: int = 3407
    resume: str | None = None
    continue_adapter: bool = False
    export_merged: bool = False
    finetune_vision: bool = True
    min_free_vram_gb: float = 12.0
    local_files_only: bool = True
    cache_dir: str = "models/huggingface"

    def validate(self) -> None:
        if type(self.last_response_only) is not bool:
            raise ValueError("last_response_only must be a boolean")
        if self.placement not in {"auto", "single_gpu"}:
            raise ValueError("placement must be auto or single_gpu")
        if type(self.offload_layers) is not int or self.offload_layers < 0:
            raise ValueError("offload_layers must be a nonnegative integer")
        if type(self.offload_prefetch) is not int or self.offload_prefetch < 1:
            raise ValueError("offload_prefetch must be a positive integer")
        if self.offload_layers and self.continue_adapter:
            raise ValueError("Block offload requires a base-model run; use --resume for its checkpoints")
        for name in ("max_seq", "max_image_size", "rank", "gradient_accumulation", "save_steps"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.eval_steps is not None and self.eval_steps < 1:
            raise ValueError("eval_steps must be positive")
        if self.load_best_model_at_end:
            if self.eval_steps is None or self.save_steps != self.eval_steps:
                raise ValueError("Best loss selection requires eval_steps equal to save_steps")
            if self.max_steps > 0 and self.max_steps < self.eval_steps:
                raise ValueError("max_steps must reach at least one evaluation for best loss selection")
        if self.early_stopping_patience is not None:
            if self.early_stopping_patience < 1 or not self.load_best_model_at_end:
                raise ValueError("Early stopping requires positive patience and best loss selection")
        if self.max_steps != -1 and self.max_steps < 1:
            raise ValueError("max_steps must be positive, or -1 to use epochs")
        if self.max_train_rows is not None and self.max_train_rows < 1:
            raise ValueError("max_train_rows must be positive")
        if not math.isfinite(self.epochs) or self.epochs <= 0:
            raise ValueError("epochs must be finite and positive")
        if not math.isfinite(self.lr) or not 0 < self.lr < 1:
            raise ValueError("lr must be finite and between 0 and 1")
        if not math.isfinite(self.min_free_vram_gb) or self.min_free_vram_gb < 0:
            raise ValueError("min_free_vram_gb must be finite and nonnegative")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.model.lower().endswith(".gguf") or "-gguf" in self.model.lower():
            raise ValueError("GGUF weights cannot be fine-tuned; use a safetensors model")
        if self.continue_adapter:
            continuation_source(self)
        if self.resume:
            step = _validate_resume_checkpoint(Path(self.resume))
            if self.max_steps > 0 and step >= self.max_steps:
                raise ValueError("Resume checkpoint global_step must be below max_steps; requested budget is exhausted")
            _validate_resume_evaluation(self)

    @property
    def effective_revision(self) -> str | None:
        return self.revision or (DEFAULT_REVISION if self.model == DEFAULT_MODEL else None)


def evaluation_arguments(config: TrainingConfig) -> dict[str, Any]:
    """Loss-only evaluation; generated-answer benchmarks remain a separate gate."""
    if config.eval_steps is None:
        return {"eval_strategy": "no"}
    return {
        "eval_strategy": "steps", "eval_steps": config.eval_steps,
        "prediction_loss_only": True,
        "load_best_model_at_end": config.load_best_model_at_end,
        "metric_for_best_model": "eval_loss", "greater_is_better": False,
        "restore_callback_states_from_checkpoint": config.early_stopping_patience is not None,
    }


def evaluation_summary(config: TrainingConfig, state, *, require_best: bool = True) -> dict[str, Any]:
    observations = [{"step": entry.get("step"), "eval_loss": entry["eval_loss"]}
                    for entry in getattr(state, "log_history", []) if "eval_loss" in entry]
    for observation in observations:
        if not math.isfinite(observation["eval_loss"]):
            raise ValueError("Validation loss is nonfinite")
    best_checkpoint = getattr(state, "best_model_checkpoint", None)
    best_metric = getattr(state, "best_metric", None)
    if config.load_best_model_at_end and require_best:
        if best_checkpoint is None or best_metric is None or not math.isfinite(best_metric):
            raise ValueError("Best loss selection completed without a valid evaluated checkpoint")
    return {
        "enabled": config.eval_steps is not None,
        "selection_metric": "eval_loss" if config.load_best_model_at_end else None,
        "best_checkpoint": best_checkpoint, "best_metric": best_metric,
        "observations": observations,
        "scope": "Teacher-forced validation loss only; generated-answer quality is not measured.",
    }


def _validate_resume_evaluation(config: TrainingConfig) -> None:
    """Bind loss history and restored early-stop state to their recorded recipe."""
    manifest_path = Path(config.resume or "").parent / "training_manifest.json"
    try:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise ValueError("Resume requires a readable training_manifest.json beside the checkpoint") from exc
    if not isinstance(previous, dict) or not isinstance(previous.get("config"), dict):
        raise ValueError("Resume manifest requires a recorded training config")
    # Manifests from before opt-in evaluation represent the disabled recipe.
    for key, default in (("eval_steps", None), ("load_best_model_at_end", False),
                         ("early_stopping_patience", None)):
        if previous["config"].get(key, default) != getattr(config, key):
            raise ValueError("Resume evaluation recipe differs from recorded run; start a separate candidate")
    if previous["config"].get("last_response_only", False) != config.last_response_only:
        raise ValueError("Resume supervision recipe differs from recorded run; start a separate candidate")


def continuation_source(config):
    """Bind a local existing adapter; continuing starts a new optimizer/run."""
    source = Path(config.model).resolve()
    output = Path(config.out).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Continuation output must be separate from the source adapter")
    if config.resume:
        checkpoint = Path(config.resume).resolve()
        if checkpoint.parent != output or not (output / "training_manifest.json").is_file():
            raise ValueError("Continuation resume requires a checkpoint from the same recorded output")
        previous = json.loads((output / "training_manifest.json").read_text())
        for key in ("model", "data", "rank", "max_steps", "max_seq", "max_image_size", "lr", "gradient_accumulation", "finetune_vision", "seed", "max_train_rows", "epochs", "four_bit"):
            if previous["config"].get(key) != getattr(config, key):
                raise ValueError("Continuation resume recipe differs from recorded run")
    files = [source / "adapter_config.json", source / "adapter_model.safetensors"]
    if not all(path.is_file() for path in files):
        raise ValueError("Continuation requires a local safetensors LoRA adapter")
    settings = json.loads(files[0].read_text())
    if (settings.get("peft_type") != "LORA" or settings.get("r") != config.rank
            or settings.get("bias") != "none" or settings.get("modules_to_save")
            or settings.get("rank_pattern") or settings.get("use_dora")
            or not settings.get("target_modules")):
        raise ValueError("Unsupported continuation adapter configuration or rank mismatch")
    return {"path": str(source), "files": {path.name: sha256(path) for path in files},
            "rank": settings["r"], "target_modules": settings["target_modules"],
            "optimizer": "new; source adapter weights retained"}


def prepare_continued_adapter(model, config, source):
    """Reuse loaded LoRA weights; never add/reinitialize a second adapter."""
    adapters = getattr(model, "peft_config", {})
    if set(adapters) != {"default"}:
        raise ValueError("Continuation requires exactly the loaded default adapter")
    loaded = adapters["default"]
    loaded_targets, source_targets = loaded.target_modules, source["target_modules"]
    if not isinstance(loaded_targets, str) and not isinstance(source_targets, str):
        loaded_targets, source_targets = set(loaded_targets), set(source_targets)
    if loaded.r != source["rank"] or loaded_targets != source_targets:
        raise ValueError("Loaded adapter configuration differs from source")
    trained = []
    for name, parameter in model.named_parameters():
        is_lora = ".lora_A.default." in name or ".lora_B.default." in name
        is_vision = any(part in name.lower() for part in ("visual", "vision", "image", "patch_embed"))
        enabled = is_lora and (config.finetune_vision or not is_vision)
        parameter.requires_grad_(enabled)
        if enabled:
            trained.append(name)
    if not trained:
        raise ValueError("Continuation found no trainable existing LoRA parameters")
    return trained


def _validate_resume_checkpoint(checkpoint: Path) -> int:
    """Require complete single-GPU PEFT state before any training side effects.

    This checks file presence and trainer metadata, not binary deserialization or
    compatibility with the current run's data and settings.
    """
    match = re.fullmatch(r"checkpoint-([1-9][0-9]*)", checkpoint.name)
    if not checkpoint.is_dir() or match is None:
        raise ValueError("resume must name a checkpoint-N directory with a positive step")
    required = ("adapter_model.safetensors", "adapter_config.json", "trainer_state.json",
                "optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin")
    for name in required:
        artifact = checkpoint / name
        if artifact.is_symlink() or not artifact.is_file() or artifact.stat().st_size == 0:
            raise ValueError(f"Resume checkpoint requires a nonempty regular file: {name}")
    from evaluate import _invalid_constant, _unique_object

    try:
        state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"),
                           object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Resume checkpoint trainer_state.json is not valid JSON") from exc
    step = state.get("global_step") if isinstance(state, dict) else None
    if type(step) is not int or step <= 0 or step != int(match[1]):
        raise ValueError("Resume checkpoint global_step must be a positive integer matching checkpoint-N")
    return step


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_image(path: object, dataset_path: Path) -> Path:
    """Support dataset-relative paths and the original repo-relative format."""
    if not isinstance(path, str) or not path or "://" in path or path.startswith("data:"):
        raise ValueError("training images must be local file paths")
    raw = Path(path).expanduser()
    candidates = [raw] if raw.is_absolute() else [dataset_path.parent / raw, dataset_path.parent.parent / raw]
    existing = {candidate.resolve() for candidate in candidates if candidate.is_file()}
    if len(existing) > 1:
        raise ValueError(f"ambiguous image path {path!r}; use an absolute path")
    if not existing:
        raise ValueError(f"missing image {path!r} referenced by {dataset_path}")
    return existing.pop()


def read_rows(path: Path, *, expected_sha256: str | None = None) -> list[dict[str, Any]]:
    data = path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError(f"Training dataset changed since preflight: {path}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    # Parse the exact bytes that were checked, rather than reopening a path.
    with io.StringIO(data.decode("utf-8")) as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("row must be an object")
                key = row.get("id")
                if not isinstance(key, str) or not key:
                    raise ValueError("row needs a nonempty string id")
                if key in seen:
                    raise ValueError(f"duplicate id {key!r}")
                seen.add(key)
                messages = row.get("messages")
                if not isinstance(messages, list) or len(messages) < 2:
                    raise ValueError("row needs a conversation")
                expected, images = "user", 0
                for index, message in enumerate(messages):
                    if not isinstance(message, dict):
                        raise ValueError("message must be an object")
                    role = message.get("role")
                    if index == 0 and role == "system":
                        pass
                    elif role != expected:
                        raise ValueError(f"expected {expected} message at index {index}")
                    else:
                        expected = "assistant" if role == "user" else "user"
                    content = message.get("content")
                    if not isinstance(content, list) or not content:
                        raise ValueError("message content must be a nonempty list")
                    has_text = False
                    for part in content:
                        if not isinstance(part, dict):
                            raise ValueError("content part must be an object")
                        if part.get("type") == "image" and role == "user":
                            part["image"] = str(resolve_image(part.get("image"), path))
                            images += 1
                        elif part.get("type") == "text" and isinstance(part.get("text"), str) and part["text"].strip():
                            has_text = True
                        else:
                            raise ValueError("expected nonempty text or a user image")
                    if role in {"assistant", "system"} and not has_text:
                        raise ValueError(f"{role} message needs text")
                if messages[-1].get("role") != "assistant" or not images:
                    raise ValueError("conversation must contain an image and end with an assistant answer")
                rows.append(row)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def image_paths(rows: list[dict[str, Any]]) -> set[Path]:
    return {Path(part["image"]) for row in rows for message in row["messages"]
            for part in message["content"] if part["type"] == "image"}


def _read_data(config: TrainingConfig, *, expected_hashes: dict[str, str] | None = None
               ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = Path(config.data).resolve()
    if expected_hashes is not None and set(expected_hashes) != {"train", "validation"}:
        raise ValueError("Preflight hashes are required for both train and validation datasets")
    train = read_rows(data / "train.jsonl", expected_sha256=expected_hashes["train"] if expected_hashes else None)
    validation = read_rows(data / "val.jsonl", expected_sha256=expected_hashes["validation"] if expected_hashes else None)
    for field in ("id", "design_id", "source_id", "family_id"):
        overlap = {row[field] for row in train if row.get(field)} & {row[field] for row in validation if row.get(field)}
        if overlap:
            raise ValueError(f"train/validation leakage in {field}: {sorted(overlap)[:5]}")
    return train, validation


def model_binding(model: str, revision: str | None = None, *, seen: frozenset[Path] = frozenset()) -> dict:
    """Bind local model bytes (including adapter bases), or an immutable remote revision."""
    path = Path(model)
    if not path.is_dir():
        if path.is_absolute() or not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
            raise ValueError("Training model requires local artifacts or an immutable revision")
        return {"model": model, "revision": revision}
    path = path.resolve()
    if path in seen:
        raise ValueError("Cyclic training adapter base reference")
    files = [item for item in sorted(path.iterdir()) if item.is_file()
             and item.suffix in {".safetensors", ".json", ".jinja", ".txt", ".model", ".tiktoken", ".vocab", ".merges", ".bpe"}
             and item.name not in {"training_manifest.json", "trainer_state.json", "deployment_manifest.json"}]
    files.extend(sorted(item for item in (path / "chat_templates").glob("*.jinja") if item.is_file()))
    weights = [item for item in files if item.suffix == ".safetensors"]
    if not weights or any(not item.stat().st_size for item in weights):
        raise ValueError("Training model requires nonempty safetensors weights")
    for index in (item for item in files if item.name.endswith(".safetensors.index.json")):
        mapping = json.loads(index.read_text()).get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("Training model index requires a nonempty weight map")
        for shard in mapping.values():
            if (not isinstance(shard, str) or Path(shard).name != shard
                    or not shard.endswith(".safetensors") or not (path / shard).is_file()):
                raise ValueError("Training model index references a missing or invalid shard")
    result: dict[str, Any] = {"path": str(path), "files": {str(item.relative_to(path)): sha256(item) for item in files}}
    adapter = path / "adapter_config.json"
    if adapter.is_file():
        settings = json.loads(adapter.read_text())
        base = settings.get("base_model_name_or_path")
        if not isinstance(base, str) or not base:
            raise ValueError("Training adapter requires a bound base model")
        base_revision = settings.get("revision") or (DEFAULT_REVISION if base == DEFAULT_MODEL else None)
        result["base_model"] = model_binding(base, base_revision, seen=seen | {path})
    return result


def validate_resume_binding(config: TrainingConfig, manifest: dict) -> None:
    """Reject changed inputs before a prior manifest or checkpoint can be replaced."""
    if not config.resume:
        return
    output = Path(config.out).resolve()
    checkpoint = Path(config.resume)
    if checkpoint.is_symlink() or checkpoint.resolve().parent != output:
        raise ValueError("Resume checkpoint must belong to the same recorded output")
    try:
        previous = json.loads((output / "training_manifest.json").read_text())
    except (OSError, ValueError) as exc:
        raise ValueError("Resume requires the existing training manifest") from exc
    # Resource admission and the selected checkpoint can change; the recipe cannot.
    ignored = {"resume", "min_free_vram_gb"}
    recipe = {key: value for key, value in manifest["config"].items() if key not in ignored}
    old_recipe = {key: value for key, value in previous.get("config", {}).items() if key not in ignored}
    # Older runs used the loader's default placement and did not record this option.
    old_recipe.setdefault("placement", "auto")
    old_recipe.setdefault("offload_layers", 0)
    old_recipe.setdefault("offload_prefetch", 1)
    # Earlier collators supervised every assistant answer.
    old_recipe.setdefault("last_response_only", False)
    if recipe != old_recipe:
        raise ValueError("Resume recipe differs from recorded run")
    for key in ("datasets", "images", "model_binding", "continued_adapter"):
        if key not in previous or previous[key] != manifest[key]:
            raise ValueError(f"Resume {key} differ from recorded run")


def installed_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("unsloth", "unsloth_zoo", "transformers", "trl", "torch", "peft", "Pillow"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def training_step_budget(config: TrainingConfig, available_rows: int) -> dict[str, Any]:
    """Mirror Trainer's single-GPU, batch-size-one optimizer-step rounding.

    A final partial accumulation group is an update. Fractional epochs round
    the total update budget upward; explicit max_steps overrides epochs.
    """
    selected_rows = min(available_rows, config.max_train_rows or available_rows)
    updates_per_epoch = max(math.ceil(selected_rows / config.gradient_accumulation), 1)
    total_steps = config.max_steps if config.max_steps > 0 else math.ceil(config.epochs * updates_per_epoch)
    if config.load_best_model_at_end and (config.eval_steps is None or total_steps < config.eval_steps):
        raise ValueError("Effective training step budget cannot reach the first evaluation for best loss selection")
    return {
        "selected_rows": selected_rows, "per_device_train_batch_size": 1, "world_size": 1,
        "gradient_accumulation_steps": config.gradient_accumulation,
        "optimizer_steps_per_epoch": updates_per_epoch, "max_optimizer_steps": total_steps,
        "source": "max_steps" if config.max_steps > 0 else "epochs",
    }


def preflight(config: TrainingConfig) -> dict[str, Any]:
    """Check complete data splits on CPU, without modifying any files."""
    config.validate()
    dataset_hashes = {split: sha256(Path(config.data, filename))
                      for split, filename in (("train", "train.jsonl"), ("validation", "val.jsonl"))}
    train, validation = _read_data(config, expected_hashes=dataset_hashes)
    step_budget = training_step_budget(config, len(train))
    if config.resume and config.max_steps == -1:
        step = _validate_resume_checkpoint(Path(config.resume))
        if step >= step_budget["max_optimizer_steps"]:
            raise ValueError("Resume checkpoint global_step must be below the effective epoch step budget; requested budget is exhausted")
    if config.continue_adapter and config.resume:
        previous = json.loads((Path(config.out) / "training_manifest.json").read_text())
        if {key: value["sha256"] for key, value in previous["datasets"].items()} != dataset_hashes:
            raise ValueError("Continuation resume datasets changed")
        if previous["continued_adapter"] != continuation_source(config):
            raise ValueError("Continuation source adapter changed")
    from PIL import Image

    records, hashes = [], {}
    for path in sorted(image_paths(train) | image_paths(validation)):
        try:
            with Image.open(path) as image:
                width, height = image.size
                image.verify()
        except OSError as exc:
            raise ValueError(f"invalid training image {path}: {exc}") from exc
        hashes[path] = sha256(path)
        records.append({"path": str(path), "sha256": hashes[path], "width": width, "height": height})
    if {hashes[p] for p in image_paths(train)} & {hashes[p] for p in image_paths(validation)}:
        raise ValueError("train/validation leakage: identical image bytes occur in both splits")
    manifest = {
        "schema_version": 1, "status": "preflight",
        "created_at": datetime.now(timezone.utc).isoformat(), "config": asdict(config),
        "model_revision": config.effective_revision, "python": platform.python_version(),
        "training_step_budget": step_budget,
        "packages": installed_versions(),
        "datasets": {
            split: {"path": str(Path(config.data, filename).resolve()), "sha256": dataset_hashes[split],
                    "rows": len(rows), "assistant_turns": sum(m["role"] == "assistant" for r in rows for m in r["messages"])}
            for split, filename, rows in (("train", "train.jsonl", train), ("validation", "val.jsonl", validation))
        },
        "images": records,
        "model_binding": model_binding(config.model, config.effective_revision),
        "continued_adapter": continuation_source(config) if config.continue_adapter else None,
        "scope": "Dataset checks only. Model loading, token lengths, GPU fit and training quality are not yet verified.",
    }
    validate_resume_binding(config, manifest)
    return manifest


def gpu_inventory() -> list[dict[str, Any]]:
    """Read physical GPU capacity without initializing a CUDA context."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("Unable to inspect NVIDIA memory with nvidia-smi") from exc
    devices = []
    for line in result.stdout.splitlines():
        index, name, total, free = [field.strip() for field in line.split(",", 3)]
        devices.append({"index": int(index), "name": name, "total_mib": int(total), "free_mib": int(free)})
    return devices


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_metrics(metrics: dict[str, Any]) -> None:
    """Never publish an adapter as completed after a numerically failed run."""
    for name, value in metrics.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(f"Training produced nonfinite {name}; inspect the run before exporting")


class CheckedVisionCollator:
    """Decode batch images only; reject truncation and empty supervision."""

    def __init__(self, collator: Any, max_seq: int, *, image_hashes: dict[str, str]):
        self.collator, self.max_seq = collator, max_seq
        self.image_hashes = {str(Path(path).resolve()): digest for path, digest in image_hashes.items()}
        collator.max_seq_length, collator.truncation = None, False

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        from PIL import Image

        converted = []
        for example in examples:
            row = copy.deepcopy(example)
            for message in row["messages"]:
                for part in message["content"]:
                    if part["type"] == "image":
                        path = Path(part["image"]).resolve()
                        expected_hash = self.image_hashes.get(str(path))
                        if expected_hash is None:
                            raise ValueError(f"Training image was not recorded at preflight: {path}")
                        data = path.read_bytes()
                        if hashlib.sha256(data).hexdigest() != expected_hash:
                            raise ValueError(f"Training image changed since preflight: {path}")
                        # Decode the verified bytes instead of reopening a path
                        # that could change between validation and decoding.
                        with Image.open(io.BytesIO(data)) as image:
                            part["image"] = image.convert("RGB")
            converted.append(row)
        batch = self.collator(converted)
        if batch["input_ids"].shape[-1] > self.max_seq:
            raise ValueError(f"Conversation has {batch['input_ids'].shape[-1]} tokens, above --max-seq {self.max_seq}; increase context or reduce image size")
        if not (batch["labels"] != -100).any(dim=-1).all().item():
            raise ValueError("An example has no supervised assistant tokens; check chat template and response mask")
        return batch


class DeadlineControl:
    """Cooperative yield only after an optimizer step; trainer saves full state."""
    def __init__(self, deadline):
        self.deadline = deadline
        self.triggered = False

    def on_step_end(self, args, state, control, **kwargs):
        if time.time() >= self.deadline and state.global_step < state.max_steps:
            control.should_save = True
            control.should_training_stop = True
            self.triggered = True
        return control


def offload_arguments(config: TrainingConfig, adapter_factory) -> dict[str, int]:
    """Require the explicit upstream API; older **kwargs loaders can ignore options."""
    if not config.offload_layers:
        return {}
    if "offload_layers" not in inspect.signature(adapter_factory).parameters:
        raise RuntimeError("Block offload requires an Unsloth runtime with explicit offload_layers support")
    return {"offload_layers": config.offload_layers, "prefetch_depth": config.offload_prefetch,
            "checkpoint_skip_layers": 0}


@_interrupt_on_sigterm()
def run_training(config: TrainingConfig) -> dict[str, Any]:
    if os.environ.get("SCHEMATIC_DEADLINE") and not math.isfinite(float(os.environ["SCHEMATIC_DEADLINE"])):
        raise ValueError("Scheduler deadline must be finite")
    manifest = preflight(config)
    output = Path(config.out).resolve()
    if output.exists() and any(output.iterdir()) and not config.resume:
        raise ValueError(f"Output directory is not empty: {output}; use a new output or --resume CHECKPOINT")
    devices = gpu_inventory()
    if not devices or max(device["free_mib"] for device in devices) < config.min_free_vram_gb * 1024:
        raise RuntimeError(f"Need at least {config.min_free_vram_gb:g} GiB free VRAM; current devices: {devices}")
    output.mkdir(parents=True, exist_ok=True)
    manifest.update(status="starting", gpu_before=devices)
    manifest_path = output / "training_manifest.json"
    started = time.monotonic()
    try:
        write_manifest(manifest_path, manifest)
        # Unsloth must patch Transformers before TRL imports it.
        from unsloth import FastVisionModel, is_bf16_supported
        offload = offload_arguments(config, FastVisionModel.get_peft_model)
        from unsloth.trainer import UnslothVisionDataCollator
        import torch
        from trl import SFTConfig, SFTTrainer
        from transformers import TrainerCallback

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this training environment")
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        if free_bytes < config.min_free_vram_gb * 1024**3:
            raise RuntimeError("The selected CUDA device does not have the requested free VRAM")
        torch.cuda.reset_peak_memory_stats()
        train, validation = _read_data(config, expected_hashes={split: record["sha256"]
                                                               for split, record in manifest["datasets"].items()})
        random.Random(config.seed).shuffle(train)
        if config.max_train_rows:
            train = train[:config.max_train_rows]
        train_data = [{"messages": row["messages"]} for row in train]
        val_data = [{"messages": row["messages"]} for row in validation]
        if model_binding(config.model, config.effective_revision) != manifest["model_binding"]:
            raise ValueError("Training model changed after preflight")
        model, processor = FastVisionModel.from_pretrained(
            model_name=config.model, revision=config.effective_revision,
            max_seq_length=config.max_seq, load_in_4bit=config.four_bit,
            dtype=torch.bfloat16 if is_bf16_supported() else torch.float16,
            use_gradient_checkpointing="unsloth", full_finetuning=False,
            local_files_only=config.local_files_only, cache_dir=str(Path(config.cache_dir).resolve()),
            trust_remote_code=False,
            **({"device_map": {"": 0}} if config.placement == "single_gpu" else {}),
        )
        if config.continue_adapter:
            if continuation_source(config) != manifest["continued_adapter"]:
                raise ValueError("Source adapter changed after preflight")
            manifest["continued_trainable_parameters"] = prepare_continued_adapter(
                model, config, manifest["continued_adapter"])
        else:
            model = FastVisionModel.get_peft_model(
                model, finetune_vision_layers=config.finetune_vision,
                finetune_language_layers=True, finetune_attention_modules=True,
                finetune_mlp_modules=True, r=config.rank, lora_alpha=config.rank,
                lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth", random_state=config.seed,
                **offload,
            )
        FastVisionModel.for_training(model)
        collator = CheckedVisionCollator(UnslothVisionDataCollator(
            model, processor, resize=config.max_image_size, resize_dimension="max",
            train_on_responses_only=True, instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
            last_response_only=config.last_response_only,
        ), config.max_seq, image_hashes={record["path"]: record["sha256"] for record in manifest["images"]})
        collator([train_data[0]])
        deadline_control = None
        if os.environ.get("SCHEMATIC_DEADLINE"):
            deadline = float(os.environ["SCHEMATIC_DEADLINE"])
            if not math.isfinite(deadline):
                raise ValueError("Scheduler deadline must be finite")
            callback_type = type("DeadlineCallback", (DeadlineControl, TrainerCallback), {})
            deadline_control = callback_type(deadline)
        callbacks = [deadline_control] if deadline_control else []
        if config.early_stopping_patience is not None:
            from transformers import EarlyStoppingCallback
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience))
        trainer = SFTTrainer(
            model=model, processing_class=processor, data_collator=collator,
            train_dataset=train_data, eval_dataset=val_data,
            callbacks=callbacks,
            args=SFTConfig(
                output_dir=str(output), per_device_train_batch_size=1, per_device_eval_batch_size=1,
                gradient_accumulation_steps=config.gradient_accumulation,
                num_train_epochs=config.epochs, max_steps=config.max_steps,
                learning_rate=config.lr, warmup_ratio=0.05, lr_scheduler_type="cosine",
                logging_steps=1, save_steps=config.save_steps, save_total_limit=2,
                **evaluation_arguments(config),
                optim="adamw_8bit", weight_decay=0.01,
                bf16=is_bf16_supported(), fp16=not is_bf16_supported(),
                max_length=config.max_seq, remove_unused_columns=False,
                dataset_text_field="", dataset_kwargs={"skip_prepare_dataset": True},
                dataset_num_proc=1, dataloader_num_workers=0, seed=config.seed, report_to="none",
            ),
        )
        manifest.update(status="training", trained_row_ids=[row["id"] for row in train],
                        resolved_model_revision=getattr(model.config, "_commit_hash", None),
                        chat_template_sha256=hashlib.sha256(str(processor.chat_template).encode()).hexdigest(),
                        cuda_device={"name": torch.cuda.get_device_name(), "total_bytes": total_bytes})
        write_manifest(manifest_path, manifest)
        result = trainer.train(resume_from_checkpoint=config.resume)
        validate_metrics(result.metrics)
        manifest["validation"] = evaluation_summary(
            config, trainer.state, require_best=not (deadline_control and deadline_control.triggered))
        if deadline_control and deadline_control.triggered:
            checkpoint = output / f"checkpoint-{trainer.state.global_step}"
            _validate_resume_checkpoint(checkpoint)
            manifest.update(status="yielded", global_step=trainer.state.global_step,
                            checkpoint=str(checkpoint), elapsed_seconds=time.monotonic() - started)
            write_manifest(manifest_path, manifest)
            if os.environ.get("SCHEMATIC_RESULT_PATH"):
                write_manifest(Path(os.environ["SCHEMATIC_RESULT_PATH"]), {"checkpoint_step": trainer.state.global_step})
            return manifest
        model.save_pretrained(str(output))
        processor.save_pretrained(str(output))
        trainer.save_state()
        manifest.update(status="adapter_saved", training_metrics=result.metrics,
                        global_step=trainer.state.global_step,
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        write_manifest(manifest_path, manifest)
        if config.export_merged:
            merged = output.with_name(output.name + "-merged")
            if merged.exists() and any(merged.iterdir()):
                raise ValueError(f"Merged output is not empty: {merged}")
            model.save_pretrained_merged(str(merged), processor, save_method="merged_16bit")
            manifest["merged_model"] = str(merged)
        manifest.update(status="completed", elapsed_seconds=round(time.monotonic() - started, 3),
                        artifacts=[{"path": str(path.relative_to(output)), "sha256": sha256(path)}
                                   for path in sorted(output.iterdir()) if path.is_file() and path != manifest_path],
                        scope="Training completed. Downstream benchmark quality and deployment require separate validation.")
        write_manifest(manifest_path, manifest)
        return manifest
    except BaseException as exc:
        interrupted = isinstance(exc, (KeyboardInterrupt, _TerminationRequested))
        manifest.update(status="interrupted" if interrupted else "failed",
                        error_type=type(exc).__name__, elapsed_seconds=round(time.monotonic() - started, 3))
        if isinstance(exc, _TerminationRequested):
            manifest.update(signal=signal.Signals(exc.signum).name, signal_number=exc.signum)
        write_manifest(manifest_path, manifest)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="directory with train.jsonl and val.jsonl")
    parser.add_argument("--out", default="outputs/schematic-lora")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="safetensors model ID or local snapshot directory")
    parser.add_argument("--revision", help="Hub commit; default model is pinned automatically")
    parser.add_argument("--4bit", dest="four_bit", action="store_true", help="optional QLoRA; bf16 is recommended for Qwen3.5")
    parser.add_argument("--placement", choices=("auto", "single_gpu"), default="auto",
                        help="loader default or initial placement on the first visible GPU; neither guarantees fit")
    parser.add_argument("--offload-layers", type=int, default=0,
                        help="frozen decoder layers to stream from host RAM; 0 disables block offload")
    parser.add_argument("--offload-prefetch", type=int, default=1,
                        help="number of offloaded layers to prefetch; greater depth uses more GPU memory")
    parser.add_argument("--max-seq", type=int, default=4096)
    parser.add_argument("--max-image-size", type=int, default=1024, help="maximum image side in pixels")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--max-steps", type=int, default=-1, help="positive step limit overrides epochs")
    parser.add_argument("--max-train-rows", type=int, help="deterministic subset for a smoke run")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, help="opt-in teacher-forced validation loss interval")
    parser.add_argument("--load-best-model-at-end", action="store_true",
                        help="save the lowest validation-loss adapter; requires equal eval/save intervals")
    parser.add_argument("--early-stopping-patience", type=int,
                        help="stop after this many unimproved evaluations; requires best loss selection")
    parser.add_argument("--last-response-only", action="store_true",
                        help="supervise only the final assistant answer; earlier answers remain context")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--resume", metavar="CHECKPOINT")
    parser.add_argument("--continue-adapter", action="store_true", help="train existing local LoRA weights in a new run with a fresh optimizer")
    parser.add_argument("--export-merged", action="store_true", help="also export a separate merged 16-bit model")
    parser.add_argument("--finetune-vision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-free-vram-gb", type=float, default=12)
    parser.add_argument("--allow-download", dest="local_files_only", action="store_false", help="allow model downloads into --cache-dir")
    parser.add_argument("--cache-dir", default="models/huggingface")
    parser.add_argument("--dry-run", action="store_true", help="validate data; no GPU imports or writes")
    args = vars(parser.parse_args(argv))
    dry_run = args.pop("dry_run")
    try:
        config = TrainingConfig(**args)
        report = preflight(config) if dry_run else run_training(config)
        print(json.dumps(report, indent=2, allow_nan=False))
        return 75 if report.get("status") == "yielded" else 0
    except _TerminationRequested as exc:
        print(f"training: interrupted by {signal.Signals(exc.signum).name}", file=sys.stderr)
        return 128 + exc.signum
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f"training: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
