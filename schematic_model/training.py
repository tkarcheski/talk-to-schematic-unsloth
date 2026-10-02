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
import io
import json
import math
from pathlib import Path
import platform
import random
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
    max_seq: int = 4096
    max_image_size: int = 1024
    epochs: float = 2.0
    max_steps: int = -1
    max_train_rows: int | None = None
    lr: float = 1e-4
    rank: int = 16
    gradient_accumulation: int = 4
    save_steps: int = 50
    seed: int = 3407
    resume: str | None = None
    export_merged: bool = False
    finetune_vision: bool = True
    min_free_vram_gb: float = 12.0
    local_files_only: bool = True
    cache_dir: str = "models/huggingface"

    def validate(self) -> None:
        for name in ("max_seq", "max_image_size", "rank", "gradient_accumulation", "save_steps"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
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
        if self.resume and not (Path(self.resume) / "trainer_state.json").is_file():
            raise ValueError("resume must name a checkpoint containing trainer_state.json")

    @property
    def effective_revision(self) -> str | None:
        return self.revision or (DEFAULT_REVISION if self.model == DEFAULT_MODEL else None)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_image(path: str, dataset_path: Path) -> Path:
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


def installed_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("unsloth", "unsloth_zoo", "transformers", "trl", "torch", "peft", "Pillow"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def preflight(config: TrainingConfig) -> dict[str, Any]:
    """Check complete data splits on CPU, without modifying any files."""
    config.validate()
    dataset_hashes = {split: sha256(Path(config.data, filename))
                      for split, filename in (("train", "train.jsonl"), ("validation", "val.jsonl"))}
    train, validation = _read_data(config, expected_hashes=dataset_hashes)
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
    return {
        "schema_version": 1, "status": "preflight",
        "created_at": datetime.now(timezone.utc).isoformat(), "config": asdict(config),
        "model_revision": config.effective_revision, "python": platform.python_version(),
        "packages": installed_versions(),
        "datasets": {
            split: {"path": str(Path(config.data, filename).resolve()), "sha256": dataset_hashes[split],
                    "rows": len(rows), "assistant_turns": sum(m["role"] == "assistant" for r in rows for m in r["messages"])}
            for split, filename, rows in (("train", "train.jsonl", train), ("validation", "val.jsonl", validation))
        },
        "images": records,
        "scope": "Dataset checks only. Model loading, token lengths, GPU fit and training quality are not yet verified.",
    }


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


@_interrupt_on_sigterm()
def run_training(config: TrainingConfig) -> dict[str, Any]:
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
        from unsloth.trainer import UnslothVisionDataCollator
        import torch
        from trl import SFTConfig, SFTTrainer

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
        model, processor = FastVisionModel.from_pretrained(
            model_name=config.model, revision=config.effective_revision,
            max_seq_length=config.max_seq, load_in_4bit=config.four_bit,
            dtype=torch.bfloat16 if is_bf16_supported() else torch.float16,
            use_gradient_checkpointing="unsloth", full_finetuning=False,
            local_files_only=config.local_files_only, cache_dir=str(Path(config.cache_dir).resolve()),
            trust_remote_code=False,
        )
        model = FastVisionModel.get_peft_model(
            model, finetune_vision_layers=config.finetune_vision,
            finetune_language_layers=True, finetune_attention_modules=True,
            finetune_mlp_modules=True, r=config.rank, lora_alpha=config.rank,
            lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth", random_state=config.seed,
        )
        FastVisionModel.for_training(model)
        collator = CheckedVisionCollator(UnslothVisionDataCollator(
            model, processor, resize=config.max_image_size, resize_dimension="max",
            train_on_responses_only=True, instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        ), config.max_seq, image_hashes={record["path"]: record["sha256"] for record in manifest["images"]})
        collator([train_data[0]])
        trainer = SFTTrainer(
            model=model, processing_class=processor, data_collator=collator,
            train_dataset=train_data, eval_dataset=val_data,
            args=SFTConfig(
                output_dir=str(output), per_device_train_batch_size=1, per_device_eval_batch_size=1,
                gradient_accumulation_steps=config.gradient_accumulation,
                num_train_epochs=config.epochs, max_steps=config.max_steps,
                learning_rate=config.lr, warmup_ratio=0.05, lr_scheduler_type="cosine",
                logging_steps=1, eval_strategy="no", save_steps=config.save_steps, save_total_limit=2,
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
    parser.add_argument("--data", default="data", help="directory with train.jsonl and val.jsonl")
    parser.add_argument("--out", default="outputs/schematic-lora")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="safetensors model ID or local snapshot directory")
    parser.add_argument("--revision", help="Hub commit; default model is pinned automatically")
    parser.add_argument("--4bit", dest="four_bit", action="store_true", help="optional QLoRA; bf16 is recommended for Qwen3.5")
    parser.add_argument("--max-seq", type=int, default=4096)
    parser.add_argument("--max-image-size", type=int, default=1024, help="maximum image side in pixels")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--max-steps", type=int, default=-1, help="positive step limit overrides epochs")
    parser.add_argument("--max-train-rows", type=int, help="deterministic subset for a smoke run")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--resume", metavar="CHECKPOINT")
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
        return 0
    except _TerminationRequested as exc:
        print(f"training: interrupted by {signal.Signals(exc.signum).name}", file=sys.stderr)
        return 128 + exc.signum
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f"training: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
