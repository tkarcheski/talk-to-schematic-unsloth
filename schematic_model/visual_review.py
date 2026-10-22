"""Create a private, opt-in raster review pack without changing source labels.

The resize mirrors serving's RGB conversion and maximum-side thumbnail step.
It does not reproduce the model processor, image tokens, or training collator.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile

from PIL import Image

from schematic_model.training import resolve_image

MAX_PIXELS = 40_000_000


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_private(path: Path, payload: bytes) -> None:
    # cspell:ignore WRONLY CREAT fchmod
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(payload)


def _hash(value) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Provenance requires a lowercase SHA-256 digest")
    return value


def _identity(value) -> str:
    if not isinstance(value, str) or not value or value.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", value):
        raise ValueError("Source identity must be nonempty and must not be an absolute path")
    return value


def _prepare(row: dict, source: Path, max_side: int, index: int, staging: Path) -> dict:
    provenance = row.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("evidence_mode") != "image_only_values_and_refusals":
        raise ValueError("Only declared image-only source rows are supported")
    identity = {key: _identity(row.get(key)) for key in ("id", "design_id", "family_id", "split")}
    if identity["split"] not in {"train", "val", "test"}:
        raise ValueError("Unknown source split")
    repository = provenance.get("repository")
    if not isinstance(repository, str) or not re.fullmatch(r"[\w.-]+/[\w.-]+", repository):
        raise ValueError("Provenance requires an owner/repository identity")
    source_path = _identity(provenance.get("source_path"))
    if ".." in PurePosixPath(source_path).parts or "\\" in source_path:
        raise ValueError("Provenance source_path must be repository-relative")
    copied_provenance = {"repository": repository, "source_path": source_path,
                         "sha256": _hash(provenance.get("sha256")),
                         "license": _identity(provenance.get("license")),
                         "evidence_mode": provenance["evidence_mode"]}
    page = provenance.get("page")
    if type(page) is not int or page < 1:
        raise ValueError("Provenance requires a positive page")
    copied_provenance["page"] = page
    records = provenance.get("images")
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise ValueError("Provenance requires image records")
    matches = [record for record in records if record.get("page") == page]
    if len(matches) != 1:
        raise ValueError("Provenance must identify exactly one image for this page")
    expected = _hash(matches[0].get("sha256"))
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Source requires messages")
    references, questions = [], []
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), list):
            raise ValueError("Malformed source message")
        for block in message["content"]:
            if not isinstance(block, dict):
                raise ValueError("Malformed content block")
            if block.get("type") == "image":
                if message.get("role") != "user":
                    raise ValueError("Images must belong to user turns")
                references.append(block.get("image"))
            elif block.get("type") == "text" and message.get("role") == "user":
                questions.append(_identity(block.get("text")))
    if len(references) != 1:
        raise ValueError("Exactly one source image is required")
    raw = resolve_image(references[0], source).read_bytes()
    if _digest(raw) != expected:
        raise ValueError("Source image does not match its provenance hash")
    # Decode the bytes that were hashed, never reopen the source file.
    with Image.open(io.BytesIO(raw)) as image:
        if image.width * image.height > MAX_PIXELS:
            raise ValueError("Image exceeds the review pixel budget")
        original_size = list(image.size)
        image.load()
        converted = image.convert("RGB")
        converted.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    name = f"images/{index:04d}.png"
    encoded = io.BytesIO()
    converted.save(encoded, format="PNG")
    _write_private(staging / name, encoded.getvalue())
    return {
        **identity, "provenance": copied_provenance, "source_image_sha256": expected,
        "source_dimensions": original_size, "review_image": name,
        "review_image_sha256": _digest((staging / name).read_bytes()),
        "review_dimensions": list(converted.size), "source_questions": questions,
        "review": {"status": "unreviewed", "readability": None, "notes": None},
    }


def build(source: Path, output: Path, *, max_side: int, max_rows: int = 25) -> dict:
    """Write an atomic new pack under ignored outputs; refuse existing paths."""
    if type(max_side) is not int or not 1 <= max_side <= 4096:
        raise ValueError("max_side must be between 1 and 4096")
    if type(max_rows) is not int or not 1 <= max_rows <= 1000:
        raise ValueError("max_rows must be between 1 and 1000")
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("Output exists; choose a new review pack")
    if not output.resolve().is_relative_to(Path.cwd().resolve() / "outputs"):
        raise ValueError("Private review packs must be under the ignored outputs directory")
    raw = source.read_bytes()
    rows: list[tuple[int, dict]] = []
    families: dict[str, str] = {}
    ids: set[str] = set()
    source_rows = 0
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("Source rows must be objects")
        identity = {key: _identity(row.get(key)) for key in ("id", "design_id", "family_id", "split")}
        if identity["split"] not in {"train", "val", "test"}:
            raise ValueError("Unknown source split")
        if identity["id"] in ids:
            raise ValueError("Duplicate source row ID")
        ids.add(identity["id"])
        split = families.setdefault(identity["family_id"], identity["split"])
        if split != identity["split"]:
            raise ValueError("A source family crosses source splits")
        source_rows += 1
        if len(rows) < max_rows:
            rows.append((line_number, row))
    if not rows:
        raise ValueError("Source has no rows")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".visual-review-", dir=output.parent) as temporary:
        staging = Path(temporary) / "pack"
        staging.mkdir(mode=0o700)
        staging.chmod(0o700)
        (staging / "images").mkdir(mode=0o700)
        (staging / "images").chmod(0o700)
        reviews = []
        for index, (line_number, row) in enumerate(rows):
            record = _prepare(row, source, max_side, index, staging)
            record["source_line"] = line_number
            reviews.append(record)
        manifest = {
            "schema_version": 1, "status": "unreviewed", "source_dataset_sha256": _digest(raw),
            "selection": {"method": "first nonempty rows", "max_rows": max_rows, "selected_rows": len(rows),
                          "source_rows_identity_checked": source_rows},
            "transform": {"mode": "RGB", "max_side": max_side, "resample": "LANCZOS", "upscale": False},
            "scope": "Private manual review only. Serving-style raster resize, not exact processor parity. "
                     "No readability verdict, source-native verification, or model-quality result. "
                     "Source provenance retained; selected images verified against recorded hashes. "
                     "All source row identities and family splits checked. Source data unchanged.",
            "rows": reviews,
        }
        _write_private(staging / "manifest.json", (json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
        if output.exists() or output.is_symlink():
            raise ValueError("Output appeared during staging; refusing overwrite")
        staging.rename(output)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new private path under outputs/")
    parser.add_argument("--max-side", type=int, required=True)
    parser.add_argument("--max-rows", type=int, default=25)
    args = parser.parse_args(argv)
    try:
        report = build(args.source, args.out, max_side=args.max_side, max_rows=args.max_rows)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"status": report["status"], "selected_rows": report["selection"]["selected_rows"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
