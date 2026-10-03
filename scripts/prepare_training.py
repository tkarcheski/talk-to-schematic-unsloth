"""Freeze evidence-assisted and image-only views with content-addressed images.

The training split contains every supplied training row once. Validation and
test rows retain their original split and board family. Test data is included
for reproducible benchmarks, never copied into training or validation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from evaluate import load_gold
from schematic_model.training import resolve_image, sha256


def prepare(evidence: str, vision: str, output: str) -> dict:
    destination = Path(output).resolve()
    if destination.exists():
        raise ValueError("Snapshot output already exists; choose a new path")
    sources = {"evidence": Path(evidence).resolve(), "vision": Path(vision).resolve()}
    views, source_files, images, ownership, identifiers = {}, [], {}, {}, set()
    for view, directory in sources.items():
        views[view] = {}
        for split in ("train", "val", "test"):
            source = directory / f"{split}.jsonl"
            rows = load_gold(source)
            source_files.append({"view": view, "split": split, "path": str(source), "sha256": sha256(source)})
            for row in rows:
                if row.get("split") != split:
                    raise ValueError(f"Row {row['id']} declares a different split from {source}")
                family = row.get("family_id")
                if not isinstance(family, str) or not family:
                    raise ValueError(f"Row {row['id']} has no board family_id")
                if family in ownership and ownership[family] != split:
                    raise ValueError(f"Board family leakage: {family}")
                ownership[family] = split
                if row["id"] in identifiers:
                    raise ValueError(f"Duplicate conversation id across views: {row['id']}")
                identifiers.add(row["id"])
                row["training_view"] = view
                for message in row["messages"]:
                    for block in message["content"]:
                        if block["type"] == "image":
                            original = resolve_image(block["image"], source)
                            digest = sha256(original)
                            name = digest + original.suffix.lower()
                            if digest in images and images[digest]["split"] != split:
                                raise ValueError("Identical image bytes occur in different splits")
                            images[digest] = {"source": original, "name": name, "split": split}
                            block["image"] = "images/" + name
            views[view][split] = rows
    # Both benchmark views must cover the same held-out families.
    for split in ("val", "test"):
        families = [{row["family_id"] for row in views[view][split]} for view in sources]
        if families[0] != families[1]:
            raise ValueError(f"Evidence and vision {split} families differ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=destination.name + ".partial-", dir=destination.parent))
    (temporary / "images").mkdir()
    for digest, image in images.items():
        copied = temporary / "images" / image["name"]
        shutil.copyfile(image["source"], copied)
        if sha256(copied) != digest:
            raise ValueError("Image changed while creating the snapshot")
    artifacts = []

    def write_rows(path: Path, rows: list[dict]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        artifacts.append({"path": str(path.relative_to(temporary)), "sha256": sha256(path), "rows": len(rows)})

    for split in ("train", "val", "test"):
        combined = [row for view in sources for row in views[view][split]]
        write_rows(temporary / f"{split}.jsonl", combined)
        for view in sources:
            rows = copy.deepcopy(views[view][split])
            for row in rows:
                for message in row["messages"]:
                    for block in message["content"]:
                        if block["type"] == "image":
                            block["image"] = "../" + block["image"]
            write_rows(temporary / view / f"{split}.jsonl", rows)
    manifest = {
        "schema_version": 1, "purpose": "mixed evidence-assisted and image-only training snapshot",
        "mix": "one copy of every row in each view; no held-out rows in training",
        "source_files": source_files,
        "families": {split: sorted(family for family, assigned in ownership.items() if assigned == split)
                     for split in ("train", "val", "test")},
        "datasets": artifacts,
        "images": [{"path": "images/" + image["name"], "sha256": digest, "split": image["split"]}
                   for digest, image in sorted(images.items())],
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["snapshot_sha256"] = hashlib.sha256(canonical).hexdigest()
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.rename(destination)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", default="data/real")
    parser.add_argument("--vision", default="data/real/vision")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    manifest = prepare(args.evidence, args.vision, args.out)
    print(json.dumps({"snapshot_sha256": manifest["snapshot_sha256"], "datasets": manifest["datasets"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
