"""Plan or freeze expanded image-only training without changing held-out data.

The default command only prints a composition plan. --write --out NEW_PATH
publishes a new snapshot after the caller has reviewed that plan. Source corpus
rows supply training images (including a separately rendered v3 corpus); the
base snapshot supplies the exact original validation/test files and images.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET

from evaluate import _invalid_constant, _unique_object, question_text, validate_gold_rows
from schematic_model.corpus import parse_eagle
from schematic_model.training import resolve_image

SPLITS = ("train", "val", "test")
REFDES = r"[A-Z][A-Z0-9_$-]*"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decode_json(data: bytes):
    return json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object,
                      parse_constant=_invalid_constant)


def read_rows(data: bytes) -> list[dict]:
    rows = [decode_json(line) for line in data.splitlines()]
    validate_gold_rows(rows)
    return rows


def _inside(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Manifest paths must be nonempty relative paths")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Manifest path escapes its snapshot/source root")
    return path


def _images(row: dict):
    for message in row["messages"]:
        content = message.get("content")
        if not isinstance(content, list):
            raise ValueError("Corpus messages must use typed content lists")
        for block in content:
            if not isinstance(block, dict):
                raise ValueError("Corpus content blocks must be objects")
            if block.get("type") == "image":
                yield block


def _without_image_paths(row: dict):
    result = copy.deepcopy(row)
    for block in _images(result):
        block["image"] = "<image>"
    return result["messages"]


@dataclass(frozen=True)
class _SourceBytes:
    """Give the native parser the exact bytes already bound to provenance."""
    data: bytes

    def read_bytes(self) -> bytes:
        return self.data


@dataclass
class ExpandedPlan:
    train_views: dict[str, list[dict]]
    preserved: dict[str, bytes]
    images: dict[str, dict]
    inputs: dict[str, str]
    manifest: dict
    protected_roots: tuple[Path, ...]

    @property
    def summary(self) -> dict:
        return {"purpose": self.manifest["purpose"], "composition": self.manifest["composition"],
                "sampling": self.manifest["sampling"], "seed": self.manifest["seed"],
                "base_snapshot_sha256": self.manifest["base_snapshot_sha256"],
                "limitations": self.manifest["limitations"]}


def _load_base(base: Path, inputs: dict) -> tuple[dict, dict, dict, dict]:
    manifest_path = base / "manifest.json"
    data = manifest_path.read_bytes()
    inputs[str(manifest_path)] = digest(data)
    manifest = decode_json(data)
    canonical = {key: value for key, value in manifest.items() if key != "snapshot_sha256"}
    if manifest.get("schema_version") != 1 or digest(json.dumps(canonical, sort_keys=True,
            separators=(",", ":")).encode()) != manifest.get("snapshot_sha256"):
        raise ValueError("Base snapshot manifest hash/schema is invalid")
    datasets, preserved, image_data = {}, {}, {}
    for section in ("datasets", "images"):
        for artifact in manifest[section]:
            path = _inside(base, artifact["path"])
            data = path.read_bytes()
            if digest(data) != artifact["sha256"]:
                raise ValueError(f"Base snapshot artifact hash mismatch: {path}")
            inputs[str(path)] = artifact["sha256"]
            if section == "datasets":
                rows = read_rows(data)
                if len(rows) != artifact["rows"]:
                    raise ValueError("Base snapshot dataset row count mismatch")
                if artifact["path"] in datasets:
                    raise ValueError("Duplicate base snapshot dataset path")
                datasets[artifact["path"]] = rows
                if Path(artifact["path"]).name in {"val.jsonl", "test.jsonl"}:
                    preserved[artifact["path"]] = data
            else:
                if artifact["path"] in image_data:
                    raise ValueError("Duplicate base snapshot image path")
                image_data[artifact["path"]] = {**artifact, "data": data}
    required = {f"{view}/{split}.jsonl" for view in ("evidence", "vision") for split in SPLITS}
    required |= {f"{split}.jsonl" for split in SPLITS}
    if not required <= datasets.keys():
        raise ValueError("Base snapshot lacks required combined/evidence/vision splits")
    ownership, source_ownership, image_ownership = {}, {}, {}
    for relative, rows in datasets.items():
        split = Path(relative).stem
        if split not in SPLITS:
            raise ValueError("Unexpected dataset split in base manifest")
        for row in rows:
            family = row.get("family_id")
            if row.get("split") != split or not isinstance(family, str) or not family:
                raise ValueError("Base rows need correct split and nonempty family_id")
            if family in ownership and ownership[family] != split:
                raise ValueError("Base snapshot family leakage")
            ownership[family] = split
            source_hash = row.get("provenance", {}).get("sha256")
            if source_hash:
                if source_hash in source_ownership and source_ownership[source_hash] != split:
                    raise ValueError("Base snapshot native source leakage")
                source_ownership[source_hash] = split
            for block in _images(row):
                path = _inside(base, str((Path(relative).parent / block["image"])))
                key = str(path.relative_to(base))
                if key not in image_data or image_data[key]["split"] != split:
                    raise ValueError("Base image is absent from manifest or assigned to another split")
                image_hash = image_data[key]["sha256"]
                if image_hash in image_ownership and image_ownership[image_hash] != split:
                    raise ValueError("Base snapshot image leakage")
                image_ownership[image_hash] = split
    actual_families = {split: sorted(f for f, s in ownership.items() if s == split) for split in SPLITS}
    if actual_families != manifest.get("families"):
        raise ValueError("Base manifest families disagree with its datasets")
    return manifest, datasets, preserved, image_data


def _absent_ref(source_refs: set[str], identity: str, seed: int) -> str:
    # Include plausible low-numbered designators and several component kinds;
    # every candidate is checked against *all* source parts, not just visible BOM.
    candidates = [f"{prefix}{number}" for prefix in ("R", "C", "U", "L", "D", "Q", "JP")
                  for number in range(1, 129) if f"{prefix}{number}" not in source_refs]
    if not candidates:
        raise ValueError("No absent component identifier available in the declared candidate pool")
    return min(candidates, key=lambda ref: digest(f"{seed}:{identity}:absent:{ref}".encode()))


def _vision_row(native: dict, suffix: str, system: dict, image: str, turns: list[tuple]) -> dict:
    row = {key: copy.deepcopy(native[key]) for key in ("design_id", "family_id", "split", "provenance")}
    row.update(id=native["id"] + suffix, training_view="vision", messages=[copy.deepcopy(system)], gold=[])
    row["provenance"]["evidence_mode"] = "image_only_values_and_refusals"
    for index, (question, answer, label) in enumerate(turns):
        content = ([{"type": "image", "image": image}] if index == 0 else [])
        content.append({"type": "text", "text": question})
        row["messages"] += [{"role": "user", "content": content},
                            {"role": "assistant", "content": [{"type": "text", "text": answer}]}]
        row["gold"].append(copy.deepcopy(label))
    return row


def plan_snapshot(corpus: str, base_snapshot: str, source_root: str, *, seed: int = 3407,
                  exclusions: str | None = None) -> ExpandedPlan:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    corpus, base, source_root = (Path(p).resolve() for p in (corpus, base_snapshot, source_root))
    inputs = {}
    excluded = {}
    if exclusions is not None:
        exclusion_path = Path(exclusions).resolve()
        exclusion_bytes = exclusion_path.read_bytes()
        inputs[str(exclusion_path)] = digest(exclusion_bytes)
        records = decode_json(exclusion_bytes)
        if not isinstance(records, list):
            raise ValueError("Target exclusions must be a list")
        for item in records:
            if not isinstance(item, dict) or any(not isinstance(item.get(k), str) or not item[k].strip()
                    for k in ("board_id", "refdes", "value", "reason")):
                raise ValueError("Each exclusion needs board_id/refdes/value/reason strings")
            key = item["board_id"], item["refdes"]
            if key in excluded:
                raise ValueError("Duplicate target exclusion")
            excluded[key] = item
    base_manifest, base_rows, preserved, images = _load_base(base, inputs)
    native_path, vision_path = corpus / "train.jsonl", corpus / "vision/train.jsonl"
    raw = {path: path.read_bytes() for path in (native_path, vision_path)}
    for path, data in raw.items():
        inputs[str(path)] = digest(data)
    native_rows, vision_rows = read_rows(raw[native_path]), read_rows(raw[vision_path])
    expected = {row["id"]: row for row in base_rows["evidence/train.jsonl"]}
    expected_vision = {row["id"]: row for row in base_rows["vision/train.jsonl"]}
    if {row["id"] for row in native_rows} != expected.keys():
        raise ValueError("Source corpus training board IDs differ from the frozen base")
    if {row["id"] for row in vision_rows} != expected_vision.keys():
        raise ValueError("Source corpus vision training IDs differ from the frozen base")
    vision_by_board = {(r["design_id"], r["provenance"]["page"]): r for r in vision_rows}
    if len(vision_by_board) != len(vision_rows):
        raise ValueError("Duplicate source vision board/page")
    native_out, vision_out, selections, refusals = [], [], [], []
    exclusions_seen, dropped_groups, visible_count = set(), [], 0
    source_cache = {}
    heldout_hashes = {item["sha256"] for item in images.values() if item["split"] != "train"}
    original_training_images = {item["sha256"] for item in images.values() if item["split"] == "train"}
    for native in sorted(native_rows, key=lambda r: r["id"]):
        frozen = expected[native["id"]]
        if native.get("split") != "train" or any(native.get(k) != frozen.get(k)
                for k in ("design_id", "family_id")):
            raise ValueError("Source training row identity/family/split changed")
        if native["gold"] != frozen["gold"] or _without_image_paths(native) != _without_image_paths(frozen):
            raise ValueError("Original native training messages/labels changed")
        provenance = native["provenance"]
        if any(provenance.get(k) != frozen["provenance"].get(k)
               for k in ("sha256", "repository", "commit", "source_path", "page")):
            raise ValueError("Source training provenance differs from frozen board identity")
        relative = str(Path(provenance["repository"].replace("/", "__")) / provenance["source_path"])
        source = _inside(source_root, relative)
        if source not in source_cache:
            data = source.read_bytes()
            if digest(data) != provenance["sha256"]:
                raise ValueError("Native source hash differs from row provenance")
            inputs[str(source)] = digest(data)
            parsed = parse_eagle(_SourceBytes(data))
            refs = {part.get("name") for part in ET.fromstring(data).findall("./drawing/schematic/parts/part")}
            source_cache[source] = (parsed, refs)
        parsed, source_refs = source_cache[source]
        page = provenance["page"]
        if type(page) is not int or not 1 <= page <= len(parsed["sheets"]):
            raise ValueError("Invalid source page")
        sheet = parsed["sheets"][page - 1]
        image_blocks = list(_images(native))
        if len(image_blocks) != 1:
            raise ValueError("Each source training row must contain exactly one image")
        original = resolve_image(image_blocks[0]["image"], native_path)
        image_bytes = original.read_bytes()
        image_hash = digest(image_bytes)
        if image_hash in heldout_hashes:
            raise ValueError("Identical image bytes leak across training and held-out splits")
        declared = [i for i in provenance["images"] if i["page"] == page]
        if len(declared) != 1 or declared[0]["sha256"] != image_hash:
            raise ValueError("Training image hash differs from row provenance")
        inputs[str(original)] = image_hash
        image_name = "images/" + image_hash + original.suffix.lower()
        images[image_name] = {"path": image_name, "sha256": image_hash, "split": "train", "data": image_bytes}
        native_copy = copy.deepcopy(native)
        native_copy["training_view"] = "evidence"
        next(_images(native_copy))["image"] = image_name
        native_out.append(native_copy)
        vision = vision_by_board.get((native["design_id"], page))
        eligible = [part for part in sheet["bom"] if part.get("value_visible") and part["value"].strip()]
        visible_count += len(eligible)
        if any(not re.fullmatch(REFDES, part["refdes"]) for part in eligible):
            raise ValueError("Visible target has an unsupported reference designator")
        if eligible and vision is None:
            raise ValueError("Visible board has no source vision system/refusal template")
        if vision is None:
            continue
        if vision.get("split") != "train" or vision.get("family_id") != native["family_id"]:
            raise ValueError("Vision row family/split differs from native training board")
        original_vision = expected_vision[vision["id"]]
        if vision["gold"] != original_vision["gold"] or _without_image_paths(vision) != _without_image_paths(original_vision):
            raise ValueError("Original vision training messages/labels changed")
        if any(vision["provenance"].get(k) != provenance.get(k)
               for k in ("sha256", "repository", "commit", "source_path", "page")):
            raise ValueError("Vision/native source provenance differs")
        vision_images = list(_images(vision))
        if len(vision_images) != 1:
            raise ValueError("Each source vision row must contain exactly one image")
        vision_image = resolve_image(vision_images[0]["image"], vision_path)
        if digest(vision_image.read_bytes()) != image_hash:
            raise ValueError("Vision/native training images differ")
        inputs[str(vision_image)] = image_hash
        if not vision["messages"] or vision["messages"][0].get("role") != "system":
            raise ValueError("Vision template must include an image-only system message")
        system = vision["messages"][0]
        by_value = {}
        for part in eligible:
            by_value.setdefault(part["value"], []).append(part)
        for value, candidates in sorted(by_value.items()):
            flags = [excluded[(native["id"], part["refdes"])] for part in candidates
                     if (native["id"], part["refdes"]) in excluded]
            if flags:
                if any(flag["value"] != value for flag in flags):
                    raise ValueError("Excluded target value differs from native source")
                exclusions_seen.update((flag["board_id"], flag["refdes"]) for flag in flags)
                dropped_groups.append({"board_id": native["id"], "value": value,
                                       "explicit_exclusions": flags,
                                       "suppressed_same_value_candidates": sorted(p["refdes"] for p in candidates)})
                # Substituting another equal-valued part would need its own
                # visual review. Exclude the whole cap group instead.
                continue
            chosen = min(candidates, key=lambda part: digest(
                f"{seed}:{native['id']}:{value}:{part['refdes']}".encode()))
            ref = chosen["refdes"]
            suffix = "-expanded-value-" + digest(ref.encode())[:12]
            row = _vision_row(native, suffix, system, image_name,
                              [(f"Read the value shown next to {ref}.", f"{ref} is marked {value}.",
                                {"type": "value", "value": value})])
            row["provenance"]["training_target"] = {"refdes": ref, "literal_value": value,
                                                     "cap": "one per board and exact literal value"}
            vision_out.append(row)
            selections.append({"row_id": row["id"], "board_id": native["id"], "refdes": ref,
                               "value": value, "same_value_candidates": sorted(p["refdes"] for p in candidates)})
        assistant = [m for m in vision["messages"] if m["role"] == "assistant"]
        turns = []
        missing_count = measurement_count = 0
        absent = _absent_ref(source_refs, native["id"], seed)
        for turn, gold in enumerate(vision["gold"]):
            if gold["type"] != "refuse":
                continue
            question = question_text(vision, turn)
            if re.fullmatch(rf"What value is {REFDES}, and what does it connect to\?", question):
                turns.append((f"What value is {absent}, and what does it connect to?",
                              f"There is no {absent} in the supplied sheet evidence. Its value and connections are not shown.", gold))
                missing_count += 1
            elif re.fullmatch(rf"What measured voltage was observed at {REFDES}\.[A-Z0-9_$+-]+ during a powered bench test\?", question):
                content = assistant[turn]["content"]
                if not isinstance(content, list) or len(content) != 1 or content[0].get("type") != "text":
                    raise ValueError("Unsupported measurement refusal answer format")
                turns.append((question, content[0]["text"], gold))
                measurement_count += 1
            else:
                raise ValueError("Unsupported source refusal question; refusing to silently drop it")
        if (missing_count, measurement_count) != (1, 1):
            raise ValueError("Expected one absent-component and one measurement refusal per vision board")
        row = _vision_row(native, "-expanded-refusals", system, image_name, turns)
        row["provenance"]["training_target"] = {"absent_refdes": absent,
            "absence_checked_against": "all original source part declarations"}
        vision_out.append(row)
        refusals.append({"row_id": row["id"], "board_id": native["id"], "absent_refdes": absent})
    if exclusions_seen != excluded.keys():
        raise ValueError(f"Excluded target is not a visible training candidate: {sorted(excluded.keys() - exclusions_seen)}")
    # Keep only referenced training images; frozen held-out image bytes remain exact.
    used_train = {block["image"] for row in native_out + vision_out for block in _images(row)}
    images = {name: item for name, item in images.items() if item["split"] != "train" or name in used_train}
    combined = native_out + vision_out
    validate_gold_rows(combined)
    labels = [gold for row in combined for gold in row["gold"]]
    composition = {"native_rows": len(native_out), "native_turns": sum(len(r["gold"]) for r in native_out),
        "expanded_value_rows": len(selections), "vision_refusal_rows": len(refusals),
        "vision_refusal_turns": 2 * len(refusals), "combined_rows": len(combined), "combined_turns": len(labels),
        "label_counts": {kind: sum(g["type"] == kind for g in labels) for kind in sorted({g["type"] for g in labels})},
        "source_training_boards": len(native_rows), "source_visible_targets": visible_count,
        "visible_targets_after_group_exclusions": sum(len(s["same_value_candidates"]) for s in selections),
        "explicit_excluded_targets": len(excluded), "excluded_board_value_groups": len(dropped_groups),
        "different_absent_identifiers": len({r["absent_refdes"] for r in refusals}),
        "training_images_changed_from_base": sum(item["sha256"] not in original_training_images
                                                    for item in images.values() if item["split"] == "train")}
    manifest = {"schema_version": 1, "purpose": "training-only expanded visible component lookup snapshot",
        "seed": seed, "base_snapshot_sha256": base_manifest["snapshot_sha256"],
        "families": base_manifest["families"], "composition": composition,
        "sampling": "one seeded representative per training board and exact literal value; original native rows once; one two-turn vision refusal row per original vision board; no held-out sampling",
        "source_files": [{"path": path, "sha256": value} for path, value in sorted(inputs.items())],
        "target_selection": selections, "refusal_selection": refusals,
        "excluded_value_groups": dropped_groups,
        "limitations": ["Source NAME/VALUE visibility flags do not certify pixel readability or absence of overlapping labels.",
                        "Native training wording and labels are preserved, including original missing-part identifiers; only image paths/training_view change.",
                        "Validation/test JSONL bytes and referenced images are copied unchanged from the base snapshot."]}
    return ExpandedPlan({"evidence": native_out, "vision": vision_out}, preserved, images, inputs,
                        manifest, (corpus, base, source_root))


def write_snapshot(plan: ExpandedPlan, output: str) -> dict:
    destination = Path(output).resolve()
    if destination.exists() or Path(output).is_symlink():
        raise ValueError("Snapshot output already exists; choose a new path")
    if any(destination.is_relative_to(root) for root in plan.protected_roots):
        raise ValueError("Snapshot output must be outside input roots")
    # A reviewed plan is bound to all inputs, including source XML and images.
    for path, expected in plan.inputs.items():
        if digest(Path(path).read_bytes()) != expected:
            raise ValueError(f"Input changed after planning: {path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name + ".partial-", dir=destination.parent))
    try:
        artifacts = []
        files = dict(plan.preserved)
        combined = plan.train_views["evidence"] + plan.train_views["vision"]
        files["train.jsonl"] = _encode_rows(combined)
        for view, rows in plan.train_views.items():
            rows = copy.deepcopy(rows)
            for row in rows:
                for block in _images(row):
                    block["image"] = "../" + block["image"]
            files[f"{view}/train.jsonl"] = _encode_rows(rows)
        for relative, data in sorted(files.items()):
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            artifacts.append({"path": relative, "sha256": digest(data), "rows": len(read_rows(data))})
        for relative, item in sorted(plan.images.items()):
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(item["data"])
            if digest(path.read_bytes()) != item["sha256"]:
                raise ValueError("Copied image differs from planned bytes")
        manifest = {**plan.manifest, "datasets": artifacts,
                    "images": [{key: value for key, value in item.items() if key != "data"}
                               for _, item in sorted(plan.images.items())]}
        manifest["snapshot_sha256"] = digest(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if destination.exists() or destination.is_symlink():
            raise ValueError("Snapshot destination appeared during generation")
        staging.rename(destination)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _encode_rows(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows).encode()


def prepare(corpus: str, base_snapshot: str, source_root: str, output: str, *, seed: int = 3407,
            exclusions: str | None = None) -> dict:
    return write_snapshot(plan_snapshot(corpus, base_snapshot, source_root, seed=seed, exclusions=exclusions), output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--base-snapshot", required=True)
    parser.add_argument("--source-root", default="data/real/sources")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--exclusions", help="reviewed board_id/refdes/value/reason records; drop the whole board/value group")
    parser.add_argument("--write", action="store_true", help="publish the planned snapshot into a new directory")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    if args.write and not args.out:
        parser.error("--write requires --out")
    if args.out and not args.write:
        parser.error("--out requires --write; omit both for a read-only plan")
    plan = plan_snapshot(args.corpus, args.base_snapshot, args.source_root, seed=args.seed, exclusions=args.exclusions)
    if args.write:
        manifest = write_snapshot(plan, args.out)
        print(json.dumps({"snapshot_sha256": manifest["snapshot_sha256"], **plan.summary}, indent=2))
    else:
        print(json.dumps(plan.summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
