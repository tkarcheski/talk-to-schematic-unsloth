"""Plan 25 validation-only prompts; publish a separate snapshot only with --write.

The default uses the frozen v2 validation images. No training/test conversations,
model answers, or model-generated labels are used to construct this set.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET

from evaluate import validate_gold_rows
from schematic_model.corpus import parse_eagle
from scripts.prepare_expanded_training import _artifact_path, decode_json, digest, read_rows

VALUE_PROMPTS = (
    "Please transcribe the component marking for {ref} exactly as printed.",
    "Which value label belongs to {ref} in this drawing?",
    "Look up {ref} on the sheet and give its printed value.",
    "Copy the value text associated with {ref}; retain any suffix.",
    "How is {ref}'s value annotated here?",
    "What text identifies the value of {ref}?",
    "For {ref}, report the adjacent value marking verbatim.",
    "I need the marked value for {ref} from this schematic.",
    "Find {ref} and quote its value label.",
    "Give the complete printed value assigned to {ref}.",
    "Using the drawing, transcribe {ref}'s value including qualifiers.",
    "Tell me the value annotation attached to {ref}.",
    "Please identify {ref}'s printed value without changing its notation.",
)
ABSENT_PROMPTS = (
    "I am looking for {ref}. What value and net membership can you establish from this drawing?",
    "Check whether {ref} appears here, then report its marked value and connections if it does.",
    "For {ref}, identify its value and connected nodes using only the sheet.",
)
MEASUREMENT_PROMPTS = (
    "What voltage did a real meter record at {pad} during operation?",
    "Give the observed bench-test voltage for {pad}; a nominal rail label is not a measurement.",
    "Does this drawing tell us the actual powered-on voltage measured at {pad}? State the measurement if available.",
)
NET_PROMPTS = (
    "Enumerate every physical package pad listed on {net} in the supplied source evidence.",
    "Using the supplied pad mapping, give the complete endpoint set for {net} on this sheet.",
    "Which package pads belong to {net}? Use REFDES.PAD notation.",
)
TRACE_PROMPTS = (
    "Starting at {pad}, list its fellow physical pads on the same source net. Omit the starting pad.",
    "According to the native connections, what other pads share {pad}'s net?",
    "Find {pad} in the provided net map and name the remaining endpoints.",
)
REFDES = r"[A-Z][A-Z0-9_$-]*"
PAD = REFDES + r"\.[A-Z0-9_$+-]+"
SOURCE_OVERLAP_EXCLUSIONS = {("real-8cfa9f969cd7abde-p1", ref) for ref in ("R1", "R2")}


@dataclass
class GeneralizationPlan:
    report: dict
    rows: dict[str, list[dict]]
    images: dict[str, bytes]
    inputs: dict[str, str]
    protected_roots: tuple[Path, ...]


def _choose(items, seed: int, identity: str, key):
    if not items:
        raise ValueError(f"No source-backed candidate for {identity}")
    return min(items, key=lambda item: digest(f"{seed}:{identity}:{key(item)}".encode()))


def _read(path: Path, inputs: dict) -> bytes:
    data = path.read_bytes()
    inputs[str(path)] = digest(data)
    return data


def _row(native: dict, vision: dict, category: str, index: int, image: str,
         question: str, answer: str, gold: dict, sheet: dict) -> dict:
    mode = "evidence" if category == "connectivity" else "vision"
    original = native if mode == "evidence" else vision
    row = {key: copy.deepcopy(native[key]) for key in ("design_id", "family_id", "split", "provenance")}
    row.update(id=f"{native['id']}-generalization-{category}-{index:02d}", training_view=mode)
    row["provenance"]["evidence_mode"] = original["provenance"]["evidence_mode"]
    row["provenance"]["generalization"] = {"category": category, "origin": "validation_only_native_source", "question": question}
    user_text = question
    if mode == "evidence":
        facts = {"scope": "this sheet only; physical package pad numbers, not symbol pin labels",
                 "bom": [{"refdes": part["refdes"], "value": part["value"]} for part in sheet["bom"]],
                 "nets": sheet["nets"]}
        user_text = "Extracted native schematic evidence (not an electrical review):\n" + json.dumps(facts) + "\n\n" + question
    row["messages"] = [copy.deepcopy(original["messages"][0]),
                       {"role": "user", "content": [{"type": "image", "image": image},
                                                    {"type": "text", "text": user_text}]},
                       {"role": "assistant", "content": [{"type": "text", "text": answer}]}]
    row["gold"] = [gold]
    return row


def plan_generalization(snapshot: str, source_root: str, *, seed: int = 3407,
                        overrides: list[dict] | None = None) -> GeneralizationPlan:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    base, source_root = Path(snapshot).resolve(), Path(source_root).resolve()
    inputs = {}
    override_by_board = {}
    if overrides is not None:
        if not isinstance(overrides, list):
            raise ValueError("Reviewed value overrides must be a list")
        for override in overrides:
            required = ("board_id", "original_refdes", "original_value", "refdes", "value", "reason",
                        "source_sha256", "image_sha256", "prior_plan_path", "prior_plan_sha256",
                        "overlap_report_path", "overlap_report_sha256", "serving_review_path")
            if not isinstance(override, dict) or any(not isinstance(override.get(key), str)
                    or not override[key].strip() for key in required):
                raise ValueError("Reviewed override lacks its target, reason or evidence binding")
            if override["board_id"] in override_by_board:
                raise ValueError("Duplicate reviewed board override")
            bound_data = {}
            for path_key, sha_key in (("prior_plan_path", "prior_plan_sha256"),
                                      ("overlap_report_path", "overlap_report_sha256")):
                data = _read(Path(override[path_key]).resolve(), inputs)
                if digest(data) != override[sha_key]:
                    raise ValueError("Reviewed override evidence hash differs")
                bound_data[path_key] = data
            prior = decode_json(bound_data["prior_plan_path"])
            prior_cases = [case for case in prior["cases"] if case["category"] == "value"
                           and case["board_id"] == override["board_id"]]
            if (len(prior_cases) != 1 or prior.get("seed") != seed
                    or prior_cases[0]["target"] != {"refdes": override["original_refdes"],
                                                   "literal_value": override["original_value"]}
                    or any(prior_cases[0][key] != override[key] for key in ("source_sha256", "image_sha256"))):
                raise ValueError("Reviewed override does not identify the original plan target")
            override_by_board[override["board_id"]] = copy.deepcopy(override)
    manifest = decode_json(_read(base / "manifest.json", inputs))
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or digest(json.dumps({k: v for k, v in manifest.items() if k != "snapshot_sha256"},
                                sort_keys=True, separators=(",", ":")).encode()) != manifest.get("snapshot_sha256")):
        raise ValueError("Frozen snapshot manifest hash/schema is invalid")
    families = manifest.get("families", {})
    if (set(families) != {"train", "val", "test"}
            or any(not isinstance(group, list) or any(not isinstance(x, str) or not x for x in group)
                   or len(set(group)) != len(group) for group in families.values())):
        raise ValueError("Snapshot requires explicit unique train/val/test family inventories")
    if (set(families["val"]) & (set(families["train"]) | set(families["test"]))
            or set(families["train"]) & set(families["test"])):
        raise ValueError("Snapshot family inventories overlap")
    inventory = {}
    for section in ("datasets", "images"):
        if not isinstance(manifest.get(section), list):
            raise ValueError("Snapshot needs dataset and image inventories")
        for artifact in manifest[section]:
            path = _artifact_path(base, artifact["path"])
            if path in inventory:
                raise ValueError("Duplicate snapshot artifact")
            inventory[path] = artifact, section
    views = {}
    for view in ("evidence", "vision"):
        path = base / view / "val.jsonl"
        artifact, section = inventory.get(path, ({}, None))
        data = _read(path, inputs)
        if section != "datasets" or digest(data) != artifact.get("sha256"):
            raise ValueError("Validation dataset differs from frozen manifest")
        rows = read_rows(data)
        if len(rows) != 13 or artifact.get("rows") != 13:
            raise ValueError("Generalization set requires exactly 13 validation boards per view")
        if ({row.get("family_id") for row in rows} != set(families["val"])
                or len(families["val"]) != 13 or any(row.get("split") != "val" for row in rows)):
            raise ValueError("Validation rows disagree with held-out family inventory")
        views[view] = {(row["design_id"], row["provenance"]["page"]): row for row in rows}
        if len(views[view]) != 13:
            raise ValueError("Duplicate validation board/page")
    if views["vision"].keys() != views["evidence"].keys():
        raise ValueError("Validation modes refer to different boards")
    boards, images, image_origins = [], {}, {}
    for identity, native in sorted(views["evidence"].items()):
        vision = views["vision"][identity]
        prov = native["provenance"]
        if (native["family_id"] != prov["repository"] or native["family_id"] != vision["family_id"]
                or any(prov.get(key) != vision["provenance"].get(key)
                       for key in ("sha256", "repository", "commit", "source_path", "page"))):
            raise ValueError("Validation source identity differs across modes")
        source = _artifact_path(source_root, prov["repository"].replace("/", "__") + "/" + prov["source_path"])
        raw = _read(source, inputs)
        if digest(raw) != prov["sha256"]:
            raise ValueError("Native source hash differs from validation provenance")
        parsed = parse_eagle(raw)
        page = prov["page"]
        if type(page) is not int or not 1 <= page <= len(parsed["sheets"]):
            raise ValueError("Invalid validation source page")
        sheet = parsed["sheets"][page - 1]
        refs = {part.get("name") for part in ET.fromstring(raw).findall("./drawing/schematic/parts/part")}
        image_digests = []
        for view, row in (("evidence", native), ("vision", vision)):
            if row["messages"][0].get("role") != "system":
                raise ValueError("Validation mode needs its original system message")
            expected_mode = "native_facts_supplied_in_prompt" if view == "evidence" else "image_only_values_and_refusals"
            if row["provenance"].get("evidence_mode") != expected_mode:
                raise ValueError("Validation evidence mode is incorrect")
            blocks = [block for message in row["messages"] for block in message["content"] if block["type"] == "image"]
            if len(blocks) != 1:
                raise ValueError("Each validation row must contain exactly one image")
            path = (base / view / blocks[0]["image"]).resolve()
            artifact, section = inventory.get(path, ({}, None))
            if not path.is_relative_to(base) or section != "images" or artifact.get("split") != "val":
                raise ValueError("Validation image is not bound to its split")
            data = _read(path, inputs)
            declared = [item for item in row["provenance"]["images"] if item["page"] == page]
            sha = digest(data)
            if sha != artifact.get("sha256") or len(declared) != 1 or sha != declared[0].get("sha256"):
                raise ValueError("Validation image hash differs from provenance")
            image_name = "images/" + sha + path.suffix.lower()
            images[image_name] = data
            image_origins[image_name] = str(path)
            image_digests.append(sha)
        if len(set(image_digests)) != 1:
            raise ValueError("Validation modes do not use the same frozen image")
        boards.append((native, vision, sheet, refs, image_name))
    result = {"vision": [], "evidence": []}
    cases = []

    def add(board, category, index, question, answer, gold, target):
        native, vision, sheet, _, image_name = board
        row = _row(native, vision, category, index, image_name, question, answer, gold, sheet)
        result[row["training_view"]].append(row)
        cases.append({"id": row["id"], "board_id": native["id"], "family_id": native["family_id"],
                      "mode": row["training_view"], "category": category, "question": question,
                      "proposed_answer": answer, "gold": gold, "target": target,
                      "source_url": native["provenance"]["source_url"], "source_sha256": native["provenance"]["sha256"],
                      "native_source_path": str(source_root / native["provenance"]["repository"].replace("/", "__")
                                                / native["provenance"]["source_path"]),
                      "frozen_image_path": image_origins[image_name],
                      "page": native["provenance"]["page"], "image": image_name,
                      "image_sha256": digest(images[image_name]), "render_method": native["provenance"].get("render_method"),
                      "readability": "unreviewed_source_visibility_flags_only" if category == "value" else "not_a_value_OCR_label"})

    applied_overrides = set()
    for index, board in enumerate(boards):
        native, _, sheet, _, _ = board
        candidates = [part for part in sheet["bom"] if part.get("value_visible") and part["value"].strip()
                      and (native["id"], part["refdes"]) not in SOURCE_OVERLAP_EXCLUSIONS
                      and re.fullmatch(REFDES, part["refdes"])]
        part = _choose(candidates, seed, native["id"] + ":value", lambda item: item["refdes"])
        override = override_by_board.get(native["id"])
        if override is not None:
            if (part["refdes"] != override["original_refdes"] or part["value"] != override["original_value"]
                    or native["provenance"]["sha256"] != override["source_sha256"]
                    or digest(images[board[4]]) != override["image_sha256"]):
                raise ValueError("Reviewed override original selection/source/image changed")
            replacements = [item for item in candidates if item["refdes"] == override["refdes"]
                            and item["value"] == override["value"]]
            if len(replacements) != 1 or override["refdes"] == override["original_refdes"]:
                raise ValueError("Reviewed replacement is not a different source-visible exact value")
            part = replacements[0]
            applied_overrides.add(native["id"])
        question = VALUE_PROMPTS[index].format(ref=part["refdes"])
        add(board, "value", index, question, f"{part['refdes']} is marked {part['value']}.",
            {"type": "value", "value": part["value"]}, {"refdes": part["refdes"], "literal_value": part["value"]})
    if applied_overrides != override_by_board.keys():
        raise ValueError("Reviewed override names a board outside the validation selection")
    ordered = sorted(boards, key=lambda board: digest(f"{seed}:{board[0]['id']}:additional".encode()))
    for index, board in enumerate(ordered[:6]):
        native, _, sheet, refs, _ = board
        if index % 2 == 0:
            prefix = ("C", "U", "L")[index // 2]
            ref = _choose([f"{prefix}{n}" for n in range(1, 129) if f"{prefix}{n}" not in refs],
                          seed, native["id"] + ":absent", str)
            question = ABSENT_PROMPTS[index // 2].format(ref=ref)
            answer = f"There is no {ref} in the supplied sheet evidence. Its value and connections are not shown."
            target = {"absent_refdes": ref, "absence_checked_against": "all source part declarations"}
            category = "absent_component"
        else:
            pad = _choose(sorted({pad for pads in sheet["nets"].values() for pad in pads if re.fullmatch(PAD, pad)}),
                          seed, native["id"] + ":measurement", str)
            question = MEASUREMENT_PROMPTS[index // 2].format(pad=pad)
            answer = "No bench measurement is supplied in this schematic or source evidence. I cannot infer a measured voltage."
            target = {"physical_pad": pad, "evidence_limit": "schematic only; no bench observations supplied"}
            category = "measurement"
        add(board, category, index, question, answer, {"type": "refuse"}, target)
    for index, board in enumerate(ordered[6:12]):
        native, _, sheet, _, _ = board
        nets = [(net, pads) for net, pads in sheet["nets"].items() if 2 <= len(pads) <= 8
                and all(re.fullmatch(PAD, pad) for pad in pads)]
        net, pads = _choose(nets, seed, native["id"] + ":net", lambda item: item[0])
        gold = {"type": "pins", "pins": list(pads), "net": net}
        target = {"net": net, "physical_pads": list(pads)}
        if index < 3:
            question = NET_PROMPTS[index].format(net=net)
            answer = "The physical pads are " + ", ".join(pads) + "."
        else:
            pad = _choose(pads, seed, native["id"] + ":query", str)
            others = [item for item in pads if item != pad]
            question = TRACE_PROMPTS[index - 3].format(pad=pad)
            answer = "The other physical pads are " + ", ".join(others) + "."
            gold.update(pins=others, query=pad)
            target["query"] = pad
        add(board, "connectivity", index, question, answer, gold, target)
    validate_gold_rows(result["vision"] + result["evidence"])
    if len({case["question"] for case in cases}) != 25:
        raise ValueError("Expected 25 distinct prompt wordings")
    report = {"schema_version": 1, "status": "proposed_not_published", "purpose": "validation-only wording generalization",
              "seed": seed, "base_snapshot_sha256": manifest["snapshot_sha256"],
              "composition": {"boards": 13, "prompts": 25, "image_only_value": 13,
                              "image_only_absent_component": 3, "image_only_measurement": 3, "native_evidence_connectivity": 6},
              "families": {"train": [], "val": families["val"], "test": []},
              "selection_policy": "seeded source-visible part per validation board; seeded six additional refusal boards and six disjoint connectivity boards; no model-output-based selection",
              "explicit_target_exclusions": [{"board_id": board, "refdes": ref,
                  "reason": "confirmed source-inherent name/value overlap; original frozen datasets unchanged"}
                  for board, ref in sorted(SOURCE_OVERLAP_EXCLUSIONS)],
              "inputs": [{"path": path, "sha256": sha} for path, sha in sorted(inputs.items())], "cases": cases,
              "limitations": ["Value visibility flags do not establish pixel readability; every new value target requires review.",
                              "Targets and question wording both change; this measures task generalization, not an isolated wording effect.",
                              "This adds validation prompts, not independent boards, training data, or held-out test evidence.",
                              "Image-only user content contains only the frozen image and question; no native facts, labels, or coordinates are injected.",
                              "Existing deterministic scorers remain unchanged; contextual refusal v2 does not recognize these new question templates and requires manual adjudication."]}
    if override_by_board:
        report["value_overrides"] = [override_by_board[key] for key in sorted(override_by_board)]
        report["revision"] = 2
        report["selection_policy"] += "; explicit source-verified visual-review replacements recorded in value_overrides"
        report["limitations"].append("Serving-scale visual review binds this versioned plan in a separate report; no model-processor readability certification is implied.")
    report["plan_sha256"] = digest(json.dumps(report, sort_keys=True, separators=(",", ":")).encode())
    return GeneralizationPlan(report, result, images, inputs, (base, source_root))


def write_snapshot(plan: GeneralizationPlan, output: str) -> dict:
    destination = Path(output).resolve()
    if destination.exists() or Path(output).is_symlink() or any(destination.is_relative_to(root) for root in plan.protected_roots):
        raise ValueError("Output must be a new path outside input roots")
    for path, expected in plan.inputs.items():
        if digest(Path(path).read_bytes()) != expected:
            raise ValueError("Input changed after generalization planning")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=destination.name + ".partial-", dir=destination.parent))
    try:
        datasets = []
        for view, original in plan.rows.items():
            if view not in {"vision", "evidence"}:
                raise ValueError("Unsupported generalization mode")
            rows = copy.deepcopy(original)
            for row in rows:
                row["messages"][1]["content"][0]["image"] = "../" + row["messages"][1]["content"][0]["image"]
            validate_gold_rows(rows)
            relative = view + "/val.jsonl"
            path = _artifact_path(stage, relative)
            path.parent.mkdir(parents=True)
            data = ''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows).encode()
            path.write_bytes(data)
            datasets.append({"path": relative, "sha256": digest(data), "rows": len(rows)})
        image_records = []
        for relative, data in sorted(plan.images.items()):
            path = _artifact_path(stage, relative)
            if path.parent != stage / "images":
                raise ValueError("Generalization images must be in the image directory")
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(data)
            image_records.append({"path": relative, "sha256": digest(data), "split": "val"})
        manifest = {"schema_version": 1, "purpose": plan.report["purpose"], "plan_sha256": plan.report["plan_sha256"],
                    "base_snapshot_sha256": plan.report["base_snapshot_sha256"], "families": plan.report["families"],
                    "datasets": datasets, "images": image_records, "source_files": plan.report["inputs"]}
        manifest["snapshot_sha256"] = digest(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + '\n')
        if destination.exists() or destination.is_symlink():
            raise ValueError("Output appeared during publication")
        stage.rename(destination)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', default='data/training/real-crop-v1')
    parser.add_argument('--source-root', default='data/real/sources')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--overrides', help='JSON override list or versioned plan containing value_overrides')
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    if bool(args.out) != args.write:
        parser.error('--write and --out must be used together; omit both to review the plan')
    overrides = decode_json(Path(args.overrides).read_bytes()) if args.overrides else None
    if isinstance(overrides, dict):
        if "value_overrides" not in overrides:
            parser.error('Override plan lacks value_overrides')
        overrides = overrides["value_overrides"]
    plan = plan_generalization(args.snapshot, args.source_root, seed=args.seed, overrides=overrides)
    result = write_snapshot(plan, args.out) if args.write else plan.report
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
