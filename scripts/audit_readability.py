"""Flag coincident text anchors for manual review without changing any dataset.

This is a limited geometry audit, not OCR or a font-metric readability gate. A
target without a flag remains unreviewed. Source-inherent overlaps are retained.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import xml.etree.ElementTree as ET

from schematic_model.render_eagle import IDENTITY, RENDER_METHOD, _Drawing, _rotation, source_svg

NAMESPACE = "{http://www.w3.org/2000/svg}"
BASELINE_GAP_FRACTION = 0.25
ANCHOR_TOLERANCE_MM = 0.0001


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_record(node: ET.Element) -> dict:
    spans = list(node.findall(NAMESPACE + "tspan")) or list(node.findall("tspan"))
    rotation = re.fullmatch(r"rotate\(([-\d.]+) [-\d.]+ [-\d.]+\)", node.get("transform", ""))
    return {
        "text": "\n".join("".join(span.itertext()) for span in spans) if spans else "".join(node.itertext()),
        "x_mm": float(node.get("x", "0")), "y_mm": float(node.get("y", "0")),
        "size_mm": float(node.get("font-size", "0")),
        "rotation_degrees": float(rotation[1]) if rotation else 0.0,
        "horizontal_anchor": node.get("text-anchor", "start"),
        "vertical_baseline": node.get("dominant-baseline", "alphabetic"),
        "line_offsets_mm": [float(span.get("dy", "0")) for span in spans],
    }


def close_anchor(first: dict, second: dict) -> dict | None:
    """Only flag aligned single-line texts with coincident baseline-axis origins."""
    if (not first["text"] or not second["text"] or first["text"] == second["text"]
            or "\n" in first["text"] or "\n" in second["text"]
            or first["horizontal_anchor"] != second["horizontal_anchor"]
            or first["vertical_baseline"] != second["vertical_baseline"]
            or abs((first["rotation_degrees"] - second["rotation_degrees"]) % 360) > 0.0001):
        return None
    angle = math.radians(first["rotation_degrees"])
    dx, dy = second["x_mm"] - first["x_mm"], second["y_mm"] - first["y_mm"]
    along = dx * math.cos(angle) + dy * math.sin(angle)
    gap = abs(-dx * math.sin(angle) + dy * math.cos(angle))
    threshold = BASELINE_GAP_FRACTION * min(first["size_mm"], second["size_mm"])
    if abs(along) > ANCHOR_TOLERANCE_MM or not gap < threshold:
        return None
    return {"reason": "coincident_baseline_axis_origin_and_small_perpendicular_gap",
            "along_baseline_delta_mm": round(along, 6), "perpendicular_gap_mm": round(gap, 6),
            "threshold_mm": round(threshold, 6), "confidence": "high_confidence_geometry_flag_not_readability_verdict"}


def source_name_value_labels(source: Path, page: int) -> dict[str, list[dict]]:
    """Associate original NAME/VALUE nodes with their actual renderer output."""
    schematic = ET.parse(source).getroot().find("./drawing/schematic")
    if schematic is None:
        raise ValueError("Source is not an EAGLE schematic")
    sheet = schematic.findall("./sheets/sheet")[page - 1]
    parts = {part.get("name"): part for part in schematic.findall("./parts/part")}
    libraries = {library.get("name"): library for library in schematic.findall("./libraries/library")}
    labels = {}
    for instance in sheet.findall("./instances/instance"):
        part = parts[instance.get("part")]
        library = libraries[part.get("library")]
        device_set = next(node for node in library.findall("./devicesets/deviceset") if node.get("name") == part.get("deviceset"))
        gate = next(node for node in device_set.findall("./gates/gate") if node.get("name") == instance.get("gate"))
        symbol = next(node for node in library.findall("./symbols/symbol") if node.get("name") == gate.get("symbol"))
        angle, mirrored = _rotation(instance.get("rot", "R0"))
        transform = (float(instance.get("x")), float(instance.get("y")), angle, mirrored)
        substitutions = {"NAME": part.get("name"), "VALUE": part.get("value") or part.get("deviceset")}
        nodes = [(node, transform) for node in symbol.findall("text") if node.text in {">NAME", ">VALUE"}]
        nodes += [(node, IDENTITY) for node in instance.findall("attribute") if node.get("name") in {"NAME", "VALUE"}]
        for node, placement in nodes:
            drawing = _Drawing()
            drawing.primitive(node, placement, substitutions, instance.get("smashed") == "yes")
            if not drawing.elements:
                continue
            kind = node.get("name") if node.tag == "attribute" else node.text[1:]
            rendered = text_record(ET.fromstring(drawing.elements[0]))
            labels.setdefault(part.get("name"), []).append({
                "kind": kind, "gate": instance.get("gate"),
                "source": {"node": node.tag, "x_mm": float(node.get("x", "0")), "y_mm": float(node.get("y", "0")),
                           "size_mm": float(node.get("size", "1.778")), "rotation": node.get("rot", "R0"),
                           "alignment": node.get("align", "bottom-left"), "instance_transform": list(placement)},
                "rendered": rendered,
            })
    return labels


def target_refdes(question: str, label: dict) -> str:
    if isinstance(label.get("refdes"), str) and label["refdes"]:
        return label["refdes"]
    match = re.fullmatch(r"(?:Read the value shown next to (.+)\.|And what value is marked for (.+)\?)", question)
    if not match:
        raise ValueError("Unknown value question format; record refdes explicitly before auditing")
    return match[1] or match[2]


def audit(data: Path, corpus: Path) -> dict:
    corpus = corpus.resolve()
    targets, inputs, cached = [], [], {}
    for split in ("train", "val", "test"):
        dataset = data / f"{split}.jsonl"
        inputs.append({"split": split, "path": str(dataset), "sha256": digest(dataset)})
        for line in dataset.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            provenance = row["provenance"]
            if provenance.get("evidence_mode") != "image_only_values_and_refusals":
                raise ValueError("Readability audit expects the image-only view")
            source = (corpus / "sources" / provenance["repository"].replace("/", "__") / provenance["source_path"]).resolve()
            if not source.is_relative_to(corpus) or digest(source) != provenance["sha256"]:
                raise ValueError("Native source is outside the corpus or fails its recorded hash")
            page = provenance["page"]
            image_record = next(image for image in provenance["images"] if image["page"] == page)
            references = [block["image"] for message in row["messages"] for block in message["content"]
                          if block["type"] == "image"]
            if len(references) != 1 or digest((dataset.parent / references[0]).resolve()) != image_record["sha256"]:
                raise ValueError("Image-only target does not match its frozen image hash")
            key = (str(source), page)
            if key not in cached:
                svg = source_svg(source, page, attribution="Adafruit Industries | CC BY-SA 3.0 | source notices retained")
                texts = [text_record(node) for node in ET.fromstring(svg).iter(NAMESPACE + "text")]
                cached[key] = source_name_value_labels(source, page), texts, hashlib.sha256(svg.encode()).hexdigest()
            labels, texts, svg_hash = cached[key]
            questions = ["\n".join(block["text"] for block in message["content"] if block["type"] == "text")
                         for message in row["messages"] if message["role"] == "user"]
            for turn, gold in enumerate(row["gold"]):
                if gold["type"] != "value":
                    continue
                refdes = target_refdes(questions[turn], gold)
                target_labels = labels.get(refdes, [])
                flags = []
                for label in target_labels:
                    for other in texts:
                        finding = close_anchor(label["rendered"], other)
                        if finding:
                            flags.append({"target_label": label["kind"], **finding, "other_text": other})
                missing = {"NAME", "VALUE"} - {label["kind"] for label in target_labels}
                if missing:
                    flags.append({"reason": "missing_rendered_source_name_or_value_label", "missing": sorted(missing)})
                targets.append({
                    "key": f"{row['id']}#{turn}", "split": split, "refdes": refdes, "value": gold["value"],
                    "source_url": provenance["source_url"], "source_sha256": provenance["sha256"], "page": page,
                    "frozen_image_sha256": image_record["sha256"],
                    "candidate_svg_sha256": svg_hash, "labels": target_labels,
                    "status": "potential_overlap_manual_review_required" if flags else "unreviewed_no_axis_collision_flag",
                    "flags": flags,
                })
    return {
        "schema_version": 1, "status": "read_only_geometry_audit_no_dataset_changes",
        "candidate_render_method": RENDER_METHOD, "inputs": inputs,
        "method": {"coordinate_units": "millimetres; SVG y axis points downward", "baseline_gap_fraction": BASELINE_GAP_FRACTION,
                   "anchor_tolerance_mm": ANCHOR_TOLERANCE_MM,
                   "scope": "single-line text with matching rotation, horizontal anchor and vertical baseline; same baseline-axis origin and gap under one quarter of the smaller source size",
                   "limitations": "No glyph-width/font-metric measurement, OCR, image downscale assessment, wire occlusion, or full visual review. Unflagged targets remain unreviewed. Flags do not automatically exclude data. SVG is generated in memory; existing PNGs are never rewritten."},
        "counts": {"targets": len(targets), "source_sheets": len(cached),
                   "by_split": dict(Counter(target["split"] for target in targets)),
                   "flagged_targets": sum(bool(target["flags"]) for target in targets),
                   "unreviewed_targets_without_flag": sum(not target["flags"] for target in targets)},
        "targets": targets,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/real/vision"))
    parser.add_argument("--corpus", type=Path, default=Path("data/real"))
    parser.add_argument("--out", type=Path, default=Path("docs/validation/readability-audit.json"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.out.exists() and not args.overwrite:
        parser.error("Audit output exists; use --overwrite")
    report = audit(args.data, args.corpus)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.out.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(args.out)
    print(json.dumps(report["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
