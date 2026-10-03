"""Build viewer-command conversations from a verified real corpus.

Each conversation teaches and tests the model to drive the schematic viewer:
show a named part or net, frame two parts together from what it sees, report a
label that is absent, and zoom out. Targets and boxes come from source EAGLE
geometry for the exact rendered image. A page is used only when its SVG is
reproduced byte-for-byte, so the boxes cannot drift from the pixels.

Rows are written in the plain text wire format from :mod:`view_tools`, so the
existing training, benchmark and evaluation code reads them unchanged.

  python -m schematic_model.view_data --corpus data/sparkfun --out data/sparkfun/view
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from .corpus import _atomic_text, _split, parse_eagle, publisher
from .view_tools import VIEW_TOOLS, VIEWER_INSTRUCTIONS, box_iou, plain_messages

SYSTEM = ("Answer only from the attached schematic image. If a part, label or measurement is not shown, "
          "say so. Do not invent facts.\n\n" + VIEWER_INSTRUCTIONS)
REFDES = re.compile(r"[A-Z][A-Z0-9_]*\d[A-Z0-9_]*")
NET = re.compile(r"[A-Z][A-Z0-9_+]{1,15}")


def _call(index: int, name: str, arguments: dict) -> dict:
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": f"call-{index}", "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)}}]}


def _result(index: int, value: dict) -> dict:
    return {"role": "tool", "tool_call_id": f"call-{index}", "content": json.dumps(value)}


def _area(box: list[int]) -> int:
    return (box[2] - box[0]) * (box[3] - box[1])


def _union(first: list[int], second: list[int]) -> list[int]:
    return [min(first[0], second[0]), min(first[1], second[1]), max(first[2], second[2]), max(first[3], second[3])]


def conversation(record: dict, sheet: dict, regions: dict, image: str) -> dict | None:
    """Five viewer tasks for one page, or None when the page lacks clear targets."""
    visible = {p["refdes"]: p for p in sheet["bom"] if p.get("value_visible") and REFDES.fullmatch(p["refdes"])}
    parts = sorted((name for name in visible if name in regions["parts"]
                    and 0 < _area(regions["parts"][name]) <= 40_000), key=lambda n: (not n.startswith(("U", "R")), n))
    # Named nets only: N$12 is an anonymous source identifier, not a visible label.
    nets = sorted((name for name, pins in sheet["nets"].items() if NET.fullmatch(name) and 2 <= len(pins)
                   and name in regions["nets"] and _area(regions["nets"][name]) <= 250_000
                   and name not in {"GND", "VCC"}), key=lambda n: (len(sheet["nets"][n]), n))
    if len(parts) < 3 or not nets:
        return None
    first = parts[0]
    pair = next(((a, b) for a in parts for b in parts if a < b and box_iou(regions["parts"][a], regions["parts"][b]) == 0
                 and _area(_union(regions["parts"][a], regions["parts"][b])) <= 250_000), None)
    if pair is None:
        return None
    net = nets[0]
    names = {p["refdes"] for p in sheet["bom"]} | set(sheet["nets"])
    missing = "R999"
    while missing in names:
        missing += "9"
    union = _union(regions["parts"][pair[0]], regions["parts"][pair[1]])
    value = visible[first]["value"]
    turns = [
        (f"Show me {first}.", "show_label", {"label": first},
         {"status": "shown", "label": first, "kind": "part", "box": regions["parts"][first]},
         f"Showing {first}, marked {value}.", {"type": "view_reply", "label": first}),
        (f"Zoom to the {net} net.", "show_label", {"label": net},
         {"status": "shown", "label": net, "kind": "net", "box": regions["nets"][net]},
         f"Showing net {net}.", {"type": "view_reply", "label": net}),
        (f"Frame {pair[0]} and {pair[1]} together in one view.", "show_region",
         {"box": union, "label": f"{pair[0]} and {pair[1]}"},
         {"status": "shown", "box": union},
         f"Showing {pair[0]} and {pair[1]} together.", {"type": "view_reply", "label": pair[1]}),
        (f"Where is {missing}? Show it to me.", "show_label", {"label": missing},
         {"status": "not_found", "label": missing},
         f"{missing} is not on this sheet, so there is nothing to show.", {"type": "view_refuse"}),
        ("Zoom back out to the whole sheet.", "show_full_sheet", {},
         {"status": "shown", "view": "full_sheet"},
         "Showing the full sheet.", {"type": "view_reply"}),
    ]
    messages = [{"role": "system", "content": SYSTEM}]
    gold = []
    for index, (question, name, arguments, result, reply, reply_gold) in enumerate(turns):
        user = question if index else [{"type": "image_url", "image_url": {"url": image}},
                                       {"type": "text", "text": question}]
        messages += [{"role": "user", "content": user}, _call(index, name, arguments),
                     _result(index, result), {"role": "assistant", "content": reply}]
        call_gold = {"type": "view", "name": name, "arguments": arguments}
        if name == "show_region":
            call_gold["min_iou"] = 0.5
        gold += [call_gold, reply_gold]
    plain = []
    for message in plain_messages(messages, VIEW_TOOLS):
        content = message["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        else:
            content = [{"type": "image", "image": b["image_url"]["url"]} if b["type"] == "image_url" else b
                       for b in content]
        plain.append({"role": message["role"], "content": content})
    return {"id": f"{record['id']}-p{sheet['page']}-view", "design_id": record["id"],
            "family_id": record["repository"], "split": _split(record["repository"]),
            "messages": plain, "gold": gold,
            "provenance": {**{k: record[k] for k in ("id", "repository", "commit", "source_path", "sha256", "license")},
                           "page": sheet["page"], "evidence_mode": "image_only_viewer_commands",
                           "box_source": "eagle_source_geometry_of_identical_svg"}}


def build(corpus: Path, output: Path, *, overwrite: bool = False) -> dict:
    from .render_eagle import source_layout

    corpus = corpus.resolve()
    if not overwrite and any((output / f"{split}.jsonl").exists() for split in ("train", "val", "test")):
        raise ValueError("Viewer dataset already exists; use --overwrite to rebuild it explicitly")
    rows, skipped = [], []
    for line in (corpus / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        source = corpus / "sources" / record["repository"].replace("/", "__") / record["source_path"]
        parsed = parse_eagle(source)
        if parsed["sha256"] != record["sha256"]:
            raise ValueError(f"{record['id']}: native source differs from the manifest")
        for image in record["images"]:
            page = image["page"]
            svg, regions = source_layout(source, page, attribution=publisher(record["repository"])["footer"])
            if hashlib.sha256(svg.encode()).hexdigest() != image.get("svg_sha256"):
                skipped.append({"id": record["id"], "page": page, "reason": "render differs from current renderer"})
                continue
            relative = Path("..") / image["path"] if output.resolve().parent == corpus else corpus / image["path"]
            row = conversation(record, parsed["sheets"][page - 1], regions, str(relative))
            if row is None:
                skipped.append({"id": record["id"], "page": page, "reason": "too few clear viewer targets"})
            else:
                rows.append(row)
    for split in ("train", "val", "test"):
        _atomic_text(output / f"{split}.jsonl", "".join(json.dumps(r) + "\n" for r in rows if r["split"] == split))
    summary = {"corpus": str(corpus.name), "conversations": len(rows),
               "turns": sum(len(r["gold"]) for r in rows),
               "split_conversations": {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")},
               "skipped": skipped,
               "claims": ("Viewer targets and boxes are derived from source geometry. Box gold is a source "
                          "bounding region, not a human annotation; final replies are template text.")}
    _atomic_text(output / "summary.json", json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = build(args.corpus, args.out, overwrite=args.overwrite)
    except (ValueError, OSError, KeyError) as error:
        print(json.dumps({"status": "incomplete", "error": str(error)}))
        return 2
    print(json.dumps({k: v for k, v in summary.items() if k != "skipped"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
