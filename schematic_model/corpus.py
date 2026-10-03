"""Pinned, attributed real hardware corpus from original EAGLE XML sources.

Images are derivative renders of source geometry, not vendor screenshots. Gold
answers describe source facts, not whether a circuit is electrically correct.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import tempfile
import urllib.parse
import urllib.request

from .eagle_source import MAX_SOURCE, load_eagle

MAX_DOWNLOAD = 40 * 1024 * 1024
SYSTEM = (
    "You answer factual questions about the attached schematic and its extracted source evidence. "
    "Use only the supplied evidence. Identify connections as REFDES.PAD, where PAD is the physical "
    "package pad number. Distinguish schematic facts from design judgments. Do not invent missing "
    "components, specifications, measurements, or safety findings."
)


# Each publisher is an explicitly reviewed license profile. Repository owners
# outside this table are rejected rather than inferred from search results.
PUBLISHERS = {
    "adafruit": {
        "name": "Adafruit Industries", "license": "CC-BY-SA-3.0",
        "license_marker": "Attribution-ShareAlike 3.0 Unported",
        "attribution": "Adafruit Industries; original notices retained in source README and license",
        "footer": "Adafruit Industries | CC BY-SA 3.0 | source notices retained",
        "discover": ["org:adafruit PCB"],
    },
    "sparkfun": {
        "name": "SparkFun Electronics", "license": "CC-BY-SA-4.0",
        "license_marker": "creativecommons.org/licenses/by-sa/4.0",
        "attribution": "SparkFun Electronics; original notices retained in source README and license",
        "footer": "SparkFun Electronics | CC BY-SA 4.0 | source notices retained",
        "discover": ["org:sparkfun Qwiic in:name", "org:sparkfun Breakout in:name"],
    },
}
REPOSITORY = re.compile(r"(adafruit|sparkfun)/[A-Za-z0-9_.-]+")


def publisher(repository: str) -> dict:
    """Return the reviewed license profile for a pinned GitHub repository name."""
    if not isinstance(repository, str) or not REPOSITORY.fullmatch(repository):
        raise ValueError("The curated importer accepts official Adafruit or SparkFun repositories")
    return PUBLISHERS[repository.split("/", 1)[0]]


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _fetch(url: str) -> bytes:
    if not url.startswith(("https://api.github.com/", "https://codeload.github.com/")):
        raise ValueError("Only explicit public GitHub source URLs are supported")
    request = urllib.request.Request(url, headers={"User-Agent": "talk-to-schematic-corpus/1"})
    with urllib.request.urlopen(request, timeout=45) as response:
        data = response.read(MAX_DOWNLOAD + 1)
    if len(data) > MAX_DOWNLOAD:
        raise ValueError("Source download exceeds the 40 MiB bound")
    return data


def parse_eagle(path: Path | bytes) -> dict:
    """Extract physical-pad nets and visible-sheet parts, rejecting unsupported files."""
    content, schematic = load_eagle(path)
    if schematic.find("./modules/module") is not None:
        raise ValueError("Hierarchical EAGLE modules require an explicit hierarchy resolver")
    libraries = {x.get("name"): x for x in schematic.findall("./libraries/library")}
    parts = {}
    pad_map = {}
    symbol_map = {}
    for part in schematic.findall("./parts/part"):
        name = part.attrib["name"]
        library = libraries.get(part.get("library"))
        if library is None:
            raise ValueError(f"Missing embedded library for {name}")
        device_set = next((x for x in library.findall("./devicesets/deviceset")
                           if x.get("name") == part.get("deviceset")), None)
        if device_set is None:
            raise ValueError(f"Missing embedded device set for {name}")
        device = next((x for x in device_set.findall("./devices/device")
                       if x.get("name", "") == part.get("device", "")), None)
        if device is None:
            raise ValueError(f"Missing embedded device for {name}")
        package = device.get("package", "")
        if name in parts:
            raise ValueError(f"Duplicate part identifier {name}")
        parts[name] = {"refdes": name, "value": part.get("value", ""),
                       "device": part.get("deviceset", ""), "package": package,
                       "library": part.get("library", ""), "physical": bool(package)}
        for gate in device_set.findall("./gates/gate"):
            symbol_map[(name, gate.get("name"))] = next(
                (x for x in library.findall("./symbols/symbol") if x.get("name") == gate.get("symbol")), None)
        for connect in device.findall("./connects/connect"):
            key = (name, connect.attrib["gate"], connect.attrib["pin"])
            pad_map[key] = [f"{name}.{pad}" for pad in connect.get("pad", "").split()]
    sheets = []
    for number, sheet in enumerate(schematic.findall("./sheets/sheet"), 1):
        instances = sheet.findall("./instances/instance")
        names = {i.attrib["part"] for i in instances}
        visible_values = set()
        visible_names = set()
        for instance in instances:
            symbol = symbol_map[(instance.get("part"), instance.get("gate"))]
            if symbol is None:
                raise ValueError("Missing embedded symbol")
            if instance.get("smashed") == "yes":
                visible = any(a.get("name") == "VALUE" and a.get("display", "value") in {"value", "both"}
                              and int(a.get("layer", "96")) in {94, 95, 96, 97, 98}
                              and a.get("value", parts[instance.get("part")]["value"]) == parts[instance.get("part")]["value"]
                              for a in instance.findall("attribute"))
                name_visible = any(a.get("name") == "NAME" and a.get("display", "value") in {"value", "both"}
                                   and int(a.get("layer", "95")) in {94, 95, 96, 97, 98}
                                   and a.get("value", instance.get("part")) == instance.get("part")
                                   for a in instance.findall("attribute"))
            else:
                visible = any(t.text == ">VALUE" and int(t.get("layer", "96")) in {94, 95, 96, 97, 98}
                              for t in symbol.findall("text"))
                name_visible = any(t.text == ">NAME" and int(t.get("layer", "95")) in {94, 95, 96, 97, 98}
                                   for t in symbol.findall("text"))
            if visible and symbol.find("pin") is not None:
                visible_values.add(instance.get("part"))
            if name_visible and symbol.find("pin") is not None:
                visible_names.add(instance.get("part"))
        bom = [{**parts[name], "value_visible": name in visible_values and name in visible_names}
               for name in sorted(names) if parts[name]["physical"]]
        nets = {}
        for net in sheet.findall("./nets/net"):
            pins = set()
            for pin in net.findall("./segment/pinref"):
                key = (pin.attrib["part"], pin.attrib["gate"], pin.attrib["pin"])
                if key not in pad_map and parts[key[0]]["physical"]:
                    raise ValueError(f"Unresolved physical pin {key}")
                pins.update(pad_map.get(key, []))
            if pins:
                nets.setdefault(net.attrib["name"], set()).update(pins)
        pin_to_net = {}
        for net_name, pins in nets.items():
            for pin in pins:
                if pin in pin_to_net and pin_to_net[pin] != net_name:
                    raise ValueError(f"Physical pad {pin} belongs to conflicting sheet nets")
                pin_to_net[pin] = net_name
        sheets.append({"page": number, "bom": bom,
                       "nets": {name: sorted(pins) for name, pins in sorted(nets.items())}})
    if not sheets:
        raise ValueError("Schematic has no sheets")
    return {"format": "eagle_xml", "sha256": sha256(content), "sheets": sheets}


LIBRARY_NAME = re.compile(r"(?i)(arduino|library|_py$|python|circuitpython|micropython|firmware|_lib)")


def _discover(pages: int = 2, publisher_name: str = "adafruit") -> list[dict]:
    rows, seen = [], set()
    for search in PUBLISHERS[publisher_name]["discover"]:
        for page in range(1, pages + 1):
            query = urllib.parse.urlencode({"q": search, "per_page": 100,
                                           "sort": "updated", "order": "desc", "page": page})
            result = json.loads(_fetch("https://api.github.com/search/repositories?" + query))
            for item in result["items"]:
                name = item["full_name"]
                if item.get("archived") or item.get("fork") or name in seen:
                    continue
                # Software-only repositories have no schematic; skip them before download.
                if publisher_name != "adafruit" and LIBRARY_NAME.search(name.split("/", 1)[1]):
                    continue
                seen.add(name)
                rows.append({"repository": name, "revision": item["default_branch"]})
    return rows


def _archive_source(source: dict, directory: Path) -> dict:
    repository, revision = source["repository"], source["revision"]
    profile = publisher(repository)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", revision):
        raise ValueError("Invalid source revision")
    repo_dir = directory / repository.replace("/", "__")
    cache = repo_dir / "source.json"
    if cache.exists():
        previous = json.loads(cache.read_text())
        if revision == previous["revision"]:
            if "files" in source and source["files"] != previous["files"]:
                raise ValueError("Cached file inventory differs from the pinned manifest")
            for item in previous["files"]:
                cached = repo_dir / item["path"]
                if cached.is_symlink() or not cached.is_file() or sha256(cached.read_bytes()) != item["sha256"]:
                    raise ValueError(f"Cached source hash mismatch: {item['path']}")
            return previous
    archive_bytes = _fetch(f"https://codeload.github.com/{repository}/legacy.tar.gz/{revision}")
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        commit = archive.pax_headers.get("comment", "")
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("Archive lacks its full immutable Git commit")
        if re.fullmatch(r"[0-9a-f]{40}", revision) and commit != revision:
            raise ValueError("Downloaded archive differs from pinned commit")
        members = [m for m in archive.getmembers() if m.isfile() and m.size <= MAX_SOURCE]
        licenses = [m for m in members if PurePosixPath(m.name).name.lower() in
                    {"license.txt", "license", "license.md"}]
        license_member = next((m for m in licenses if profile["license_marker"] in
                               archive.extractfile(m).read().decode("utf-8", errors="replace")), None)
        if license_member is None:
            raise ValueError(f"No explicit {profile['license']} license found; manual review required")
        selected = [m for m in members if m.name.lower().endswith(".sch")]
        selected += licenses
        selected += [m for m in members if PurePosixPath(m.name).name.lower().startswith("readme")]
        records = []
        for member in selected:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 2:
                raise ValueError("Unsafe archive path")
            relative = PurePosixPath(*relative.parts[1:])
            data = archive.extractfile(member).read(MAX_SOURCE + 1)
            if len(data) > MAX_SOURCE:
                raise ValueError("Oversized source member")
            destination = repo_dir.joinpath(*relative.parts)
            if not destination.resolve().is_relative_to(directory.resolve()):
                raise ValueError("Source destination escapes its corpus directory")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            records.append({"path": str(relative), "sha256": sha256(data), "bytes": len(data)})
    result = {"repository": repository, "revision": commit, "license": profile["license"],
              "attribution": profile["attribution"],
              "license_path": str(PurePosixPath(*PurePosixPath(license_member.name).parts[1:])),
              "archive_sha256": sha256(archive_bytes), "files": records}
    if "files" in source and source["files"] != records:
        raise ValueError("Downloaded source hashes differ from the pinned manifest")
    cache.write_text(json.dumps(result, indent=2) + "\n")
    return result


def _split(family: str) -> str:
    # All revisions/pages of a board repository stay together.
    bucket = int(hashlib.sha256(family.encode()).hexdigest()[:8], 16) % 10
    return "test" if bucket < 2 else "val" if bucket == 2 else "train"


def conversation(record: dict, sheet: dict, image: str, *, image_only: bool = False) -> dict | None:
    """Six evidence-grounded turns; no invented electrical issue annotations."""
    bom = [part for part in sheet["bom"] if part["value"] and (not image_only or part.get("value_visible")) and
           re.fullmatch(r"[A-Z][A-Z0-9_$-]*", part["refdes"])]
    nets = [(name, pins) for name, pins in sheet["nets"].items() if 2 <= len(pins) <= 8
            and all(re.fullmatch(r"[A-Z][A-Z0-9_$-]*\.[A-Z0-9_$+-]+", pin) for pin in pins)]
    if len(bom) < 2 or not nets:
        return None
    bom.sort(key=lambda p: (not p["refdes"].startswith("R"), p["refdes"]))
    nets.sort(key=lambda pair: (pair[0].startswith("N$"), len(pair[1]), pair[0]))
    first, second = bom[:2]
    net_name, pins = nets[0]
    query, others = pins[0], pins[1:]
    missing = "R9999"
    while any(p["refdes"] == missing for p in sheet["bom"]):
        missing += "9"
    turns = [
        (f"What value is recorded for {first['refdes']}?",
         f"{first['refdes']} has value {first['value']} in the source schematic.",
         {"type": "value", "value": first["value"]}),
        (f"Which physical component pads connect to net {net_name} on this sheet?",
         f"On this sheet, {net_name} connects {', '.join(pins)}.", {"type": "pins", "pins": pins, "net": net_name}),
        (f"Trace {query}: which other physical pads share its net on this sheet?",
         f"{query} is on {net_name}, together with {', '.join(others)}.",
         {"type": "pins", "pins": others, "query": query, "net": net_name}),
        (f"What is the specified value of {second['refdes']}?",
         f"{second['refdes']} has value {second['value']} in the source schematic.",
         {"type": "value", "value": second["value"]}),
        (f"What value is {missing}, and what does it connect to?",
         f"There is no {missing} in the supplied sheet evidence. Its value and connections are not shown.",
         {"type": "refuse"}),
        (f"What measured voltage was observed at {query} during a powered bench test?",
         "No bench measurement is supplied in this schematic or source evidence. I cannot infer a measured voltage.",
         {"type": "refuse"}),
    ]
    evidence = {
        "scope": "this sheet only; physical package pad numbers, not symbol pin labels",
        "bom": [{"refdes": p["refdes"], "value": p["value"]} for p in sheet["bom"]],
        "nets": sheet["nets"],
    }
    if image_only:
        turns = [turns[index] for index in (0, 3, 4, 5)]
        turns[0] = (f"Read the value shown next to {first['refdes']}.",
                    f"{first['refdes']} is marked {first['value']}.", turns[0][2])
        turns[1] = (f"And what value is marked for {second['refdes']}?",
                    f"{second['refdes']} is marked {second['value']}.", turns[1][2])
    system = ("Answer only from the attached schematic image. Read the marked component values. "
              "If a part or measurement is not shown, say so. Do not invent facts.") if image_only else SYSTEM
    messages = [{"role": "system", "content": [{"type": "text", "text": system}]}]
    for index, (question, answer, _) in enumerate(turns):
        content = []
        if index == 0:
            content.append({"type": "image", "image": image})
            if not image_only:
                question = "Extracted native schematic evidence (not an electrical review):\n" + json.dumps(evidence) + "\n\n" + question
        content.append({"type": "text", "text": question})
        messages.extend([{"role": "user", "content": content},
                         {"role": "assistant", "content": [{"type": "text", "text": answer}]}])
    mode = "image_only_values_and_refusals" if image_only else "native_facts_supplied_in_prompt"
    return {"id": f"{record['id']}-p{sheet['page']}" + ("-vision" if image_only else ""), "design_id": record["id"],
            "family_id": record["repository"], "split": _split(record["repository"]),
            "messages": messages, "gold": [t[2] for t in turns],
            "provenance": {**record, "page": sheet["page"], "evidence_mode": mode}}


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_corpus(output: Path, *, limit: int = 120, manifest: Path | None = None,
                 workers: int = 6, render: bool = True, overwrite: bool = False,
                 publisher_name: str = "adafruit") -> dict:
    """Fetch licensed sources and build attributable facts plus source-geometry images."""
    if limit < 1 or not 1 <= workers <= 8:
        raise ValueError("limit must be positive and workers must be 1..8")
    if not overwrite and any((output / name).exists() for name in ("train.jsonl", "val.jsonl", "test.jsonl", "summary.json")):
        raise ValueError("Corpus outputs already exist; use --overwrite to rebuild them explicitly")
    output.mkdir(parents=True, exist_ok=True)
    source_dir = output / "sources"
    source_dir.mkdir(exist_ok=True)
    if publisher_name not in PUBLISHERS:
        raise ValueError("Unknown publisher profile")
    sources = json.loads(manifest.read_text())["sources"] if manifest else _discover(publisher_name=publisher_name)
    failures, downloaded = [], []

    def download(source):
        try:
            return _archive_source(source, source_dir), None
        except (OSError, ValueError, tarfile.TarError) as error:
            return None, {"repository": source.get("repository"), "reason": str(error)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for source, error in pool.map(download, sources):
            if error:
                failures.append(error)
            else:
                downloaded.append(source)
    records, rows, vision_rows, used, fingerprints = [], [], [], [], set()
    from .render_eagle import CROP_POLICY, RENDER_METHOD, render_eagle
    for source in downloaded:
        directory = source_dir / source["repository"].replace("/", "__")
        candidates = sorted((f for f in source["files"] if f["path"].lower().endswith(".sch")),
                            key=lambda f: f["path"], reverse=True)
        for item in candidates:
            if item["sha256"] in fingerprints:
                continue
            path = directory / item["path"]
            try:
                parsed = parse_eagle(path)
                if parsed["sha256"] != item["sha256"]:
                    raise ValueError("Cached native file changed since source manifest")
                identifier = "real-" + item["sha256"][:16]
                record = {"id": identifier, "repository": source["repository"],
                          "commit": source["revision"], "source_path": item["path"],
                          "source_url": "https://github.com/" + source["repository"] + "/blob/" + source["revision"] + "/" + urllib.parse.quote(item["path"]),
                          "sha256": item["sha256"], "license": source["license"],
                          "render_method": RENDER_METHOD, "crop_policy": CROP_POLICY, "source_kind": "real_hardware",
                          "image_kind": "derivative_render_of_original_source_geometry", "images": []}
                candidate_rows = []
                candidate_vision_rows = []
                for sheet in parsed["sheets"]:
                    image_path = output / "images" / f"{identifier}-p{sheet['page']}.png"
                    row = conversation(record, sheet, "images/" + image_path.name)
                    if row is None:
                        continue
                    if render:
                        render_eagle(path, sheet["page"], image_path,
                                     attribution=publisher(source["repository"])["footer"])
                        record["images"].append({"page": sheet["page"], "path": "images/" + image_path.name,
                                                 "sha256": sha256(image_path.read_bytes()),
                                                 "svg_sha256": sha256(image_path.with_suffix(".svg").read_bytes())})
                    candidate_rows.append(row)
                    vision = conversation(record, sheet, "../images/" + image_path.name, image_only=True)
                    if vision is not None:
                        candidate_vision_rows.append(vision)
                if not candidate_rows:
                    raise ValueError("No supported sheet with enough explicit factual evidence")
                (output / "evidence").mkdir(exist_ok=True)
                _atomic_text(output / "evidence" / f"{identifier}.json", json.dumps(parsed, indent=2) + "\n")
                records.append(record)
                rows.extend(candidate_rows)
                vision_rows.extend(candidate_vision_rows)
                fingerprints.add(item["sha256"])
                used.append(source)
                break  # one source design per board repository; revisions remain together
            except (OSError, ValueError) as error:
                failures.append({"repository": source["repository"], "path": item["path"], "reason": str(error)})
        if len(records) >= limit:
            break
    for split in ("train", "val", "test"):
        _atomic_text(output / f"{split}.jsonl", "".join(json.dumps(row) + "\n" for row in rows if row["split"] == split))
        _atomic_text(output / "vision" / f"{split}.jsonl", "".join(json.dumps(row) + "\n" for row in vision_rows if row["split"] == split))
    _atomic_text(output / "manifest.jsonl", "".join(json.dumps(record) + "\n" for record in records))
    pinned = {"version": 1, "sources": used}
    _atomic_text(output / "pinned-sources.json", json.dumps(pinned, indent=2) + "\n")
    summary = {"requested_designs": limit, "distinct_source_schematics": len(records),
               "distinct_board_repositories": len({r["repository"] for r in records}),
               "conversations": len(rows), "turns": sum(len(r["gold"]) for r in rows),
               "split_conversations": {split: sum(r["split"] == split for r in rows) for split in ("train", "val", "test")},
               "image_only_conversations": len(vision_rows),
               "image_only_split_conversations": {split: sum(r["split"] == split for r in vision_rows) for split in ("train", "val", "test")},
               "rendered": render, "status": "complete" if len(records) >= limit else "incomplete",
               "render_method": RENDER_METHOD, "crop_policy": CROP_POLICY,
               "claims": "Source-fact Q&A with native evidence supplied; not human-reviewed defect gold or image-only OCR evaluation",
               "excluded": failures}
    _atomic_text(output / "summary.json", json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/real"))
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--publisher", choices=sorted(PUBLISHERS), default="adafruit",
                        help="license profile used for discovery when no pinned manifest is given")
    args = parser.parse_args(argv)
    try:
        summary = build_corpus(args.out, limit=args.limit, manifest=args.manifest, workers=args.workers,
                               overwrite=args.overwrite, publisher_name=args.publisher)
    except (ValueError, OSError) as error:
        print(json.dumps({"status": "incomplete", "error": str(error)}))
        return 2
    print(json.dumps({k: v for k, v in summary.items() if k != "excluded"}, indent=2))
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
