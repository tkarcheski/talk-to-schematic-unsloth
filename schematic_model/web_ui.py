"""Static browser UI and an allowlisted catalog of attributable real schematics.

The catalog reads source manifests and native files, never conversation gold or
stored assistant answers. No model is imported or started by this module.
"""

from __future__ import annotations

import hashlib
from importlib.resources import files
import json
from pathlib import Path
import re
from urllib.parse import quote
import xml.etree.ElementTree as ET

from .corpus import PUBLISHERS, parse_eagle, publisher


ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/favicon.svg": ("favicon.svg", "image/svg+xml"),
    "/assets/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/assets/style.css": ("style.css", "text/css; charset=utf-8"),
    "/assets/vendor/pdf.min.mjs": ("vendor/pdf.min.mjs", "text/javascript; charset=utf-8"),
    "/assets/vendor/pdf.worker.min.mjs": ("vendor/pdf.worker.min.mjs", "text/javascript; charset=utf-8"),
}


def static_asset(route: str) -> tuple[bytes, str] | None:
    asset = ASSETS.get(route)
    if asset is None:
        return None
    name, content_type = asset
    return files("schematic_model").joinpath("web", *name.split("/")).read_bytes(), content_type


def _inside(directory: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("Example file reference must be a nonempty relative path")
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise ValueError("Example file is missing or outside the selected corpus")
    return path


class ExampleCatalog:
    """Load and verify a finite manifest; request paths never become file paths."""

    def __init__(self, directories: str | Path | list[str | Path] | tuple[str | Path, ...]):
        self.entries = {}
        self.images = {}
        self.sources = {}
        self._regions = {}
        if isinstance(directories, (str, Path)):
            directories = [directories]
        if not directories:
            raise ValueError("At least one example corpus directory is required")
        for directory in directories:
            self._load(Path(directory).resolve())
        if not self.entries:
            raise ValueError("Example catalog contains no rendered schematic pages")

    def _load(self, directory: Path) -> None:
        manifest = _inside(directory, "manifest.jsonl")
        if manifest.stat().st_size > 10 * 1024 * 1024:
            raise ValueError("Example manifest exceeds the 10 MiB limit")
        for line in manifest.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("Example records must be objects")
            if any(not isinstance(record.get(key), str) for key in ("repository", "commit", "id", "source_path")):
                raise ValueError("Example source identifiers must be strings")
            repository, commit = record.get("repository", ""), record.get("commit", "")
            try:
                profile = publisher(repository)
            except ValueError as exc:
                raise ValueError("Examples require pinned, attributed Adafruit or SparkFun source records") from exc
            if (not re.fullmatch(r"[0-9a-f]{40}", commit)
                    or not re.fullmatch(r"real-[0-9a-f]{16}", record.get("id", ""))
                    or record.get("license") != profile["license"]):
                raise ValueError("Examples require pinned, attributed Adafruit or SparkFun source records")
            source = _inside(directory, "sources/" + repository.replace("/", "__") + "/" + record["source_path"])
            try:
                evidence = parse_eagle(source)
            except (ET.ParseError, KeyError, TypeError) as exc:
                raise ValueError("Example native schematic could not be parsed") from exc
            if evidence["sha256"] != record.get("sha256"):
                raise ValueError("Example native source hash differs from its manifest")
            image_records = record.get("images", [])
            if not isinstance(image_records, list):
                raise ValueError("Example images must be a list")
            for image in image_records:
                if not isinstance(image, dict):
                    raise ValueError("Example image records must be objects")
                page = image.get("page")
                if type(page) is not int or not 1 <= page <= len(evidence["sheets"]):
                    raise ValueError("Example page is outside the source schematic")
                identifier = f"{record['id']}-p{page}"
                if identifier in self.entries:
                    raise ValueError("Duplicate example identifier")
                path = _inside(directory, image["path"])
                if path.suffix.lower() != ".png" or path.stat().st_size > 16 * 1024 * 1024:
                    raise ValueError("Example images must be PNG files under 16 MiB")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != image.get("sha256"):
                    raise ValueError("Example image hash differs from its manifest")
                sheet = evidence["sheets"][page - 1]
                visible_parts = [p for p in sheet["bom"] if p.get("value_visible") and p.get("value")]
                visible_parts.sort(key=lambda p: (not p["refdes"].startswith("R"), p["refdes"]))
                prompts = [f"What value is shown for {p['refdes']}?" for p in visible_parts[:2]]
                prompts.append("What information would require a bench measurement?")
                title = re.sub(r"[-_]+", " ", repository.split("/", 1)[1])
                title = title.removeprefix("Adafruit ").removeprefix("SparkFun ").removesuffix(" PCB")
                self.entries[identifier] = {
                    "id": identifier,
                    "title": title,
                    "publisher": profile["name"],
                    "page": page, "repository": repository, "revision": commit,
                    "source_url": f"https://github.com/{repository}/blob/{commit}/{quote(record['source_path'])}",
                    "license": record["license"], "attribution": profile["name"],
                    "source_sha256": evidence["sha256"], "image_sha256": digest,
                    "image_url": f"/api/examples/{identifier}/image",
                    "image_kind": "Derivative render of original source geometry",
                    "render_method": record.get("render_method", ""), "crop_policy": record.get("crop_policy", ""),
                    "suggested_questions": prompts,
                    "source_evidence": {"scope": "this sheet only; physical package pad numbers, not symbol pin labels",
                                        "bom": [{"refdes": p["refdes"], "value": p["value"]} for p in sheet["bom"]],
                                        "nets": sheet["nets"]},
                }
                self.images[identifier] = (path, digest)
                self.sources[identifier] = (source, page, image.get("svg_sha256"))
                if len(self.entries) > 1000:
                    raise ValueError("Example catalog exceeds the 1000-page limit")

    def index(self) -> list[dict]:
        order = {profile["name"]: rank for rank, profile in enumerate(reversed(list(PUBLISHERS.values())))}
        return [{key: entry[key] for key in ("id", "title", "page", "publisher")}
                for entry in sorted(self.entries.values(),
                                    key=lambda entry: (order[entry["publisher"]], entry["title"], entry["id"]))]

    def regions(self, identifier: str) -> dict | None:
        """Source-geometry part and net boxes, only when the image's SVG is reproduced exactly.

        A corpus rendered by an older renderer keeps its image, but its boxes
        are withheld rather than drawn against geometry that may not match.
        """
        if identifier not in self.sources:
            return None
        if identifier not in self._regions:
            from .render_eagle import source_layout

            source, page, svg_digest = self.sources[identifier]
            regions = None
            if isinstance(svg_digest, str):
                svg, layout = source_layout(source, page, attribution=publisher(
                    self.entries[identifier]["repository"])["footer"])
                if hashlib.sha256(svg.encode()).hexdigest() == svg_digest:
                    regions = layout
            self._regions[identifier] = regions
        return self._regions[identifier]

    def details(self, identifier: str) -> dict | None:
        entry = self.entries.get(identifier)
        if entry is None:
            return None
        regions = self.regions(identifier)
        prompts = list(entry["suggested_questions"])
        if regions:
            shown = [p for p in re.findall(r"for ([A-Z][A-Z0-9_]*)\?", " ".join(prompts)) if p in regions["parts"]]
            if shown:
                prompts.insert(0, f"Show me {shown[0]}.")
            prompts.append("Zoom back out to the whole sheet.")
        return {**entry, "regions": regions, "suggested_questions": prompts}

    def image(self, identifier: str) -> bytes | None:
        if identifier not in self.images:
            return None
        path, expected = self.images[identifier]
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("Example image changed after catalog validation; restart with verified data")
        return content
