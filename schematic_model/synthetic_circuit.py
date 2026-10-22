"""Generate bounded instructional EAGLE circuits and verify their native facts.

Synthetic two-pad packages are identifiers, not fabrication-ready footprints.
All variants of a circuit family must stay in the same dataset split.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from .corpus import parse_eagle


def validate_spec(spec: dict) -> None:
    if not isinstance(spec, dict) or set(spec) != {"family", "components", "nets"}:
        raise ValueError("Expected family, components, and nets only")
    if not isinstance(spec["family"], str) or not re.fullmatch(r"[a-z0-9-]{1,64}", spec["family"]):
        raise ValueError("Family must be a stable lowercase identifier shared by related variants")
    parts, nets = spec["components"], spec["nets"]
    if not isinstance(parts, list) or not 1 <= len(parts) <= 12:
        raise ValueError("Provide 1 to 12 components")
    names = set()
    for part in parts:
        if not isinstance(part, dict) or set(part) != {"refdes", "kind", "value"}:
            raise ValueError("Each component requires refdes, kind, and value only")
        name = part["refdes"]
        if (not isinstance(name, str) or not re.fullmatch(r"[RC][1-9][0-9]{0,2}", name)
                or name in names or not isinstance(part["kind"], str)
                or part["kind"] not in {"R", "C"} or not name.startswith(part["kind"])):
            raise ValueError("Unique matching R/C reference and kind required")
        if (not isinstance(part["value"], str) or not part["value"].strip()
                or not re.fullmatch(r"[A-Za-z0-9.µΩ +/%-]{1,24}", part["value"])):
            raise ValueError("Value must be a short printable component label")
        names.add(name)
    if not isinstance(nets, dict) or not 1 <= len(nets) <= 24:
        raise ValueError("Provide 1 to 24 named nets")
    assigned = set()
    allowed = {f"{name}.{pad}" for name in names for pad in (1, 2)}
    for name, pads in nets.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Z][A-Z0-9_+-]{0,15}", name):
            raise ValueError("Invalid net name")
        if not isinstance(pads, list) or not pads:
            raise ValueError("Every net requires endpoints")
        for pad in pads:
            if not isinstance(pad, str) or pad not in allowed or pad in assigned:
                raise ValueError("Every endpoint must exist and belong to exactly one net")
            assigned.add(pad)
    if assigned != allowed:
        raise ValueError("Assign every component pad explicitly, including isolated pads")


def generate(spec: dict) -> tuple[bytes, dict]:
    """Return XML and checked facts without writing or changing existing datasets."""
    validate_spec(spec)
    root = ET.Element("eagle", version="9.6.2")
    drawing = ET.SubElement(root, "drawing")
    layers = ET.SubElement(drawing, "layers")
    for number, name in ((1, "Top"), (91, "Nets"), (94, "Symbols"), (95, "Names"), (96, "Values")):
        ET.SubElement(layers, "layer", number=str(number), name=name, color="1", fill="1", visible="yes", active="yes")
    schematic = ET.SubElement(drawing, "schematic")
    library = ET.SubElement(ET.SubElement(schematic, "libraries"), "library", name="instructional")
    package = ET.SubElement(ET.SubElement(library, "packages"), "package", name="SYNTHETIC_TWO_PAD")
    ET.SubElement(package, "description").text = "Instructional pad identifiers; not a manufacturing footprint."
    for pad, pad_x in (("1", "-1"), ("2", "1")):
        ET.SubElement(package, "smd", name=pad, x=pad_x, y="0", dx="1", dy="1", layer="1")
    symbols = ET.SubElement(library, "symbols")
    devices = ET.SubElement(library, "devicesets")
    for kind in ("R", "C"):
        symbol = ET.SubElement(symbols, "symbol", name=kind)
        for pin, pin_x, rot in (("1", "-5.08", "R0"), ("2", "5.08", "R180")):
            ET.SubElement(symbol, "pin", name=pin, x=pin_x, y="0", length="short", direction="pas", rot=rot, visible="off")
        lines = ([(-2.54, -1.27, 2.54, -1.27), (-2.54, 1.27, 2.54, 1.27),
                  (-2.54, -1.27, -2.54, 1.27), (2.54, -1.27, 2.54, 1.27)] if kind == "R" else
                 [(-2.54, 0, -0.635, 0), (0.635, 0, 2.54, 0),
                  (-0.635, -1.9, -0.635, 1.9), (0.635, -1.9, 0.635, 1.9)])
        for x1, y1, x2, y2 in lines:
            ET.SubElement(symbol, "wire", x1=str(x1), y1=str(y1), x2=str(x2), y2=str(y2), width="0.254", layer="94")
        for text, text_y, layer in ((">NAME", "3", "95"), (">VALUE", "-4", "96")):
            ET.SubElement(symbol, "text", x="-2.54", y=text_y, size="1.27", layer=layer).text = text
        # cspell:ignore uservalue
        device_set = ET.SubElement(devices, "deviceset", name=kind, prefix=kind, uservalue="yes")
        ET.SubElement(ET.SubElement(device_set, "gates"), "gate", name="G$1", symbol=kind, x="0", y="0")
        device = ET.SubElement(ET.SubElement(device_set, "devices"), "device", name="", package="SYNTHETIC_TWO_PAD")
        connects = ET.SubElement(device, "connects")
        for pin in ("1", "2"):
            ET.SubElement(connects, "connect", gate="G$1", pin=pin, pad=pin)
        ET.SubElement(ET.SubElement(device, "technologies"), "technology", name="")
    parts = ET.SubElement(schematic, "parts")
    sheet = ET.SubElement(ET.SubElement(schematic, "sheets"), "sheet")
    ET.SubElement(sheet, "plain")
    instances = ET.SubElement(sheet, "instances")
    positions = {}
    for index, part in enumerate(sorted(spec["components"], key=lambda p: p["refdes"])):
        name = part["refdes"]
        x, y = (index % 3) * 65, -(index // 3) * 22
        positions[name] = x, y
        ET.SubElement(parts, "part", name=name, library="instructional", deviceset=part["kind"], device="", value=part["value"])
        ET.SubElement(instances, "instance", part=name, gate="G$1", x=str(x), y=str(y))
    nets = ET.SubElement(sheet, "nets")
    for name, endpoints in sorted(spec["nets"].items()):
        net = ET.SubElement(nets, "net", {"name": name, "class": "0"})
        for endpoint in sorted(endpoints):
            ref, pin = endpoint.split(".")
            x, y = positions[ref]
            sign = -1 if pin == "1" else 1
            segment = ET.SubElement(net, "segment")
            ET.SubElement(segment, "pinref", part=ref, gate="G$1", pin=pin)
            ET.SubElement(segment, "wire", x1=str(x + sign * 5.08), y1=str(y), x2=str(x + sign * 10.16), y2=str(y), width="0.1524", layer="91")
            ET.SubElement(segment, "label", x=str(x + sign * 10.16), y=str(y), size="1.27", layer="95", rot="R180" if sign < 0 else "R0")
    ET.indent(root)
    content = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    extracted = parse_eagle(content)["sheets"][0]
    expected_values = {p["refdes"]: p["value"] for p in spec["components"]}
    if {p["refdes"]: p["value"] for p in extracted["bom"]} != expected_values or extracted["nets"] != {k: sorted(v) for k, v in spec["nets"].items()}:
        raise ValueError("Generated circuit failed native round-trip validation")
    report = {"schema_version": 1, "family": "synthetic:" + spec["family"],
              "source_sha256": hashlib.sha256(content).hexdigest(),
              "status": "synthetic_native_round_trip_verified", "facts": extracted,
              "limitations": "Instructional circuit only; no electrical, manufacturing, or model-quality qualification."}
    return content, report


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="new .sch output; refuses overwrite")
    args = parser.parse_args(argv)
    try:
        if args.out.suffix != ".sch":
            raise ValueError("Output must end in .sch")
        content, _ = generate(json.loads(args.spec.read_text(), object_pairs_hook=_unique_object))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("xb") as stream:
            stream.write(content)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
