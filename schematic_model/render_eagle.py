"""Render original EAGLE XML primitives; never reconstruct inferred connections.

This intentionally bounded renderer rejects unknown primitives. It retains source
wire positions, symbols, instance transforms, pin labels and component values.
It is a derivative renderer, not Autodesk's rendering engine.
"""
from __future__ import annotations

import html
import math
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

from .eagle_source import load_eagle

Transform = tuple[float, float, float, bool]
IDENTITY: Transform = (0, 0, 0, False)
RENDER_METHOD = "eagle_xml_geometry_v3_text_alignment"
CROP_POLICY = "electrical_symbols_and_source_nets_with_text_bounds; frames_and_fiducials_do_not_expand_view"


def _rotation(value: str) -> tuple[float, bool]:
    match = re.fullmatch(r"S?(M?)R(-?\d+(?:\.\d+)?)", value)
    if not match:
        raise ValueError(f"Unsupported EAGLE rotation {value!r}")
    return float(match[2]), bool(match[1])


def _point(x: float, y: float, transform: Transform) -> tuple[float, float]:
    tx, ty, degrees, mirror = transform
    if mirror:
        x = -x
    angle = math.radians(degrees)
    return (tx + x * math.cos(angle) - y * math.sin(angle),
            ty + x * math.sin(angle) + y * math.cos(angle))


class _Drawing:
    def __init__(self):
        self.elements: list[str] = []
        self.points: list[tuple[float, float]] = []

    def point(self, x, y, transform=IDENTITY):
        x, y = _point(float(x), float(y), transform)
        self.points.append((x, -y))
        return x, -y

    def text(self, value, x, y, size=1.778, transform=IDENTITY, rotation="R0", align="bottom-left", color="#111827"):
        if not value:
            return
        xx, yy = self.point(x, y, transform)
        local_angle, local_mirror = _rotation(rotation)
        angle = transform[2] + (-local_angle if transform[3] else local_angle)
        # EAGLE's first alignment axis is vertical, its second horizontal.
        # In particular center-left and center-right are not centered on x.
        alignments = {
            "bottom-left": ("start", "bottom"), "bottom-center": ("middle", "bottom"),
            "bottom-right": ("end", "bottom"), "center-left": ("start", "center"),
            "center": ("middle", "center"), "center-right": ("end", "center"),
            "top-left": ("start", "top"), "top-center": ("middle", "top"),
            "top-right": ("end", "top"),
        }
        if align not in alignments:
            raise ValueError(f"Unsupported EAGLE text alignment {align!r}")
        anchor, vertical = alignments[align]
        mirrored = transform[3] != local_mirror
        if mirrored:
            anchor = {"start": "end", "end": "start", "middle": "middle"}[anchor]
        angle %= 360
        if 90 < angle <= 270:
            angle = (angle + 180) % 360
            anchor = {"start": "end", "end": "start", "middle": "middle"}[anchor]
            vertical = {"bottom": "top", "top": "bottom", "center": "center"}[vertical]
        size = float(size)
        lines = str(value).splitlines() or [""]
        # Conservative glyph bounds are rotated with the rendered text. Ignoring
        # text rotation clips long vertical values in tightly cropped drawings.
        extent = max(map(len, lines)) * size
        left = -extent if anchor == "end" else -extent / 2 if anchor == "middle" else 0
        line_span = (len(lines) - 1) * size * 1.25
        first_line_offset = {"bottom": -line_span, "center": -line_span / 2, "top": 0}[vertical]
        baseline = {"bottom": "text-after-edge", "center": "central", "top": "text-before-edge"}[vertical]
        # Conservative crop bounds, not a claim about measured glyph extents.
        top, bottom = {
            "bottom": (-line_span - size * 1.3, size * .1),
            "center": (-line_span / 2 - size * .7, line_span / 2 + size * .7),
            "top": (-size * .1, line_span + size * 1.3),
        }[vertical]
        radians = math.radians(-angle)
        for dx, dy in ((left, top), (left + extent, top), (left, bottom), (left + extent, bottom)):
            self.points.append((xx + dx * math.cos(radians) - dy * math.sin(radians),
                                yy + dx * math.sin(radians) + dy * math.cos(radians)))
        spans = "".join(f'<tspan x="{xx:.4f}" dy="{first_line_offset if i == 0 else size * 1.25:.4f}">{html.escape(line)}</tspan>'
                        for i, line in enumerate(lines))
        self.elements.append(f'<text x="{xx:.4f}" y="{yy:.4f}" font-size="{size:.4f}" '
                             f'text-anchor="{anchor}" dominant-baseline="{baseline}" fill="{color}" stroke="none" '
                             f'transform="rotate({-angle:.4f} {xx:.4f} {yy:.4f})">{spans}</text>')

    def wire(self, node, transform, color):
        x1, y1 = self.point(node.get("x1"), node.get("y1"), transform)
        x2, y2 = self.point(node.get("x2"), node.get("y2"), transform)
        width = max(float(node.get("width", "0.1524")), 0.08)
        curve = float(node.get("curve", "0"))
        if abs(curve) > 0.001:
            radius = math.hypot(x2 - x1, y2 - y1) / (2 * abs(math.sin(math.radians(curve / 2))))
            sweep = int((curve < 0) != transform[3])
            path = f'M{x1},{y1} A{radius},{radius} 0 {int(abs(curve) > 180)},{sweep} {x2},{y2}'
            self.elements.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{width}"/>')
        else:
            self.elements.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"/>')

    def primitive(self, node, transform=IDENTITY, substitutions=None, smashed=False, pads=None):
        tag = node.tag
        layer = int(node.get("layer", "94"))
        if layer not in {91, 92, 93, 94, 95, 96, 97, 98}:
            return
        color = "#126333" if layer == 91 else "#1e40af" if layer == 92 else "#382b38"
        if tag == "wire":
            self.wire(node, transform, color)
        elif tag in {"rectangle", "frame"}:
            # Frame coordinates are retained; row/column decorations are omitted.
            x1, y1, x2, y2 = (float(node.get(k)) for k in ("x1", "y1", "x2", "y2"))
            points = [self.point(x, y, transform) for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))]
            text = " ".join(f"{x},{y}" for x, y in points)
            self.elements.append(f'<polygon points="{text}" fill="{color if tag == "rectangle" else "none"}" stroke="{color}" stroke-width="0.15"/>')
        elif tag in {"circle", "junction"}:
            x, y = self.point(node.get("x"), node.get("y"), transform)
            radius = float(node.get("radius", "0.45"))
            self.points.extend([(x - radius, y - radius), (x + radius, y + radius)])
            self.elements.append(f'<circle cx="{x}" cy="{y}" r="{radius}" stroke="{color}" '
                                 f'stroke-width="{max(float(node.get("width", ".15")), .08)}" '
                                 f'fill="{color if tag == "junction" else "none"}"/>')
        elif tag == "polygon":
            vertices = list(node.findall("vertex"))
            if not vertices:
                return
            transformed = [self.point(v.get("x"), v.get("y"), transform) for v in vertices]
            pieces = [f"M{transformed[0][0]},{transformed[0][1]}"]
            for index, start in enumerate(vertices):
                end = transformed[(index + 1) % len(vertices)]
                curve = float(start.get("curve", "0"))
                if curve:
                    begin = transformed[index]
                    radius = math.dist(begin, end) / (2 * abs(math.sin(math.radians(curve / 2))))
                    pieces.append(f'A{radius},{radius} 0 {int(abs(curve)>180)},{int((curve<0)!=transform[3])} {end[0]},{end[1]}')
                else:
                    pieces.append(f'L{end[0]},{end[1]}')
            self.elements.append(f'<path d="{" ".join(pieces)} Z" stroke="{color}" fill="{color}" stroke-width=".1"/>')
        elif tag in {"text", "label", "attribute"}:
            value = node.text or ""
            if tag == "label":
                value = (substitutions or {}).get("NET", "")
            if tag == "attribute":
                value = node.get("value", (substitutions or {}).get(node.get("name"), ""))
                display = node.get("display", "value")
                if display == "off":
                    return
                if display == "name":
                    value = node.get("name", "")
                elif display == "both":
                    value = node.get("name", "") + " = " + value
            if value.startswith(">"):
                if smashed and value in {">NAME", ">VALUE"}:
                    return
                value = (substitutions or {}).get(value[1:], "")
            self.text(value, node.get("x", 0), node.get("y", 0), node.get("size", 1.778),
                      transform, node.get("rot", "R0"), node.get("align", "bottom-left"))
        elif tag == "pin":
            x, y = float(node.get("x")), float(node.get("y"))
            angle, _ = _rotation(node.get("rot", "R0"))
            angle = math.radians(angle)
            length = {"point": 0, "short": 2.54, "middle": 5.08, "long": 7.62}[node.get("length", "long")]
            endx, endy = x + length * math.cos(angle), y + length * math.sin(angle)
            wire = ET.Element("wire", {"x1": str(x), "y1": str(y), "x2": str(endx), "y2": str(endy), "width": ".1524"})
            self.wire(wire, transform, "#8b1b1b")
            visible = node.get("visible", "both")
            name = node.get("name", "")
            if visible in {"both", "pin"}:
                self.text(name.split("@")[0], endx + .6 * math.cos(angle), endy + .6 * math.sin(angle),
                          1.15, transform, node.get("rot", "R0"), color="#4b153a")
            if visible in {"both", "pad"} and pads and name in pads:
                self.text(pads[name], (x + endx) / 2, (y + endy) / 2 + .45, .95,
                          transform, node.get("rot", "R0"), color="#374151")
            function = node.get("function", "none")
            if function in {"dot", "dotclk"}:
                bubble = ET.Element("circle", {"x": str(endx), "y": str(endy), "radius": ".5", "width": ".15"})
                self.primitive(bubble, transform)
            if function in {"clk", "dotclk"}:
                # Retain clock indicator from source pin function.
                for sign in (-1, 1):
                    wire = ET.Element("wire", {"x1": str(endx - sign * .6 * math.sin(angle)),
                        "y1": str(endy + sign * .6 * math.cos(angle)),
                        "x2": str(endx + .9 * math.cos(angle)), "y2": str(endy + .9 * math.sin(angle)), "width": ".15"})
                    self.wire(wire, transform, color)
        elif tag in {"description", "pinref", "portref", "probe"}:
            return
        else:
            raise ValueError(f"Unsupported visible EAGLE primitive: {tag}")


def source_svg(path: Path, page: int, *, attribution: str = "") -> str:
    _, schematic = load_eagle(path)
    sheets = schematic.findall("./sheets/sheet")
    if not 1 <= page <= len(sheets):
        raise ValueError("Requested sheet does not exist")
    sheet = sheets[page - 1]
    drawing = _Drawing()
    circuit_points = []
    libraries = {x.get("name"): x for x in schematic.findall("./libraries/library")}
    parts = {x.get("name"): x for x in schematic.findall("./parts/part")}
    for node in sheet.findall("./plain/*"):
        drawing.primitive(node)
    for instance in sheet.findall("./instances/instance"):
        bounds_start = len(drawing.points)
        part = parts[instance.attrib["part"]]
        library = libraries[part.get("library")]
        device_set = next(x for x in library.findall("./devicesets/deviceset") if x.get("name") == part.get("deviceset"))
        gate = next(x for x in device_set.findall("./gates/gate") if x.get("name") == instance.get("gate"))
        symbol = next(x for x in library.findall("./symbols/symbol") if x.get("name") == gate.get("symbol"))
        device = next(x for x in device_set.findall("./devices/device") if x.get("name", "") == part.get("device", ""))
        pads = {x.get("pin"): x.get("pad") for x in device.findall("./connects/connect") if x.get("gate") == gate.get("name")}
        angle, mirrored = _rotation(instance.get("rot", "R0"))
        transform = (float(instance.get("x")), float(instance.get("y")), angle, mirrored)
        substitutions = {"NAME": part.get("name"), "VALUE": part.get("value") or part.get("deviceset"),
                         "DRAWING_NAME": path.stem, "SHEET": str(page)}
        for primitive in symbol:
            drawing.primitive(primitive, transform, substitutions, instance.get("smashed") == "yes", pads)
        for attribute in instance.findall("attribute"):
            drawing.primitive(attribute, substitutions=substitutions)
        # An electrical symbol has pins, including disconnected components and
        # power symbols. Page frames, logos, and fiducials have none. They remain
        # in the source geometry but cannot force the circuit into a tiny view.
        if symbol.find("pin") is not None:
            circuit_points.extend(drawing.points[bounds_start:])
    bounds_start = len(drawing.points)
    for container in ("nets/net", "busses/bus"):
        for net in sheet.findall("./" + container):
            for primitive in net.findall("./segment/*"):
                drawing.primitive(primitive, substitutions={"NET": net.get("name")})
    circuit_points.extend(drawing.points[bounds_start:])
    if not circuit_points:
        raise ValueError("No electrical source geometry available for circuit crop")
    xmin, xmax = min(x for x, _ in circuit_points) - 4, max(x for x, _ in circuit_points) + 4
    ymin, body_bottom = min(y for _, y in circuit_points) - 4, max(y for _, y in circuit_points) + 4
    width = xmax - xmin
    footer_lines = [attribution, f"Source geometry render | circuit crop | {path.name} | sheet {page}"]
    footer_lines = [line for line in footer_lines if line]
    font_size = min(1.8, (width - 6) / max(len(line) for line in footer_lines))
    footer_height = max(5, font_size * (len(footer_lines) * 1.5 + 2))
    height = body_bottom - ymin + footer_height
    footer = "".join(f'<text x="{xmin + 3}" y="{body_bottom + font_size * (2 + index * 1.5)}" '
                     f'font-size="{font_size}" fill="#4b5563">{html.escape(line)}</text>'
                     for index, line in enumerate(footer_lines))
    metadata = html.escape(f"{RENDER_METHOD}; {CROP_POLICY}; original source retained separately")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="{max(1, round(2400 * height / width))}" '
            f'viewBox="{xmin} {ymin} {width} {height}"><metadata>{metadata}</metadata>'
            f'<defs><clipPath id="circuit"><rect x="{xmin}" y="{ymin}" width="{width}" height="{body_bottom - ymin}"/></clipPath></defs>'
            f'<rect x="{xmin}" y="{ymin}" width="{width}" height="{height}" fill="white"/>'
            '<g font-family="DejaVu Sans, sans-serif" stroke-linecap="round" stroke-linejoin="round">'
            + '<g clip-path="url(#circuit)">' + "".join(drawing.elements) + '</g>' + footer + '</g></svg>')


def render_eagle(path: Path, page: int, output: Path, *, attribution: str = "") -> None:
    renderer = shutil.which("rsvg-convert")
    if not renderer:
        raise ValueError("rsvg-convert is required to render EAGLE SVGs to PNG")
    svg = source_svg(path, page, attribution=attribution)
    output.parent.mkdir(parents=True, exist_ok=True)
    svg_path = output.with_suffix(".svg")
    svg_path.write_text(svg)
    try:
        subprocess.run([renderer, "--output", str(output), str(svg_path)], check=True,
                       timeout=30, capture_output=True)
    except (subprocess.SubprocessError, OSError) as error:
        raise ValueError(f"Source geometry rasterization failed: {error}") from error
