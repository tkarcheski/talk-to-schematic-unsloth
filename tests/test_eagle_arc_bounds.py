"""Curved source geometry must remain inside circuit crops and viewer targets."""

import math
import xml.etree.ElementTree as ET

import pytest

from schematic_model.render_eagle import _Drawing, IDENTITY, source_layout


def wire(curve):
    return ET.Element("wire", x1="0", y1="0", x2="20", y2="0", width="0.15", layer="91", curve=str(curve))


@pytest.mark.parametrize("curve,expected", [
    (180, (0, 0, 20, 10)),
    (-180, (0, -10, 20, 0)),
    (270, (10 - math.sqrt(200), 0, 10 + math.sqrt(200), 10 + math.sqrt(200))),
    (-270, (10 - math.sqrt(200), -10 - math.sqrt(200), 10 + math.sqrt(200), 0)),
])
def test_semicircle_and_major_arc_bounds(curve, expected):
    drawing = _Drawing()
    drawing.wire(wire(curve), IDENTITY, "black")
    xs, ys = zip(*drawing.points)
    assert (min(xs), min(ys), max(xs), max(ys)) == pytest.approx(expected)


@pytest.mark.parametrize("rotation", [0, 37, 90, 230])
@pytest.mark.parametrize("mirrored", [False, True])
def test_transformed_arc_bounds_contain_independent_semicircle_samples(rotation, mirrored):
    drawing = _Drawing()
    drawing.wire(wire(180), (3, 7, rotation, mirrored), "black")
    xs, ys = zip(*drawing.points)
    angle = math.radians(rotation)
    for step in range(101):
        theta = math.pi * step / 100
        x, y = 10 - 10 * math.cos(theta), -10 * math.sin(theta)
        x *= -1 if mirrored else 1
        actual_x = 3 + x * math.cos(angle) - y * math.sin(angle)
        actual_y = -(7 + x * math.sin(angle) + y * math.cos(angle))
        assert min(xs) - 1e-9 <= actual_x <= max(xs) + 1e-9
        assert min(ys) - 1e-9 <= actual_y <= max(ys) + 1e-9


def test_curved_polygon_includes_the_same_visible_extrema():
    drawing = _Drawing()
    drawing.primitive(ET.fromstring('<polygon layer="94"><vertex x="0" y="0" curve="180"/>'
                                   '<vertex x="20" y="0"/><vertex x="10" y="5"/></polygon>'))
    assert max(y for _, y in drawing.points) == pytest.approx(10)


def test_net_arc_is_inside_crop_and_navigation_box(tmp_path):
    source = tmp_path / "arc.sch"
    source.write_text('<?xml version="1.0"?><eagle version="9.6.2"><drawing><schematic><libraries/><parts/><sheets>'
                      '<sheet><plain/><instances/><nets><net name="CURVED"><segment>'
                      + ET.tostring(wire(180), encoding="unicode")
                      + '</segment></net></nets></sheet></sheets></schematic></drawing></eagle>')
    svg, regions = source_layout(source, 1)
    document = ET.fromstring(svg)
    xmin, ymin, width, height = map(float, document.attrib["viewBox"].split())
    clip = document.find('.//{http://www.w3.org/2000/svg}clipPath/{http://www.w3.org/2000/svg}rect')
    assert float(clip.attrib["y"]) + float(clip.attrib["height"]) >= 10
    left, top, right, bottom = regions["nets"]["CURVED"]
    assert left <= (10 - xmin) / width * 1000 <= right
    assert top <= (10 - ymin) / height * 1000 <= bottom


@pytest.mark.parametrize("curve", [360, -360, float("inf"), float("nan")])
def test_unrepresentable_arc_fails_explicitly(curve):
    with pytest.raises(ValueError, match="degenerate"):
        _Drawing().wire(wire(curve), IDENTITY, "black")
