import copy
import json

import pytest

from schematic_model.corpus import parse_eagle
from schematic_model.render_eagle import source_layout
from schematic_model.synthetic_circuit import generate, main


@pytest.fixture
def circuit():
    return {"family": "rc-filter", "components": [
        {"refdes": "R1", "kind": "R", "value": "10k"},
        {"refdes": "C1", "kind": "C", "value": "100nF"}],
        "nets": {"IN": ["R1.1"], "OUT": ["R1.2", "C1.1"], "GND": ["C1.2"]}}


def test_round_trip_and_render(circuit, tmp_path):
    content, report = generate(circuit)
    assert report["family"] == "synthetic:rc-filter"
    assert report["source_sha256"] == parse_eagle(content)["sha256"]
    source = tmp_path / "circuit.sch"
    source.write_bytes(content)
    svg, regions = source_layout(source, 1)
    assert set(regions["parts"]) == {"R1", "C1"}
    assert set(regions["nets"]) == {"IN", "OUT", "GND"}
    assert "100nF" in svg and "10k" in svg
    assert all(part["value_visible"] for part in report["facts"]["bom"])


def test_value_counterfactual_changes_only_value(circuit):
    _, before = generate(circuit)
    changed = copy.deepcopy(circuit)
    changed["components"][0]["value"] = "22k"
    _, after = generate(changed)
    assert before["family"] == after["family"]
    assert before["source_sha256"] != after["source_sha256"]
    assert before["facts"]["nets"] == after["facts"]["nets"]
    assert {p["refdes"]: p["value"] for p in after["facts"]["bom"]}["R1"] == "22k"


def test_connection_counterfactual_keeps_values(circuit):
    _, before = generate(circuit)
    circuit["nets"]["OUT"].remove("C1.1")
    circuit["nets"]["IN"].append("C1.1")
    _, after = generate(circuit)
    assert before["facts"]["bom"] == after["facts"]["bom"]
    assert after["facts"]["nets"]["IN"] == ["C1.1", "R1.1"]


@pytest.mark.parametrize("mutation", [
    lambda s: s["nets"]["OUT"].append("C1.2"),
    lambda s: s["nets"]["OUT"].append("U1.1"),
    lambda s: s["nets"].pop("GND"),
    lambda s: s["components"].append(s["components"][0]),
    lambda s: s["components"][0].update(kind="IC"),
    lambda s: s["components"][0].update(kind=[]),
    lambda s: s["components"][0].update(value="   "),
    lambda s: s["components"][0].update(value="<script>"),
    lambda s: s.update(family="../escape"),
    lambda s: s.update(extra="unreviewed"),
])
def test_invalid_spec_fails_closed(circuit, mutation):
    mutation(circuit)
    with pytest.raises(ValueError):
        generate(circuit)


def test_cli_preserves_existing_artifact(circuit, tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(circuit))
    output = tmp_path / "output.sch"
    assert main([str(spec), "--out", str(output)]) == 0
    first = output.read_bytes()
    with pytest.raises(SystemExit):
        main([str(spec), "--out", str(output)])
    assert output.read_bytes() == first


def test_cli_does_not_follow_dangling_symlink(circuit, tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(circuit))
    output = tmp_path / "output.sch"
    target = tmp_path / "missing.sch"
    output.symlink_to(target)
    with pytest.raises(SystemExit):
        main([str(spec), "--out", str(output)])
    assert not target.exists()


def test_duplicate_json_key_rejected(tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text('{"family":"one","family":"two"}')
    output = tmp_path / "output.sch"
    with pytest.raises(SystemExit):
        main([str(spec), "--out", str(output)])
    assert not output.exists()
