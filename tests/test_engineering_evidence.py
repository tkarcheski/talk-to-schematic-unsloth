from pathlib import Path

import pytest

from schematic_model.corpus import parse_eagle
from schematic_model.engineering_evidence import REVIEW_REQUIREMENTS, answer


SOURCE = Path(__file__).parent / "fixtures/real/txb0104.sch"


def test_native_facts_match_real_source():
    sheet = parse_eagle(SOURCE)["sheets"][0]
    part = next(p for p in sheet["bom"] if p["value"])
    result = answer(SOURCE, {"kind": "value", "page": 1, "target": part["refdes"]})
    assert result["status"] == "source_verified" and result["answer"] == part["value"]
    net, pads = next(iter(sheet["nets"].items()))
    assert answer(SOURCE, {"kind": "net_members", "page": 1, "target": net})["answer"] == pads
    assert answer(SOURCE, {"kind": "pad_net", "page": 1, "target": pads[0]})["answer"] == net


@pytest.mark.parametrize("kind", REVIEW_REQUIREMENTS)
def test_engineering_conclusions_are_never_guessed(kind):
    result = answer(SOURCE, {"kind": kind, "page": 1, "target": "U1"})
    assert result["status"] == "additional_evidence_required"
    assert result["answer"] is None and result["training_gold_eligible"] is False
    assert result["required_evidence"]


@pytest.mark.parametrize("kind", ["value", "net_members", "pad_net"])
def test_absent_target_does_not_become_training_gold(kind):
    result = answer(SOURCE, {"kind": kind, "page": 1, "target": "ABSENT"})
    assert result["status"] == "not_recorded_on_page"
    assert result["answer"] is None and not result["training_gold_eligible"]


@pytest.mark.parametrize("question", [
    {"kind": "unknown", "page": 1, "target": "U1"},
    {"kind": "value", "page": True, "target": "U1"},
    {"kind": "value", "page": 999, "target": "U1"},
    {"kind": "value", "page": 1, "target": ""},
])
def test_invalid_question_rejected(question):
    with pytest.raises(ValueError):
        answer(SOURCE, question)
