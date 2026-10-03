"""Viewer-command conversations: source-derived targets, wire format and gold."""

import json

from evaluate import grade, validate_gold_rows
from schematic_model.view_data import conversation

RECORD = {"id": "real-0123456789abcdef", "repository": "sparkfun/SparkFun_Qwiic_Test", "commit": "1" * 40,
          "source_path": "board.sch", "sha256": "2" * 64, "license": "CC-BY-SA-4.0"}
SHEET = {"page": 1,
         "bom": [{"refdes": "U1", "value": "AP2112", "value_visible": True},
                 {"refdes": "R1", "value": "10k", "value_visible": True},
                 {"refdes": "R2", "value": "2.2k", "value_visible": True},
                 {"refdes": "C1", "value": "1uF", "value_visible": False}],
         "nets": {"SCL": ["R1.1", "U1.2"], "N$4": ["R2.2", "U1.3"], "GND": ["C1.2", "U1.4"]}}
REGIONS = {"parts": {"U1": [100, 100, 200, 200], "R1": [300, 100, 340, 160], "R2": [400, 100, 440, 160],
                     "C1": [500, 100, 540, 160]},
           "nets": {"SCL": [200, 120, 300, 130], "N$4": [200, 150, 400, 160], "GND": [0, 0, 1000, 1000]}}


def text(message):
    return "".join(block.get("text", "") for block in message["content"])


def test_conversation_covers_part_net_region_missing_label_and_full_sheet():
    row = conversation(RECORD, SHEET, REGIONS, "../images/sheet.png")
    assert row["split"] in {"train", "val", "test"} and row["provenance"]["license"] == "CC-BY-SA-4.0"
    assert row["messages"][0]["role"] == "system" and "show_label" in text(row["messages"][0])
    assert row["messages"][1]["content"][0] == {"type": "image", "image": "../images/sheet.png"}
    assert [g["type"] for g in row["gold"]] == ["view", "view_reply"] * 3 + ["view", "view_refuse", "view", "view_reply"]
    calls = [g for g in row["gold"] if g["type"] == "view"]
    assert calls[0]["arguments"] == {"label": "R1"}  # R and U parts come first; C1 has a hidden value.
    assert calls[1]["arguments"] == {"label": "SCL"}  # Anonymous N$ nets and GND are never targets.
    assert calls[2]["name"] == "show_region" and calls[2]["arguments"]["box"] == [300, 100, 440, 160]
    assert calls[3]["arguments"] == {"label": "R999"}
    assert calls[4] == {"type": "view", "name": "show_full_sheet", "arguments": {}}
    # Calls and results are plain wire text: tool use is learned as ordinary tokens.
    wire = [text(message) for message in row["messages"] if message["role"] == "assistant"]
    assert wire[0].startswith("<tool_call>\n<function=show_label>") and "R1" in wire[0]
    tool_turns = [text(message) for message in row["messages"]
                  if message["role"] == "user" and "<tool_response>" in text(message)]
    assert json.loads(tool_turns[3].split("<tool_response>\n")[1].split("\n</tool_response>")[0]) == {
        "status": "not_found", "label": "R999"}
    validate_gold_rows([row])


def test_gold_replies_score_perfectly_and_a_plain_answer_does_not():
    row = conversation(RECORD, SHEET, REGIONS, "image.png")
    replies = [text(message) for message in row["messages"] if message["role"] == "assistant"]
    assert len(replies) == len(row["gold"])
    for gold, reply in zip(row["gold"], replies):
        assert all(grade(gold, reply).values()), (gold, reply)
    assert not any(grade(row["gold"][0], "U1 is at the top left.").values())


def test_pages_without_clear_targets_are_skipped():
    sparse = {**SHEET, "bom": SHEET["bom"][:2]}
    assert conversation(RECORD, sparse, REGIONS, "image.png") is None
    no_net = {**SHEET, "nets": {"N$4": ["R2.2", "U1.3"]}}
    assert conversation(RECORD, no_net, REGIONS, "image.png") is None
