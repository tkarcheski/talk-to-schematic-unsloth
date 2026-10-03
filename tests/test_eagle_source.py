"""Bound native input before parsing, with identical extractor/renderer rules."""

import hashlib
import io
from pathlib import Path

import pytest

from schematic_model.corpus import parse_eagle
from schematic_model.eagle_source import load_eagle
from schematic_model.render_eagle import source_svg


@pytest.mark.parametrize("read_source", [parse_eagle, lambda path: source_svg(path, 1)])
def test_file_reads_stop_at_one_byte_beyond_limit(monkeypatch, read_source):
    requested = []

    class RecordingStream(io.BytesIO):
        def read(self, size=-1):
            requested.append(size)
            return super().read(size)

    def open_source(path, mode):
        assert mode == "rb"
        return RecordingStream(b"x" * 4096)

    monkeypatch.setattr("schematic_model.eagle_source.MAX_SOURCE", 64)
    monkeypatch.setattr(Path, "open", open_source)
    with pytest.raises(ValueError, match="exceeds"):
        read_source(Path("large.sch"))
    assert requested == [65]


def test_verified_bytes_produce_the_same_facts_and_source_hash():
    path = Path(__file__).parent / "fixtures/real/txb0104.sch"
    content = path.read_bytes()
    facts = parse_eagle(content)
    assert facts == parse_eagle(path)
    assert facts["sha256"] == hashlib.sha256(content).hexdigest()


def test_limit_is_inclusive_for_file_and_byte_inputs(tmp_path, monkeypatch):
    content = b'<?xml version="1.0"?><eagle><drawing><schematic/></drawing></eagle>'
    path = tmp_path / "exact.sch"
    path.write_bytes(content)
    monkeypatch.setattr("schematic_model.eagle_source.MAX_SOURCE", len(content))
    assert load_eagle(path)[0] == content
    assert load_eagle(content)[0] == content
    with pytest.raises(ValueError, match="exceeds"):
        load_eagle(content + b" ")
    path.write_bytes(content + b" ")
    with pytest.raises(ValueError, match="exceeds"):
        load_eagle(path)


@pytest.mark.parametrize("read_source", [parse_eagle, lambda path: source_svg(path, 1)])
@pytest.mark.parametrize("content, error", [
    (b'<?xml version="1.0"?><eagle><drawing>', "Malformed EAGLE XML"),
    (b'<?xml version="1.0" encoding="unsupported_encoding"?><eagle/>', "Malformed EAGLE XML"),
    (b'<?xml version="1.0"?><other><drawing><schematic/></drawing></other>', "Not an EAGLE"),
    (b'<?xml version="1.0"?><eagle><drawing><board/></drawing></eagle>', "Not an EAGLE"),
    (b'<eagle><drawing><schematic/></drawing></eagle>', "Only XML EAGLE"),
    (b'<?xml version="1.0"?><!DOCTYPE eagle [<!ENTITY x "value">]><eagle/>', "entity declarations"),
])
def test_extracting_and_rendering_reject_invalid_source_consistently(tmp_path, read_source, content, error):
    path = tmp_path / "invalid.sch"
    path.write_bytes(content)
    with pytest.raises(ValueError, match=error):
        read_source(path)
