"""Bound and validate native EAGLE XML before extraction or rendering."""

from pathlib import Path
import xml.etree.ElementTree as ET

MAX_SOURCE = 12 * 1024 * 1024


def load_eagle(source: Path | bytes) -> tuple[bytes, ET.Element]:
    """Return exact source bytes and the schematic element, or a readable error.

    Bytes input lets hash-verified dataset builders parse the same bytes they
    verified. File input reads at most one byte beyond the accepted size limit.
    """
    if isinstance(source, bytes):
        content = source
    else:
        with source.open("rb") as stream:
            content = stream.read(MAX_SOURCE + 1)
    if len(content) > MAX_SOURCE:
        raise ValueError("EAGLE source exceeds the 12 MiB limit")
    if b"<!ENTITY" in content.upper():
        raise ValueError("XML entity declarations are not supported")
    if not content.lstrip().startswith(b"<?xml"):
        raise ValueError("Only XML EAGLE schematics are supported; binary/KiCad files are excluded")
    try:
        root = ET.fromstring(content)
    except (ET.ParseError, LookupError) as exc:
        raise ValueError(f"Malformed EAGLE XML: {exc}") from exc
    schematic = root.find("./drawing/schematic")
    if root.tag != "eagle" or schematic is None:
        raise ValueError("Not an EAGLE schematic")
    return content, schematic
