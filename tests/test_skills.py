import pytest

from schematic_model.skill_bundle import export_skills


def test_export_preserves_existing_destination(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    existing = target / "my-skill.md"
    existing.write_text("user content")
    with pytest.raises(ValueError):
        export_skills(target)
    assert existing.read_text() == "user content"


def test_export_contains_complete_packaged_skills(tmp_path):
    output = tmp_path / "bundle"
    paths = export_skills(output)
    assert len(paths) == 3
    for path in output.glob("*/SKILL.md"):
        content = path.read_text()
        assert content.startswith("---\nname: ")
        assert "schematic-model" in content
