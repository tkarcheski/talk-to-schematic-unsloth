"""Audit inputs must remain frozen and may not hide unsupported packages."""
import pytest

from scripts.audit_dependencies import registry_pins, verify_vcs_tool


def test_export_respects_platform_markers_and_exact_pins():
    assert registry_pins('pillow==12.3.0\n# dependency\nunused==1 ; python_version < "2"\n') == {'pillow': '12.3.0'}


@pytest.mark.parametrize('text', ['pillow>=12', 'pillow==12.*', 'pillow @ https://example.invalid/package.whl',
                                 'pillow==12\npillow==13', '# empty'])
def test_unpinned_or_unsupported_exports_fail(text):
    with pytest.raises(ValueError):
        registry_pins(text)


def test_vcs_exclusion_cannot_expand_silently(tmp_path):
    revision = 'a' * 40
    (tmp_path / 'pyproject.toml').write_text('[tool.uv.sources]\nai-pilled = {rev = "' + revision + '"}\n')
    (tmp_path / 'uv.lock').write_text('[[package]]\nname = "another-tool"\n[package.source]\ngit = "https://example.invalid/repo#' + revision + '"\n')
    with pytest.raises(ValueError, match='new VCS'):
        verify_vcs_tool(tmp_path)
