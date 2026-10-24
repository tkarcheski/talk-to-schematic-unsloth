"""Fail-closed dependency coverage and exact-commit model-status publication."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import audit_dependencies as dependencies
from scripts import audit_repository as repository
from ai_pilled.runtime import CommandError


HEAD = "a" * 40
OTHER = "b" * 40


def lock_file(extra: str = "") -> bytes:
    return ('''
[[package]]
name = "talk-to-schematic"
version = "0.1.0"
source = {editable = "."}
[[package]]
name = "ai-pilled"
version = "0.1.0"
source = {git = "https://github.com/tkarcheski/ai-pilled.git?rev=''' + HEAD + "#" + HEAD + '''"}
[[package]]
name = "pillow"
version = "13.0.0"
source = {registry = "https://pypi.org/simple"}
''' + extra).encode()


def model_report() -> dict:
    return {
        "status": "pass", "snapshot": "source-hash", "metrics": {"model_reviews_run": 3},
        "checks": [{"check": name, "status": "pass", "metrics": {"model_invocations": 1}}
                   for name in sorted(repository.PERSPECTIVES)],
    }


def prepare_audit(monkeypatch, report=None):
    report = model_report() if report is None else report
    states = []
    monkeypatch.setattr(repository, "clean_head", lambda root: HEAD)
    monkeypatch.setattr(repository, "verify_remote", lambda root, head: None)
    monkeypatch.setattr(repository, "publish_status", lambda root, head, state: states.append((head, state)))
    monkeypatch.setattr(repository, "full_audit", lambda *args, **kwargs: SimpleNamespace(to_dict=lambda: report))
    monkeypatch.setattr(repository, "scan", lambda *args, **kwargs:
                        SimpleNamespace(status="pass", snapshot="source-hash"))
    return states


def test_python_lock_pins_include_complete_registry_set():
    extra = '''[[package]]
name = "transitive-library"
version = "1.2.3"
source = {registry = "https://pypi.org/simple"}
'''
    assert dependencies.python_pins(lock_file(extra)) == "pillow==13.0.0\ntransitive-library==1.2.3\n"


@pytest.mark.parametrize("replacement", [
    b'source = {git = "https://example.invalid/package.git#main"}',
    b'source = {registry = "https://example.invalid/simple"}',
    b'source = {editable = "../package"}',
])
def test_unknown_dependency_source_blocks(replacement):
    content = lock_file().replace(b'source = {registry = "https://pypi.org/simple"}', replacement)
    with pytest.raises(ValueError, match="Unsupported dependency source"):
        dependencies.python_pins(content)


def test_git_tool_requires_complete_matching_commit_pin():
    with pytest.raises(ValueError, match="Unsupported dependency source"):
        dependencies.python_pins(lock_file().replace(("#" + HEAD).encode(), ("#" + OTHER).encode()))


def test_duplicate_versions_are_not_silently_dropped():
    extra = '''[[package]]
name = "pillow"
version = "12.0.0"
source = {registry = "https://pypi.org/simple"}
'''
    with pytest.raises(ValueError, match="Duplicate"):
        dependencies.python_pins(lock_file(extra))


def test_malformed_lock_fails_without_invoking_registry(tmp_path, monkeypatch):
    (tmp_path / "uv.lock").write_text("not a lock")
    monkeypatch.setattr(dependencies, "record", lambda *args: None)
    monkeypatch.setattr(dependencies, "audit", lambda root: pytest.fail("Registry should not be queried"))
    assert dependencies.dependency_audit(tmp_path).status == "incomplete"


def test_dependency_lock_change_blocks(tmp_path, monkeypatch):
    (tmp_path / "uv.lock").write_bytes(lock_file())
    monkeypatch.setattr(dependencies, "record", lambda *args: None)

    def python_audit(root, files, executable):
        assert (root / files[0]).read_text() == "pillow==13.0.0\n"
        assert Path(executable).name == "pip-audit"
        (root / "uv.lock").write_bytes(lock_file() + b"\n")
        return dependencies.PipelineReport("python-dependency-vulnerabilities")

    monkeypatch.setattr(dependencies, "audit_python", python_audit)
    monkeypatch.setattr(dependencies, "audit", lambda root: dependencies.PipelineReport("dependency-vulnerabilities"))
    assert dependencies.dependency_audit(tmp_path).status == "incomplete"


@pytest.mark.parametrize("failure", ["missing", "duplicate", "incomplete", "never-invoked", "wrong-total"])
def test_model_status_requires_every_perspective(failure):
    report = model_report()
    if failure == "missing":
        report["checks"].pop()
    elif failure == "duplicate":
        report["checks"][1] = report["checks"][0]
    elif failure == "incomplete":
        report["checks"][0]["status"] = "incomplete"
    elif failure == "never-invoked":
        report["checks"][0]["metrics"]["model_invocations"] = 0
    else:
        report["metrics"]["model_reviews_run"] = 0
    assert not repository.model_checks_passed(report)


def test_exact_commit_model_success_publishes_only_sanitized_status(tmp_path, monkeypatch, capsys):
    report = model_report()
    report["findings"] = [{"message": "private operational detail"}]
    states = prepare_audit(monkeypatch, report)
    assert repository.audit_repository(tmp_path, publish=True) == 0
    assert states == [(HEAD, "pending"), (HEAD, "success")]
    output = capsys.readouterr().out
    assert "private" not in output
    assert json.loads(output)["commit"] == HEAD


def test_source_change_after_model_audit_never_publishes_success(tmp_path, monkeypatch):
    states = prepare_audit(monkeypatch)
    monkeypatch.setattr(repository, "scan", lambda *args, **kwargs:
                        SimpleNamespace(status="pass", snapshot="changed"))
    assert repository.audit_repository(tmp_path, publish=True) == 1
    assert states == [(HEAD, "pending"), (HEAD, "error")]


def test_missing_model_result_blocks_even_with_top_level_pass(tmp_path, monkeypatch):
    report = model_report()
    report["checks"] = []
    states = prepare_audit(monkeypatch, report)
    assert repository.audit_repository(tmp_path, publish=True) == 1
    assert states[-1] == (HEAD, "error")


def test_unavailable_publication_blocks_delivery(tmp_path, monkeypatch):
    prepare_audit(monkeypatch)

    def unavailable(*args):
        raise CommandError("unavailable")

    monkeypatch.setattr(repository, "publish_status", unavailable)
    assert repository.audit_repository(tmp_path, publish=True) == 1


def test_deterministic_ci_never_claims_or_publishes_model_review(tmp_path, monkeypatch, capsys):
    report = {"status": "pass", "snapshot": "source-hash", "checks": [], "metrics": {}}
    states = prepare_audit(monkeypatch, report)
    assert repository.audit_repository(tmp_path) == 0
    assert states == []
    assert json.loads(capsys.readouterr().out)["model_reviews"] == "not_run"


def test_wrong_origin_blocks_before_github_request(tmp_path, monkeypatch):
    calls = []

    def command(argv, root):
        calls.append(argv)
        return b"https://github.com/another/repository.git\n"

    monkeypatch.setattr(repository, "run", command)
    with pytest.raises(ValueError, match="origin"):
        repository.verify_remote(tmp_path, HEAD)
    assert len(calls) == 1


def test_remote_commit_must_match(tmp_path, monkeypatch):
    responses = iter([f"git@github.com:{repository.REPOSITORY}.git\n".encode(),
                      json.dumps({"sha": OTHER}).encode()])
    monkeypatch.setattr(repository, "run", lambda *args: next(responses))
    with pytest.raises(ValueError, match="already exist"):
        repository.verify_remote(tmp_path, HEAD)


def test_push_requires_the_audited_head(tmp_path, monkeypatch):
    monkeypatch.setenv("PRE_COMMIT_TO_REF", OTHER)
    monkeypatch.setattr(repository, "require_visible_index", lambda root: None)
    responses = iter([(HEAD + "\n").encode(), b"", (OTHER + "\n").encode()])
    monkeypatch.setattr(repository, "run", lambda *args: next(responses))
    with pytest.raises(ValueError, match="pushed commit"):
        repository.clean_head(tmp_path)
