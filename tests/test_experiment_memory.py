"""Private command evidence contracts; child commands are small CPU fixtures."""
# cspell:words mkfifo
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time

import pytest

from schematic_model import experiment_memory as memory


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def command(code):
    return [sys.executable, "-c", code]


def test_success_records_exact_argv_inputs_and_artifact_changes(project):
    (project / "input.txt").write_text("input")
    (project / "stable.txt").write_text("prior")
    argv = command("from pathlib import Path; Path('new.txt').write_text('result'); print('done')")
    report = memory.run(argv, inputs=["input.txt"], artifacts=["new.txt", "stable.txt", "absent.txt"])
    start, end = report["events"]
    assert report["status"] == "succeeded"
    assert start["argv"] == argv and start["cwd"] == str(project)
    assert start["inputs"][0]["sha256"] == hashlib.sha256(b"input").hexdigest()
    assert end["exit_code"] == 0 and end["elapsed_seconds"] >= 0
    assert [a["changed_since_start"] for a in end["artifacts_after"]] == [True, False, False]
    assert end["artifacts_after"][2]["missing"] is True
    assert start["started_at"] <= end["ended_at"]
    directory = project / "outputs/experiments" / report["id"]
    assert (directory / "stdout.log").read_text() == "done\n"
    assert len((directory / "events.jsonl").read_text().splitlines()) == 2


def test_failures_are_retained_and_launch_failure_is_distinct(project):
    failed = memory.run(command("import sys; print('failure'); sys.exit(7)"))
    assert failed["status"] == "failed" and failed["events"][-1]["exit_code"] == 7
    missing = memory.run([str(project / "missing-executable")])
    assert missing["status"] == "failed"
    assert missing["events"][-1]["exit_code"] is None
    assert missing["events"][-1]["error_type"] == "FileNotFoundError"
    assert memory.read(Path("outputs/experiments"), failed["id"]) == failed


def test_stream_caps_and_private_permissions_ignore_umask(project):
    previous = os.umask(0)
    try:
        result = memory.run(command("import sys; print('x'*10000); print('y'*10000,file=sys.stderr)"), output_limit=31)
    finally:
        os.umask(previous)
    root = project / "outputs/experiments"
    for path in [root, *root.rglob("*")]:
        assert stat.S_IMODE(path.stat().st_mode) == (0o700 if path.is_dir() else 0o600)
    for log in result["events"][-1]["logs"].values():
        assert log["retained_bytes"] == 31 and log["observed_bytes"] == 10001 and log["truncated"]


def test_rerun_rejects_changed_inputs_and_variant_records_lineage(project):
    (project / "input").write_text("one")
    argv = command("pass")
    first = memory.run(argv, inputs=["input"])
    again = memory.run(argv, inputs=["input"], parent=first["id"], relation="rerun")
    assert again["id"] != first["id"]
    (project / "input").write_text("two")
    with pytest.raises(ValueError, match="Rerun argv, cwd and input hashes"):
        memory.run(argv, inputs=["input"], parent=first["id"], relation="rerun")
    changed = memory.run(argv, inputs=["input"], parent=first["id"], relation="variant")
    assert changed["events"][0]["parent"] == first["id"]
    assert changed["events"][0]["relation"] == "variant"


def test_read_and_list_never_execute_recorded_commands(project, monkeypatch, capsys):
    result = memory.run(command("pass"))
    monkeypatch.setattr(memory.subprocess, "Popen", lambda *a, **kw: pytest.fail("reading must not launch"))
    assert memory.main(["show", result["id"]]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == result["id"]
    assert memory.main(["list"]) == 0
    assert json.loads(capsys.readouterr().out) == [{"id": result["id"], "status": "succeeded"}]


def test_attempt_is_durable_before_launch_and_environment_not_inventoried(project, monkeypatch):
    monkeypatch.setenv("PRIVATE_EXPERIMENT_TEST_SECRET", "not-in-metadata")
    original = memory.subprocess.Popen
    def inspect_then_launch(*args, **kwargs):
        path, = (project / "outputs/experiments").glob("*/events.jsonl")
        record = memory.read(path.parent.parent, path.parent.name)
        assert record["status"] == "incomplete"
        assert "not-in-metadata" not in path.read_text()
        return original(*args, **kwargs)
    monkeypatch.setattr(memory.subprocess, "Popen", inspect_then_launch)
    assert memory.run(command("pass"))["status"] == "succeeded"


def test_timeout_bounds_pipe_inheriting_descendant(project):
    code = "import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])"
    start = time.monotonic()
    report = memory.run(command(code), timeout=0.2)
    assert report["status"] == "timed_out"
    assert time.monotonic() - start < 5


def test_incomplete_and_torn_tail_never_claim_success(project):
    root = project / "outputs/experiments"
    directory = root / ("a" * 32)
    directory.mkdir(parents=True)
    payload = json.dumps({"event": "attempted", "status": "succeeded", "argv": ["never-execute"]}) + '\n{"event":"fin'
    (directory / "events.jsonl").write_text(payload)
    result = memory.read(root, directory.name)
    assert result["status"] == "incomplete" and result["partial_tail"] is True


@pytest.mark.parametrize("options", [
    {"timeout": float("nan")}, {"timeout": 0}, {"output_limit": 0},
    {"parent": "a" * 32}, {"root": Path("tracked-records")},
])
def test_invalid_metadata_rejected_without_launch(project, monkeypatch, options):
    monkeypatch.setattr(memory.subprocess, "Popen", lambda *a, **kw: pytest.fail("must not launch"))
    with pytest.raises(ValueError):
        memory.run(command("pass"), **options)
    assert not (project / "outputs").exists()


def test_hashing_fifo_is_rejected_without_blocking(project):
    os.mkfifo(project / "pipe")
    with pytest.raises(ValueError, match="regular files"):
        memory.run(command("pass"), inputs=["pipe"])


def test_literal_argv_is_not_shell_expanded(project):
    value = "$(touch should-not-exist); * $HOME"
    report = memory.run([sys.executable, "-c", "import sys; print(sys.argv[1])", value])
    assert report["status"] == "succeeded"
    assert not (project / "should-not-exist").exists()
    log = project / "outputs/experiments" / report["id"] / "stdout.log"
    assert log.read_text().strip() == value


def test_sigterm_preserves_interrupted_terminal_record(project):
    argv = [sys.executable, "-m", "schematic_model.experiment_memory", "run", "--timeout", "10", "--",
            sys.executable, "-c", "import time; time.sleep(10)"]
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 5
        while not list((project / "outputs/experiments").glob("*/stdout.log")):
            assert time.monotonic() < deadline and process.poll() is None
            time.sleep(0.01)
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 1, stderr.decode()
        result = json.loads(stdout)
        assert result["status"] == "interrupted"
        record = memory.read(Path("outputs/experiments"), result["id"])
        assert record["events"][-1]["exit_code"] == -signal.SIGKILL
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_changed_argv_is_variant_not_rerun(project):
    first = memory.run(command("pass"))
    with pytest.raises(ValueError, match="Rerun argv"):
        memory.run(command("print('changed')"), parent=first["id"], relation="rerun")


def test_record_path_symlink_is_rejected(project):
    (project / "outputs").mkdir()
    (project / "target").mkdir()
    (project / "outputs/experiments").symlink_to(project / "target", target_is_directory=True)
    with pytest.raises(ValueError):
        memory.run(command("pass"))
