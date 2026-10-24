# cspell:words worktrees mkfifo
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/check_worktrees.py"


def git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path):
    primary = tmp_path / "primary"
    primary.mkdir()
    git(primary, "init", "-b", "main")
    git(primary, "config", "user.name", "Fixture")
    git(primary, "config", "user.email", "fixture@example.invalid")
    (primary / ".gitignore").write_text("outputs/\n")
    (primary / "source.txt").write_text("original\n")
    git(primary, "add", ".")
    git(primary, "commit", "-m", "Fixture")
    return primary


def secondary(repo, name="candidate"):
    path = repo.parent / name
    git(repo, "worktree", "add", "-b", name, str(path))
    return path


def command(cwd, *args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=cwd, capture_output=True, text=True)


def claim(path, hours="24"):
    return command(path, "claim", "--owner", "fixture", "--reference", "#12", "--hours", hours)


def state(repo):
    return repo / ".git/worktree-policy.json"


def test_single_checkout_ci_and_ignored_artifacts_need_no_lease(repo):
    (repo / "outputs").mkdir()
    (repo / "outputs/private.txt").write_text("fixture")
    git(repo, "checkout", "--detach")
    result = command(repo, "check")
    assert result.returncode == 0, result.stderr
    assert "1/3" in result.stdout
    assert not state(repo).exists()


def test_missing_lease_then_claim_and_atomic_renewal(repo):
    linked = secondary(repo)
    assert "missing, expired" in command(repo, "check").stderr
    assert claim(linked).returncode == 0
    original_inode = state(repo).stat().st_ino
    assert command(repo, "check").returncode == 0
    assert claim(linked, "48").returncode == 0
    assert state(repo).stat().st_ino != original_inode
    assert state(repo).stat().st_mode & 0o777 == 0o600
    data = json.loads(state(repo).read_text())
    assert len(data["leases"]) == 1
    assert data["leases"][0]["expires"] - data["leases"][0]["issued"] == 48 * 3600
    assert sorted(p.name for p in state(repo).parent.glob("worktree-policy.json*")) == [
        "worktree-policy.json", "worktree-policy.json.lock"]


@pytest.mark.parametrize("change,expected", [
    ("expired", "expired"), ("branch", "branch-mismatched"), ("detached", "detached"),
    ("missing", "missing"), ("locked", "locked"),
])
def test_stale_or_unavailable_secondary_fails(repo, change, expected):
    linked = secondary(repo)
    assert claim(linked).returncode == 0
    if change == "expired":
        data = json.loads(state(repo).read_text())
        data["leases"][0].update(issued=int(time.time()) - 7200, expires=int(time.time()) - 3600)
        state(repo).write_text(json.dumps(data))
    elif change == "branch":
        git(linked, "checkout", "-b", "different")
    elif change == "detached":
        git(linked, "checkout", "--detach")
    elif change == "locked":
        git(repo, "worktree", "lock", str(linked))
    else:
        linked.rename(linked.with_name("moved"))
    result = command(repo, "check")
    assert result.returncode == 1
    assert expected in result.stderr
    assert str(repo.parent) not in result.stderr


def test_checkout_budget_is_enforced_even_with_valid_leases(repo):
    for name in ("one", "two", "three"):
        assert claim(secondary(repo, name)).returncode == 0
    result = command(repo, "check")
    assert result.returncode == 1
    assert "exceed the limit of 3" in result.stderr


@pytest.mark.parametrize("name", ["source.txt", "new.txt"])
def test_primary_source_must_be_clean(repo, name):
    (repo / name).write_text("changed")
    result = command(repo, "check")
    assert result.returncode == 1
    assert "primary source is dirty" in result.stderr


@pytest.mark.parametrize("payload", ['{}', '{"version":1,"leases":[{}]}', '{"version":1,"version":1,"leases":[]}', '[]', 'null'])
def test_malformed_private_state_fails_closed(repo, payload):
    state(repo).write_text(payload)
    state(repo).chmod(0o600)
    result = command(repo, "check")
    assert result.returncode == 1
    assert "Malformed" in result.stderr


@pytest.mark.parametrize("hours", ["0", "73", "-1"])
def test_claim_duration_is_bounded(repo, hours):
    result = claim(secondary(repo), hours)
    assert result.returncode == 1
    assert not state(repo).exists()


@pytest.mark.parametrize("filename", ["worktree-policy.json", "worktree-policy.json.lock"])
def test_symlink_state_cannot_read_or_overwrite_outside_file(repo, filename):
    linked = secondary(repo)
    outside = repo.parent / "outside"
    outside.write_text("untouched")
    (repo / ".git" / filename).symlink_to(outside)
    assert claim(linked).returncode == 1
    assert outside.read_text() == "untouched"


def test_symlink_checkout_is_rejected_and_paths_are_not_printed(repo):
    linked = secondary(repo)
    assert claim(linked).returncode == 0
    moved = linked.with_name("actual")
    linked.rename(moved)
    linked.symlink_to(moved, target_is_directory=True)
    result = command(repo, "check")
    assert result.returncode == 1
    assert "symbolic-link" in result.stderr
    assert str(repo.parent) not in result.stderr


def test_concurrent_claims_preserve_both_leases(repo):
    paths = [secondary(repo, "one"), secondary(repo, "two")]
    processes = [subprocess.Popen([sys.executable, str(SCRIPT), "claim", "--owner", "fixture",
                                  "--reference", "#12", "--hours", "24"], cwd=path,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE) for path in paths]
    for process in processes:
        _, stderr = process.communicate(timeout=20)
        assert process.returncode == 0, stderr
    assert {item["path"] for item in json.loads(state(repo).read_text())["leases"]} == set(map(str, paths))
    assert command(repo, "check").returncode == 0


def test_failed_atomic_replace_leaves_old_state(repo, monkeypatch):
    from scripts import check_worktrees

    linked = secondary(repo)
    assert claim(linked).returncode == 0
    before = state(repo).read_bytes()
    def fail(*args, **kwargs):
        raise OSError("fixture replacement failure")
    monkeypatch.setattr(check_worktrees.os, "replace", fail)
    with check_worktrees.locked(repo / ".git") as folder:
        with pytest.raises(OSError, match="fixture"):
            check_worktrees.write_leases(folder, [])
    assert state(repo).read_bytes() == before
    assert sorted(p.name for p in (repo / ".git").glob("worktree-policy.json*")) == [
        "worktree-policy.json", "worktree-policy.json.lock"]


@pytest.mark.parametrize("kind", ["fifo", "public", "oversize"])
def test_invalid_state_files_fail_without_hanging(repo, kind):
    if kind == "fifo":
        os.mkfifo(state(repo), 0o600)
    else:
        state(repo).write_text(" " * 65_537 if kind == "oversize" else '{"version":1,"leases":[]}')
        state(repo).chmod(0o644 if kind == "public" else 0o600)
    result = subprocess.run([sys.executable, str(SCRIPT), "check"], cwd=repo,
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 1


def test_missing_secondary_git_file_fails(repo):
    linked = secondary(repo)
    assert claim(linked).returncode == 0
    (linked / ".git").unlink()
    assert command(repo, "check").returncode == 1


def test_lease_allows_active_secondary_edits_but_no_primary_lease_is_created(repo):
    assert claim(repo).returncode == 0
    assert not state(repo).exists()
    linked = secondary(repo)
    assert claim(linked, "72").returncode == 0
    (linked / "source.txt").write_text("in progress")
    assert command(linked, "check").returncode == 0


@pytest.mark.parametrize("overrides", ["directory", "index", "combined"])
def test_hook_git_environment_does_not_change_per_checkout_inspection(repo, overrides):
    linked = secondary(repo)
    (linked / "candidate.txt").write_text("committed candidate\n")
    git(linked, "add", ".")
    git(linked, "commit", "-m", "Candidate")
    assert claim(linked).returncode == 0
    git_dir = git(linked, "rev-parse", "--absolute-git-dir").strip()
    values = {"GIT_DIR": git_dir} if overrides == "directory" else {"GIT_INDEX_FILE": str(Path(git_dir) / "index")}
    if overrides == "combined":
        values.update(GIT_DIR=git_dir, GIT_WORK_TREE=str(linked), GIT_PREFIX="", GIT_COMMON_DIR=str(repo / ".git"))
    environment = dict(os.environ, **values)
    def inspect():
        return subprocess.run([sys.executable, str(SCRIPT), "check"], cwd=linked,
                              env=environment, capture_output=True, text=True)
    assert not git(repo, "status", "--porcelain")
    result = inspect()
    assert result.returncode == 0, result.stderr
    (repo / "source.txt").write_text("actual primary edit\n")
    result = inspect()
    assert result.returncode == 1
    assert "primary source is dirty" in result.stderr


def test_real_pre_push_hook_checks_primary_from_linked_checkout(repo):
    linked = secondary(repo)
    (linked / "candidate.txt").write_text("committed candidate\n")
    git(linked, "add", ".")
    git(linked, "commit", "-m", "Candidate")
    assert claim(linked).returncode == 0
    hook = repo / ".git/hooks/pre-push"
    hook.write_text(f"#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(SCRIPT))} check\n")
    hook.chmod(0o700)
    remote = repo.parent / "remote.git"
    git(repo, "init", "--bare", str(remote))
    result = subprocess.run(["git", "-C", str(linked), "push", str(remote), "HEAD:refs/heads/candidate"],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "Worktree policy passed" in result.stdout
    assert git(remote, "rev-parse", "refs/heads/candidate") == git(linked, "rev-parse", "HEAD")


def test_other_owner_cannot_replace_unexpired_claim_but_can_claim_after_expiry(repo):
    linked = secondary(repo)
    assert claim(linked).returncode == 0
    before = state(repo).read_bytes()
    takeover = ["claim", "--owner", "different-owner", "--reference", "#13"]
    result = command(linked, *takeover)
    assert result.returncode == 1
    assert "explicit handoff" in result.stderr
    assert state(repo).read_bytes() == before
    data = json.loads(before)
    data["leases"][0].update(issued=int(time.time()) - 7200, expires=int(time.time()) - 3600)
    state(repo).write_text(json.dumps(data))
    assert command(linked, *takeover).returncode == 0
    assert json.loads(state(repo).read_text())["leases"][0]["owner"] == "different-owner"


def test_concurrent_owners_cannot_overwrite_each_others_active_claim(repo):
    linked = secondary(repo)
    processes = [subprocess.Popen([sys.executable, str(SCRIPT), "claim", "--owner", owner,
                                  "--reference", "#12"], cwd=linked,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE) for owner in ("one", "two")]
    errors = []
    for process in processes:
        _, stderr = process.communicate(timeout=20)
        errors.append(stderr.decode())
    assert sorted(process.returncode for process in processes) == [0, 1]
    assert sum("explicit handoff" in error for error in errors) == 1
    assert len(json.loads(state(repo).read_text())["leases"]) == 1
