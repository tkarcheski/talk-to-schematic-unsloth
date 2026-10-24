"""Local checkout hygiene guard. Leases record intent; they do not grant authority."""
# cspell:words fsmonitor RDONLY fstat getuid RDWR CREAT NONBLOCK WRONLY worktrees
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
import uuid

STATE = "worktree-policy.json"
MAX_HOURS = 72


class PolicyError(Exception):
    pass


def git(cwd, *args):
    result = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "-C", str(cwd), *args],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=15, check=False,
    )
    if result.returncode or len(result.stdout) > 2_000_000:
        raise PolicyError("Cannot inspect Git metadata; repair the checkout and retry.")
    return result.stdout.decode("utf-8", errors="strict")


def inventory(cwd):
    records = []
    for block in git(cwd, "worktree", "list", "--porcelain", "-z").split("\0\0"):
        if block:
            records.append(dict(field.partition(" ")[::2] for field in block.split("\0") if field))
    if not records or any("worktree" not in row for row in records):
        raise PolicyError("Invalid worktree inventory; repair Git metadata.")
    return records


def label(path):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", Path(path).name)[:80] or "checkout"


def directory(path):
    """Open each directory component without following symbolic links."""
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise PolicyError("Checkout and Git metadata directories must be absolute and not symbolic links.")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def private_file(fd):
    info = os.fstat(fd)
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600):
        raise PolicyError("Lease files must be private regular files owned by this user (mode 0600).")


@contextmanager
def locked(common):
    folder = directory(common)
    lock = None
    try:
        lock = os.open(STATE + ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=folder)
        private_file(lock)
        deadline = time.monotonic() + 5
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise PolicyError("Lease state is busy; retry after the current update.") from None
                time.sleep(0.05)
        yield folder
    finally:
        if lock is not None:
            os.close(lock)
        os.close(folder)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate lease field")
        result[key] = value
    return result


def read_leases(folder):
    try:
        fd = os.open(STATE, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=folder)
    except FileNotFoundError:
        return []
    with os.fdopen(fd, "rb") as source:
        private_file(source.fileno())
        raw = source.read(65_537)
    try:
        data = json.loads(raw, object_pairs_hook=unique_object)
        if len(raw) > 65_536 or set(data) != {"version", "leases"} or type(data["version"]) is not int or data["version"] != 1:
            raise ValueError()
        leases = data["leases"]
        if not isinstance(leases, list) or len(leases) > 32:
            raise ValueError()
        seen = set()
        for lease in leases:
            if not isinstance(lease, dict) or set(lease) != {"owner", "reference", "branch", "path", "issued", "expires"}:
                raise ValueError()
            if not all(isinstance(lease[k], str) for k in ("owner", "reference", "branch", "path")):
                raise ValueError()
            if (not re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", lease["owner"])
                    or not re.fullmatch(r"#[1-9][0-9]{0,9}", lease["reference"])
                    or not 1 <= len(lease["branch"]) <= 240
                    or not lease["path"].startswith("/") or len(lease["path"]) > 4096
                    or any(ord(c) < 32 for c in lease["branch"] + lease["path"])
                    or lease["path"] in seen):
                raise ValueError()
            if (type(lease["issued"]) is not int or type(lease["expires"]) is not int
                    or lease["issued"] < 0 or not 0 < lease["expires"] - lease["issued"] <= MAX_HOURS * 3600):
                raise ValueError()
            seen.add(lease["path"])
        return leases
    except (ValueError, TypeError, KeyError, UnicodeError):
        raise PolicyError("Malformed private lease state; inspect and repair local leases.") from None


def write_leases(folder, leases):
    temporary = STATE + "." + uuid.uuid4().hex
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=folder)
        with os.fdopen(fd, "w", encoding="utf-8") as target:
            json.dump({"version": 1, "leases": leases}, target, indent=2)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, STATE, src_dir_fd=folder, dst_dir_fd=folder)
        os.fsync(folder)
    finally:
        try:
            os.unlink(temporary, dir_fd=folder)
        except FileNotFoundError:
            pass


def run(args, cwd):
    rows = inventory(cwd)
    common = git(cwd, "rev-parse", "--path-format=absolute", "--git-common-dir").rstrip("\n")
    current = git(cwd, "rev-parse", "--show-toplevel").rstrip("\n")
    now = int(time.time())
    with locked(common) as folder:
        leases = read_leases(folder)
        if args.command == "claim":
            if (not re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", args.owner)
                    or not re.fullmatch(r"#[1-9][0-9]{0,9}", args.reference) or not 1 <= args.hours <= MAX_HOURS):
                raise PolicyError("Use a short owner, an issue/PR reference such as #12, and 1–72 hours.")
            row = next((row for row in rows if row["worktree"] == current), None)
            if row is rows[0]:
                return "Primary checkout needs no lease."
            if (row is None or not row.get("branch") or len(row["branch"]) > 240
                    or len(current) > 4096 or any(ord(c) < 32 for c in current)
                    or "locked" in row or "prunable" in row):
                raise PolicyError("Claim requires an available named-branch checkout with a bounded plain path.")
            os.close(directory(current))
            active = {row["worktree"] for row in rows[1:]}
            leases = [lease for lease in leases if lease["path"] in active and lease["path"] != current]
            leases.append(dict(owner=args.owner, reference=args.reference, branch=row["branch"], path=current,
                               issued=now, expires=now + args.hours * 3600))
            write_leases(folder, leases)
            return f"Claimed {label(current)} for {args.hours} hours; renew only while actively working."
        failures = []
        if len(rows) > 3:
            failures.append(f"{len(rows)} registered checkouts exceed the limit of 3; retire completed worktrees.")
        for index, row in enumerate(rows):
            name, path = label(row["worktree"]), row["worktree"]
            try:
                os.close(directory(path))
                if git(path, "rev-parse", "--show-toplevel").rstrip("\n") != path:
                    raise PolicyError("Invalid checkout registration")
            except (OSError, PolicyError):
                failures.append(f"{name}: missing or symbolic-link checkout; repair its registration.")
                continue
            if "locked" in row or "prunable" in row:
                failures.append(f"{name}: locked or prunable checkout; resolve its lifecycle before delivery.")
            if index == 0:
                if git(path, "status", "--porcelain=v1", "-z", "--untracked-files=normal"):
                    failures.append(f"{name}: primary source is dirty; commit or resolve tracked/untracked changes.")
                continue
            lease = next((item for item in leases if item["path"] == path), None)
            if not row.get("branch"):
                failures.append(f"{name}: detached checkout; use a named branch or retire it.")
            elif lease is None or lease["branch"] != row["branch"] or not lease["issued"] <= now < lease["expires"]:
                failures.append(f"{name}: missing, expired or branch-mismatched lease; run claim here or retire it.")
        if failures:
            raise PolicyError("\n".join(failures))
        return f"Worktree policy passed ({len(rows)}/3 checkouts)."


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check")
    claim = commands.add_parser("claim")
    claim.add_argument("--owner", required=True)
    claim.add_argument("--reference", required=True)
    claim.add_argument("--hours", type=int, default=24)
    args = parser.parse_args(argv)
    try:
        print(run(args, Path.cwd()))
        return 0
    except (PolicyError, OSError, UnicodeError, subprocess.TimeoutExpired) as exc:
        print(str(exc) if isinstance(exc, PolicyError) else "Cannot safely inspect checkout or private lease state; repair it and retry.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
