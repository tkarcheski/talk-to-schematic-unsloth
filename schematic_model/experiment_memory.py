"""Explicit local command recording; reading records never executes commands.

This is a private evidence log, not a sandbox or a GPU scheduler. Environment
values are inherited by the child but are never inventoried or recorded.
"""
# cspell:words WRONLY CREAT fchmod RDONLY NONBLOCK fstat keepends killpg
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import subprocess
import time
import threading
import uuid

from evaluate import _invalid_constant, _unique_object


def _now():
    return datetime.now(timezone.utc).isoformat()


def _private_directory(path: Path):
    if path.is_symlink():
        raise ValueError("Record directories must not be symbolic links")
    if not path.exists():
        _private_directory(path.parent)
        path.mkdir(mode=0o700)
    if not path.is_dir():
        raise ValueError("Record path must be a directory")
    # Only directories below outputs are passed here; existing ancestors are
    # checked by the caller; their permissions are not changed recursively.
    path.chmod(0o700)


def _create(path: Path):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    os.fchmod(descriptor, 0o600)
    return os.fdopen(descriptor, "wb", buffering=0)


def _append(path: Path, event: dict):
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "ab") as stream:
        stream.write((json.dumps(event, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def _directory_sync(path: Path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _root(path: Path):
    project = Path.cwd().resolve()
    resolved = path.absolute()
    if not resolved.resolve().is_relative_to(project / "outputs"):
        raise ValueError("Experiment records must stay in this project's ignored outputs directory")
    for ancestor in [resolved, *resolved.parents]:
        if ancestor == project:
            break
        if ancestor.is_symlink():
            raise ValueError("Record paths must not traverse symbolic links")
    return resolved


def _snapshot(paths: list[str], cwd: Path, *, required: bool) -> list[dict]:
    if len(paths) > 32 or len(set(paths)) != len(paths):
        raise ValueError("Use at most 32 distinct input or artifact paths")
    result = []
    for value in paths:
        path = (cwd / value).resolve()
        if not path.is_relative_to(Path.cwd().resolve()):
            raise ValueError("Recorded inputs and artifacts must stay within the project")
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
            with os.fdopen(descriptor, "rb") as stream:
                before = os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise ValueError("Only regular files can be hashed")
                digest = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
                after = os.fstat(stream.fileno())
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError("File changed while hashing")
            result.append({"path": str(path), "bytes": before.st_size, "sha256": digest.hexdigest()})
        except FileNotFoundError:
            if required:
                raise ValueError("A declared input is missing") from None
            result.append({"path": str(path), "missing": True})
    return result


def read(root: Path, experiment_id: str) -> dict:
    """Read evidence only. A missing terminal event is incomplete, never success."""
    if not re.fullmatch(r"[0-9a-f]{32}", experiment_id):
        raise ValueError("Invalid experiment ID")
    path = _root(root) / experiment_id / "events.jsonl"
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("Record paths must not be symbolic links")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("Experiment events must be a regular file")
        raw = stream.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError("Experiment record exceeds the metadata budget")
    lines = raw.splitlines(keepends=True)
    events = [json.loads(line, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
              for line in lines if line.endswith(b"\n")]
    if any(not isinstance(event, dict) for event in events):
        raise ValueError("Experiment events must be objects")
    if not events or events[0].get("event") != "attempted" or len(events) > 2:
        raise ValueError("Invalid experiment event sequence")
    if len(events) == 2 and events[1].get("event") != "finished":
        raise ValueError("Invalid terminal experiment event")
    status = "incomplete" if len(events) == 1 else events[1].get("status")
    if status not in {"incomplete", "succeeded", "failed", "timed_out", "interrupted", "recording_failed"}:
        raise ValueError("Invalid recorded status")
    if status == "succeeded" and events[1].get("exit_code") != 0:
        raise ValueError("Success requires recorded zero exit status")
    return {"id": experiment_id, "status": status, "events": events,
            "partial_tail": bool(lines and not lines[-1].endswith(b"\n"))}


def _stop(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def _capture(process, directory, timeout, output_limit):
    counts = {"stdout": 0, "stderr": 0}
    retained = {"stdout": 0, "stderr": 0}
    deadline = time.monotonic() + timeout
    with selectors.DefaultSelector() as selector, ExitStack() as stack:
        files = {name: stack.enter_context(_create(directory / (name + ".log"))) for name in counts}
        for name in counts:
            pipe = getattr(process, name)
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, name)
        timed_out = False
        while selector.get_map() or process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                _stop(process)
                break
            for key, _ in selector.select(min(remaining, 0.1)):
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                name = key.data
                counts[name] += len(chunk)
                kept = chunk[:max(0, output_limit - retained[name])]
                files[name].write(kept)
                retained[name] += len(kept)
        for stream in files.values():
            stream.flush()
            os.fsync(stream.fileno())
    return timed_out, {name: {"file": name + ".log", "observed_bytes": counts[name],
                              "retained_bytes": retained[name], "truncated": counts[name] > retained[name]}
                       for name in counts}


class _Interrupted(BaseException):
    pass


@contextmanager
def _interrupt_on_sigterm():
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    def interrupted(_signum, _frame):
        raise _Interrupted()
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def run(argv: list[str], *, root=Path("outputs/experiments"), cwd=None, inputs=None,
        artifacts=None, parent=None, relation=None, timeout=300.0, output_limit=1024 * 1024):
    """Run only explicit argv. 'rerun' verifies lineage; it never supplies argv."""
    if (not argv or any(not isinstance(arg, str) or "\0" in arg for arg in argv)
            or not argv[0] or len(json.dumps(argv).encode()) > 65536):
        raise ValueError("Supply a nonempty argv of at most 64 KiB")
    if not math.isfinite(timeout) or not 0 < timeout <= 3600:
        raise ValueError("timeout must be between zero and 3600 seconds")
    if type(output_limit) is not int or not 1 <= output_limit <= 16 * 1024 * 1024:
        raise ValueError("output_limit must be 1 byte through 16 MiB per stream")
    root = _root(Path(root))
    cwd = Path(cwd or Path.cwd()).resolve()
    if not cwd.is_dir() or not cwd.is_relative_to(Path.cwd().resolve()):
        raise ValueError("Command cwd must be an existing project directory")
    input_records = _snapshot(inputs or [], cwd, required=True)
    prior_artifacts = _snapshot(artifacts or [], cwd, required=False)
    if bool(parent) != bool(relation) or relation not in {None, "rerun", "variant"}:
        raise ValueError("Parent and rerun/variant relation must be supplied together")
    if parent:
        previous = read(root, parent)["events"][0]
        if relation == "rerun" and (previous["argv"] != argv or previous["cwd"] != str(cwd)
                                   or previous["inputs"] != input_records):
            raise ValueError("Rerun argv, cwd and input hashes must match; use variant for changes")
    # Create parents without modifying permissions outside the private store.
    root.parent.mkdir(parents=True, exist_ok=True)
    _private_directory(root)
    experiment_id = uuid.uuid4().hex
    directory = root / experiment_id
    directory.mkdir(mode=0o700)
    directory.chmod(0o700)
    with _create(directory / "events.jsonl"):
        pass
    started = time.monotonic()
    _append(directory / "events.jsonl", {
        "event": "attempted", "schema_version": 1, "started_at": _now(), "argv": argv,
        "cwd": str(cwd), "inputs": input_records, "artifacts_before": prior_artifacts,
        "parent": parent, "relation": relation, "timeout_seconds": timeout,
        "output_limit_per_stream": output_limit,
        "environment": "inherited; not recorded; reproducibility not guaranteed",
    })
    _directory_sync(directory)
    _directory_sync(root)
    process = None
    status, exit_code, logs, error_type = "failed", None, {}, None
    try:
        with _interrupt_on_sigterm():
            process = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, start_new_session=True, shell=False)
            timed_out, logs = _capture(process, directory, timeout, output_limit)
            exit_code = process.wait(timeout=5)
            status = "timed_out" if timed_out else "succeeded" if exit_code == 0 else "failed"
    except BaseException as exc:
        if process is not None:
            _stop(process)
            exit_code = process.returncode
        status = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit, _Interrupted)) else "failed"
        error_type = type(exc).__name__
    finally:
        if process is not None:
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
    try:
        after = _snapshot(artifacts or [], cwd, required=False)
        for old, new in zip(prior_artifacts, after):
            new["changed_since_start"] = old != new
    except (ValueError, OSError) as exc:
        after = []
        error_type = type(exc).__name__
        status = "recording_failed"
    _append(directory / "events.jsonl", {
        "event": "finished", "ended_at": _now(), "elapsed_seconds": time.monotonic() - started,
        "status": status, "exit_code": exit_code, "logs": logs, "artifacts_after": after,
        "error_type": error_type,
        "claim_scope": "Process exit status only; no assertion of scientific correctness or artifact validity.",
    })
    return read(root, experiment_id)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/experiments"))
    commands = parser.add_subparsers(dest="action", required=True)
    show = commands.add_parser("show", help="read private metadata; never execute")
    show.add_argument("id")
    commands.add_parser("list", help="list IDs and process statuses only")
    launch = commands.add_parser("run", help="execute explicit argv and retain private evidence")
    launch.add_argument("--cwd", type=Path)
    launch.add_argument("--input", action="append", default=[])
    launch.add_argument("--artifact", action="append", default=[])
    launch.add_argument("--parent")
    launch.add_argument("--relation", choices=["rerun", "variant"])
    launch.add_argument("--timeout", type=float, default=300)
    launch.add_argument("--output-limit", type=int, default=1024 * 1024)
    launch.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.action == "show":
        print(json.dumps(read(args.root, args.id), indent=2))
        return 0
    if args.action == "list":
        records = []
        for path in sorted(_root(args.root).glob("*/events.jsonl")):
            record = read(args.root, path.parent.name)
            records.append({"id": record["id"], "status": record["status"]})
        print(json.dumps(records, indent=2))
        return 0
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    report = run(command, root=args.root, cwd=args.cwd, inputs=args.input, artifacts=args.artifact,
                 parent=args.parent, relation=args.relation, timeout=args.timeout, output_limit=args.output_limit)
    print(json.dumps({"id": report["id"], "status": report["status"]}))
    return 0 if report["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
