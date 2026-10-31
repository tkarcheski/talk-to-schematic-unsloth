"""Audit every registry pin in the Python and npm locks without installing fixes."""

import json
from pathlib import Path
import re
import sys
import tempfile
import tomllib

from ai_pilled.dependencies import audit
from ai_pilled.pipeline import PipelineReport, combine
from ai_pilled.python_dependencies import audit_python
from ai_pilled.runtime import CommandError
from ai_pilled.state import record
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def registry_pins(export: str) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in export.splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        requirement = Requirement(line)
        if requirement.url or requirement.extras:
            raise ValueError('Audit export contains unsupported non-registry dependencies')
        if requirement.marker and not requirement.marker.evaluate():
            continue
        specifiers = list(requirement.specifier)
        if len(specifiers) != 1 or specifiers[0].operator != '==' or '*' in specifiers[0].version:
            raise ValueError('Audit dependencies must have exact versions')
        name, version = canonicalize_name(requirement.name), specifiers[0].version
        if name in pins and pins[name] != version:
            raise ValueError('Audit export contains conflicting package versions')
        pins[name] = version
    if not pins:
        raise ValueError('Audit export is empty')
    return pins


def verify_vcs_tool(root: Path) -> None:
    project = tomllib.loads((root / 'pyproject.toml').read_text())
    configured = project['tool']['uv']['sources']['ai-pilled']
    lock = tomllib.loads((root / 'uv.lock').read_text())
    vcs = [p for p in lock['package'] if 'git' in p['source']]
    if len(vcs) != 1 or vcs[0]['name'] != 'ai-pilled':
        raise ValueError('Review new VCS dependencies before excluding them from registry audits')
    revision = configured['rev']
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('The VCS audit tool must use an immutable revision')
    if vcs[0]['source']['git'].split('#')[-1] != revision:
        raise ValueError('VCS audit tool revision differs from the reviewed configuration')


def python_pins(lock: bytes) -> str:
    """Reject source types without advisory coverage except the two explicit sources."""
    packages = tomllib.loads(lock.decode())["package"]
    pins: dict[str, str] = {}
    exceptions: set[str] = set()
    for package in packages:
        name, version, source = package["name"], package["version"], package["source"]
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", name) or not re.fullmatch(r"[0-9][a-zA-Z0-9.!+_-]*", version):
            raise ValueError("Unsupported package identity")
        if name in pins or name in exceptions:
            raise ValueError("Duplicate or conflicting lock entries require explicit audit support")
        if source == {"registry": "https://pypi.org/simple"}:
            pins[name] = version
        elif name == "talk-to-schematic" and source == {"editable": "."}:
            exceptions.add(name)
        elif name == "ai-pilled" and set(source) == {"git"} and re.fullmatch(
            r"https://github.com/tkarcheski/ai-pilled\.git\?rev=([0-9a-f]{40})#\1", source["git"]
        ):
            exceptions.add(name)
        else:
            raise ValueError("Unsupported dependency source requires explicit audit support")
    if not pins or exceptions != {"ai-pilled", "talk-to-schematic"}:
        raise ValueError("Expected registry dependencies and the two reviewed source exceptions")
    return "".join(f"{name}=={version}\n" for name, version in sorted(pins.items()))


def dependency_audit(root: Path) -> PipelineReport:
    result = PipelineReport("locked-dependencies")
    try:
        lock = (root / "uv.lock").read_bytes()
        pins = python_pins(lock)
        private = root / ".ai-pilled"
        if private.is_symlink():
            raise ValueError("Audit report directory must not be a symlink")
        private.mkdir(mode=0o700, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="dependencies-", dir=private) as temporary:
            requirements = Path(temporary) / "requirements.txt"
            requirements.write_text(pins)
            requirements.chmod(0o600)
            combine(result, audit_python(root, [str(requirements.relative_to(root))],
                                        executable=str(Path(sys.executable).with_name("pip-audit"))))
        combine(result, audit(root))
        if (root / "uv.lock").read_bytes() != lock:
            raise ValueError("Python lock changed during the audit")
    except (CommandError, OSError, ValueError, KeyError, TypeError):
        result.add("dependency-audit-unavailable", "Cannot audit the complete locked dependency set.",
                   severity="warning")
    record(root, result, "locked-dependencies")
    return result


def main() -> int:
    result = dependency_audit(Path.cwd())
    print(json.dumps({"check": result.check, "status": result.status}))
    return 0 if result.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
