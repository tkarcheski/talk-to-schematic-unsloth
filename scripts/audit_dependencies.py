"""Audit frozen registry dependencies without installing or changing packages."""
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib

from ai_pilled.dependencies import audit
from ai_pilled.json_data import loads
from ai_pilled.python_dependencies import parse_audit
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


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    try:
        verify_vcs_tool(root)
        exported = subprocess.run(
            ['uv', 'export', '--locked', '--no-emit-project', '--no-emit-package', 'ai-pilled',
             '--no-hashes', '--no-header', '--format', 'requirements-txt'],
            cwd=root, check=True, capture_output=True, text=True, timeout=60,
        )
        pins = registry_pins(exported.stdout)
        with tempfile.TemporaryDirectory(prefix='schematic-dependency-audit-') as directory:
            requirements = Path(directory) / 'requirements.txt'
            requirements.write_text(''.join(f'{name}=={version}\n' for name, version in sorted(pins.items())))
            result = subprocess.run(
                [sys.executable, '-m', 'pip_audit', '-r', str(requirements), '--no-deps', '--disable-pip',
                 '--strict', '--format', 'json', '--progress-spinner', 'off', '--desc', 'off', '--aliases', 'off'],
                cwd=root, capture_output=True, text=True, timeout=180,
            )
        if result.returncode not in (0, 1):
            raise ValueError('Python advisory service failed; audit is incomplete')
        findings = parse_audit(loads(result.stdout), pins)
        if result.returncode != bool(findings):
            raise ValueError('Python audit exit status contradicts its evidence')
        print(f'Python registry packages audited: {len(pins)}; vulnerabilities: {len(findings)}')
        print('Pinned ai-pilled Git source requires source review; not covered by registry advisories.')
        print('GPU extras outside uv.lock require a separate qualified environment audit.')
        for _, message in findings:
            print(message)
        npm = audit(root)
        print(f'npm dependency audit: {npm.status}')
        for finding in npm.findings:
            print(finding.message)
        return int(bool(findings) or npm.status != 'pass')
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f'Dependency audit incomplete ({type(error).__name__}); check lock consistency and advisory connectivity.', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
