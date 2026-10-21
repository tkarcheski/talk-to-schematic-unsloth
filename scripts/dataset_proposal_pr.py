"""Explicit public-manifest draft PR adapter. Never merges or closes PRs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit

try:
    from .public_research import ALLOWED_LICENSES
except ImportError:
    from public_research import ALLOWED_LICENSES


def publish(repo, manifest, branch, github_repo, *, run=subprocess.run):
    repo = Path(repo).resolve()
    manifest = Path(manifest).resolve()
    if not re.fullmatch(r'research/[a-z0-9][a-z0-9-]{1,60}', branch):
        raise ValueError('dedicated research branch required')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', github_repo):
        raise ValueError('explicit GitHub repository required')
    root = repo / 'data' / 'public-proposals'
    if not manifest.is_relative_to(root) or manifest.is_symlink():
        raise ValueError('manifest must be under data/public-proposals')
    proposal = json.loads(manifest.read_text())
    if proposal.get('public_sources_only') is not True or proposal.get('publication_approved') is not True:
        raise ValueError('explicit reviewed public publication required')
    for key in ('schema', 'requested_sources'):
        if key in proposal and (type(proposal[key]) is not int or not 0 <= proposal[key] <= 8):
            raise ValueError('invalid numeric manifest metadata')
    if 'ready_for_training' in proposal and type(proposal['ready_for_training']) is not bool:
        raise ValueError('invalid training readiness metadata')
    sources = proposal.get('sources', [])
    if not 1 <= len(sources) <= 8:
        raise ValueError('bounded reviewed source list required')
    paths = [manifest]
    reviewed_hashes = {str(manifest.relative_to(repo)): hashlib.sha256(manifest.read_bytes()).hexdigest()}
    for source in sources:
        if source.get('review_status') != 'approved-for-publication' or source.get('license') not in ALLOWED_LICENSES or source.get('public') is not True:
            raise ValueError('source license and public review required')
        for field in ('url', 'license_url'):
            target = urlsplit(source.get(field, ''))
            if target.scheme != 'https' or not target.hostname or target.username or target.password or target.query or target.fragment:
                raise ValueError('public HTTPS provenance required')
        if 'bytes' in source and (type(source['bytes']) is not int or not 0 <= source['bytes'] <= 8_000_000):
            raise ValueError('invalid byte count')
        if not isinstance(source.get('file'), str) or not re.fullmatch(r'[a-zA-Z0-9_.-]{1,128}', source['file']):
            raise ValueError('invalid source filename')
        if not isinstance(source.get('sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', source['sha256']):
            raise ValueError('invalid hash')
        path = (manifest.parent / source['file']).resolve()
        if not path.is_relative_to(manifest.parent) or path == manifest:
            raise ValueError('source escapes proposal directory')
        if hashlib.sha256(path.read_bytes()).hexdigest() != source['sha256']:
            raise ValueError('reviewed source hash mismatch')
        paths.append(path)
        reviewed_hashes[str(path.relative_to(repo))] = source['sha256']
    # Do not transmit arbitrary additional manifest keys/private annotations.
    allowed = {'schema', 'public_sources_only', 'publication_approved', 'sources', 'requested_sources', 'ready_for_training'}
    source_allowed = {'url', 'license', 'license_url', 'public', 'sha256', 'bytes', 'file', 'review_status'}
    if set(proposal) - allowed or any(set(s) - source_allowed for s in sources):
        raise ValueError('unreviewed manifest fields')
    def command(argv):
        return run(argv, cwd=repo, check=True, capture_output=True, text=True).stdout.strip()
    status = command(['git', 'status', '--porcelain', '--untracked-files=no'])
    if status:
        raise ValueError('tracked working tree/index must be clean')
    remote = command(['git', 'remote', 'get-url', 'origin'])
    if remote not in (f'https://github.com/{github_repo}.git', f'git@github.com:{github_repo}.git'):
        raise ValueError('origin does not match explicit repository')
    # Branch only from the already-published public baseline: never transmit unrelated local ancestors.
    head = command(['git', 'rev-parse', 'HEAD'])
    base = command(['git', 'rev-parse', 'refs/remotes/origin/main'])
    if not head or head != base:
        raise ValueError('checkout must be exactly the approved public origin/main base')
    command(['git', 'switch', '-c', branch])
    relative = [str(p.relative_to(repo)) for p in paths]
    command(['git', 'add', '--', *relative])
    staged = command(['git', 'diff', '--cached', '--name-only']).splitlines()
    if set(staged) != set(relative):
        raise ValueError('staged paths differ from reviewed public proposal; inspect index')
    for relative_path, digest in reviewed_hashes.items():
        blob = run(['git', 'show', ':' + relative_path], cwd=repo, check=True, capture_output=True).stdout
        if hashlib.sha256(blob).hexdigest() != digest:
            raise ValueError('staged bytes differ from reviewed content; inspect index')
    command(['git', 'diff', '--cached', '--check'])
    command(['git', 'commit', '-m', 'data: propose reviewed public schematic sources'])
    if command(['git', 'rev-parse', 'HEAD^']) != base:
        raise ValueError('unexpected commit ancestry; refusing publication')
    command(['git', 'push', '-u', 'origin', branch])
    return command(['gh', 'pr', 'create', '--repo', github_repo, '--head', branch, '--draft',
                    '--title', 'Public schematic dataset proposal', '--body',
                    'Reviewed public source proposal with hashes and license provenance. Requires dataset and training evaluation before merge. No private schematics or conversations included.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--branch', required=True)
    parser.add_argument('--github-repo', required=True)
    args = parser.parse_args()
    print(publish(args.repo, args.manifest, args.branch, args.github_repo))


if __name__ == '__main__':
    main()
