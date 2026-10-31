"""Reject stale bases and empty or Markdown-only delivery proposals."""
# cspell:words surrogateescape
import argparse
import os
from pathlib import Path
import subprocess
import sys


def _git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ['git', *arguments], check=check, capture_output=True, text=True,
        encoding='utf-8', errors='surrogateescape', timeout=30,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--push', action='store_true', help='Verify the live remote and committed proposal')
    args = parser.parse_args(argv)
    try:
        remote = os.environ.get('PRE_COMMIT_REMOTE_NAME', 'origin') if args.push else 'origin'
        tracking_remote = remote if remote in _git('remote').stdout.splitlines() else 'origin'
        base = _git('rev-parse', '--verify', f'refs/remotes/{tracking_remote}/main^{{commit}}').stdout.strip()
        # CI and pre-push supply the actual source tip, not a synthetic merge or another checkout's HEAD.
        explicit_tip = os.environ.get('PRE_COMMIT_TO_REF')
        tip = _git('rev-parse', '--verify', '--end-of-options',
                   f'{explicit_tip or "HEAD"}^{{commit}}').stdout.strip()
        if args.push:
            advertised = _git('ls-remote', '--exit-code', '--', remote, 'refs/heads/main').stdout.split()
            if advertised != [base, 'refs/heads/main']:
                raise ValueError('Remote main moved. Fetch and integrate main, then rerun checks before pushing.')

        parents = [tip]
        if not args.push and not explicit_tip:
            merge = _git('rev-parse', '--verify', '--quiet', 'MERGE_HEAD^{commit}', check=False)
            if merge.returncode == 0:
                parents.append(merge.stdout.strip())
            elif merge.returncode != 1:
                merge.check_returncode()
        for parent in parents:
            ancestry = _git('merge-base', '--is-ancestor', base, parent, check=False)
            if ancestry.returncode == 0:
                break
            if ancestry.returncode != 1:
                ancestry.check_returncode()
        else:
            raise ValueError('Branch is behind fetched main. Integrate origin/main and rerun checks.')

        comparison = [base, tip] if args.push or explicit_tip else ['--cached', base]
        paths = _git('diff', '--no-renames', '--name-only', '-z', *comparison, '--').stdout.split('\0')
        paths = [path for path in paths if path]
        if not paths:
            if not args.push and tip == base:
                print('Delivery checks passed on the current base.')
                return 0
            raise ValueError('Proposal has no file changes against main. Do not publish an empty or duplicate PR.')
        if paths and all(Path(path).suffix.lower() in {'.md', '.markdown'} for path in paths):
            raise ValueError('Markdown-only proposal. Bundle documentation with relevant substantive work in one PR.')
    except ValueError as error:
        print(f'Delivery guard: {error}', file=sys.stderr)
        return 1
    except (OSError, subprocess.SubprocessError):
        print('Delivery guard: cannot verify Git metadata or remote state. Fetch the base and retry; do not bypass checks.',
              file=sys.stderr)
        return 1
    print('Delivery checks passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
