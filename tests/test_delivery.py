"""Delivery checks must bind the actual proposal to a current base."""
# cspell:words delenv NOSYSTEM
import os
from pathlib import Path
import subprocess

import pytest
import yaml

from scripts import check_delivery


def git(repo, *arguments):
    return subprocess.run(
        ['git', *arguments], cwd=repo, check=True, capture_output=True, text=True, timeout=10,
    ).stdout.strip()


def stage(repo, name='feature.py'):
    (repo / name).write_text('value = 2\n')
    git(repo, 'add', '--', name)


def commit(repo, name='feature.py'):
    stage(repo, name)
    git(repo, 'commit', '-m', 'test: add fixture change')
    return git(repo, 'rev-parse', 'HEAD')


@pytest.fixture
def repo(tmp_path, monkeypatch):
    # Never inherit the real hook's index, repository, credentials, or signing configuration.
    for name in list(os.environ):
        if name.startswith('GIT_') or name in {'PRE_COMMIT_TO_REF', 'PRE_COMMIT_REMOTE_NAME'}:
            monkeypatch.delenv(name, raising=False)
    for name, value in {
        'GIT_CONFIG_GLOBAL': '/dev/null', 'GIT_CONFIG_NOSYSTEM': '1', 'GIT_TERMINAL_PROMPT': '0',
        'GIT_AUTHOR_NAME': 'Delivery Test', 'GIT_COMMITTER_NAME': 'Delivery Test',
        'GIT_AUTHOR_EMAIL': 'delivery@example.invalid', 'GIT_COMMITTER_EMAIL': 'delivery@example.invalid',
    }.items():
        monkeypatch.setenv(name, value)
    git(tmp_path, 'init', '--bare', '--initial-branch=main', 'remote.git')
    git(tmp_path, 'clone', 'remote.git', 'work')
    work = tmp_path / 'work'
    commit(work, 'project.py')
    git(work, 'push', 'origin', 'main')
    git(work, 'switch', '-c', 'feature')
    monkeypatch.chdir(work)
    return work


def advance_main(repo):
    peer = repo.parent / 'peer'
    git(repo.parent, 'clone', str(repo.parent / 'remote.git'), str(peer))
    revision = commit(peer, 'main_update.py')
    git(peer, 'push', 'origin', 'main')
    return revision


@pytest.mark.parametrize('ci', [False, True])
def test_clean_base_and_main_ci_can_run_checks_without_publishing(repo, monkeypatch, ci):
    if ci:
        monkeypatch.setenv('PRE_COMMIT_TO_REF', git(repo, 'rev-parse', 'HEAD'))
    assert check_delivery.main([]) == 0


def test_substantive_staged_proposal_passes(repo):
    stage(repo)
    stage(repo, 'AGENTS.md')
    assert check_delivery.main([]) == 0


@pytest.mark.parametrize('name', ['AGENTS.md', 'notes.MD', 'notes.markdown', 'line\nbreak.md'])
def test_markdown_only_staged_proposal_fails(repo, name, capsys):
    stage(repo, name)
    assert check_delivery.main([]) == 1
    assert 'Markdown-only' in capsys.readouterr().err


def test_unstaged_code_cannot_disguise_markdown_only_proposal(repo):
    stage(repo, 'AGENTS.md')
    (repo / 'project.py').write_text('value = 3\n')
    assert check_delivery.main([]) == 1


def test_documentation_can_join_an_existing_substantive_branch(repo):
    commit(repo)
    stage(repo, 'AGENTS.md')
    assert check_delivery.main([]) == 0


def test_source_removal_is_a_substantive_change(repo):
    git(repo, 'rm', 'project.py')
    assert check_delivery.main([]) == 0


def test_empty_branch_cannot_be_published(repo, capsys):
    assert check_delivery.main(['--push']) == 1
    assert 'no file changes' in capsys.readouterr().err


def test_commits_with_no_net_proposal_cannot_be_delivered(repo):
    commit(repo)
    git(repo, 'rm', 'feature.py')
    git(repo, 'commit', '-m', 'test: remove fixture change')
    assert check_delivery.main([]) == 1
    assert check_delivery.main(['--push']) == 1


@pytest.mark.parametrize('arguments', [[], ['--push']])
def test_branch_behind_fetched_main_fails(repo, arguments, capsys):
    commit(repo)
    advance_main(repo)
    git(repo, 'fetch', 'origin', 'main')
    assert check_delivery.main(arguments) == 1
    assert 'behind fetched main' in capsys.readouterr().err


def test_push_detects_remote_changes_without_refreshing_refs(repo, capsys):
    commit(repo)
    cached = git(repo, 'rev-parse', 'origin/main')
    assert advance_main(repo) != cached
    assert check_delivery.main(['--push']) == 1
    assert 'Remote main moved' in capsys.readouterr().err
    assert git(repo, 'rev-parse', 'origin/main') == cached


def test_push_of_current_substantive_branch_passes(repo):
    commit(repo)
    assert check_delivery.main(['--push']) == 0


def test_direct_push_url_uses_the_configured_tracking_base(repo, monkeypatch):
    tip = commit(repo)
    monkeypatch.setenv('PRE_COMMIT_REMOTE_NAME', str(repo.parent / 'remote.git'))
    monkeypatch.setenv('PRE_COMMIT_TO_REF', tip)
    assert check_delivery.main(['--push']) == 0


@pytest.mark.parametrize('remove_origin', [False, True])
def test_named_push_remote_uses_its_own_tracking_base(repo, monkeypatch, remove_origin):
    commit(repo)
    advance_main(repo)
    git(repo, 'remote', 'add', 'upstream', str(repo.parent / 'remote.git'))
    git(repo, 'fetch', 'upstream', 'main')
    git(repo, 'merge', '--no-edit', 'upstream/main')
    if remove_origin:
        git(repo, 'remote', 'remove', 'origin')
    monkeypatch.setenv('PRE_COMMIT_REMOTE_NAME', 'upstream')
    monkeypatch.setenv('PRE_COMMIT_TO_REF', git(repo, 'rev-parse', 'HEAD'))
    assert check_delivery.main(['--push']) == 0


def test_pending_main_merge_can_fix_a_stale_branch(repo):
    commit(repo)
    advance_main(repo)
    git(repo, 'fetch', 'origin', 'main')
    git(repo, 'merge', '--no-commit', '--no-ff', 'origin/main')
    assert check_delivery.main([]) == 0
    assert check_delivery.main(['--push']) == 1


@pytest.mark.parametrize('arguments', [[], ['--push']])
def test_actual_source_tip_cannot_hide_behind_a_fresh_checkout(repo, monkeypatch, arguments, capsys):
    stale_tip = commit(repo)
    advance_main(repo)
    git(repo, 'fetch', 'origin', 'main')
    git(repo, 'merge', '--no-edit', 'origin/main')
    monkeypatch.setenv('PRE_COMMIT_TO_REF', stale_tip)
    assert check_delivery.main(arguments) == 1
    assert 'behind fetched main' in capsys.readouterr().err


def test_staged_code_cannot_disguise_a_markdown_only_push(repo):
    commit(repo, 'AGENTS.md')
    stage(repo)
    assert check_delivery.main([]) == 0
    assert check_delivery.main(['--push']) == 1


def test_missing_tracking_base_fails_closed(repo, capsys):
    stage(repo)
    git(repo, 'update-ref', '-d', 'refs/remotes/origin/main')
    assert check_delivery.main([]) == 1
    assert 'cannot verify' in capsys.readouterr().err


def test_unavailable_remote_fails_without_disclosing_paths(repo, capsys):
    commit(repo)
    (repo.parent / 'remote.git').rename(repo.parent / 'unavailable.git')
    assert check_delivery.main(['--push']) == 1
    message = capsys.readouterr().err
    assert 'cannot verify' in message
    assert str(repo.parent) not in message


def test_git_timeout_fails_closed(repo, monkeypatch, capsys):
    def unavailable(*arguments, **options):
        raise subprocess.TimeoutExpired('git', 30)

    monkeypatch.setattr(check_delivery, '_git', unavailable)
    assert check_delivery.main(['--push']) == 1
    assert 'cannot verify' in capsys.readouterr().err


def test_delivery_guards_run_first_and_ci_checks_the_original_tip():
    root = Path(__file__).resolve().parents[1]
    configuration = yaml.safe_load((root / '.pre-commit-config.yaml').read_text())
    commit_hook, push_hook = configuration['repos'][0]['hooks']
    assert commit_hook['id'] == 'delivery-checks'
    assert push_hook['id'] == 'delivery-push-checks'
    assert commit_hook['stages'] == ['pre-commit']
    assert push_hook['stages'] == ['pre-push']
    assert push_hook['entry'].endswith('scripts/check_delivery.py --push')
    for hook in (commit_hook, push_hook):
        assert hook['fail_fast'] and hook['always_run']
        assert hook['pass_filenames'] is False

    workflow = yaml.safe_load((root / '.github/workflows/checks.yml').read_text())
    steps = workflow['jobs']['cpu']['steps']
    assert steps[0]['with']['fetch-depth'] == 0
    checks = steps[2]['run']
    assert 'PRE_COMMIT_TO_REF=${{ github.event.pull_request.head.sha || github.sha }}' in checks
    assert 'for hook in delivery-checks ' in checks
