import json
from types import SimpleNamespace

import pytest

from scripts.research_scheduler import Ledger, run_once, validate
from scripts.public_research import fetch_public, studio_recommend


def config(tmp_path):
    return {'budgets': {'research': 60, 'training': 120, 'proposal': 60}, 'reserve_mb': 512,
            'adapters': {'test': {'argv': ['/bin/true'], 'cwd': str(tmp_path), 'kind': 'training',
                                  'ready': True, 'gpu_mb': 1024, 'cooperative': True}}}


def test_busy_gpu_never_launches(tmp_path):
    ledger = Ledger(tmp_path / 'jobs.db')
    ledger.submit('training', 'test')
    assert run_once(ledger, config(tmp_path), probe=lambda: (24000, True), runner=lambda *a, **k: pytest.fail('launched')) == 'blocked'
    assert ledger.jobs()[0]['reason'] == 'GPU handoff/resource budget required'


def test_checkpoint_and_privacy(tmp_path, monkeypatch):
    monkeypatch.setenv('PRIVATE_TOKEN', 'secret')
    ledger = Ledger(tmp_path / 'jobs.db')
    ledger.submit('training', 'test')
    def runner(argv, **kwargs):
        assert argv == ['/bin/true']
        assert 'PRIVATE_TOKEN' not in kwargs['env']
        assert 'SCHEMATIC_DEADLINE' in kwargs['env']
        assert 'shell' not in kwargs
        return SimpleNamespace(returncode=75)
    assert run_once(ledger, config(tmp_path), probe=lambda: (24000, False), runner=runner) == 'queued'
    assert 'secret' not in json.dumps(ledger.jobs())


def test_stale_running_blocks(tmp_path):
    ledger = Ledger(tmp_path / 'jobs.db')
    job = ledger.submit('training', 'test')
    ledger.update(job, 'running')
    assert run_once(ledger, config(tmp_path)).startswith('blocked')


def test_unknown_adapter_and_advisor_cannot_execute(tmp_path):
    ledger = Ledger(tmp_path / 'jobs.db')
    ledger.submit('training', 'unknown')
    assert run_once(ledger, config(tmp_path)) == 'blocked'


def test_config_and_public_boundaries(tmp_path):
    cfg = config(tmp_path)
    cfg['adapters']['test']['argv'] = ['sh', '-c', 'echo unsafe']
    with pytest.raises(ValueError):
        validate(cfg)
    with pytest.raises(ValueError):
        fetch_public({'url': 'http://127.0.0.1/private', 'license': 'CC0-1.0', 'license_url': 'https://example.com', 'public': True}, ['example.com'])
    with pytest.raises(ValueError):
        studio_recommend('https://external.ai', 'model', [])


def test_real_noop_and_dashboard(tmp_path):
    from scripts.training_dashboard import queue_snapshot
    ledger = Ledger(tmp_path / 'jobs.db')
    ledger.submit('training', 'test')
    cfg = config(tmp_path)
    cfg['adapters']['test']['gpu_mb'] = 0
    assert run_once(ledger, cfg) == 'completed'
    assert queue_snapshot(ledger.path)[0]['status'] == 'completed'


def test_single_worker_lock(tmp_path):
    import fcntl
    ledger = Ledger(tmp_path / 'jobs.db')
    ledger.submit('training', 'test')
    with open(str(ledger.path) + '.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert run_once(ledger, config(tmp_path)) == 'busy'


def test_concrete_collection_proposal(tmp_path, monkeypatch):
    from scripts import research_step
    source = {'url': 'https://example.com/board.pdf', 'license': 'CC-BY-4.0',
              'license_url': 'https://example.com/license', 'public': True}
    monkeypatch.setattr(research_step, 'fetch_public', lambda source, hosts: (b'public schematic', {**source, 'sha256': 'a' * 64, 'bytes': 16}))
    assert research_step.collect({'allowed_hosts': ['example.com'], 'sources': [source]}, tmp_path) == 0
    proposal = json.loads((tmp_path / 'proposal.json').read_text())
    assert proposal['ready_for_training'] is False
    assert proposal['sources'][0]['review_status'].startswith('needs-human')


def test_local_agent_search_has_public_only_context(tmp_path, monkeypatch):
    from scripts import research_agent_step as agent
    calls = []
    def request(endpoint, path, payload=None, **kwargs):
        calls.append((endpoint, path, payload))
        if path == '/v1/models':
            return {'data': [{'id': 'resident'}]}
        if payload:
            assert payload['messages'][1]['content'] == 'open multi-page motor controller'
            return {'choices': [{'message': {'content': 'open motor controller schematic'}}]}
        return {'results': [{'url': 'https://example.org/board.pdf'}, {'url': 'http://127.0.0.1/private'}]}
    monkeypatch.setattr(agent, 'local_json', request)
    plan = {'public_topic': 'open multi-page motor controller', 'model': 'resident', 'model_endpoint': 'http://127.0.0.1:8892', 'search_endpoint': 'http://127.0.0.1:8894', 'allowed_hosts': ['example.org'], 'allow_public_search': True, 'backend': 'native-unsloth'}
    proposal = agent.research(plan)
    assert len(proposal['candidates']) == 1
    assert proposal['candidates'][0]['license'] is None
    assert not proposal['ready_for_training']
    assert len(calls) == 3


def test_publication_requires_review_before_any_git(tmp_path):
    from scripts.dataset_proposal_pr import publish
    folder = tmp_path / 'data/public-proposals/review'
    folder.mkdir(parents=True)
    manifest = folder / 'proposal.json'
    manifest.write_text(json.dumps({'public_sources_only': True, 'publication_approved': False}))
    with pytest.raises(ValueError, match='publication'):
        publish(tmp_path, manifest, 'research/test', 'owner/repo', run=lambda *a, **k: pytest.fail('git called'))


def test_draft_pr_stages_only_hashed_public_sources(tmp_path):
    import hashlib
    from scripts.dataset_proposal_pr import publish
    folder = tmp_path / 'data/public-proposals/review'
    folder.mkdir(parents=True)
    source = folder / 'board.pdf'
    source.write_bytes(b'public')
    manifest = folder / 'proposal.json'
    manifest.write_text(json.dumps({'public_sources_only': True, 'publication_approved': True, 'sources': [
        {'url': 'https://example.org/board.pdf', 'license_url': 'https://example.org/license', 'public': True, 'license': 'CC-BY-4.0', 'review_status': 'approved-for-publication', 'file': 'board.pdf', 'sha256': hashlib.sha256(b'public').hexdigest()}]}))
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        if argv[:2] == ['git', 'show']:
            return SimpleNamespace(stdout=(tmp_path / argv[2][1:]).read_bytes())
        output = ''
        if argv[:2] == ['git', 'rev-parse']:
            output = 'a' * 40
        if argv[:3] == ['git', 'remote', 'get-url']:
            output = 'https://github.com/owner/repo.git'
        if argv[:4] == ['git', 'diff', '--cached', '--name-only']:
            output = '\n'.join(str(p.relative_to(tmp_path)) for p in (manifest, source))
        return SimpleNamespace(stdout=output)
    publish(tmp_path, manifest, 'research/test', 'owner/repo', run=run)
    assert '--draft' in calls[-1]
    assert all('merge' not in c and 'close' not in c for c in calls)


def test_gpu_blocked_does_not_starve_cpu_and_retries(tmp_path):
    ledger = Ledger(tmp_path / 'jobs.db')
    gpu_job = ledger.submit('training', 'test')
    cfg = config(tmp_path)
    cfg['adapters']['cpu'] = {**cfg['adapters']['test'], 'kind': 'research', 'gpu_mb': 0}
    ledger.submit('research', 'cpu')
    assert run_once(ledger, cfg, probe=lambda: (24000, True)) == 'completed'
    assert ledger.jobs()[0]['id'] == gpu_job
    assert ledger.jobs()[0]['status'] == 'blocked'
    assert run_once(ledger, cfg, probe=lambda: (24000, False)) == 'completed'


def test_research_resumes_without_refetch(tmp_path, monkeypatch):
    import hashlib
    from scripts import research_step
    source = {'url': 'https://example.com/board.pdf', 'license': 'CC-BY-4.0', 'license_url': 'https://example.com/license', 'public': True}
    monkeypatch.setattr(research_step, 'fetch_public', lambda source, hosts: (b'public', {**source, 'sha256': hashlib.sha256(b'public').hexdigest(), 'bytes': 6}))
    plan = {'allowed_hosts': ['example.com'], 'sources': [source]}
    assert research_step.collect(plan, tmp_path) == 0
    monkeypatch.setattr(research_step, 'fetch_public', lambda *a: pytest.fail('fetched again'))
    assert research_step.collect(plan, tmp_path) == 0


def test_training_slice_binds_dataset_and_resumes(tmp_path):
    from scripts.training_slice import training_argv
    data = tmp_path / 'data'
    data.mkdir()
    for name in ('train.jsonl', 'val.jsonl'):
        (data / name).write_text('{}\n')
    out = tmp_path / 'out'
    plan = {'data': str(data), 'out': str(out), 'model': str(tmp_path / 'model'), 'max_steps': 10, 'save_steps': 2, 'max_seq': 4096, 'rank': 16, 'gradient_accumulation': 4}
    assert '--resume' not in training_argv(plan)
    checkpoint = out / 'checkpoint-2'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'trainer_state.json').write_text('{}')
    assert training_argv(plan)[-2:] == ['--resume', str(checkpoint)]
    (data / 'train.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='different'):
        training_argv(plan)


def test_co_resident_requires_identity_and_memory_cap():
    from scripts.research_scheduler import unexpected_residents
    resident = {'pid': 123, 'starttime': '456', 'exe': '/usr/bin/trusted', 'used_mb': 142}
    approved = [{**resident, 'max_mb': 200}]
    assert not unexpected_residents([resident], approved)
    assert unexpected_residents([{**resident, 'starttime': '999'}], approved)
    assert unexpected_residents([{**resident, 'used_mb': 201}], approved)
    assert unexpected_residents([{**resident, 'pid': 999}], approved)


def test_training_continuation_cli_contract(tmp_path):
    from scripts.training_slice import training_argv
    data = tmp_path / 'data'
    data.mkdir()
    for name in ('train.jsonl', 'val.jsonl'):
        (data / name).write_text('{}\n')
    out = tmp_path / 'fresh-output'
    plan = {'data': str(data), 'out': str(out), 'model': str(tmp_path / 'adapter'), 'max_steps': 10, 'save_steps': 2, 'max_seq': 4096, 'rank': 16, 'gradient_accumulation': 4, 'continue_adapter': True}
    argv = training_argv(plan)
    assert argv[-1] == '--continue-adapter'
    assert '--no-finetune-vision' in argv
    assert argv[argv.index('--lr') + 1] == '5e-05'
    assert not out.exists()


def test_native_owned_research_and_advice_no_server(tmp_path, monkeypatch):
    import sys
    from scripts import research_agent_step as agent
    loaded = []
    next_id = 'a' * 32
    class Model:
        def __init__(self, config):
            loaded.append(config)
        def complete(self, model, messages, *, max_tokens):
            assert model == 'schematic'
            return {'output': 'public schematic' if max_tokens == 96 else next_id}
    monkeypatch.setitem(sys.modules, 'schematic_model.deployment', SimpleNamespace(
        DeploymentConfig=lambda **kwargs: kwargs, LocalModel=Model))
    monkeypatch.setenv('SCHEMATIC_QUEUE_METADATA', json.dumps([{'id': next_id, 'kind': 'training'}]))
    monkeypatch.setattr(agent, 'local_json', lambda endpoint, path: {'results': [{'url': 'https://example.org/board.pdf'}]})
    plan = {'public_topic': 'public motor schematic', 'model': '/local/model', 'model_endpoint': 'unused',
            'search_endpoint': 'http://127.0.0.1:8894', 'allowed_hosts': ['example.org'],
            'allow_public_search': True, 'backend': 'native-owned'}
    result = agent.research(plan)
    assert len(loaded) == 1
    assert loaded[0]['model'] == '/local/model'
    assert result['next_job_id'] == next_id
    assert len(result['candidates']) == 1


def test_worker_is_finite_on_idle_and_checkpoint_slices(tmp_path):
    from scripts.research_scheduler import worker
    calls = []
    def idle(*args, **kwargs):
        calls.append(1)
        return 'blocked'
    slept = []
    result = worker(None, {}, max_jobs=3, max_idle_polls=2, idle_seconds=1, dispatch=idle, sleep=slept.append)
    assert len(calls) == 3 and slept == [1, 1] and result['slices'] == 0
    result = worker(None, {}, max_jobs=3, dispatch=lambda *a, **k: 'queued', sleep=lambda _: pytest.fail('sleep'))
    assert result['slices'] == 3


def test_persisted_advice_revalidated_and_consumed(tmp_path):
    ledger = Ledger(tmp_path / 'jobs.db')
    cfg = config(tmp_path)
    cfg['adapters']['test']['gpu_mb'] = 0
    first = ledger.submit('training', 'test')
    second = ledger.submit('training', 'test')
    third = ledger.submit('training', 'test')
    def runner(argv, **kwargs):
        assert kwargs['env']['SCHEMATIC_JOB_ID'] == first
        from pathlib import Path
        Path(kwargs['env']['SCHEMATIC_RESULT_PATH']).write_text(json.dumps({'next_job_id': third, 'private_text': 'never retained'}))
        return SimpleNamespace(returncode=0)
    assert run_once(ledger, cfg, runner=runner) == 'completed'
    def next_runner(argv, **kwargs):
        assert kwargs['env']['SCHEMATIC_JOB_ID'] == third
        return SimpleNamespace(returncode=0)
    assert run_once(ledger, cfg, runner=next_runner) == 'completed'
    assert ledger.db.execute('SELECT * FROM scheduler_meta').fetchall() == []
    assert ledger.jobs()[1]['id'] == second and ledger.jobs()[1]['status'] == 'queued'
    assert 'private_text' not in json.dumps(ledger.jobs())


def test_module_cli_real_subprocess_imports(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for module in ('scripts.research_agent_step', 'scripts.training_slice', 'scripts.research_scheduler'):
        result = subprocess.run([sys.executable, '-m', module, '--help'], cwd=root, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    result = subprocess.run([sys.executable, '-c', 'from schematic_model.deployment import LocalModel, DeploymentConfig'], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_sanitized_failure_report(tmp_path, monkeypatch):
    from scripts.public_research import report_failure
    path = tmp_path / 'result.json'
    monkeypatch.setenv('SCHEMATIC_RESULT_PATH', str(path))
    assert report_failure(ValueError('private secret content')) == 'invalid_configuration'
    assert json.loads(path.read_text()) == {'error_code': 'invalid_configuration'}
    assert 'private' not in path.read_text()


def test_followed_run_discovers_current_log_and_preserves_override(tmp_path):
    import os
    from scripts.training_dashboard import progress_log, progress
    run_a = tmp_path / 'outputs' / 'first'
    run_b = tmp_path / 'outputs' / 'second'
    run_a.mkdir(parents=True)
    run_b.mkdir()
    results = tmp_path / 'results'
    results.mkdir()
    first = results / 'first.log'
    first.write_text('40/68 [02:00<01:24, 3.00s/it]')
    assert progress_log(run_b) is None
    second = results / 'second.log'
    second.write_text('51/68 [02:33<00:51, 3.00s/it]')
    assert progress_log(run_a) == first
    assert progress(progress_log(run_b), 68)['step'] == 51
    resumed = results / 'second-resume.log'
    resumed.write_text('60/68 [03:00<00:24, 3.00s/it]')
    os.utime(resumed, (second.stat().st_mtime + 2, second.stat().st_mtime + 2))
    assert progress_log(run_b) == resumed
    assert progress_log(run_b, first) == first
