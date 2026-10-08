"""Bounded, operator-configured research/training queue. No shell or model switching."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time
import tempfile
import uuid


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = sqlite3.connect(self.path)
        os.chmod(self.path, 0o600)
        self.db.row_factory = sqlite3.Row
        self.db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, kind TEXT, adapter TEXT, status TEXT, created REAL, elapsed REAL DEFAULT 0, reason TEXT DEFAULT "")')
        columns = {row[1] for row in self.db.execute('PRAGMA table_info(jobs)')}
        if 'metrics' not in columns:
            self.db.execute("ALTER TABLE jobs ADD COLUMN metrics TEXT DEFAULT '{}'")
        self.db.execute('CREATE TABLE IF NOT EXISTS scheduler_meta (key TEXT PRIMARY KEY, value TEXT)')
        self.db.commit()

    def jobs(self):
        return [{**dict(row), 'metrics': json.loads(row['metrics'])} for row in self.db.execute('SELECT * FROM jobs ORDER BY created')]

    def submit(self, kind, adapter):
        if kind not in {'research', 'training', 'proposal'}:
            raise ValueError('unknown workload')
        job = uuid.uuid4().hex
        self.db.execute('INSERT INTO jobs(id,kind,adapter,status,created) VALUES(?,?,?,?,?)', (job, kind, adapter, 'queued', time.time()))
        self.db.commit()
        return job

    def update(self, job, status, reason='', elapsed=None):
        self.db.execute('UPDATE jobs SET status=?,reason=?,elapsed=COALESCE(?,elapsed) WHERE id=?', (status, reason, elapsed, job))
        self.db.commit()


def validate(config):
    if set(config) - {'adapters', 'budgets', 'reserve_mb', 'allow_proposals', 'advisor', 'co_residents'}:
        raise ValueError('unknown scheduler setting')
    for kind in ('research', 'training', 'proposal'):
        value = config['budgets'][kind]
        if type(value) is not int or not 1 <= value <= 3600:
            raise ValueError('budgets must be 1..3600 seconds')
    if type(config.get('reserve_mb')) is not int or config['reserve_mb'] < 512:
        raise ValueError('reserve_mb must be at least 512')
    for spec in config['adapters'].values():
        if set(spec) - {'argv', 'cwd', 'kind', 'ready', 'gpu_mb', 'cooperative', 'credential_env'} or not {'argv', 'cwd', 'kind', 'ready', 'gpu_mb', 'cooperative'} <= set(spec):
            raise ValueError('invalid adapter fields')
        if spec.get('credential_env') not in (None, 'UNSLOTH_API_KEY'):
            raise ValueError('only named Studio credential forwarding is supported')
        if spec['kind'] not in config['budgets'] or spec['cooperative'] is not True:
            raise ValueError('adapter must support cooperative deadlines')
        if not isinstance(spec['argv'], list) or not spec['argv'] or not all(isinstance(a, str) and a for a in spec['argv']):
            raise ValueError('argv must be a nonempty string list')
        if not Path(spec['argv'][0]).is_absolute() or not Path(spec['cwd']).is_absolute():
            raise ValueError('executable and cwd must be absolute')
        if type(spec['gpu_mb']) is not int or spec['gpu_mb'] < 0 or type(spec['ready']) is not bool:
            raise ValueError('invalid readiness/resource budget')
    return config


def gpu_state():
    """Fail closed without trustworthy inventory; never unload another resident."""
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5, check=True)
    values = result.stdout.strip().splitlines()
    if len(values) != 1:
        raise ValueError('exactly one GPU required')
    active = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5, check=True)
    residents = []
    for line in active.stdout.splitlines():
        pid_text, used = [part.strip() for part in line.split(',')]
        pid = int(pid_text)
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        residents.append({'pid': pid, 'starttime': stat[19], 'exe': os.readlink(f'/proc/{pid}/exe'), 'used_mb': int(used)})
    return int(values[0]), residents


def unexpected_residents(residents, approved):
    if isinstance(residents, bool):
        return residents
    for resident in residents:
        matches = [a for a in approved if all(a.get(k) == resident[k] for k in ('pid', 'starttime', 'exe'))]
        if not matches or type(matches[0].get('max_mb')) is not int or resident['used_mb'] > matches[0]['max_mb']:
            return True
    return False


def run_once(ledger, config, *, probe=gpu_state, runner=subprocess.run, recommend=None):
    validate(config)
    with open(str(ledger.path) + '.lock', 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 'busy'
        # A stale running record may represent an orphan still using the GPU. Never infer completion.
        if any(j['status'] == 'running' for j in ledger.jobs()):
            return 'blocked: unresolved running job'
        for pending in ledger.jobs():
            if pending['status'] == 'blocked' and pending['reason'] in ('GPU handoff/resource budget required', 'GPU inventory unavailable'):
                ledger.update(pending['id'], 'queued', elapsed=pending['elapsed'])
        queued = [j for j in ledger.jobs() if j['status'] == 'queued']
        if not queued:
            return 'idle'
        # Alternate workload classes by accumulated usage; AI can only choose from this ready set.
        usage = {k: sum(j['elapsed'] for j in ledger.jobs() if j['kind'] == k) / config['budgets'][k] for k in config['budgets']}
        eligible = []
        for job in queued:
            spec = config['adapters'].get(job['adapter'])
            reason = ''
            if not spec or spec['kind'] != job['kind'] or not spec['ready']:
                reason = 'adapter not ready'
            elif job['kind'] == 'proposal' and config.get('allow_proposals') is not True:
                reason = 'publication disabled'
            if reason:
                ledger.update(job['id'], 'blocked', reason)
            else:
                eligible.append(job)
        if not eligible:
            return 'blocked'
        gpu_info = None
        runnable = []
        for candidate in eligible:
            adapter = config['adapters'][candidate['adapter']]
            if adapter['gpu_mb']:
                try:
                    if gpu_info is None:
                        gpu_info = probe()
                    free, occupied = gpu_info
                except (OSError, ValueError, subprocess.SubprocessError):
                    ledger.update(candidate['id'], 'blocked', 'GPU inventory unavailable')
                    continue
                if unexpected_residents(occupied, config.get('co_residents', [])) or free < adapter['gpu_mb'] + config['reserve_mb']:
                    ledger.update(candidate['id'], 'blocked', 'GPU handoff/resource budget required')
                    continue
            runnable.append(candidate)
        if not runnable:
            return 'blocked'
        runnable.sort(key=lambda j: (usage[j['kind']], j['created']))
        job = runnable[0]
        saved = ledger.db.execute("SELECT value FROM scheduler_meta WHERE key='next_job_id'").fetchone()
        if saved:
            job = next((j for j in runnable if j['id'] == saved[0]), job)
            ledger.db.execute("DELETE FROM scheduler_meta WHERE key='next_job_id'")
            ledger.db.commit()
        if recommend:
            chosen = recommend([{'id': j['id'], 'kind': j['kind']} for j in runnable])
            job = next((j for j in runnable if j['id'] == chosen), job)
        spec = config['adapters'][job['adapter']]
        budget = config['budgets'][job['kind']]
        # Minimal environment deliberately excludes API keys, cloud credentials and private prompts.
        env = {k: os.environ[k] for k in ('PATH', 'LANG', 'HOME') if k in os.environ}
        if spec.get('credential_env'):
            credential = os.environ.get(spec['credential_env'])
            if not credential:
                ledger.update(job['id'], 'blocked', 'Studio authentication required', job['elapsed'])
                return 'blocked'
            env[spec['credential_env']] = credential
        env['SCHEMATIC_QUEUE_METADATA'] = json.dumps([{'id': j['id'], 'kind': j['kind']} for j in runnable if j['id'] != job['id']][:100])
        env.update(SCHEMATIC_JOB_ID=job['id'], SCHEMATIC_DEADLINE=str(time.time() + budget), SCHEMATIC_BUDGET_SECONDS=str(budget))
        ledger.update(job['id'], 'running')
        start = time.monotonic()
        result_dir = tempfile.TemporaryDirectory(prefix='schematic-job-')
        result_path = Path(result_dir.name) / 'result.json'
        env['SCHEMATIC_RESULT_PATH'] = str(result_path)
        try:
            result = runner(spec['argv'], cwd=spec['cwd'], env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except OSError:
            result_dir.cleanup()
            ledger.update(job['id'], 'failed', 'adapter launch failed', time.monotonic() - start)
            return 'failed'
        metrics = {}
        if result_path.is_file() and result_path.stat().st_size <= 4096:
            try:
                supplied = json.loads(result_path.read_text())
                try:
                    from .public_research import SAFE_FAILURE_CODES
                except ImportError:
                    from public_research import SAFE_FAILURE_CODES
                if isinstance(supplied.get('error_code'), str) and supplied['error_code'] in SAFE_FAILURE_CODES:
                    metrics['error_code'] = supplied['error_code']
                next_job = supplied.get('next_job_id')
                if isinstance(next_job, str) and next_job in {j['id'] for j in ledger.jobs() if j['status'] == 'queued' and j['id'] != job['id']}:
                    ledger.db.execute("INSERT OR REPLACE INTO scheduler_meta(key,value) VALUES('next_job_id',?)", (next_job,))
                for name in ('checkpoint_step', 'source_count', 'proposal_count'):
                    value = supplied.get(name)
                    if type(value) is int and 0 <= value <= 1_000_000:
                        metrics[name] = value
            except (ValueError, OSError, AttributeError):
                pass
        result_dir.cleanup()
        ledger.db.execute('UPDATE jobs SET metrics=? WHERE id=?', (json.dumps(metrics), job['id']))
        ledger.db.commit()
        elapsed = time.monotonic() - start
        status = 'completed' if result.returncode == 0 else 'queued' if result.returncode == 75 else 'failed'
        reason = 'checkpoint yielded' if result.returncode == 75 else (metrics.get('error_code', f'adapter exit {result.returncode}') if result.returncode else '')
        if elapsed > budget + 30:
            status, reason = 'blocked', 'adapter exceeded cooperative budget; inspect before retry'
        ledger.update(job['id'], status, reason, job['elapsed'] + elapsed)
        return status


def worker(ledger, config, *, max_jobs=10, max_idle_polls=5, idle_seconds=10,
           dispatch=run_once, sleep=time.sleep, recommend=None):
    """Finite worker: checkpoint yields count as slices; idle waits are also bounded."""
    if type(max_jobs) is not int or not 1 <= max_jobs <= 100:
        raise ValueError('max_jobs must be 1..100 slices')
    if type(max_idle_polls) is not int or not 0 <= max_idle_polls <= 100:
        raise ValueError('max_idle_polls must be 0..100')
    if type(idle_seconds) is not int or not 1 <= idle_seconds <= 60:
        raise ValueError('idle_seconds must be 1..60')
    slices = idle = 0
    outcomes = []
    while slices < max_jobs:
        result = dispatch(ledger, config, recommend=recommend)
        outcomes.append(result)
        if result in ('completed', 'queued', 'failed'):
            slices += 1
            idle = 0
        else:
            if idle >= max_idle_polls:
                break
            idle += 1
            sleep(idle_seconds)
    return {'slices': slices, 'outcomes': outcomes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, default=Path('results/research-jobs.sqlite3'))
    parser.add_argument('--config', type=Path)
    sub = parser.add_subparsers(dest='command', required=True)
    submit = sub.add_parser('submit')
    submit.add_argument('kind', choices=['research', 'training', 'proposal'])
    submit.add_argument('adapter')
    sub.add_parser('status')
    sub.add_parser('run-once')
    work = sub.add_parser('worker')
    work.add_argument('--max-jobs', type=int, default=10)
    work.add_argument('--max-idle-polls', type=int, default=5)
    work.add_argument('--idle-seconds', type=int, default=10)
    args = parser.parse_args()
    ledger = Ledger(args.ledger)
    if args.command == 'submit':
        print(ledger.submit(args.kind, args.adapter))
    elif args.command == 'status':
        print(json.dumps(ledger.jobs(), indent=2))
    else:
        if not args.config:
            parser.error('--config is required for run-once')
        config = validate(json.loads(args.config.read_text()))
        advisor = config.get('advisor')
        recommend = None
        if advisor:
            try:
                from .public_research import studio_recommend
            except ImportError:
                from public_research import studio_recommend
            if set(advisor) != {'endpoint', 'model', 'enabled'} or advisor['enabled'] is not True:
                parser.error('advisor requires explicit enabled, endpoint and resident model')
            recommend = lambda candidates: studio_recommend(advisor['endpoint'], advisor['model'], candidates)
        if args.command == 'worker':
            print(json.dumps(worker(ledger, config, max_jobs=args.max_jobs, max_idle_polls=args.max_idle_polls, idle_seconds=args.idle_seconds, recommend=recommend)))
        else:
            print(run_once(ledger, config, recommend=recommend))


if __name__ == '__main__':
    main()
