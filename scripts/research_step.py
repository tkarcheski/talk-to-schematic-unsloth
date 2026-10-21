"""One bounded public-source collection step, producing a local review proposal."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

try:
    from .public_research import fetch_public
except ImportError:
    from public_research import fetch_public


def collect(plan, output):
    if set(plan) != {'allowed_hosts', 'sources'}:
        raise ValueError('expected allowed_hosts and reviewed sources')
    if not isinstance(plan['sources'], list) or not 1 <= len(plan['sources']) <= 8:
        raise ValueError('one to eight public sources required')
    if not isinstance(plan['allowed_hosts'], list) or not all(isinstance(h, str) for h in plan['allowed_hosts']):
        raise ValueError('explicit host allowlist required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    deadline = float(os.environ.get('SCHEMATIC_DEADLINE', time.time() + 120))
    proposal_path = output / 'proposal.json'
    records = []
    if proposal_path.is_file():
        previous = json.loads(proposal_path.read_text())
        for entry in previous.get('sources', []):
            import hashlib
            path = output / (entry['sha256'] + '.source')
            if {k: entry[k] for k in ('url', 'license', 'license_url', 'public')} in plan['sources'] and path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']:
                records.append(entry)
    for source in plan['sources']:
        if any(entry['url'] == source['url'] for entry in records):
            continue
        if time.time() + 25 >= deadline:
            break
        data, provenance = fetch_public(source, plan['allowed_hosts'])
        name = provenance['sha256'] + '.source'
        destination = output / name
        with destination.open('wb') as handle:
            handle.write(data)
        os.chmod(destination, 0o600)
        records.append({**provenance, 'file': name, 'review_status': 'needs-human-license-and-content-review'})
    proposal = {'schema': 1, 'public_sources_only': True, 'sources': records,
                'requested_sources': len(plan['sources']), 'ready_for_training': False}
    path = output / 'proposal.json'
    path.write_text(json.dumps(proposal, indent=2) + '\n')
    os.chmod(path, 0o600)
    if os.environ.get('SCHEMATIC_RESULT_PATH'):
        Path(os.environ['SCHEMATIC_RESULT_PATH']).write_text(json.dumps({'source_count': len(records), 'proposal_count': 1}))
    return 0 if len(records) == len(plan['sources']) else 75


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(collect(json.loads(args.plan.read_text()), args.output))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        try:
            from .public_research import report_failure
        except ImportError:
            from public_research import report_failure
        report_failure(error)
        raise SystemExit(1)
