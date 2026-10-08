"""Bounded local-model research: public topic -> approved-policy search -> review candidates."""
import argparse
import json
import os
import time
from pathlib import Path
from urllib.parse import urlencode, urlsplit

try:
    from .public_research import local_json
except ImportError:
    from public_research import local_json


def research(plan):
    fields = {'public_topic', 'model', 'model_endpoint', 'search_endpoint', 'allowed_hosts', 'allow_public_search', 'backend'}
    if set(plan) != fields or plan['allow_public_search'] is not True:
        raise ValueError('explicit public research search policy required')
    if plan['backend'] not in ('native-unsloth', 'native-owned', 'studio'):
        raise ValueError('unknown self-hosted backend')
    if not isinstance(plan['public_topic'], str) or not 1 <= len(plan['public_topic']) <= 500:
        raise ValueError('bounded operator-reviewed public topic required')
    if not isinstance(plan['allowed_hosts'], list) or not plan['allowed_hosts'] or not all(isinstance(h, str) and h and '/' not in h for h in plan['allowed_hosts']):
        raise ValueError('public host allowlist required')
    key = os.environ.get('UNSLOTH_API_KEY') if plan['backend'] == 'studio' else None
    if plan['backend'] == 'studio' and not key:
        raise ValueError('Studio authentication required')
    messages = [{'role': 'system', 'content': 'Produce only a short public web search query for open hardware schematics and datasheets. No tools or commands.'},
                {'role': 'user', 'content': plan['public_topic']}]
    if float(os.environ.get('SCHEMATIC_DEADLINE', time.time() + 300)) - time.time() < 30:
        raise TimeoutError('insufficient research slice remaining')
    if plan['backend'] == 'native-owned':
        from schematic_model.deployment import DeploymentConfig, LocalModel
        # This short-lived scheduler child owns its loaded model. Normal process exit releases residency.
        engine = LocalModel(DeploymentConfig(model=plan['model'], max_tokens=96, max_seq=4096))
        result = engine.complete('schematic', messages, max_tokens=96)
        response = {'choices': [{'message': {'content': result['output']}}]}
    else:
        prefix = '/api/inference' if plan['backend'] == 'studio' else '/v1'
        if plan['backend'] == 'studio':
            status = local_json(plan['model_endpoint'], '/api/inference/status', key=key)
            if status.get('loading') or plan['model'] != status.get('model_identifier'):
                raise ValueError('Studio resident model mismatch; handoff required')
        models = local_json(plan['model_endpoint'], prefix + '/models', key=key)
        if plan['model'] not in {m.get('id') for m in models.get('data', [])}:
            raise ValueError('requested model unavailable; no automatic load')
        response = local_json(plan['model_endpoint'], prefix + '/chat/completions', {
            'model': plan['model'], 'stream': False, 'max_tokens': 96, 'messages': messages}, key=key)
    query = response['choices'][0]['message']['content'].strip()
    if not 1 <= len(query) <= 250 or '\n' in query or any(ord(c) < 32 for c in query):
        raise ValueError('model query requires review')
    results = local_json(plan['search_endpoint'], '/search?' + urlencode({'q': query, 'format': 'json'}))
    candidates = []
    for result in results.get('results', [])[:30]:
        url = result.get('url', '')
        parsed = urlsplit(url)
        if parsed.scheme == 'https' and parsed.hostname in plan['allowed_hosts'] and not parsed.username and not parsed.password and not parsed.query and not parsed.fragment and parsed.port in (None, 443):
            candidates.append({'url': url, 'license': None, 'review_status': 'needs-provenance-and-license-review'})
    next_job_id = None
    if plan['backend'] == 'native-owned':
        candidates = json.loads(os.environ.get('SCHEMATIC_QUEUE_METADATA', '[]'))
        if not isinstance(candidates, list) or len(candidates) > 100 or any(set(c) != {'id', 'kind'} or c['kind'] not in ('research', 'training', 'proposal') or not isinstance(c['id'], str) or len(c['id']) != 32 or any(x not in '0123456789abcdef' for x in c['id']) for c in candidates):
            raise ValueError('invalid opaque scheduler metadata')
        remaining = float(os.environ.get('SCHEMATIC_DEADLINE', time.time() + 300)) - time.time()
        if candidates and remaining >= 30:
            advice = engine.complete('schematic', [{'role': 'user', 'content': 'Choose the next queued job to balance public research and training. Return only one exact job ID: ' + json.dumps(candidates)}], max_tokens=64)['output'].strip()
            if advice in {c['id'] for c in candidates}:
                next_job_id = advice
    return {'next_job_id': next_job_id, 'schema': 1, 'public_topic': plan['public_topic'], 'query': query, 'backend': plan['backend'],
            'model': plan['model'], 'candidates': candidates, 'ready_for_training': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        proposal = research(json.loads(args.plan.read_text()))
    except TimeoutError:
        raise SystemExit(75)
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.output.write_text(json.dumps(proposal, indent=2) + '\n')
    os.chmod(args.output, 0o600)
    if os.environ.get('SCHEMATIC_RESULT_PATH'):
        Path(os.environ['SCHEMATIC_RESULT_PATH']).write_text(json.dumps({'source_count': len(proposal['candidates']), 'proposal_count': 1, 'next_job_id': proposal['next_job_id']}))


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
