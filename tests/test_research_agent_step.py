"""Source discovery and scheduler advice stay separate; all model/search calls are fixtures."""
import json
import time

import pytest

from schematic_model import deployment
from scripts import research_agent_step as research


@pytest.mark.parametrize('jobs,advice,expected_next', [
    ([], 'unused', None),
    ([{'id': 'a' * 32, 'kind': 'training'}], 'a' * 32, 'a' * 32),
    ([{'id': 'a' * 32, 'kind': 'training'}], 'b' * 32, None),
])
def test_native_owned_scheduler_advice_preserves_discovered_sources(monkeypatch, jobs, advice, expected_next):
    calls = []
    class Model:
        def __init__(self, config):
            pass

        def complete(self, model, messages, **kwargs):
            calls.append(messages)
            return {'output': 'public circuit datasheets' if len(calls) == 1 else advice}

    monkeypatch.setattr(deployment, 'LocalModel', Model)
    monkeypatch.setenv('SCHEMATIC_QUEUE_METADATA', json.dumps(jobs))
    monkeypatch.setenv('SCHEMATIC_DEADLINE', str(time.time() + 300))
    def search(endpoint, path, **kwargs):
        assert endpoint == 'http://127.0.0.1:8889' and path.startswith('/search?')
        return {'results': [{'url': 'https://example.org/circuit.pdf'}, {'url': 'https://other.invalid/private'}]}
    monkeypatch.setattr(research, 'local_json', search)
    result = research.research({'public_topic': 'Public circuit references', 'model': 'fixture',
                                'model_endpoint': 'http://127.0.0.1:8893', 'search_endpoint': 'http://127.0.0.1:8889',
                                'allowed_hosts': ['example.org'], 'allow_public_search': True, 'backend': 'native-owned'})
    assert result['candidates'] == [{'url': 'https://example.org/circuit.pdf', 'license': None,
                                     'review_status': 'needs-provenance-and-license-review'}]
    assert result['next_job_id'] == expected_next
    assert result['ready_for_training'] is False
    assert len(calls) == (2 if jobs else 1)
    if jobs:
        assert json.dumps(jobs) in calls[1][0]['content']
