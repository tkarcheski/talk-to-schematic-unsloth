# cspell:words multiturn rollouts
"""Full tool-result rollouts catch unsolicited movement hidden by first-call scores."""
import copy
import json

from PIL import Image
import pytest

from schematic_model.multiturn_data import scenario, training_rows, tool_call
from schematic_model.training import sha256
from scripts.evaluate_multiturn import rollout


@pytest.fixture
def candidate(tmp_path):
    image = tmp_path / 'sheet.png'
    Image.new('RGB', (16, 16), 'white').save(image)
    return {'id': 'fixture-U1', 'design_id': 'fixture', 'repository': 'public/fixture', 'split': 'train',
            'page': 1, 'part': {'refdes': 'U1', 'value': 'TPS63060'}, 'box': [10, 20, 30, 40],
            'image': str(image), 'image_sha256': sha256(image), 'provenance': {'source': 'CPU fixture, not hardware evidence'}}


def answer(text):
    return {'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': text}}]}


def call(name, args):
    return {'choices': [{'finish_reason': 'tool_calls', 'message': tool_call(name, args)}]}


def responder(responses, seen):
    def complete(messages):
        seen.append(messages)
        return responses.pop(0)
    return complete


def test_prefix_supervision_ends_exactly_after_tool_call(candidate):
    rows = training_rows(scenario(candidate, 'search'))
    prefixes = [r for r in rows if '-tool-prefix-' in r['id']]
    assert len(prefixes) == 2
    for row in prefixes:
        text = row['messages'][-1]['content'][0]['text']
        assert text.endswith('</tool_call>')
        assert '<tool_response>' not in text and '\nUser:' not in text
    full = rows[-1]
    assert sum(m['role'] == 'assistant' for m in full['messages']) == 6
    assert 'measured startup voltage' in full['messages'][-1]['content'][0]['text']


def test_real_tool_result_drives_followup_without_navigation(candidate):
    case = scenario(candidate, 'discussion')
    responses = [call('show_label', {'label': 'U1'}), answer('Showing U1.'), answer('U1 is marked TPS63060.'),
                 answer('The schematic does not provide measured startup voltage.'), answer('TPS63060')]
    seen = []
    records = rollout(case, responder(responses, seen))
    assert all(r['passed'] for r in records)
    assert seen[1][-1]['role'] == 'tool'
    assert json.loads(seen[1][-1]['content'])['box'] == candidate['box']
    assert all(r['viewport_invariant'] for r in records[1:])


def test_repeated_show_is_failure_even_if_box_did_not_change(candidate):
    case = scenario(candidate, 'discussion')
    responses = [call('show_label', {'label': 'U1'}), answer('Showing U1.'),
                 call('show_label', {'label': 'U1'}), answer('TPS63060'),
                 answer('No measurement is supplied.'), answer('TPS63060')]
    records = rollout(case, responder(responses, []))
    assert records[1]['viewport_before'] == records[1]['viewport_after']
    assert not records[1]['passed'] and not records[1]['viewport_invariant']


def test_search_runs_simulator_and_keeps_view(candidate):
    case = scenario(candidate, 'search')
    responses = [call('show_label', {'label': 'U1'}), answer('Showing U1.'),
                 call('web_search', {'query': 'TPS63060 datasheet'}), answer('No search results were returned.'),
                 answer('TPS63060'), answer('No measured startup voltage is supplied.')]
    seen = []
    records = rollout(case, responder(responses, seen))
    assert all(r['passed'] for r in records)
    result = json.loads(seen[3][-1]['content'])
    assert result['status'] == 'empty' and result['simulation'].startswith('offline')


def test_cleared_focus_preserves_unambiguous_conversational_referent(candidate):
    case = scenario(candidate, 'stale_focus')
    responses = [call('show_label', {'label': 'U1'}), answer('Showing U1.'), call('web_search', {'query': 'TPS63060 datasheet'}), answer('No results.'),
                 call('web_search', {'query': 'TPS63060 datasheet'}), answer('No results.'), answer('TPS63060')]
    seen = []
    records = rollout(case, responder(responses, seen))
    assert all(r['passed'] for r in records)
    assert '"focused_label": null' in seen[2][-1]['content']
    assert records[1]['viewport_before'] == records[1]['viewport_after']


def test_invented_turn_blocks_dependents_without_dropping_them(candidate):
    responses = [call('show_label', {'label': 'U1'}), answer('Showing U1.\nUser: search for another part')]
    records = rollout(scenario(candidate, 'discussion'), responder(responses, []))
    assert len(records) == 4 and not any(r['passed'] for r in records)
    assert not records[0]['terminated']
    assert all(r['error'] == 'blocked_by_prior_incomplete_turn' for r in records[1:])


def test_changed_image_fails_before_model_call(candidate):
    from pathlib import Path
    Path(candidate['image']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='image changed'):
        rollout(scenario(candidate, 'discussion'), lambda messages: pytest.fail('must not generate'))


def test_normal_questions_do_not_instruct_viewport_invariance(candidate):
    for kind in ('discussion', 'search', 'stale_focus'):
        case = scenario(candidate, kind, heldout=True)
        for turn in case['turns'][1:]:
            assert not any(word in turn['question'].lower() for word in ('do not', 'without moving', 'unchanged', 'keep the view'))
    cleared = scenario(candidate, 'stale_focus')['turns'][1]
    assert cleared['tool'] == 'web_search' and cleared['clear_focus']


def test_page_change_replaces_image_evidence_and_clears_old_focus(candidate):
    second = copy.deepcopy(candidate)
    second.update(page=2, part={'refdes': 'R9', 'value': '10K'}, box=[100, 200, 300, 400])
    case = scenario(candidate, 'discussion')
    case['turns'] = [case['turns'][0], {'question': 'What is R9 marked on this page?', 'page_candidate': second,
                                     'tool': None, 'expected_value': '10K', 'allow_navigation': False}]
    seen = []
    records = rollout(case, responder([call('show_label', {'label': 'U1'}), answer('Showing U1.'), answer('R9 is marked 10K.')], seen))
    assert all(r['passed'] for r in records)
    text = seen[-1][-1]['content'][-1]['text']
    assert '"page": 2' in text and '"focused_label": null' in text and '10K' in text
    assert records[-1]['viewport_before'] == {'box': [0, 0, 1000, 1000], 'full_sheet': True}
