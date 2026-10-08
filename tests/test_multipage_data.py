# cspell:words multipage
"""Candidate review queue integrity: source binding, family isolation, and budgets."""
import json
from pathlib import Path

import pytest

from schematic_model import multipage_data as module
from schematic_model.training import sha256


def evidence():
    return {'sheets': [
        {'page': 1, 'bom': [{'refdes': 'U1'}], 'nets': {'CLK': ['U1.5'], 'N$1': ['U1.6']}},
        {'page': 2, 'bom': [{'refdes': 'TP1'}, {'refdes': 'U2'}], 'nets': {'CLK': ['U2.7', 'TP1.1'], 'N$1': ['U2.8']}},
    ]}


def test_page_endpoints_are_not_flattened_and_anonymous_names_not_joined():
    rows = module.endpoint_candidates(evidence(), {'repository': 'public/board', 'split': 'train', 'source_sha256': 'a' * 64}, lambda messages: 9000, 8192, 256)
    assert len(rows) == 1 and rows[0]['net'] == 'CLK'
    assert rows[0]['candidate_answer'] == [{'page': 1, 'physical_pads': ['U1.5']}, {'page': 2, 'physical_pads': ['TP1.1', 'U2.7']}]
    assert not rows[0]['training_eligible'] and rows[0]['review_status'] == 'required'
    assert not rows[0]['variants']['full']['fits_text_context']
    assert 'U2.7' in rows[0]['variants']['full']['messages'][0]['content']


def test_empty_second_page_does_not_qualify():
    parsed = evidence()
    parsed['sheets'][1]['bom'] = []
    with pytest.raises(ValueError, match='populated'):
        module.endpoint_candidates(parsed, {}, lambda messages: 10, 8192, 256)


@pytest.fixture
def source_plan(tmp_path, monkeypatch):
    root = tmp_path / 'source'
    root.mkdir()
    (root / 'board.sch').write_text('public source fixture')
    (root / 'LICENSE').write_text('CC BY-SA fixture notice')
    family = {'repository': 'public/board', 'split': module._split('public/board'), 'revision': 'a' * 40,
              'license': 'CC-BY-SA-4.0', 'license_path': 'LICENSE', 'license_sha256': sha256(root / 'LICENSE'),
              'attribution': 'Public hardware author', 'sources': [{'path': 'board.sch', 'sha256': sha256(root / 'board.sch')}]}
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'families': [family]}))
    monkeypatch.setattr(module, 'parse_eagle', lambda path: evidence())
    return root, plan, family


def test_repeat_build_is_deterministic_and_never_overwrites(source_plan, tmp_path):
    root, plan, _ = source_plan
    def build(name):
        return module.build(plan, {'public/board': str(root)}, tmp_path / name, lambda messages: 100, {'fixture': True})
    a, b = build('one'), build('two')
    assert a == b
    assert (tmp_path / 'one/review-queue.jsonl').read_bytes() == (tmp_path / 'two/review-queue.jsonl').read_bytes()
    with pytest.raises(ValueError, match='exists'):
        build('one')


@pytest.mark.parametrize('change', ['source', 'license', 'split', 'diagnostic', 'duplicate'])
def test_invalid_inputs_do_not_publish_a_queue(source_plan, tmp_path, change):
    root, plan, family = source_plan
    families = [family]
    if change == 'source':
        (root / 'board.sch').write_text('changed')
    elif change == 'license':
        (root / 'LICENSE').write_text('changed')
    elif change == 'split':
        family['split'] = 'val' if family['split'] != 'val' else 'train'
    elif change == 'diagnostic':
        family['role'] = 'diagnostic'
    else:
        families.append(family.copy())
    plan.write_text(json.dumps({'families': families}))
    with pytest.raises(ValueError):
        module.build(plan, {'public/board': str(root)}, tmp_path / 'output', lambda messages: 100, {})
    assert not (tmp_path / 'output').exists()


def test_source_root_escape_rejected(tmp_path):
    outside = tmp_path / 'outside.sch'
    outside.write_text('public but outside approved root')
    root = tmp_path / 'root'
    root.mkdir()
    with pytest.raises(ValueError, match='escapes'):
        module.verified_path(root, '../outside.sch', sha256(outside))


def test_missing_embedded_library_does_not_publish_partial_candidates(source_plan, tmp_path, monkeypatch):
    root, plan, _ = source_plan
    def unsupported(path):
        raise ValueError('Missing embedded device set for SUPPLY2')
    monkeypatch.setattr(module, 'parse_eagle', unsupported)
    with pytest.raises(ValueError, match='SUPPLY2'):
        module.build(plan, {'public/board': str(root)}, tmp_path / 'output', lambda messages: 100, {})
    assert not (tmp_path / 'output').exists()


@pytest.mark.parametrize('encoded', [[1, 2, 3], {'input_ids': [1, 2, 3], 'attention_mask': [1, 1, 1]}, {'input_ids': [[1, 2, 3]]}])
def test_token_count_uses_ids_not_mapping_field_count(encoded):
    from types import SimpleNamespace
    tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: encoded)
    assert module.chat_token_count(tokenizer, []) == 3


def test_unknown_tokenizer_shape_fails_instead_of_estimating():
    from types import SimpleNamespace
    tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: {'unexpected': [1, 2, 3]})
    with pytest.raises(ValueError, match='shape'):
        module.chat_token_count(tokenizer, [])
