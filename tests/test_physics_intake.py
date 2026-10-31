# cspell:words pdftotext
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from schematic_model import physics_intake as physics


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pdf = tmp_path / 'source.pdf'
    pdf.write_bytes(b'%PDF-1.4\nfixture')
    plan = {
        'source': {'url': 'https://example.org/book.pdf', 'title': 'Circuit facts', 'author': 'Author',
                   'sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(),
                   'license': {'id': 'CC-BY-4.0', 'url': physics.LICENSES['CC-BY-4.0'],
                               'pages': [3], 'evidence_text': 'Creative Commons Attribution 4.0'}},
        'family_id': 'circuit-book', 'pages': [1, 2],
        'calculations': [{'id': 'current-1', 'page': 1, 'anchor': 'Ohm law',
                          'kind': 'ohms_law_current', 'inputs': {'voltage_v': '9', 'resistance_ohm': '300'}}],
        'reasoning_proposals': [{'id': 'reason-1', 'page': 2, 'anchor': 'capacitor',
                                'question': 'Does this capacitor suffice?', 'proposed_answer': 'A proposal only.'}],
    }

    def fake_run(argv, **kwargs):
        page = int(argv[argv.index('-f') + 1])
        texts = {1: 'Ohm law\nI = V / R', 2: 'An ideal capacitor', 3: 'Creative Commons\nAttribution 4.0'}
        Path(argv[-1]).write_text(texts[page])

    monkeypatch.setattr(physics.subprocess, 'run', fake_run)
    return pdf, plan, tmp_path / 'data/external/unit'


def test_pins_evidence_and_keeps_calculation_separate_from_review(fixture):
    pdf, plan, out = fixture
    result = physics.intake(pdf, plan, out)
    proposals = json.loads((out / 'proposals.json').read_text())
    assert result['proposal_count'] == 2
    assert proposals[0]['calculation']['answer_fraction'] == '3/100'
    assert proposals[0]['citation']['pdf_sha256'] == hashlib.sha256(pdf.read_bytes()).hexdigest()
    assert proposals[0]['evidence_level'] == 'arithmetic-checked-proposal'
    assert proposals[1]['evidence_level'] == 'unverified-reasoning-proposal'
    assert proposals[0]['review_states']['ai_review'] == 'not-performed'
    assert proposals[0]['review_states']['human_review'] == 'not-performed'
    assert proposals[0]['review_states']['formula_check'] == 'exact-arithmetic-only; source-alignment-pending'
    assert all(p['split'] == 'unassigned' and not p['ready_for_training'] for p in proposals)
    assert all(not p['source_formula_verified'] and not p['reasoning_verified'] for p in proposals)
    assert all('messages' not in p and 'gold' not in p for p in proposals)
    assert (out / 'source.pdf').read_bytes() == pdf.read_bytes()
    assert not result['ready_for_training']


def test_private_modes_independent_of_umask(fixture):
    pdf, plan, out = fixture
    previous = os.umask(0)
    try:
        physics.intake(pdf, plan, out)
    finally:
        os.umask(previous)
    assert out.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in out.iterdir())


@pytest.mark.parametrize('change', ['hash', 'license', 'anchor', 'page', 'duplicate', 'injected-gold'])
def test_rejects_invalid_evidence_without_partial_output(fixture, change):
    pdf, plan, out = fixture
    if change == 'hash':
        pdf.write_bytes(b'%PDF-1.4\nchanged')
    elif change == 'license':
        plan['source']['license']['evidence_text'] = 'Not in the PDF'
    elif change == 'anchor':
        plan['calculations'][0]['anchor'] = 'Wrong source'
    elif change == 'page':
        plan['calculations'][0]['page'] = 99
    elif change == 'duplicate':
        plan['reasoning_proposals'][0]['id'] = 'current-1'
    else:
        plan['calculations'][0]['ready_for_training'] = True
    with pytest.raises(ValueError):
        physics.intake(pdf, plan, out)
    assert not out.exists()


@pytest.mark.parametrize('values', [{'voltage_v': '9', 'resistance_ohm': '0'},
                                   {'voltage_v': 'NaN', 'resistance_ohm': '300'},
                                   {'voltage_v': '9 V', 'resistance_ohm': '300'},
                                   {'voltage_v': True, 'resistance_ohm': '300'},
                                   {'voltage_v': '9', 'resistance_ohm': '300', 'execute': 'something'}])
def test_invalid_calculations_cannot_be_checked(values):
    with pytest.raises(ValueError):
        physics.calculate('ohms_law_current', values)


def test_exact_decimal_circuit_calculations():
    assert physics.calculate('rc_time_constant', {'resistance_ohm': '10000', 'capacitance_f': '0.0000001'})['answer_fraction'] == '1/1000'
    assert physics.calculate('series_resistance', {'r1_ohm': '100.5', 'r2_ohm': '200.25'})['answer_fraction'] == '1203/4'
    with pytest.raises(ValueError, match='Unsupported'):
        physics.calculate('arbitrary-expression', {})


def test_preserves_existing_output_and_symlinks(fixture, tmp_path):
    pdf, plan, out = fixture
    out.parent.mkdir(parents=True)
    target = tmp_path / 'preserve'
    target.mkdir()
    (target / 'keep').write_text('unchanged')
    out.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match='exists'):
        physics.intake(pdf, plan, out)
    assert (target / 'keep').read_text() == 'unchanged'


def test_rejects_output_outside_private_roots(fixture, tmp_path):
    pdf, plan, _ = fixture
    with pytest.raises(ValueError, match='ignored'):
        physics.intake(pdf, plan, tmp_path / 'public')


def test_strict_json_refuses_duplicates_and_nonfinite(tmp_path):
    for content in ('{"source": 1, "source": 2}', '{"value": NaN}'):
        path = tmp_path / 'plan.json'
        path.write_text(content)
        with pytest.raises(ValueError):
            physics.read_plan(path)


def test_extraction_failure_leaves_no_proposal(fixture, monkeypatch):
    pdf, plan, out = fixture
    def fail(*args, **kwargs):
        raise subprocess.TimeoutExpired('pdftotext', 20)
    monkeypatch.setattr(physics.subprocess, 'run', fail)
    with pytest.raises(subprocess.TimeoutExpired):
        physics.intake(pdf, plan, out)
    assert not out.exists()


def test_cli_intake(fixture, tmp_path, capsys):
    pdf, plan, out = fixture
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text(json.dumps(plan))
    assert physics.main(['--pdf', str(pdf), '--plan', str(plan_path), '--out', str(out)]) == 0
    assert json.loads(capsys.readouterr().out) == {'proposal_count': 2, 'ready_for_training': False}


@pytest.mark.parametrize('field,value', [('pages', [1, 1]), ('pages', [True]), ('family_id', '../bad'),
                                        ('calculations', [None]), ('reasoning_proposals', [None])])
def test_malformed_plan_fields(fixture, field, value):
    _, plan, _ = fixture
    malformed = copy.deepcopy(plan)
    malformed[field] = value
    with pytest.raises(ValueError):
        physics.validate_plan(malformed)
