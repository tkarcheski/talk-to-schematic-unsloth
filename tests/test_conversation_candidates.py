# cspell:words multiturn
import copy
import hashlib
import json
import os
from pathlib import Path

from PIL import Image
import pytest

from evaluate import load_gold, question_text
from schematic_model import conversation_candidates as candidates
from schematic_model.training import TrainingConfig, _read_data


def message(role, text):
    return {'role': role, 'content': [{'type': 'text', 'text': text}]}


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows = []
    for index in range(3):
        image = tmp_path / f'sheet-{index}.png'
        Image.new('RGB', (30, 20), (index * 50, 30, 80)).save(image)
        row = {'id': f'board-{index}', 'design_id': f'design-{index}', 'family_id': f'family-{index}',
               'split': 'val' if index == 2 else 'train',
               'messages': [message('system', 'Use the supplied schematic.'),
                            message('user', 'Read the value shown next to R1.'),
                            message('assistant', 'R1 is marked 10K.'),
                            message('user', 'And what value is marked for R2?'),
                            message('assistant', 'R2 is marked 22K.'),
                            message('user', 'What value is R9999, and what does it connect to?'),
                            message('assistant', 'There is no R9999 in the supplied sheet evidence. Its value and connections are not shown.'),
                            message('user', 'What measured voltage was observed at IC1.6 during a powered bench test?'),
                            message('assistant', 'No bench measurement is supplied in this schematic or source evidence. I cannot infer a measured voltage.')],
               'gold': [{'type': 'value', 'value': '10K'}, {'type': 'value', 'value': '22K'},
                        {'type': 'refuse'}, {'type': 'refuse'}],
               'provenance': {'repository': 'public/circuits', 'source_path': 'circuit.sch',
                              'sha256': hashlib.sha256(f'source-{index}'.encode()).hexdigest(),
                              'license': 'CC-BY-4.0', 'page': 1,
                              'images': [{'page': 1, 'sha256': hashlib.sha256(image.read_bytes()).hexdigest()}],
                              'evidence_mode': 'image_only_values_and_refusals'}}
        row['messages'][1]['content'].insert(0, {'type': 'image', 'image': image.name})
        rows.append(row)
    train, val = tmp_path / 'train-source.jsonl', tmp_path / 'val-source.jsonl'
    train.write_text(''.join(json.dumps(row) + '\n' for row in rows[:2]))
    val.write_text(json.dumps(rows[2]) + '\n')
    return train, val, tmp_path / 'outputs/candidates'


def test_matched_inventory_return_and_correction_preserve_source_answers(source):
    train, _, _ = source
    row = load_gold(train)[0]
    original = copy.deepcopy(row)
    singles, prefixes = candidates.variants(row)
    multi = prefixes[-1]
    assert row == original
    assert len(singles) == 6 and len(multi['gold']) == 6
    assert len(prefixes) == 6
    assert [len(item['gold']) for item in prefixes] == list(range(1, 7))
    assert candidates.answer_inventory(singles) == candidates.answer_inventory(prefixes)
    assert all(single['messages'][-1] == prefix['messages'][-1]
               for single, prefix in zip(singles, prefixes))
    assert question_text(multi, 2) == 'Returning to R1: what value is marked for it?'
    assert question_text(multi, 3) == 'Correction: I meant R2, not R1. What value is marked for it?'
    assert multi['messages'][6] == original['messages'][2]
    assert multi['messages'][8] == original['messages'][4]
    assert [question_text(row, 0) for row in singles[:4]] == [
        'Read the value shown next to R1.', 'Read the value shown next to R2.',
        'Read the value shown next to R1.', 'Read the value shown next to R2.']
    assert multi['messages'][-4:] == original['messages'][-4:]
    assert all(item['provenance'] == original['provenance'] for item in singles + prefixes)


def test_real_training_reader_accepts_both_arms_and_validation_is_unchanged(source):
    train, val, output = source
    before = (train.read_bytes(), val.read_bytes())
    manifest = candidates.build(train, val, output, families=2)
    assert manifest['answer_inventory_each_arm']['assistant_answers'] == 12
    assert manifest['recommended_budget']['gradient_accumulation'] == {'standalone': 4, 'multiturn': 4}
    assert manifest['recommended_budget']['last_response_only'] is True
    validation_outputs = []
    for arm, count in [('standalone', 12), ('multiturn', 12)]:
        rows, validation = _read_data(TrainingConfig(data=str(output / arm), last_response_only=True))
        assert len(rows) == count and len(validation) == 1
        validation_outputs.append(load_gold(output / arm / 'val.jsonl'))
        for row in rows + validation:
            for msg in row['messages']:
                for part in msg['content']:
                    if part['type'] == 'image':
                        assert Path(part['image']).is_file()
    assert validation_outputs[0] == validation_outputs[1]
    expected = load_gold(val)[0]
    actual = validation_outputs[0][0]
    actual['messages'][1]['content'][0] = expected['messages'][1]['content'][0]
    assert actual == expected
    assert (train.read_bytes(), val.read_bytes()) == before
    assert str(train.parent) not in (output / 'manifest.json').read_text()


def test_selection_deterministic_and_independent_of_input_order(source):
    train, val, output = source
    first = candidates.build(train, val, output, families=1, seed=19)
    train.write_text('\n'.join(reversed(train.read_text().splitlines())) + '\n')
    second = candidates.build(train, val, output.with_name('second'), families=1, seed=19)
    assert first['selected_source_rows'] == second['selected_source_rows']
    assert first['datasets'] == second['datasets']


@pytest.mark.parametrize('field', ['family_id', 'design_id', 'source_sha256', 'image_sha256'])
def test_overlap_rejected_even_before_selecting_families(source, field):
    train, val, output = source
    rows, heldout = load_gold(train), load_gold(val)
    if field == 'source_sha256':
        heldout[0]['provenance']['sha256'] = rows[0]['provenance']['sha256']
    elif field == 'image_sha256':
        heldout[0]['messages'][1]['content'][0] = rows[0]['messages'][1]['content'][0]
        heldout[0]['provenance']['images'] = rows[0]['provenance']['images']
    else:
        heldout[0][field] = rows[0][field]
    val.write_text(json.dumps(heldout[0]) + '\n')
    with pytest.raises(ValueError, match='overlap'):
        candidates.build(train, val, output, families=1)
    assert not output.exists()


@pytest.mark.parametrize('mutation', ['answer', 'refusal', 'image', 'question', 'split', 'turns'])
def test_bad_source_cannot_enter_candidate(source, mutation):
    train, val, output = source
    rows = load_gold(train)
    if mutation == 'answer':
        rows[0]['messages'][2] = message('assistant', 'R1 is marked 100K.')
    elif mutation == 'refusal':
        rows[0]['messages'][6] = message('assistant', 'R9999 is 1K.')
    elif mutation == 'image':
        (train.parent / 'sheet-0.png').write_bytes(b'changed')
    elif mutation == 'question':
        rows[0]['messages'][3] = message('user', 'And the other part?')
    elif mutation == 'split':
        rows[0]['split'] = 'test'
    else:
        rows[0]['messages'] = rows[0]['messages'][:5]
        rows[0]['gold'] = rows[0]['gold'][:2]
    train.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError):
        candidates.build(train, val, output, families=1)
    assert not output.exists()


def test_private_snapshot_and_existing_output_preserved(source):
    train, val, output = source
    old = os.umask(0)
    try:
        candidates.build(train, val, output, families=1)
    finally:
        os.umask(old)
    before = (output / 'manifest.json').read_bytes()
    assert all(path.stat().st_mode & 0o777 == (0o700 if path.is_dir() else 0o600)
               for path in [output, *output.rglob('*')])
    with pytest.raises(ValueError, match='exists'):
        candidates.build(train, val, output, families=1)
    assert (output / 'manifest.json').read_bytes() == before


def test_source_image_change_after_preflight_leaves_no_snapshot(source, monkeypatch):
    train, val, output = source
    original = candidates._relocate
    def changed(rows, path, images):
        (train.parent / 'sheet-0.png').write_bytes(b'changed')
        return original(rows, path, images)
    monkeypatch.setattr(candidates, '_relocate', changed)
    with pytest.raises(ValueError, match='changed'):
        candidates.build(train, val, output, families=2)
    assert not output.exists()


def test_bounds_and_output_location(source):
    train, val, output = source
    for options in ({'families': 0}, {'families': 3}, {'seed': -1}, {'accumulation': True}):
        with pytest.raises(ValueError):
            candidates.build(train, val, output, **options)
    with pytest.raises(ValueError, match='ignored'):
        candidates.build(train, val, train.parent / 'public', families=1)


def test_cli_outputs_auditable_budget(source, capsys):
    train, val, output = source
    assert candidates.main(['--train-source', str(train), '--validation-source', str(val),
                            '--out', str(output), '--families', '1']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['answer_inventory_each_arm']['assistant_answers'] == 6
    assert report['recommended_budget']['optimizer_steps_per_epoch_each_arm'] == 2
