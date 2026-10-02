"""Expanded snapshots preserve frozen holdouts and bind all training evidence."""
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image
import pytest

from schematic_model.corpus import conversation, parse_eagle
from schematic_model.training import TrainingConfig, preflight
from scripts.prepare_expanded_training import main, plan_snapshot, write_snapshot
from scripts.prepare_training import prepare as prepare_base


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


@pytest.fixture
def case(tmp_path):
    corpus, source_root = tmp_path / 'corpus', tmp_path / 'sources'
    source = source_root / 'fixture__board' / 'board.sch'
    source.parent.mkdir(parents=True)
    source.write_bytes((Path(__file__).parent / 'fixtures/real/txb0104.sch').read_bytes())
    native = parse_eagle(source)
    for split, color in [('train', 'red'), ('val', 'green'), ('test', 'blue')]:
        path = corpus / 'images' / (split + '.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (24, 24), color).save(path)
        record = {'id': 'board-' + split, 'repository': 'fixture/board' if split == 'train' else 'fixture/' + split,
                  'commit': 'pinned', 'source_path': 'board.sch',
                  'sha256': native['sha256'] if split == 'train' else hashlib.sha256(split.encode()).hexdigest(),
                  'images': [{'page': 1, 'path': 'images/' + split + '.png',
                              'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}]}
        for view, directory in [('evidence', corpus), ('vision', corpus / 'vision')]:
            image = ('../' if view == 'vision' else '') + 'images/' + split + '.png'
            row = conversation(record, native['sheets'][0], image, image_only=view == 'vision')
            row['split'] = split
            write_rows(directory / (split + '.jsonl'), [row])
    base = tmp_path / 'base'
    prepare_base(str(corpus), str(corpus / 'vision'), str(base))
    return corpus, base, source_root


def plan(case, **kwargs):
    return plan_snapshot(*(str(path) for path in case), **kwargs)


def test_cap_one_uses_all_visible_values_and_retains_native_and_refusal_turns(case, tmp_path):
    planned = plan(case)
    assert planned.summary['composition'] == {
        'native_rows': 1, 'native_turns': 6, 'expanded_value_rows': 2,
        'vision_refusal_rows': 1, 'vision_refusal_turns': 2,
        'combined_rows': 4, 'combined_turns': 10,
        'label_counts': {'pins': 2, 'refuse': 4, 'value': 4},
        'source_training_boards': 1, 'source_visible_targets': 3,
        'visible_targets_after_group_exclusions': 3,
        'explicit_excluded_targets': 0, 'excluded_board_value_groups': 0,
        'different_absent_identifiers': 1, 'training_images_changed_from_base': 0}
    selected = planned.manifest['target_selection']
    assert {row['value'] for row in selected} == {'0.1uF', '10K'}
    cap = next(row for row in selected if row['value'] == '0.1uF')
    assert cap['same_value_candidates'] == ['C1', 'C2']
    assert cap['refdes'] in {'C1', 'C2'}
    output = tmp_path / 'expanded'
    manifest = write_snapshot(planned, str(output))
    assert manifest['composition']['combined_rows'] == 4
    checked = preflight(TrainingConfig(data=str(output)))
    assert checked['datasets']['train']['rows'] == 4
    base = case[1]
    for view in ('', 'evidence', 'vision'):
        for split in ('val', 'test'):
            assert (output / view / (split + '.jsonl')).read_bytes() == (base / view / (split + '.jsonl')).read_bytes()
    for artifact in manifest['images']:
        assert hashlib.sha256((output / artifact['path']).read_bytes()).hexdigest() == artifact['sha256']


def test_absent_refs_vary_but_never_name_any_actual_source_part(case):
    source = case[2] / 'fixture__board/board.sch'
    existing = {p.get('name') for p in ET.parse(source).findall('./drawing/schematic/parts/part')}
    refs = set()
    for seed in (1, 2, 3):
        planned = plan(case, seed=seed)
        row = planned.train_views['vision'][-1]
        ref = row['provenance']['training_target']['absent_refdes']
        assert ref not in existing
        assert ref != 'R9999'
        assert ref in row['messages'][1]['content'][-1]['text']
        assert ref in row['messages'][2]['content'][0]['text']
        refs.add(ref)
    assert len(refs) > 1


def test_planning_does_not_write_and_repeated_plan_is_deterministic(case, tmp_path, capsys):
    before = sorted(str(path) for path in tmp_path.rglob('*'))
    assert main(['--corpus', str(case[0]), '--base-snapshot', str(case[1]), '--source-root', str(case[2])]) == 0
    assert json.loads(capsys.readouterr().out)['composition']['expanded_value_rows'] == 2
    assert before == sorted(str(path) for path in tmp_path.rglob('*'))
    first = write_snapshot(plan(case), str(tmp_path / 'one'))
    second = write_snapshot(plan(case), str(tmp_path / 'two'))
    assert first['snapshot_sha256'] == second['snapshot_sha256']


def test_v3_training_images_can_change_without_changing_holdout_bytes(case, tmp_path):
    corpus, base, _ = case
    image = corpus / 'images/train.png'
    Image.new('RGB', (24, 24), 'magenta').save(image)
    sha = hashlib.sha256(image.read_bytes()).hexdigest()
    for path in (corpus / 'train.jsonl', corpus / 'vision/train.jsonl'):
        row = json.loads(path.read_text())
        row['provenance']['images'][0]['sha256'] = sha
        row['provenance']['render_method'] = 'eagle_xml_geometry_v3_text_alignment'
        write_rows(path, [row])
    planned = plan(case)
    assert planned.summary['composition']['training_images_changed_from_base'] == 1
    output = tmp_path / 'expanded-v3'
    manifest = write_snapshot(planned, str(output))
    assert (output / 'vision/test.jsonl').read_bytes() == (base / 'vision/test.jsonl').read_bytes()
    assert any(item['sha256'] == sha for item in manifest['images'])


def test_source_hash_change_is_rejected_before_planning_output(case, tmp_path):
    source = case[2] / 'fixture__board/board.sch'
    source.write_bytes(source.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='Native source hash'):
        plan(case)
    assert not (tmp_path / 'expanded').exists()


@pytest.mark.parametrize('relative, message', [
    ('train.jsonl', 'messages/labels changed'),
    ('vision/train.jsonl', 'messages/labels changed'),
])
def test_changed_training_answers_cannot_be_silently_substituted(case, relative, message):
    path = case[0] / relative
    row = json.loads(path.read_text())
    next(m for m in row['messages'] if m['role'] == 'assistant')['content'][0]['text'] = 'Different answer'
    write_rows(path, [row])
    with pytest.raises(ValueError, match=message):
        plan(case)


def test_family_or_source_provenance_changes_are_rejected(case):
    path = case[0] / 'train.jsonl'
    original = json.loads(path.read_text())
    for field, value in [('family_id', 'fixture/val'), ('design_id', 'board-test')]:
        row = copy.deepcopy(original)
        row[field] = value
        write_rows(path, [row])
        with pytest.raises(ValueError, match='identity/family/split'):
            plan(case)
    row = copy.deepcopy(original)
    row['provenance']['source_path'] = '../elsewhere.sch'
    write_rows(path, [row])
    with pytest.raises(ValueError, match='provenance differs'):
        plan(case)


def test_changed_input_after_reviewed_plan_prevents_publication(case, tmp_path):
    planned = plan(case)
    (case[0] / 'images/train.png').write_bytes(b'changed')
    output = tmp_path / 'expanded'
    with pytest.raises(ValueError, match='Input changed after planning'):
        write_snapshot(planned, str(output))
    assert not output.exists()
    assert not list(tmp_path.glob('expanded.partial-*'))


def test_staging_failure_rolls_back_without_changing_inputs(case, tmp_path, monkeypatch):
    planned = plan(case)
    original_hashes = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in planned.inputs}
    actual_write = Path.write_bytes
    def fail_image_write(path, data):
        if 'expanded.partial-' in str(path) and path.suffix == '.png':
            raise OSError('simulated disk write failure')
        return actual_write(path, data)
    monkeypatch.setattr(Path, 'write_bytes', fail_image_write)
    with pytest.raises(OSError, match='simulated disk write'):
        write_snapshot(planned, str(tmp_path / 'expanded'))
    assert not (tmp_path / 'expanded').exists()
    assert not list(tmp_path.glob('expanded.partial-*'))
    assert original_hashes == {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in planned.inputs}


def test_existing_output_and_input_roots_are_never_overwritten(case, tmp_path):
    planned = plan(case)
    output = tmp_path / 'existing'
    output.mkdir()
    sentinel = output / 'keep.txt'
    sentinel.write_text('keep')
    with pytest.raises(ValueError, match='already exists'):
        write_snapshot(planned, str(output))
    assert sentinel.read_text() == 'keep'
    for root in case:
        with pytest.raises(ValueError, match='outside input roots'):
            write_snapshot(planned, str(root / 'nested-output'))


def test_training_image_cannot_match_a_heldout_image(case):
    corpus, _, _ = case
    original = (corpus / 'images/val.png').read_bytes()
    (corpus / 'images/train.png').write_bytes(original)
    for path in (corpus / 'train.jsonl', corpus / 'vision/train.jsonl'):
        row = json.loads(path.read_text())
        row['provenance']['images'][0]['sha256'] = hashlib.sha256(original).hexdigest()
        write_rows(path, [row])
    with pytest.raises(ValueError, match='leak across'):
        plan(case)


def test_corrupt_base_artifact_cannot_be_carried_into_snapshot(case):
    (case[1] / 'vision/test.jsonl').write_text('tampered')
    with pytest.raises(ValueError, match='artifact hash mismatch'):
        plan(case)


def test_exclusion_drops_entire_value_group_without_substitution_or_native_change(case, tmp_path):
    exclusion = tmp_path / 'exclusions.json'
    exclusion.write_text(json.dumps([{'board_id': 'board-train-p1', 'refdes': 'C1',
                                     'value': '0.1uF', 'reason': 'Reviewed ambiguous target'}]))
    unfiltered = plan(case)
    filtered = plan(case, exclusions=str(exclusion))
    assert filtered.train_views['evidence'] == unfiltered.train_views['evidence']
    assert {item['value'] for item in filtered.manifest['target_selection']} == {'10K'}
    dropped = filtered.manifest['excluded_value_groups']
    assert dropped[0]['suppressed_same_value_candidates'] == ['C1', 'C2']
    assert filtered.summary['composition']['source_visible_targets'] == 3
    assert filtered.summary['composition']['visible_targets_after_group_exclusions'] == 1
    assert filtered.summary['composition']['explicit_excluded_targets'] == 1
    assert filtered.summary['composition']['excluded_board_value_groups'] == 1


@pytest.mark.parametrize('change', [{'refdes': 'R9999'}, {'board_id': 'board-test-p1'}, {'value': 'wrong'}])
def test_invalid_exclusion_cannot_silently_drop_or_replace_targets(case, tmp_path, change):
    exclusion = tmp_path / 'exclusions.json'
    item = {'board_id': 'board-train-p1', 'refdes': 'C1', 'value': '0.1uF', 'reason': 'Reviewed'}
    item.update(change)
    exclusion.write_text(json.dumps([item]))
    with pytest.raises(ValueError, match='Excluded target'):
        plan(case, exclusions=str(exclusion))
