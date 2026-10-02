"""CPU fixtures verify validation-only provenance and prompt/evidence separation."""
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image
import pytest

from evaluate import grade, question_text
from scripts.prepare_expanded_training import digest
from scripts.prepare_generalization_eval import (
    SOURCE_OVERLAP_EXCLUSIONS, plan_generalization, write_snapshot,
)
from tests.test_prepare_expanded_training import case as small_case


def rehash_manifest(base, manifest):
    manifest.pop('snapshot_sha256', None)
    manifest['snapshot_sha256'] = digest(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode())
    (base / 'manifest.json').write_text(json.dumps(manifest))


@pytest.fixture
def case(tmp_path):
    corpus, base, sources = small_case.__wrapped__(tmp_path)
    manifest = json.loads((base / 'manifest.json').read_text())
    originals = {view: json.loads((base / view / 'val.jsonl').read_text()) for view in ('evidence', 'vision')}
    collected = {'evidence': [], 'vision': []}
    families = []
    for index in range(13):
        family = f'fixture/validation-{index:02d}'
        families.append(family)
        source = sources / family.replace('/', '__') / 'board.sch'
        source.parent.mkdir(parents=True)
        source.write_bytes((sources / 'fixture__board/board.sch').read_bytes())
        image = corpus / 'images' / f'validation-{index}.png'
        Image.new('RGB', (24, 24), (index, 50, 100)).save(image)
        raw = image.read_bytes()
        relative = 'images/' + digest(raw) + '.png'
        (base / relative).write_bytes(raw)
        manifest['images'].append({'path': relative, 'sha256': digest(raw), 'split': 'val'})
        for view in collected:
            row = copy.deepcopy(originals[view])
            row.update(id=f'validation-{index}-p1' + ('-vision' if view == 'vision' else ''),
                       design_id=f'validation-{index}', family_id=family)
            row['provenance'].update(repository=family, source_url='https://example.test/board.sch',
                                     sha256=digest(source.read_bytes()))
            row['provenance']['images'] = [{'page': 1, 'path': relative, 'sha256': digest(raw)}]
            row['messages'][1]['content'][0]['image'] = '../' + relative
            collected[view].append(row)
    manifest['families']['val'] = families
    for view, rows in collected.items():
        path = base / view / 'val.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        record = next(item for item in manifest['datasets'] if item['path'] == f'{view}/val.jsonl')
        record.update(sha256=digest(path.read_bytes()), rows=13)
    rehash_manifest(base, manifest)
    return base, sources


def plan(case):
    return plan_generalization(*(str(path) for path in case))


def test_exact_25_bounded_cases_and_source_gold(case):
    planned = plan(case)
    assert planned.report['composition'] == {
        'boards': 13, 'prompts': 25, 'image_only_value': 13, 'image_only_absent_component': 3,
        'image_only_measurement': 3, 'native_evidence_connectivity': 6}
    assert len(planned.rows['vision']) == 19 and len(planned.rows['evidence']) == 6
    assert len({item['question'] for item in planned.report['cases']}) == 25
    values = [item for item in planned.report['cases'] if item['category'] == 'value']
    assert len({item['board_id'] for item in values}) == 13
    assert {item['gold']['value'] for item in values} <= {'10K', '0.1uF'}
    for item in planned.report['cases']:
        assert all(grade(item['gold'], item['proposed_answer']).values())
        assert Path(item['native_source_path']).is_file()
        assert digest(Path(item['frozen_image_path']).read_bytes()) == item['image_sha256']
        if item['category'] == 'absent_component':
            source_refs = {part.get('name') for part in ET.parse(item['native_source_path']).findall(
                './drawing/schematic/parts/part')}
            assert item['target']['absent_refdes'] not in source_refs
    assert len({item['target']['absent_refdes'] for item in planned.report['cases']
                if item['category'] == 'absent_component'}) == 3


def test_image_only_never_includes_source_facts_or_gold_context(case):
    planned = plan(case)
    by_id = {item['id']: item for item in planned.report['cases']}
    for row in planned.rows['vision']:
        assert len(row['messages']) == 3
        assert row['messages'][1]['content'] == [
            {'type': 'image', 'image': by_id[row['id']]['image']},
            {'type': 'text', 'text': by_id[row['id']]['question']}]
        assert 'Extracted native' not in question_text(row, 0)
        assert row['provenance']['evidence_mode'] == 'image_only_values_and_refusals'
    for row in planned.rows['evidence']:
        text = question_text(row, 0)
        evidence = json.loads(text.split('\n', 1)[1].split('\n\n', 1)[0])
        gold = row['gold'][0]
        assert set(gold['pins']) == set(evidence['nets'][gold['net']]) - {gold.get('query')}
        assert row['provenance']['evidence_mode'] == 'native_facts_supplied_in_prompt'


def test_planning_reads_only_validation_conversations_and_is_deterministic(case, monkeypatch):
    original = Path.read_bytes
    reads = []

    def record_read(path):
        reads.append(path)
        assert path.name not in {'train.jsonl', 'test.jsonl'}
        return original(path)

    monkeypatch.setattr(Path, 'read_bytes', record_read)
    before = set(case[0].parent.rglob('*'))
    first, second = plan(case), plan(case)
    assert first.report == second.report and first.rows == second.rows
    assert set(case[0].parent.rglob('*')) == before
    assert {path for path in reads if path.suffix == '.jsonl'} == {
        case[0] / 'evidence/val.jsonl', case[0] / 'vision/val.jsonl'}


def test_snapshot_preserves_exact_image_bytes_and_keeps_modes_separate(case, tmp_path):
    planned = plan(case)
    output = tmp_path / 'generalization'
    manifest = write_snapshot(planned, str(output))
    assert manifest['families']['train'] == manifest['families']['test'] == []
    assert {entry['path']: entry['rows'] for entry in manifest['datasets']} == {
        'evidence/val.jsonl': 6, 'vision/val.jsonl': 19}
    for relative, raw in planned.images.items():
        assert (output / relative).read_bytes() == raw
    for view in ('vision', 'evidence'):
        for row in map(json.loads, (output / view / 'val.jsonl').read_text().splitlines()):
            image = output / view / row['messages'][1]['content'][0]['image']
            assert image.is_file() and row['split'] == 'val'


@pytest.mark.parametrize('change', ['source', 'image', 'dataset', 'families'])
def test_changed_or_leaking_evidence_rejected(case, change):
    base, sources = case
    if change == 'source':
        path = sources / 'fixture__validation-00/board.sch'
    elif change == 'image':
        row = json.loads((base / 'vision/val.jsonl').read_text().splitlines()[0])
        path = base / 'vision' / row['messages'][1]['content'][0]['image']
    elif change == 'dataset':
        path = base / 'vision/val.jsonl'
    else:
        manifest = json.loads((base / 'manifest.json').read_text())
        manifest['families']['train'].append(manifest['families']['val'][0])
        rehash_manifest(base, manifest)
        with pytest.raises(ValueError, match='overlap'):
            plan(case)
        return
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='hash|differs'):
        plan(case)


def test_post_plan_changes_reject_without_publishing(case, tmp_path):
    planned = plan(case)
    (case[0] / 'vision/val.jsonl').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Input changed'):
        write_snapshot(planned, str(tmp_path / 'new-eval'))
    assert not (tmp_path / 'new-eval').exists()


def test_output_aliases_and_path_tampering_cannot_overwrite_inputs(case, tmp_path):
    planned = plan(case)
    for root in case:
        with pytest.raises(ValueError, match='new path outside'):
            write_snapshot(planned, str(root / 'new-eval'))
    victim = tmp_path / 'sentinel.png'
    victim.write_bytes(b'keep')
    planned.images['../sentinel.png'] = b'wrong'
    with pytest.raises(ValueError, match='escapes|canonical'):
        write_snapshot(planned, str(tmp_path / 'new-eval'))
    assert victim.read_bytes() == b'keep'
    assert not (tmp_path / 'new-eval').exists()
    assert not list(tmp_path.glob('new-eval.partial-*'))


def test_explicit_overlap_exclusions_and_task_generalization_limitation(case):
    planned = plan(case)
    assert {(record['board_id'], record['refdes']) for record in planned.report['explicit_target_exclusions']} == SOURCE_OVERLAP_EXCLUSIONS
    assert any('not an isolated wording effect' in text for text in planned.report['limitations'])


def test_malformed_manifest_duplicate_keys_are_rejected(case):
    path = case[0] / 'manifest.json'
    path.write_text(path.read_text().replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1'))
    with pytest.raises(ValueError, match='Duplicate JSON field'):
        plan(case)


def test_manifest_path_alias_rejected_even_with_valid_manifest_digest(case):
    base = case[0]
    manifest = json.loads((base / 'manifest.json').read_text())
    alias = dict(next(item for item in manifest['datasets'] if item['path'] == 'vision/val.jsonl'))
    alias['path'] = '../base/vision/val.jsonl'
    manifest['datasets'].append(alias)
    rehash_manifest(base, manifest)
    with pytest.raises(ValueError, match='canonical'):
        plan(case)


def test_failed_snapshot_write_rolls_back_and_preserves_source_evidence(case, tmp_path, monkeypatch):
    planned = plan(case)
    expected = {path: Path(path).read_bytes() for path in planned.inputs}
    actual_write = Path.write_bytes

    def fail_images(path, data):
        if 'new-eval.partial-' in str(path) and path.suffix == '.png':
            raise OSError('simulated full disk')
        return actual_write(path, data)

    monkeypatch.setattr(Path, 'write_bytes', fail_images)
    with pytest.raises(OSError, match='full disk'):
        write_snapshot(planned, str(tmp_path / 'new-eval'))
    assert not (tmp_path / 'new-eval').exists()
    assert not list(tmp_path.glob('new-eval.partial-*'))
    assert expected == {path: Path(path).read_bytes() for path in planned.inputs}


def reviewed_override(case, tmp_path):
    original = plan(case)
    first = next(item for item in original.report['cases'] if item['category'] == 'value')
    prior = tmp_path / 'prior-plan.json'
    prior.write_text(json.dumps(original.report))
    overlap = tmp_path / 'prior-overlap-review.json'
    overlap.write_text(json.dumps({'status': 'fixture manual review'}))
    replacement = {'refdes': 'R1', 'value': '10K'} if first['target']['refdes'] != 'R1' else {'refdes': 'C1', 'value': '0.1uF'}
    return original, {
        'board_id': first['board_id'], 'original_refdes': first['target']['refdes'],
        'original_value': first['target']['literal_value'], **replacement,
        'reason': 'Explicit reviewed fixture label replacement',
        'source_sha256': first['source_sha256'], 'image_sha256': first['image_sha256'],
        'prior_plan_path': str(prior), 'prior_plan_sha256': digest(prior.read_bytes()),
        'overlap_report_path': str(overlap), 'overlap_report_sha256': digest(overlap.read_bytes()),
        'serving_review_path': 'fixture-serving-review.json'}


def test_reviewed_override_changes_only_declared_value_case_and_preserves_original(case, tmp_path):
    original, override = reviewed_override(case, tmp_path)
    old_bytes = Path(override['prior_plan_path']).read_bytes()
    revised = plan_generalization(*(str(path) for path in case), overrides=[override])
    assert revised.report['revision'] == 2
    assert revised.report['value_overrides'] == [override]
    assert revised.report['composition'] == original.report['composition']
    assert revised.report['plan_sha256'] != original.report['plan_sha256']
    changed = [(old, new) for old, new in zip(original.report['cases'], revised.report['cases'], strict=True) if old != new]
    assert len(changed) == 1
    old, new = changed[0]
    assert new['board_id'] == override['board_id']
    assert new['target'] == {'refdes': override['refdes'], 'literal_value': override['value']}
    assert new['gold']['value'] == override['value']
    assert new['image_sha256'] == old['image_sha256']
    assert Path(override['prior_plan_path']).read_bytes() == old_bytes
    row = next(row for row in revised.rows['vision'] if row['id'] == new['id'])
    assert question_text(row, 0) == new['question']
    assert row['gold'] == [new['gold']]


@pytest.mark.parametrize('change', [
    {'refdes': 'R9999'}, {'value': 'invented value'}, {'original_value': 'not the prior label'},
    {'image_sha256': '0' * 64}, {'board_id': 'outside-validation'},
])
def test_override_cannot_invent_value_or_silently_change_original_binding(case, tmp_path, change):
    _, override = reviewed_override(case, tmp_path)
    override.update(change)
    with pytest.raises(ValueError, match='override|replacement'):
        plan_generalization(*(str(path) for path in case), overrides=[override])


def test_changed_review_evidence_prevents_publication(case, tmp_path):
    _, override = reviewed_override(case, tmp_path)
    revised = plan_generalization(*(str(path) for path in case), overrides=[override])
    Path(override['overlap_report_path']).write_text('changed review')
    with pytest.raises(ValueError, match='Input changed'):
        write_snapshot(revised, str(tmp_path / 'new-eval'))
    assert not (tmp_path / 'new-eval').exists()


def test_builder_passes_verified_bytes_to_shared_eagle_parser(case, monkeypatch):
    import scripts.prepare_generalization_eval as builder
    original = builder.parse_eagle
    parsed = []

    def checked(source):
        assert isinstance(source, bytes)
        parsed.append(source)
        return original(source)

    monkeypatch.setattr(builder, 'parse_eagle', checked)
    plan(case)
    assert len(parsed) == 13
