"""Diagnostic derivation preserves gold and isolates conversation context."""
import copy
import hashlib
import json

import pytest

from schematic_model.conversation_diagnostics import build, derive
from evaluate import load_gold, validate_gold_rows


def message(role, text):
    return {'role': role, 'content': [{'type': 'text', 'text': text}]}


@pytest.fixture
def source_row():
    row = {'id': 'fixture-p1-vision', 'design_id': 'fixture', 'family_id': 'public/fixture', 'split': 'val',
           'messages': [message('system', 'Use the attached image.'),
                        message('user', 'Read the value shown next to R1.'),
                        message('assistant', 'R1 is marked 10K.'),
                        message('user', 'And what value is marked for R2?'),
                        message('assistant', 'R2 is marked 22K.')],
           'gold': [{'type': 'value', 'value': '10K'}, {'type': 'value', 'value': '22K'}],
           'provenance': {'repository': 'public/fixture', 'source_path': 'example.sch', 'sha256': 'a' * 64,
                          'license': 'CC-BY-SA-4.0', 'page': 1,
                          'images': [{'page': 1, 'sha256': hashlib.sha256(b'fixture image').hexdigest()}],
                          'evidence_mode': 'image_only_values_and_refusals'}}
    row['messages'][1]['content'].insert(0, {'type': 'image', 'image': 'sheet.png'})
    return row


def test_triplet_preserves_answers_identity_and_image(source_row):
    original = copy.deepcopy(source_row)
    probes, counts = derive([source_row])
    validate_gold_rows(probes)
    assert source_row == original
    assert counts['paired_source_rows'] == 1 and counts['derived_rows'] == 3
    standalone, explicit, contextual = probes
    assert standalone['gold'] == [source_row['gold'][1]]
    assert explicit['gold'] == contextual['gold'] == source_row['gold']
    assert standalone['messages'][-2]['content'][-1] == explicit['messages'][-2]['content'][-1]
    assert contextual['messages'][-2]['content'][0]['text'] == 'And R2?'
    for probe in probes:
        assert probe['messages'][-1] == source_row['messages'][-1]
        assert probe['provenance'] == source_row['provenance']
        assert probe['family_id'] == source_row['family_id'] and probe['split'] == 'val'
        assert sum(block['type'] == 'image' for m in probe['messages'] for block in m['content']) == 1
        assert probe['diagnostic']['target_turn'] == len(probe['gold']) - 1


def test_single_turn_is_reported_not_silently_fabricated(source_row):
    source_row['messages'] = source_row['messages'][:3]
    source_row['gold'] = source_row['gold'][:1]
    probes, counts = derive([source_row])
    assert probes == []
    assert counts['skipped_no_initial_value_pair'] == 1
    assert counts['source_rows_with_1_turns'] == 1


def test_ambiguous_target_is_skipped(source_row):
    source_row['messages'][3] = message('user', 'And the other resistor?')
    assert derive([source_row])[1]['skipped_unsupported_question'] == 1


def test_same_target_is_skipped_to_prevent_answer_repetition(source_row):
    source_row['messages'][3] = message('user', 'And what value is marked for R1?')
    assert derive([source_row])[1]['skipped_repeated_target'] == 1


@pytest.mark.parametrize('mutation', ['native', 'gold', 'malformed', 'missing_image', 'provenance', 'split'])
def test_invalid_sources_fail_closed(source_row, mutation):
    if mutation == 'native':
        source_row['provenance']['evidence_mode'] = 'native_facts_supplied_in_prompt'
    elif mutation == 'gold':
        source_row['gold'][1]['value'] = '33K'
    elif mutation == 'malformed':
        source_row['messages'][2]['content'] = [{'type': 'tool_call'}]
    elif mutation == 'missing_image':
        source_row['messages'][1]['content'].pop(0)
    elif mutation == 'provenance':
        del source_row['provenance']['license']
    elif mutation == 'split':
        source_row['split'] = 'unknown'
    with pytest.raises(ValueError):
        derive([source_row])


def test_cross_split_family_rejected(source_row):
    other = copy.deepcopy(source_row)
    other.update(id='another', split='train')
    with pytest.raises(ValueError, match='crosses'):
        derive([source_row, other])


def test_build_is_new_snapshot_with_valid_relocated_images(tmp_path, source_row):
    source = tmp_path / 'source.jsonl'
    source.write_text(json.dumps(source_row) + '\n')
    (tmp_path / 'sheet.png').write_bytes(b'fixture image')
    before = source.read_bytes()
    output = tmp_path / 'diagnostics'
    manifest = build(source, output)
    assert source.read_bytes() == before
    assert manifest['counts']['derived_rows'] == 3
    rows = load_gold(output / 'probes.jsonl')
    assert rows[0]['messages'][1]['content'][0]['image'] == '../sheet.png'
    assert str(tmp_path) not in (output / 'manifest.json').read_text()
    with pytest.raises(ValueError, match='exists'):
        build(source, output)


def test_missing_image_and_duplicate_json_fail_before_writing(tmp_path, source_row):
    source = tmp_path / 'source.jsonl'
    output = tmp_path / 'diagnostics'
    source.write_text(json.dumps(source_row) + '\n')
    with pytest.raises(ValueError, match='image is missing'):
        build(source, output)
    assert not output.exists()
    source.write_text('{"id":"a","id":"b"}\n')
    with pytest.raises(ValueError, match='Duplicate JSON'):
        build(source, output)
    assert not output.exists()


def benchmark_records(probes):
    return [{'key': f"{row['id']}#{index}", 'attempted': True, 'status': 'success',
             'finish_reason': 'stop', 'output': label['value']}
            for row in probes for index, label in enumerate(row['gold'])]


def test_target_reducer_counts_completion_separately_from_correctness(source_row):
    from schematic_model.conversation_diagnostics import summarize_probes
    probes, _ = derive([source_row])
    records = benchmark_records(probes)
    records[1]['output'] = 'wrong prefix does not affect target-only score'
    records[2]['output'] = '220K'  # must not match 22K
    records[4].update(status='failure', output='', error={'code': 'generation_failed'}, finish_reason='length')
    report = summarize_probes(probes, records, history='gold')
    assert report['paired_sources'] == 1
    assert report['history_label'] == 'teacher-forced source answers'
    assert report['variants']['standalone']['value_accuracy'] == 1
    assert report['variants']['explicit_second_turn']['completed'] == 1
    assert report['variants']['explicit_second_turn']['value_accuracy'] == 0
    assert report['variants']['contextual_followup']['targets'] == 1
    assert report['variants']['contextual_followup']['completed'] == 0
    assert report['variants']['contextual_followup']['value_accuracy'] == 0


def test_generated_dependency_remains_in_denominator(source_row):
    from schematic_model.conversation_diagnostics import summarize_probes
    probes, _ = derive([source_row])
    records = benchmark_records(probes)
    records[1].update(status='failure', output='', error={'code': 'generation_failed'}, finish_reason='error')
    records[2].update(status='failure', attempted=False, output='', error={'code': 'dependency_failed'},
                      finish_reason='dependency_failed')
    report = summarize_probes(probes, records, metadata={'history': 'generated'})
    assert report['variants']['explicit_second_turn']['blocked'] == 1
    assert report['variants']['explicit_second_turn']['targets'] == 1
    assert report['variants']['explicit_second_turn']['value_accuracy'] == 0
    with pytest.raises(ValueError, match='Gold history'):
        summarize_probes(probes, records, history='gold')
    records[2].update(status='success', attempted=True, output='22K', finish_reason='stop')
    with pytest.raises(ValueError, match='contradicts'):
        summarize_probes(probes, records, history='generated')


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'extra', 'partial', 'unknown_status', 'wrong_key'])
def test_reducer_rejects_incomplete_or_invalid_records(source_row, change):
    from schematic_model.conversation_diagnostics import summarize_probes
    probes, _ = derive([source_row])
    records = benchmark_records(probes)
    if change == 'missing':
        records.pop()
    elif change == 'duplicate':
        records.append(copy.deepcopy(records[0]))
    elif change == 'extra':
        records.append({**records[0], 'key': 'unknown#0'})
    elif change == 'partial':
        records[0]['finish_reason'] = 'length'
    elif change == 'unknown_status':
        records[0]['status'] = 'unknown'
    elif change == 'wrong_key':
        records = {'mismatched': records[0]}
    with pytest.raises(ValueError):
        summarize_probes(probes, records, history='gold')


def test_reducer_requires_history_and_complete_matching_triplets(source_row):
    from schematic_model.conversation_diagnostics import summarize_probes
    probes, _ = derive([source_row])
    records = benchmark_records(probes)
    with pytest.raises(ValueError, match='history policy'):
        summarize_probes(probes, records)
    with pytest.raises(ValueError, match='history policy'):
        summarize_probes(probes, records, history='gold', metadata={'history': 'generated'})
    with pytest.raises(ValueError, match='matched triplets'):
        summarize_probes(probes[:1], records[:1], history='gold')
    probes[1]['gold'][1]['value'] = '33K'
    with pytest.raises(ValueError, match='matched triplets'):
        summarize_probes(probes, records, history='gold')


def test_snapshot_permissions_are_private_even_under_permissive_umask(tmp_path, source_row):
    import os
    import stat
    source = tmp_path / 'source.jsonl'
    source.write_text(json.dumps(source_row) + '\n')
    (tmp_path / 'sheet.png').write_bytes(b'fixture image')
    output = tmp_path / 'private-probes'
    previous = os.umask(0)
    try:
        build(source, output)
    finally:
        os.umask(previous)
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in output.iterdir())


def test_prediction_cli_binds_metadata_dataset_and_history(tmp_path, source_row, monkeypatch, capsys):
    import hashlib
    from schematic_model.conversation_diagnostics import main
    probes, _ = derive([source_row])
    source = tmp_path / 'probes.jsonl'
    source.write_text(''.join(json.dumps(row) + '\n' for row in probes))
    predictions = tmp_path / 'predictions.jsonl'
    predictions.write_text(''.join(json.dumps(record) + '\n' for record in benchmark_records(probes)))
    metadata = predictions.with_suffix('.jsonl.meta.json')
    metadata.write_text(json.dumps({'history': 'gold', 'dataset_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}))
    monkeypatch.setattr('sys.argv', ['diagnostics', '--source', str(source), '--predictions', str(predictions)])
    main()
    assert json.loads(capsys.readouterr().out)['history'] == 'gold'
    metadata.write_text(json.dumps({'history': 'gold', 'dataset_sha256': 'different'}))
    with pytest.raises(SystemExit):
        main()
    assert 'different probe dataset' in capsys.readouterr().err


@pytest.mark.parametrize('change', ['image', 'system', 'control_question', 'control_answer', 'control_gold'])
def test_reducer_rejects_unmatched_contexts(source_row, change):
    from schematic_model.conversation_diagnostics import summarize_probes
    probes, _ = derive([source_row])
    if change == 'image':
        probes[2]['messages'][1]['content'][0]['image'] = 'different.png'
    elif change == 'system':
        probes[1]['messages'][0]['content'][0]['text'] = 'A different instruction.'
    elif change == 'control_question':
        probes[2]['messages'][1]['content'][1]['text'] = 'Ask about a different component.'
    elif change == 'control_answer':
        probes[2]['messages'][2]['content'][0]['text'] = 'Different prior answer.'
    else:
        probes[2]['gold'][0]['value'] = '33K'
    with pytest.raises(ValueError, match='differs'):
        summarize_probes(probes, benchmark_records(probes), history='gold')


@pytest.mark.parametrize('change', ['modified_image', 'duplicate_page', 'missing_page', 'missing_hash'])
def test_build_requires_unique_matching_image_fingerprint(tmp_path, source_row, change):
    source = tmp_path / 'source.jsonl'
    (tmp_path / 'sheet.png').write_bytes(b'fixture image')
    if change == 'modified_image':
        (tmp_path / 'sheet.png').write_bytes(b'altered image')
    elif change == 'duplicate_page':
        source_row['provenance']['images'].append(copy.deepcopy(source_row['provenance']['images'][0]))
    elif change == 'missing_page':
        source_row['provenance']['images'][0]['page'] = 2
    else:
        del source_row['provenance']['images'][0]['sha256']
    source.write_text(json.dumps(source_row) + '\n')
    output = tmp_path / 'diagnostics'
    with pytest.raises(ValueError, match='fingerprint'):
        build(source, output)
    assert not output.exists()
