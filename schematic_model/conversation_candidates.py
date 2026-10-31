"""Build private standalone/multiturn candidates with matched answer content.

Uses existing source answers, images and frozen validation families. No model
calls, new factual labels, tokenizer claims, or training are performed.
"""
# cspell:words multiturn multiset
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile

from PIL import Image

from evaluate import contextual_refusal, load_gold, question_text, validate_gold_rows
from schematic_model.conversation_diagnostics import VALUE_QUESTION, _text, _verify_source_images, derive


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def validate_sources(rows, split):
    """Accept only the source format whose labels and reference rewrites we know."""
    _, counts = derive(rows)
    if counts.get('paired_source_rows') != len(rows):
        raise ValueError('Every source needs two supported value questions with distinct references')
    for row in rows:
        if row['split'] != split:
            raise ValueError('Source split differs from its declared input role')
        if [gold['type'] for gold in row['gold']] != ['value', 'value', 'refuse', 'refuse']:
            raise ValueError('Require exactly two values followed by two refusals')
        if not re.fullmatch('[0-9a-f]{64}', row['provenance']['sha256']):
            raise ValueError('Require a source-file SHA-256')
        for turn in (2, 3):
            verdict = contextual_refusal(question_text(row, turn), _text(row['messages'][2 + 2 * turn]))
            if not verdict['accepted']:
                raise ValueError('Source refusal does not match its question and evidence scope')


def _identities(rows, field):
    if field == 'source_sha256':
        return {row['provenance']['sha256'] for row in rows}
    if field == 'image_sha256':
        return {image['sha256'] for row in rows for image in row['provenance']['images']
                if image['page'] == row['provenance']['page']}
    return {row[field] for row in rows if row.get(field)}


def check_holdouts(train, validation):
    for field in ('id', 'design_id', 'source_id', 'family_id', 'source_sha256', 'image_sha256'):
        if _identities(train, field) & _identities(validation, field):
            raise ValueError(f'Training and validation overlap in {field}')


def variants(row):
    matches = [VALUE_QUESTION.fullmatch(question_text(row, turn)) for turn in (0, 1)]
    if not all(matches):
        raise ValueError('Unsupported source value question')
    refs = [next(group for group in match.groups() if group) for match in matches if match]
    source_turns = [0, 1, 0, 1, 2, 3]
    explicit = [f'Read the value shown next to {refs[turn]}.' for turn in (0, 1, 0, 1)]
    contextual = [explicit[0], f'And {refs[1]}?',
                  f'Returning to {refs[0]}: what value is marked for it?',
                  f'Correction: I meant {refs[1]}, not {refs[0]}. What value is marked for it?']
    suffix = [question_text(row, turn) for turn in (2, 3)]
    explicit += suffix
    contextual += suffix
    image = next(copy.deepcopy(part) for message in row['messages'] for part in message['content']
                 if part['type'] == 'image')
    multi = copy.deepcopy(row)
    multi['id'] += '-multiturn'
    multi['messages'] = [copy.deepcopy(row['messages'][0])]
    multi['gold'] = []
    multi['experiment'] = {'arm': 'multiturn', 'source_row_id': row['id'], 'source_turns': source_turns}
    singles, prefixes = [], []
    for index, original_turn in enumerate(source_turns):
        answer = row['messages'][2 + 2 * original_turn]
        gold = row['gold'][original_turn]
        content = [{'type': 'text', 'text': contextual[index]}]
        if index == 0:
            content.insert(0, copy.deepcopy(image))
        multi['messages'].extend([{'role': 'user', 'content': content}, copy.deepcopy(answer)])
        multi['gold'].append(copy.deepcopy(gold))
        prefix = copy.deepcopy(multi)
        prefix['id'] = row['id'] + f'-multiturn-{index}'
        prefix['experiment']['source_turns'] = source_turns[:index + 1]
        prefixes.append(prefix)
        single = copy.deepcopy(row)
        single['id'] += f'-standalone-{index}'
        single['messages'] = [copy.deepcopy(row['messages'][0]),
                              {'role': 'user', 'content': [copy.deepcopy(image), {'type': 'text', 'text': explicit[index]}]},
                              copy.deepcopy(answer)]
        single['gold'] = [copy.deepcopy(gold)]
        single['experiment'] = {'arm': 'standalone', 'source_row_id': row['id'], 'source_turns': [original_turn]}
        singles.append(single)
    return singles, prefixes


def answer_inventory(rows):
    """Only the final answer is a supervised target; earlier answers are context."""
    answers = [_canonical(row['messages'][-1]['content']) for row in rows]
    labels = [_canonical(row['gold'][-1]) for row in rows]
    return {'assistant_answers': len(answers),
            'answer_multiset_sha256': _digest(_canonical(sorted(answers)).encode()),
            'label_multiset_sha256': _digest(_canonical(sorted(labels)).encode()),
            'answer_utf8_bytes': sum(len(answer.encode()) for answer in answers),
            'task_counts': dict(Counter(row['gold'][-1]['type'] for row in rows))}


def _read_verified(path, split):
    payload = path.read_bytes()
    rows = load_gold(path)
    if path.read_bytes() != payload:
        raise ValueError('Source changed while reading')
    validate_sources(rows, split)
    _verify_source_images(rows, path)
    return rows, _digest(payload)


def _relocate(rows, source, directory):
    rows = copy.deepcopy(rows)
    for row in rows:
        declared = next(image for image in row['provenance']['images']
                        if image['page'] == row['provenance']['page'])
        for message in row['messages']:
            for part in message['content']:
                if part['type'] != 'image':
                    continue
                path = Path(part['image'])
                if not path.is_absolute():
                    path = source.parent / path
                data = path.read_bytes()
                if _digest(data) != declared['sha256']:
                    raise ValueError('Image changed since preflight')
                with Image.open(io.BytesIO(data)) as image:
                    suffix = {'PNG': '.png', 'JPEG': '.jpg'}.get(image.format or '')
                    image.verify()
                if suffix is None:
                    raise ValueError('Expected PNG or JPEG source image')
                name = declared['sha256'] + suffix
                destination = directory / name
                if not destination.exists():
                    with destination.open('xb') as handle:
                        os.chmod(destination, 0o600)
                        handle.write(data)
                part['image'] = '../images/' + name
    return rows


def _write(path, payload):
    with path.open('x', encoding='utf-8') as stream:
        os.chmod(path, 0o600)
        stream.write(payload)


def build(train_source, validation_source, output, *, families=24, seed=3407, accumulation=4):
    if type(families) is not int or not 1 <= families <= 128:
        raise ValueError('Choose 1..128 training families')
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError('Seed must be an unsigned 32-bit integer')
    if type(accumulation) is not int or not 1 <= accumulation <= 64:
        raise ValueError('Choose 1..64 accumulation steps for both arms')
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError('Output exists; choose a fresh candidate directory')
    allowed = [Path.cwd().resolve() / 'outputs', Path.cwd().resolve() / 'data' / 'training']
    if not any(root in output.resolve().parents for root in allowed):
        raise ValueError('Keep candidates under ignored outputs/ or data/training/')
    train_source, validation_source = Path(train_source), Path(validation_source)
    train, train_hash = _read_verified(train_source, 'train')
    validation, val_hash = _read_verified(validation_source, 'val')
    check_holdouts(train, validation)
    available = {row['family_id'] for row in train}
    if families > len(available):
        raise ValueError('Requested more families than the source contains')
    chosen = sorted(available, key=lambda family: (_digest(f'{seed}:{family}'.encode()), family))[:families]
    selected = sorted((row for row in train if row['family_id'] in chosen), key=lambda row: row['id'])
    if len(selected) > 512:
        raise ValueError('Selected families exceed 512 source rows')
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix='.conversation-candidates-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'candidate'
        stage.mkdir(mode=0o700)
        images = stage / 'images'
        images.mkdir(mode=0o700)
        selected = _relocate(selected, train_source, images)
        validation = _relocate(validation, validation_source, images)
        single, multi = [], []
        for row in selected:
            singles, prefixes = variants(row)
            single.extend(singles)
            multi.extend(prefixes)
        if answer_inventory(single) != answer_inventory(multi):
            raise ValueError('Candidate answer/label inventories differ')
        datasets = {}
        for arm, rows in [('standalone', single), ('multiturn', multi)]:
            directory = stage / arm
            directory.mkdir(mode=0o700)
            for split, content in [('train', rows), ('val', validation)]:
                validate_gold_rows(content)
                payload = ''.join(_canonical(row) + '\n' for row in content)
                _write(directory / f'{split}.jsonl', payload)
                datasets[f'{arm}/{split}.jsonl'] = {'sha256': _digest(payload.encode()), 'rows': len(content)}
        manifest = {'schema_version': 1, 'source_sha256': {'train': train_hash, 'validation': val_hash},
                    'seed': seed, 'selected_families': chosen, 'selected_source_rows': [row['id'] for row in selected],
                    'datasets': datasets, 'answer_inventory_each_arm': answer_inventory(single),
                    'images': [{'file': 'images/' + path.name, 'sha256': _digest(path.read_bytes())}
                               for path in sorted(images.iterdir())],
                    'recommended_budget': {'per_device_train_batch_size': 1,
                        'last_response_only': True,
                        'gradient_accumulation': {'standalone': accumulation, 'multiturn': accumulation},
                        'optimizer_steps_per_epoch_each_arm': math.ceil(len(single) / accumulation),
                        'assistant_answers_per_complete_optimizer_step_each_arm': accumulation,
                        'epochs_and_max_steps': 'Use identical values for both arms.'},
                    'validation_policy': 'Same original validation rows/answers in both arms; only image references relocated.',
                    'limitations': ['Answers and labels are inherited, not independently source-reverified.',
                        'Matched assistant answer content does not prove equal processed supervised tokens.',
                        'Audit actual labels, answer boundaries, truncation and loss normalization before training.',
                        'Each arm repeats the image six times per source row; multiturn prefixes add context.',
                        'Both arms require last_response_only; prior assistant answers must remain masked.',
                        'No training or measured quality comparison has been performed.']}
        _write(stage / 'manifest.json', json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        if output.exists() or output.is_symlink():
            raise ValueError('Output appeared during candidate creation')
        stage.rename(output)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-source', type=Path, required=True)
    parser.add_argument('--validation-source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--families', type=int, default=24)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--accumulation', type=int, default=4, help='same accumulation steps for both arms')
    args = vars(parser.parse_args(argv))
    args['output'] = args.pop('out')
    result = build(**args)
    print(json.dumps({'datasets': result['datasets'], 'answer_inventory_each_arm': result['answer_inventory_each_arm'],
                      'recommended_budget': result['recommended_budget']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
