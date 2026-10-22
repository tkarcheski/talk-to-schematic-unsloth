"""Build offline, paired value probes from existing image-only gold conversations.

These derived probes are diagnostic artifacts, not additions to frozen benchmarks
or training data. Gold and provenance are copied, never generated. Supported
source questions intentionally match the corpus templates exactly; ambiguous
questions are counted as unsupported instead of guessing a component identity.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from evaluate import _read_jsonl, grade, load_gold, question_text, strip, validate_gold_rows

VALUE_QUESTION = re.compile(r"(?:Read the value shown next to ([A-Z][A-Z0-9_$-]*)\.|"
                            r"And what value is marked for ([A-Z][A-Z0-9_$-]*)\?)")


def _text(message):
    content = message.get('content')
    if not isinstance(content, list) or not content:
        raise ValueError('Messages require nonempty typed content')
    parts = []
    for block in content:
        if not isinstance(block, dict):
            raise ValueError('Malformed content block')
        if block.get('type') == 'text' and isinstance(block.get('text'), str) and block['text'].strip():
            parts.append(block['text'])
        elif (block.get('type') == 'image' and message['role'] == 'user'
              and isinstance(block.get('image'), str) and block['image'].strip()):
            continue
        else:
            raise ValueError('Unsupported or malformed content block')
    if not parts:
        raise ValueError('Messages require nonempty text')
    return '\n'.join(parts)


def derive(rows):
    """Return paired evaluator-compatible rows and coverage counts without I/O.

    The two-turn probes include the original first value question and answer.
    Standalone and explicit follow-up use identical target question text. The
    contextual version uses 'And REF?' after a value question, never a guessed
    pronoun referent. Evaluation must score the target turn separately and state
    whether preceding answers were teacher-forced or generated.
    """
    validate_gold_rows(rows)
    counts: Counter[str] = Counter()
    output = []
    families: dict[str, str] = {}
    for row in rows:
        for field in ('design_id', 'family_id', 'split'):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError('Missing source identity or split')
        if row['split'] not in {'train', 'val', 'test'}:
            raise ValueError('Unknown split')
        previous = families.setdefault(row['family_id'], row['split'])
        if previous != row['split']:
            raise ValueError('A family crosses source splits')
        provenance = row.get('provenance')
        if not isinstance(provenance, dict) or provenance.get('evidence_mode') != 'image_only_values_and_refusals':
            raise ValueError('Only declared image-only source rows are supported')
        for field in ('repository', 'source_path', 'sha256', 'license'):
            if not isinstance(provenance.get(field), str) or not provenance[field]:
                raise ValueError('Incomplete source provenance')
        messages = row.get('messages', [])
        if not messages or messages[0]['role'] != 'system':
            raise ValueError('A source system message is required')
        for message in messages:
            _text(message)
        images = [(i, block) for i, message in enumerate(messages) for block in message['content']
                  if block['type'] == 'image']
        if len(images) != 1 or images[0][0] != 1:
            raise ValueError('Exactly one image on the first user turn is required')
        counts['source_rows'] += 1
        counts[f"source_rows_with_{len(row['gold'])}_turns"] += 1
        counts['source_turns'] += len(row['gold'])
        counts['source_value_turns'] += sum(label['type'] == 'value' for label in row['gold'])
        counts['source_value_followups'] += sum(label['type'] == 'value' for label in row['gold'][1:])
        if len(row['gold']) < 2 or any(label['type'] != 'value' for label in row['gold'][:2]):
            counts['skipped_no_initial_value_pair'] += 1
            continue
        refs = [VALUE_QUESTION.fullmatch(question_text(row, i)) for i in range(2)]
        if not all(refs):
            counts['skipped_unsupported_question'] += 1
            continue
        references = [next(group for group in match.groups() if group) for match in refs if match is not None]
        if references[0] == references[1]:
            counts['skipped_repeated_target'] += 1
            continue
        # Match the original source-derived answer template too. A gold value
        # alone does not identify a component or verify a rewritten answer.
        for index, ref in enumerate(references):
            if _text(messages[2 + 2 * index]) != f"{ref} is marked {row['gold'][index]['value']}.":
                raise ValueError('Source answer does not agree with its value target')
        target = references[1]
        for variant in ('standalone', 'explicit_second_turn', 'contextual_followup'):
            probe = copy.deepcopy(row)
            probe['id'] = row['id'] + '-diagnostic-' + variant
            prompt = f'And {target}?' if variant == 'contextual_followup' else f'Read the value shown next to {target}.'
            content = [{'type': 'text', 'text': prompt}]
            if variant == 'standalone':
                content.insert(0, copy.deepcopy(images[0][1]))
                prefix = [messages[0]]
                labels = [row['gold'][1]]
            else:
                prefix = messages[:3]
                labels = row['gold'][:2]
            probe['messages'] = copy.deepcopy(prefix) + [{'role': 'user', 'content': content}, copy.deepcopy(messages[4])]
            probe['gold'] = copy.deepcopy(labels)
            probe['diagnostic'] = {'source_id': row['id'], 'source_target_turn': 1, 'variant': variant,
                                   'target_turn': len(labels) - 1, 'target_refdes': target,
                                   'scope': 'derived diagnostic; not independent benchmark evidence'}
            output.append(probe)
        counts['paired_source_rows'] += 1
    counts['derived_rows'] = len(output)
    return output, dict(sorted(counts.items()))


def build(source: Path, output: Path):
    """Write a new local snapshot; do not alter or overwrite the source."""
    if output.exists():
        raise ValueError('Output exists; choose a new diagnostic snapshot')
    source_bytes = source.read_bytes()
    rows = load_gold(source)
    if source.read_bytes() != source_bytes:
        raise ValueError('Source changed during loading')
    probes, counts = derive(rows)
    if not probes:
        raise ValueError('No supported value pairs; inspect source conversation coverage')
    # Keep image references valid after relocation without recording local
    # absolute paths in the new artifact.
    for probe in probes:
        for message in probe['messages']:
            for block in message['content']:
                if block['type'] == 'image':
                    original = Path(block['image'])
                    resolved = original if original.is_absolute() else source.parent / original
                    if not resolved.is_file():
                        raise ValueError('Source image is missing')
                    block['image'] = os.path.relpath(resolved.resolve(), output.resolve())
    payload = ''.join(json.dumps(row, sort_keys=True) + '\n' for row in probes)
    manifest = {'schema_version': 1, 'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
                'probes_sha256': hashlib.sha256(payload.encode()).hexdigest(), 'counts': counts,
                'scope': 'Private derived diagnostic; source labels inherited, not independently reverified. '
                         'Preserve family splits. Not a training snapshot or replacement frozen benchmark.'}
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name, content in [('probes.jsonl', payload), ('manifest.json', json.dumps(manifest, indent=2) + '\n')]:
        # cspell:ignore WRONLY CREAT
        descriptor = os.open(output / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            handle.write(content)
    return manifest


def summarize_probes(rows, records, *, history=None, metadata=None):
    """Reduce only target turns from complete failure-accounted benchmark records.

    History must be explicitly declared or taken from benchmark metadata. Literal
    value matching uses the existing evaluator's token-boundary scorer; it does
    not certify absence of contradictory prose or electrical reasoning quality.
    """
    metadata = {} if metadata is None else metadata
    if not isinstance(metadata, dict):
        raise ValueError('Benchmark metadata must be an object')
    bound_history = metadata.get('history')
    if history is None:
        history = bound_history
    if history not in {'gold', 'generated'} or (bound_history is not None and bound_history != history):
        raise ValueError('Declare a valid history policy consistent with benchmark metadata')
    expected = validate_gold_rows(rows)
    if isinstance(records, dict):
        entries = list(records.values())
        if any(not isinstance(record, dict) or record.get('key') != key for key, record in records.items()):
            raise ValueError('Record mapping keys disagree with prediction keys')
    elif isinstance(records, list):
        entries = records
    else:
        raise ValueError('Benchmark records must be a list or keyed mapping')
    by_key = {}
    for record in entries:
        if not isinstance(record, dict) or not isinstance(record.get('key'), str):
            raise ValueError('Benchmark record requires a prediction key')
        key = record['key']
        if key in by_key:
            raise ValueError('Duplicate prediction key')
        if type(record.get('attempted')) is not bool or not isinstance(record.get('output'), str):
            raise ValueError('Malformed benchmark attempt or output')
        if record.get('status') == 'success':
            if not record['attempted'] or record.get('finish_reason') != 'stop' or not strip(record['output']):
                raise ValueError('Successful record must contain a complete attempted answer')
        elif record.get('status') == 'failure':
            if record['output'] != '' or not isinstance(record.get('error'), dict):
                raise ValueError('Failed records require empty scored output and structured error')
        else:
            raise ValueError('Unknown benchmark status')
        by_key[key] = record
    if set(by_key) != expected:
        raise ValueError('Prediction coverage mismatch: missing or unexpected turn keys')
    variants = ('standalone', 'explicit_second_turn', 'contextual_followup')
    totals = {variant: {'targets': 0, 'attempted': 0, 'completed': 0, 'value_correct': 0,
                        'failed': 0, 'blocked': 0} for variant in variants}
    pairs: dict[str, dict[str, Any]] = {}
    for row in rows:
        diagnostic = row.get('diagnostic', {})
        variant = diagnostic.get('variant')
        target = diagnostic.get('target_turn')
        source_id = diagnostic.get('source_id')
        if (variant not in variants or type(target) is not int or target != len(row['gold']) - 1
                or target != (0 if variant == 'standalone' else 1)
                or not isinstance(source_id, str) or not source_id
                or row['gold'][target]['type'] != 'value'):
            raise ValueError('Invalid paired diagnostic target metadata')
        pair = pairs.setdefault(source_id, {})
        if variant in pair:
            raise ValueError('Duplicate variant for a diagnostic source')
        pair[variant] = (row['gold'][target], row.get('family_id'), row.get('split'),
                         diagnostic.get('target_refdes'))
        if history == 'generated':
            blocked = False
            for index in range(len(row['gold'])):
                record = by_key[f"{row['id']}#{index}"]
                if blocked and (record['attempted'] or record['status'] != 'failure'):
                    raise ValueError('Generated history contradicts a prior failed turn')
                if not blocked and not record['attempted']:
                    raise ValueError('Unattempted turn has no failed dependency')
                blocked = blocked or record['status'] == 'failure'
        elif any(not by_key[f"{row['id']}#{index}"]['attempted'] for index in range(len(row['gold']))):
            raise ValueError('Gold history cannot contain dependency-blocked turns')
        record = by_key[f"{row['id']}#{target}"]
        total = totals[variant]
        total['targets'] += 1
        total['attempted'] += record['attempted']
        complete = record['status'] == 'success'
        total['completed'] += complete
        total['failed'] += not complete
        total['blocked'] += not record['attempted']
        total['value_correct'] += bool(complete and grade(row['gold'][target], record['output'])['value.acc'])
    for pair in pairs.values():
        if set(pair) != set(variants) or any(value != pair['standalone'] for value in pair.values()):
            raise ValueError('Require complete matched triplets with identical source target gold and split')
    report_variants = {variant: {**total, 'completion_rate': total['completed'] / total['targets'],
                                'value_accuracy': total['value_correct'] / total['targets']}
                       for variant, total in totals.items()}
    return {'schema_version': 1, 'history': history,
            'history_label': 'teacher-forced source answers' if history == 'gold' else 'generated answers',
            'paired_sources': len(pairs), 'variants': report_variants,
            'scoring': 'Target turns only; exact literal value token match using the shared evaluator. '
                       'All failed and blocked targets remain in denominators. '
                       'Literal matching does not certify contradictory prose or electrical reasoning.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--audit-only', action='store_true')
    mode.add_argument('--predictions', type=Path, help='Reduce benchmark records for an existing probe source')
    parser.add_argument('--history', choices=('gold', 'generated'))
    args = parser.parse_args()
    if args.predictions:
        metadata_path = args.predictions.with_suffix(args.predictions.suffix + '.meta.json')
        metadata = None
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if metadata.get('dataset_sha256') != hashlib.sha256(args.source.read_bytes()).hexdigest():
                parser.error('Benchmark metadata is bound to a different probe dataset')
        report = summarize_probes(load_gold(args.source), _read_jsonl(args.predictions),
                                  history=args.history, metadata=metadata)
    elif args.audit_only:
        report = derive(load_gold(args.source))[1]
    else:
        if args.out is None:
            parser.error('--out is required unless --audit-only or --predictions is selected')
        report = build(args.source, args.out)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
