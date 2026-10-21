# cspell:words multiturn rollouts
"""Public source-correct page changes, kept separate from held-out routing scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

from schematic_model.corpus import _split, parse_eagle, publisher
from schematic_model.multiturn_data import current_context, training_rows
from schematic_model.render_eagle import render_eagle, source_layout
from schematic_model.training import sha256


def build(source, metadata, output):
    if output.exists():
        raise ValueError('Use a new page-change diagnostic output')
    info = json.loads(metadata.read_text())
    root = metadata.parent.resolve()
    source = source.resolve()
    if not source.is_relative_to(root):
        raise ValueError('Source must belong to metadata directory')
    relative = str(source.relative_to(root))
    files = {f['path']: f for f in info['files']}
    if relative not in files or sha256(source) != files[relative]['sha256']:
        raise ValueError('Source hash differs from pinned metadata')
    license_path = root / info['license_path']
    if sha256(license_path) != files[info['license_path']]['sha256']:
        raise ValueError('License changed')
    if info['license'] not in {'CC-BY-SA-3.0', 'CC-BY-SA-4.0'} or _split(info['repository']) != 'train':
        raise ValueError('Only reviewed public train-family sources are allowed')
    parsed = parse_eagle(source)
    if len(parsed['sheets']) < 2:
        raise ValueError('Need a real multi-page source')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + '.partial-', dir=output.parent))
    try:
        candidates = []
        for sheet in parsed['sheets'][:2]:
            page = sheet['page']
            _, regions = source_layout(source, page, attribution=publisher(info['repository'])['footer'])
            parts = [p for p in sheet['bom'] if p['refdes'].startswith(('U', 'IC', 'R', 'C')) and p['value'] and p['refdes'] in regions['parts']]
            if not parts:
                raise ValueError('Both pages need a source-grounded valued component target')
            part = parts[0]
            name = f'page-{page}.png'
            render_eagle(source, page, staging / name, attribution=publisher(info['repository'])['footer'])
            candidates.append({'id': 'page-change-' + str(page), 'design_id': parsed['sha256'][:16], 'repository': info['repository'],
                'split': 'train', 'page': page, 'part': part, 'box': regions['parts'][part['refdes']],
                'image': str((output / name).resolve()), 'image_sha256': sha256(staging / name),
                'provenance': {'repository': info['repository'], 'commit': info['revision'], 'source_path': relative,
                               'sha256': parsed['sha256'], 'license': info['license']}})
        first, second = candidates
        ref = second['part']['refdes']
        case = {'id': 'page-change-' + parsed['sha256'][:16], 'family_id': info['repository'], 'split': 'train',
            'candidate': first, 'initial_context': current_context(first, False), 'kind': 'page_change',
            'scope': 'TRAIN-DIAGNOSTIC; not a held-out score; source-correct page transition', 'turns': [
                {'question': f"Show me {first['part']['refdes']}.", 'tool': 'show_label', 'arguments': {'label': first['part']['refdes']}, 'answer': f"Showing {first['part']['refdes']}.", 'allow_navigation': True},
                {'question': f'I changed to page {second["page"]}. Read the value of {ref} here without moving the view.', 'page_candidate': second, 'tool': None, 'answer': f'{ref} is marked {second["part"]["value"]}.', 'expected_value': second['part']['value'], 'allow_navigation': False},
                {'question': f'Discuss only: what startup voltage was measured at {ref} on this page?', 'tool': None, 'answer': 'The schematic does not provide measured startup voltage. A bench measurement is required.', 'refusal': True, 'allow_navigation': False},
                {'question': f'Now explicitly show {ref} on this page.', 'tool': 'show_label', 'arguments': {'label': ref}, 'answer': f'Showing {ref}.', 'allow_navigation': True}]}
        (staging / 'diagnostic-cases.jsonl').write_text(json.dumps(case) + '\n')
        (staging / 'train-candidates.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in training_rows(case)))
        manifest = {'source_sha256': parsed['sha256'], 'family_id': info['repository'], 'split': 'train',
                    'source_revision': info['revision'], 'license_sha256': sha256(license_path),
                    'status': 'Separate train-family workflow diagnostic; not merged into any training snapshot',
                    'artifacts': {p.name: sha256(p) for p in staging.iterdir() if p.is_file()}}
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.metadata, args.out), indent=2))


if __name__ == '__main__':
    main()
