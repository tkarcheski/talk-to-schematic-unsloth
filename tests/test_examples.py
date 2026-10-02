"""Reference-dialogue export integrity; CPU only and no downloads."""
import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.export_examples import export_examples


class ExampleExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'data'
        self.data.mkdir()
        self.output = self.root / 'docs' / 'examples.md'
        self.images = self.root / 'docs' / 'images'
        self.manifest = self.root / 'docs' / 'manifest.json'
        png = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aWZkAAAAASUVORK5CYII=')
        (self.data / 'images').mkdir()
        (self.data / 'images' / 'board.png').write_bytes(png)
        source = self.data / 'sources' / 'adafruit__Example-PCB'
        source.mkdir(parents=True)
        native = (Path(__file__).parent / 'fixtures' / 'real' / 'vcnl4000.sch').read_bytes()
        (source / 'board.sch').write_bytes(native)
        (source / 'license.txt').write_text('Attribution-ShareAlike 3.0 Unported')
        (source / 'source.json').write_text(json.dumps({'revision': 'a' * 40, 'license_path': 'license.txt'}))
        messages = [{'role': 'system', 'content': [{'type': 'text', 'text': 'Use supplied evidence.'}]}]
        for turn in range(6):
            question = ('Extracted native evidence:\n{}\n\n' if turn == 0 else '') + f'Read C1, turn {turn + 1}.'
            messages += [{'role': 'user', 'content': ([{'type': 'image', 'image': 'images/board.png'}] if turn == 0 else []) + [{'type': 'text', 'text': question}]},
                         {'role': 'assistant', 'content': [{'type': 'text', 'text': 'C1 is 10uF.'}]}]
        self.row = {'id': 'example-p1', 'design_id': 'example', 'family_id': 'adafruit/Example-PCB',
                    'split': 'test', 'messages': messages, 'gold': [{'type': 'value', 'value': '10uF'}] * 6,
                    'provenance': {'repository': 'adafruit/Example-PCB', 'commit': 'a' * 40,
                                   'source_path': 'board.sch', 'sha256': hashlib.sha256(native).hexdigest(),
                                   'license': 'CC-BY-SA-3.0', 'render_method': 'test_fixture',
                                   'evidence_mode': 'native_facts_supplied_in_prompt',
                                   'images': [{'path': 'images/board.png', 'sha256': hashlib.sha256(png).hexdigest()}]}}
        for split in ('train', 'val', 'test'):
            (self.data / f'{split}.jsonl').write_text(json.dumps(self.row) + '\n' if split == 'test' else '')

    def export(self, **kwargs):
        return export_examples(self.data, self.output, self.images, self.manifest, count=kwargs.pop('count', 1), **kwargs)

    def test_full_dialogue_exact_image_and_deterministic_manifest(self):
        original = {p.name: p.read_bytes() for p in self.data.glob('*.jsonl')}
        report = self.export()
        text = self.output.read_text()
        self.assertEqual(text.count('**User '), 6)
        self.assertEqual(text.count('**Assistant '), 6)
        self.assertIn('not model transcripts', text)
        self.assertIn('Extracted native evidence:', text)
        self.assertIn('/blob/' + 'a' * 40 + '/board.sch', text)
        self.assertEqual((self.images / 'board.png').read_bytes(), (self.data / 'images' / 'board.png').read_bytes())
        self.assertEqual(report['markdown_sha256'], hashlib.sha256(self.output.read_bytes()).hexdigest())
        self.assertEqual(report, self.export(overwrite=True))
        self.assertEqual(original, {p.name: p.read_bytes() for p in self.data.glob('*.jsonl')})

    def test_insufficient_board_count_writes_nothing(self):
        with self.assertRaisesRegex(ValueError, 'only 1 available'):
            self.export(count=25)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.images.exists())

    def test_incomplete_conversation_does_not_publish(self):
        self.row['messages'].pop()
        (self.data / 'test.jsonl').write_text(json.dumps(self.row) + '\n')
        with self.assertRaisesRegex(ValueError, 'six complete'):
            self.export()
        self.assertFalse(self.manifest.exists())

    def test_missing_or_changed_image_does_not_publish(self):
        (self.data / 'images' / 'board.png').write_bytes(b'\x89PNG\r\n\x1a\nchanged')
        with self.assertRaisesRegex(ValueError, 'image hash'):
            self.export()
        self.assertFalse(self.output.exists())
        (self.data / 'images' / 'board.png').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing'):
            self.export()

    def test_changed_native_source_does_not_publish(self):
        (self.data / 'sources' / 'adafruit__Example-PCB' / 'board.sch').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Native schematic hash'):
            self.export()
        self.assertFalse(self.output.exists())

    def test_existing_artifact_requires_explicit_overwrite(self):
        self.export()
        with self.assertRaisesRegex(ValueError, '--overwrite'):
            self.export()

    def test_duplicate_board_identity_cannot_inflate_count(self):
        self.row['family_id'] = 'invented-distinct-board'
        (self.data / 'test.jsonl').write_text(json.dumps(self.row) + '\n')
        with self.assertRaisesRegex(ValueError, 'families must match'):
            self.export()


if __name__ == '__main__':
    unittest.main()
