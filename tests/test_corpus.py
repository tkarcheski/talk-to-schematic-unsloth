"""Native evidence and corpus provenance tests; never fetch or load a model."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from schematic_model.corpus import _archive_source, _split, build_corpus, conversation, parse_eagle
from schematic_model.render_eagle import _Drawing, _point, source_svg


EAGLE = '''<?xml version="1.0"?>
<!DOCTYPE eagle SYSTEM "eagle.dtd">
<eagle version="9.6.2"><drawing><schematic><libraries><library name="passives">
<symbols><symbol name="R"><wire x1="-1" y1="-1" x2="1" y2="-1" width="0.1" layer="94"/>
<pin name="LEFT" x="-2.54" y="0" length="short"/><pin name="RIGHT" x="2.54" y="0" length="short" rot="R180"/>
<text x="0" y="2" size="1" layer="95">&gt;NAME</text><text x="0" y="-2" size="1" layer="96">&gt;VALUE</text>
</symbol></symbols><devicesets><deviceset name="RESISTOR"><gates><gate name="G$1" symbol="R"/></gates>
<devices><device name="" package="0603"><connects><connect gate="G$1" pin="LEFT" pad="1"/>
<connect gate="G$1" pin="RIGHT" pad="2"/></connects></device></devices></deviceset></devicesets>
</library></libraries><parts><part name="R1" library="passives" deviceset="RESISTOR" device="" value="10k"/>
<part name="R2" library="passives" deviceset="RESISTOR" device="" value="22k"/></parts><sheets><sheet>
<plain/><instances><instance part="R1" gate="G$1" x="0" y="0"/><instance part="R2" gate="G$1" x="10" y="0" rot="MR90"/></instances>
<nets><net name="SIGNAL"><segment><pinref part="R1" gate="G$1" pin="RIGHT"/><pinref part="R2" gate="G$1" pin="LEFT"/>
<wire x1="2.54" y1="0" x2="10" y2="2.54" width=".15" layer="91"/>
<label x="4" y="1" size="1" layer="95"/></segment></net></nets></sheet></sheets></schematic></drawing></eagle>'''


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.source = self.directory / "example.sch"
        self.source.write_text(EAGLE)

    def test_native_pins_resolve_to_physical_pads(self):
        result = parse_eagle(self.source)
        self.assertEqual(result["sheets"][0]["nets"], {"SIGNAL": ["R1.2", "R2.1"]})
        self.assertEqual(result["sheets"][0]["bom"][1]["value"], "22k")
        self.assertEqual(result["sha256"], hashlib.sha256(EAGLE.encode()).hexdigest())

    def test_missing_pin_mapping_is_incomplete_instead_of_guessing(self):
        self.source.write_text(EAGLE.replace('pin="RIGHT" pad="2"', 'pin="UNKNOWN" pad="2"'))
        with self.assertRaisesRegex(ValueError, "Unresolved physical pin"):
            parse_eagle(self.source)

    def test_conflicting_pad_nets_are_rejected(self):
        self.source.write_text(EAGLE.replace('</nets>', '<net name="OTHER"><segment><pinref part="R1" gate="G$1" pin="RIGHT"/></segment></net></nets>'))
        with self.assertRaisesRegex(ValueError, "conflicting"):
            parse_eagle(self.source)

    def test_binary_and_xml_entities_are_rejected(self):
        for content in (b"binary eagle source", EAGLE.replace('<eagle ', '<!ENTITY x "injected"><eagle ').encode()):
            with self.subTest(content=content[:20]):
                self.source.write_bytes(content)
                with self.assertRaises(ValueError):
                    parse_eagle(self.source)

    def test_unresolved_hierarchy_is_not_flattened(self):
        self.source.write_text(EAGLE.replace('<schematic>', '<schematic><modules><module name="child"/></modules>'))
        with self.assertRaisesRegex(ValueError, "hierarchy"):
            parse_eagle(self.source)

    def test_source_geometry_and_values_are_rendered(self):
        svg = source_svg(self.source, 1)
        for literal in ('10k', '22k', 'R1', 'SIGNAL', 'Source geometry render'):
            self.assertIn(literal, svg)
        self.assertNotIn('&gt;VALUE', svg)
        self.assertAlmostEqual(_point(2, 0, (10, 20, 90, True))[0], 10)
        self.assertAlmostEqual(_point(2, 0, (10, 20, 90, True))[1], 18)

    def test_unknown_drawing_geometry_is_rejected(self):
        self.source.write_text(EAGLE.replace('<plain/>', '<plain><spline layer="94"/></plain>'))
        with self.assertRaisesRegex(ValueError, "Unsupported visible"):
            source_svg(self.source, 1)

    def test_circuit_crop_ignores_large_page_frame_and_retains_attribution(self):
        initial = ET.fromstring(source_svg(self.source, 1)).get('viewBox')
        self.source.write_text(EAGLE.replace('<plain/>', '<plain><frame x1="-1000" y1="-1000" x2="1000" y2="1000" layer="94"/></plain>'))
        svg = source_svg(self.source, 1, attribution='Adafruit Industries | CC BY-SA 3.0')
        self.assertEqual(ET.fromstring(svg).get('viewBox'), initial)
        self.assertIn('Adafruit Industries | CC BY-SA 3.0', svg)
        self.assertIn('eagle_xml_geometry_v2_circuit_crop', svg)
        self.assertIn('circuit crop', svg)

    def test_crop_keeps_disconnected_electrical_symbols(self):
        self.source.write_text(EAGLE.replace('part="R2" gate="G$1" x="10"', 'part="R2" gate="G$1" x="200"')
                              .replace('<pinref part="R2" gate="G$1" pin="LEFT"/>', ''))
        xmin, _, width, _ = map(float, ET.fromstring(source_svg(self.source, 1)).get('viewBox').split())
        self.assertGreater(xmin + width, 200)

    def test_rotated_text_bounds_cover_vertical_values(self):
        drawing = _Drawing()
        drawing.text('VERTICAL VALUE', 0, 0, size=2, rotation='R90')
        self.assertLess(min(y for _, y in drawing.points), -20)
        self.assertLess(max(x for x, _ in drawing.points), 3)

    def test_chat_has_replayable_gold_and_explicit_evidence_scope(self):
        record = {'id': 'board-1', 'repository': 'adafruit/Example-PCB', 'sha256': 'a' * 64}
        row = conversation(record, parse_eagle(self.source)['sheets'][0], 'sheet.png')
        self.assertEqual(len(row['gold']), 6)
        self.assertEqual(row['gold'][1], {'type': 'pins', 'pins': ['R1.2', 'R2.1'], 'net': 'SIGNAL'})
        self.assertEqual(row['gold'][2]['query'], 'R1.2')
        self.assertEqual(row['gold'][2]['pins'], ['R2.1'])
        self.assertIn('not an electrical review', row['messages'][1]['content'][1]['text'])
        self.assertEqual(row['provenance']['evidence_mode'], 'native_facts_supplied_in_prompt')
        self.assertEqual(row['split'], _split(record['repository']))

    def test_image_only_view_never_supplies_answer_evidence(self):
        record = {'id': 'board-1', 'repository': 'adafruit/Example-PCB', 'sha256': 'a' * 64}
        row = conversation(record, parse_eagle(self.source)['sheets'][0], '../images/sheet.png', image_only=True)
        self.assertEqual([g['type'] for g in row['gold']], ['value', 'value', 'refuse', 'refuse'])
        questions = ' '.join(c.get('text', '') for m in row['messages'] if m['role'] == 'user' for c in m['content'])
        self.assertNotIn('10k', questions)
        self.assertNotIn('22k', questions)
        self.assertNotIn('Extracted native', questions)

    def test_hidden_values_do_not_become_image_only_gold(self):
        self.source.write_text(EAGLE.replace('&gt;VALUE', 'unrelated text'))
        sheet = parse_eagle(self.source)['sheets'][0]
        self.assertFalse(any(p['value_visible'] for p in sheet['bom']))
        record = {'id': 'board-1', 'repository': 'adafruit/Example-PCB', 'sha256': 'a' * 64}
        self.assertIsNone(conversation(record, sheet, 'sheet.png', image_only=True))

    def test_visible_value_with_hidden_name_is_not_image_only_gold(self):
        self.source.write_text(EAGLE.replace('&gt;NAME', 'unnamed'))
        sheet = parse_eagle(self.source)['sheets'][0]
        self.assertFalse(any(p['value_visible'] for p in sheet['bom']))

    def test_existing_corpus_needs_explicit_overwrite(self):
        (self.directory / 'train.jsonl').write_text('user data')
        with patch('schematic_model.corpus._fetch', side_effect=AssertionError('must not fetch')):
            with self.assertRaisesRegex(ValueError, '--overwrite'):
                build_corpus(self.directory)
        self.assertEqual((self.directory / 'train.jsonl').read_text(), 'user data')

    def test_real_vcnl4000_native_facts(self):
        directory = Path(__file__).parent / 'fixtures' / 'real'
        sheet = parse_eagle(directory / 'vcnl4000.sch')['sheets'][0]
        values = {part['refdes']: part['value'] for part in sheet['bom']}
        self.assertEqual(values['U1'], 'VCNL4000-GS08')
        self.assertEqual(values['C3'], '0.1uF')
        self.assertEqual(sheet['nets']['SDA'], ['JP1.3', 'R1.1', 'U1.4'])
        self.assertEqual(sheet['nets']['SCL'], ['JP1.2', 'R2.1', 'U1.5'])

    def test_real_txb0104_native_facts_and_source_hashes(self):
        directory = Path(__file__).parent / 'fixtures' / 'real'
        sheet = parse_eagle(directory / 'txb0104.sch')['sheets'][0]
        self.assertEqual(sheet['nets']['OE'], ['JP4.1', 'R1.1', 'U2.8'])
        self.assertEqual(sheet['nets']['VLOW'], ['C1.1', 'JP4.6', 'R1.2', 'U2.1'])
        for source in json.loads((directory / 'sources.json').read_text()):
            self.assertEqual(hashlib.sha256((directory / source['fixture']).read_bytes()).hexdigest(), source['sha256'])
            self.assertTrue((directory / source['license_file']).is_file())
            self.assertTrue((directory / source['original_readme']).is_file())

    def archive(self, files, commit='a' * 40):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode='w:gz', format=tarfile.PAX_FORMAT,
                          pax_headers={'comment': commit}) as archive:
            for name, content in files.items():
                content = content.encode()
                member = tarfile.TarInfo('source-root/' + name)
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
        return buffer.getvalue()

    def test_download_pins_commit_license_and_hashes(self):
        data = self.archive({'board.sch': EAGLE, 'license.txt': 'Attribution-ShareAlike 3.0 Unported', 'README.md': 'Original attribution'})
        with patch('schematic_model.corpus._fetch', return_value=data):
            result = _archive_source({'repository': 'adafruit/Example-PCB', 'revision': 'main'}, self.directory)
        self.assertEqual(result['revision'], 'a' * 40)
        self.assertEqual(result['license'], 'CC-BY-SA-3.0')
        self.assertTrue(all(len(x['sha256']) == 64 for x in result['files']))
        cache = self.directory / 'adafruit__Example-PCB' / 'source.json'
        self.assertEqual(json.loads(cache.read_text()), result)
        with patch('schematic_model.corpus._fetch', side_effect=AssertionError('network not needed')):
            self.assertEqual(_archive_source(result, self.directory), result)
        (cache.parent / 'board.sch').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            _archive_source(result, self.directory)

    def test_wrong_commit_and_missing_license_fail(self):
        source = {'repository': 'adafruit/Example-PCB', 'revision': 'b' * 40}
        data = self.archive({'license.txt': 'Attribution-ShareAlike 3.0 Unported'})
        with patch('schematic_model.corpus._fetch', return_value=data):
            with self.assertRaisesRegex(ValueError, 'pinned commit'):
                _archive_source(source, self.directory)
        data = self.archive({'board.sch': EAGLE})
        with patch('schematic_model.corpus._fetch', return_value=data):
            with self.assertRaisesRegex(ValueError, 'license'):
                _archive_source({**source, 'revision': 'main'}, self.directory)

    def test_fresh_download_checks_pinned_file_hashes(self):
        data = self.archive({'license.txt': 'Attribution-ShareAlike 3.0 Unported'})
        source = {'repository': 'adafruit/Example-PCB', 'revision': 'a' * 40,
                  'files': [{'path': 'license.txt', 'sha256': 'b' * 64, 'bytes': 39}]}
        with patch('schematic_model.corpus._fetch', return_value=data):
            with self.assertRaisesRegex(ValueError, 'hashes differ'):
                _archive_source(source, self.directory)

    def test_archive_path_escape_is_rejected(self):
        data = self.archive({'../../escaped.sch': EAGLE, 'license.txt': 'Attribution-ShareAlike 3.0 Unported'})
        with patch('schematic_model.corpus._fetch', return_value=data):
            with self.assertRaisesRegex(ValueError, 'Unsafe archive path'):
                _archive_source({'repository': 'adafruit/Example-PCB', 'revision': 'main'}, self.directory)
        self.assertFalse((self.directory.parent / 'escaped.sch').exists())


if __name__ == '__main__':
    unittest.main()
