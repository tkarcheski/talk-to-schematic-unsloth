# cspell:words Poppler pdftotext pdftoppm
import json
import os
from pathlib import Path
import subprocess

from PIL import Image
import pytest

from schematic_model import physics_intake as intake
from schematic_model import physics_review as review


@pytest.fixture
def unit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pdf = tmp_path / 'input.pdf'
    pdf.write_bytes(b'%PDF-1.4\nfixture')
    plan = {'source': {'url': 'https://example.org/book.pdf', 'title': 'Circuit facts', 'author': 'Author',
                      'sha256': review.digest(pdf.read_bytes()),
                      'license': {'id': 'CC-BY-4.0', 'url': intake.LICENSES['CC-BY-4.0'],
                                  'pages': [2], 'evidence_text': 'Creative Commons Attribution 4.0'}},
            'family_id': 'book', 'pages': [1],
            'calculations': [{'id': 'current', 'page': 1, 'anchor': 'Ohm law',
                              'kind': 'ohms_law_current', 'inputs': {'voltage_v': '9', 'resistance_ohm': '300'}}],
            'reasoning_proposals': [{'id': '</script><script>alert(1)</script>', 'page': 1, 'anchor': 'Ohm law',
                                    'question': '<img src=https://example.org/x>', 'proposed_answer': 'Unverified'}]}
    def fake_run(argv, **kwargs):
        assert kwargs['timeout'] <= 30
        page = int(argv[argv.index('-f') + 1])
        if argv[0] == 'pdftotext':
            Path(argv[-1]).write_text({1: 'Ohm law I = V/R', 2: 'Creative Commons Attribution 4.0'}[page])
        else:
            assert argv[0] == 'pdftoppm' and argv[argv.index('-scale-to') + 1] == '1600'
            Image.new('RGB', (80, 100), 'white').save(Path(argv[-1]).with_suffix('.png'))
    monkeypatch.setattr(intake.subprocess, 'run', fake_run)
    source = tmp_path / 'data/external/unit'
    intake.intake(pdf, plan, source)
    return source, tmp_path / 'outputs/nested/review'


def build(unit):
    source, out = unit
    return review.build_review(source, out, review.digest((source / 'manifest.json').read_bytes()),
                               review.digest((source / 'proposals.json').read_bytes()))


def test_bound_private_pack_under_permissive_umask(unit):
    old = os.umask(0)
    try:
        binding = build(unit)
    finally:
        os.umask(old)
    source, out = unit
    assert binding['source_sha256'] == review.digest((source / 'source.pdf').read_bytes())
    assert len(binding['rendered_pages']) == 2
    assert out.parent.stat().st_mode & 0o777 == 0o700
    for path in [out, *out.rglob('*')]:
        assert path.stat().st_mode & 0o777 == (0o700 if path.is_dir() else 0o600)
    page = (out / 'index.html').read_text()
    assert '<img src=https://' not in page
    assert '</script><script>alert' not in page
    assert "connect-src 'none'" in page
    assert 'localStorage' not in page and 'sessionStorage' not in page
    assert '9 / 300 = 3/100 A' in page
    assert '<strong>Formula</strong>' in page
    assert '<strong>Assumptions and scope</strong>' in page
    assert page.index('exact-rational-arithmetic-v1') > page.index('<details><summary>Provenance')
    assert str(source) not in page
    assert 'HUMAN REVIEW NOT PERFORMED AT BUILD' in page


@pytest.mark.parametrize('name', ['source.pdf', 'page-00001.txt', 'manifest.json', 'proposals.json', 'intake-plan.json'])
def test_tampering_is_rejected_without_pack(unit, name):
    source, out = unit
    manifest_hash = review.digest((source / 'manifest.json').read_bytes())
    proposals_hash = review.digest((source / 'proposals.json').read_bytes())
    (source / name).write_bytes((source / name).read_bytes() + b'changed')
    with pytest.raises((ValueError, UnicodeError)):
        review.build_review(source, out, manifest_hash, proposals_hash)
    assert not out.exists()


@pytest.mark.parametrize('change', ['answer', 'human', 'path', 'page'])
def test_rehashed_forged_metadata_fails_regeneration(unit, change):
    source, out = unit
    name = 'manifest.json' if change in ('path', 'page') else 'proposals.json'
    value = json.loads((source / name).read_text())
    if change == 'answer':
        value[0]['calculation']['answer_fraction'] = '999'
    elif change == 'human':
        value[0]['review_states']['human_review'] = 'accepted'
    elif change == 'path':
        value['pages'][0]['text_file'] = '../../outside'
    else:
        value['pages'][0]['pdf_page'] = 10001
    (source / name).write_text(json.dumps(value, indent=2) + '\n')
    with pytest.raises(ValueError, match='regenerated'):
        build(unit)
    assert not out.exists()


@pytest.mark.parametrize('name', ['source.pdf', 'page-00001.txt', 'manifest.json', 'proposals.json', 'intake-plan.json'])
def test_symlinked_evidence_is_rejected(unit, name):
    source, out = unit
    original = source / name
    moved = source / ('saved-' + name)
    original.rename(moved)
    original.symlink_to(moved)
    with pytest.raises(ValueError, match='linked'):
        build(unit)
    assert not out.exists()


def test_existing_output_and_escape_refused(unit, tmp_path):
    source, out = unit
    build(unit)
    before = (out / 'index.html').read_bytes()
    with pytest.raises(ValueError, match='new private'):
        build(unit)
    assert (out / 'index.html').read_bytes() == before
    with pytest.raises(ValueError, match='new private'):
        build((source, tmp_path / 'outputs/../../escape'))


def test_render_failure_preserves_no_partial_output(unit, monkeypatch):
    def fail(*args):
        raise subprocess.TimeoutExpired('pdftoppm', 30)
    monkeypatch.setattr(review, 'render_page', fail)
    with pytest.raises(subprocess.TimeoutExpired):
        build(unit)
    assert not unit[1].exists()
    assert not list(unit[1].parent.glob('.physics-review-*'))


def test_render_pixel_bound(unit, monkeypatch):
    def huge(argv, **kwargs):
        Image.new('RGB', (1601, 2)).save(Path(argv[-1]).with_suffix('.png'))
    monkeypatch.setattr(review.subprocess, 'run', huge)
    with pytest.raises(ValueError, match='pixel bound'):
        review.render_page(unit[0] / 'source.pdf', 1, unit[0])


def test_cli_requires_explicit_pins_and_records_no_acceptance(unit, capsys):
    source, out = unit
    assert review.main(['--intake', str(source), '--out', str(out), '--manifest-sha256',
                        review.digest((source / 'manifest.json').read_bytes()), '--proposals-sha256',
                        review.digest((source / 'proposals.json').read_bytes())]) == 0
    assert 'no human decisions' in capsys.readouterr().out


def test_browser_session_export_and_responsive_layout(unit):
    """Scripted decisions test controls only; this is not human dataset review."""
    build(unit)
    modules = Path(__file__).resolve().parents[1] / 'node_modules/@playwright/test'
    if not modules.exists():
        pytest.skip('Optional browser check requires npm ci and installed Playwright Chromium')
    script = r'''
const {chromium}=require(process.argv[1]);
const fs=require('node:fs');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.CI?undefined:process.env.SCHEMATIC_CHROMIUM||'/usr/bin/chromium'});
 try {
 const page=await browser.newPage({acceptDownloads:true});
 const external=[],errors=[];let downloads=0;
 page.on('request',r=>{if(!r.url().startsWith('file:')&&!r.url().startsWith('data:'))external.push(r.url());});
 page.on('pageerror',e=>errors.push(e.message));page.on('download',()=>downloads++);
 await page.goto('file://'+process.argv[2]);
 for(const width of [1280,390]){
  await page.setViewportSize({width,height:900});
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  assert.equal(await page.locator('article select').first().inputValue(),'unreviewed');
 }
 assert.equal(downloads,0);
 await page.locator('article select').first().selectOption('needs-correction');
 await page.locator('textarea').first().fill('Scripted test only. Check units.');
 assert.equal(downloads,0);
 const downloadEvent=page.waitForEvent('download');await page.getByRole('button',{name:'Export review'}).click();
 const download=await downloadEvent;
 const result=JSON.parse(fs.readFileSync(await download.path(),'utf8'));
 assert.equal(result.ready_for_training,false);
 assert.equal(result.decisions[0].decision,'needs-correction');
 assert.equal(result.decisions[0].notes,'Scripted test only. Check units.');
 assert.equal(result.decisions[1].human_review,'not-performed');
 assert.match(result.binding.source_sha256,/^[a-f0-9]{64}$/);
 assert.equal(result.binding.proposals.length,2);
 await page.locator('textarea').first().fill('Changed after export');
 assert.match(await page.locator('#status').textContent(),/Not exported/);
 await page.reload();assert.equal(await page.locator('article select').first().inputValue(),'unreviewed');
 assert.equal(await page.locator('textarea').first().inputValue(),'');
 assert.deepEqual(external,[]);assert.deepEqual(errors,[]);
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
'''
    # Popen bypasses the fixture's mocked Poppler subprocess.run only.
    process = subprocess.Popen(['node', '-e', script, str(modules), str(unit[1] / 'index.html')],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(timeout=45)
    assert process.returncode == 0, (stdout + stderr).decode()


def test_repeated_inline_evidence_is_bounded_before_document_allocation(unit, monkeypatch):
    # Both proposals cite the same page; the license contributes a third occurrence.
    monkeypatch.setattr(review, 'render_page', lambda *args: b'x' * 90)
    monkeypatch.setattr(review, 'MAX_HTML_BYTES', 359)
    def forbidden_document(*args):
        raise AssertionError('Oversized inline evidence must fail before HTML allocation')
    monkeypatch.setattr(review, 'document', forbidden_document)
    with pytest.raises(ValueError, match='Inline evidence'):
        build(unit)
    assert not unit[1].exists()
