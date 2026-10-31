"""Build a private, offline human review pack from pinned physics intake evidence."""
# cspell:words Poppler pdftoppm singlefile noopener
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from PIL import Image
from schematic_model.physics_intake import intake, read_plan, _write


MAX_HTML_BYTES = 48_000_000


def digest(data):
    return hashlib.sha256(data).hexdigest()


def bounded_file(root, name, limit):
    path = root / name
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError('Missing, linked, or oversized intake evidence')
    return path.read_bytes()


def render_page(pdf, page, directory):
    prefix = directory / f'page-{page:05}'
    subprocess.run(['pdftoppm', '-f', str(page), '-l', str(page), '-singlefile',
                    '-scale-to', '1600', '-png', str(pdf), str(prefix)],
                   check=True, timeout=30, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    path = prefix.with_suffix('.png')
    raw = bounded_file(directory, path.name, 4_000_000)
    with Image.open(path) as image:
        if image.format != 'PNG' or max(image.size) > 1600 or min(image.size) < 1:
            raise ValueError('Rendered page exceeds pixel bound')
        image.verify()
    os.chmod(path, 0o600)
    return raw


STYLE = '''
*{box-sizing:border-box}body{margin:0;background:#f1f4f8;color:#182638;font:16px/1.55 system-ui,sans-serif}
header,main{max-width:1240px;margin:auto;padding:24px}header{padding-bottom:8px}
h1{font-size:clamp(1.7rem,4vw,2.5rem);margin:8px 0}h2{font-size:1.2rem}p{overflow-wrap:anywhere}
.tag{font-size:.8rem;font-weight:700;letter-spacing:.08em;color:#48647e}article,details.source{background:white;border:1px solid #d5dfe9;border-radius:12px;padding:24px;margin-bottom:24px}
.grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:24px}.evidence img{width:100%;height:auto;border:1px solid #d5dfe9}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.85rem;background:#f1f4f8;padding:12px}
label{display:block;font-weight:600;margin-top:16px}select,textarea,button{font:inherit;padding:10px;border:1px solid #9eafc1;border-radius:6px;max-width:100%}select,textarea{width:100%}textarea{min-height:100px}button{background:#174f76;color:white;cursor:pointer}button:focus-visible,select:focus-visible,textarea:focus-visible,summary:focus-visible{outline:3px solid #de9d22;outline-offset:3px}
.toolbar{display:flex;align-items:center;flex-wrap:wrap;gap:16px;margin:20px 0}.muted{color:#526273}summary{cursor:pointer;font-weight:600}.answer{border-left:4px solid #328288;padding-left:14px}
@media(max-width:700px){header,main{padding:16px}article,details.source{padding:16px}.grid{grid-template-columns:1fr}}
'''
SCRIPT = '''
const pack = JSON.parse(document.getElementById('binding').textContent);
const decisions = pack.proposals.map(p => ({id:p.id, decision:'unreviewed', notes:''}));
const status = document.getElementById('status');
document.querySelectorAll('article').forEach((card,i) => {
  card.querySelector('select').addEventListener('change', e => { decisions[i].decision=e.target.value; update(); });
  card.querySelector('textarea').addEventListener('input', e => { decisions[i].notes=e.target.value; update(); });
});
function update(){status.textContent=decisions.filter(d=>d.decision!=='unreviewed').length+' / '+decisions.length+' decisions entered. Not exported.';}
document.getElementById('export').addEventListener('click', () => {
  const result={schema_version:1, kind:'physics-human-review-export', binding:pack,
    exported_at:new Date().toISOString(), ready_for_training:false,
    scope:'User-entered decisions; not authenticated reviewer identity or training authorization.',
    decisions:decisions.map(d=>({...d,human_review:d.decision==='unreviewed'?'not-performed':'decision-entered'}))};
  const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)+'\\n'],{type:'application/json'}));
  const link=document.createElement('a');link.href=url;link.download='physics-review-'+pack.manifest_sha256.slice(0,12)+'.json';link.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
  status.textContent='Download requested. Keep this review file private; browser download permissions apply.';
});
'''


def document(manifest, proposals, binding, images):
    escape = html.escape
    def pretty(value):
        return '<pre>' + escape(json.dumps(value, indent=2, ensure_ascii=False)) + '</pre>'
    def evidence(page):
        return ('<div class="evidence"><p>PDF page ' + str(page) + '</p><img alt="Source PDF page '
                + str(page) + '" src="data:image/png;base64,' + base64.b64encode(images[page]).decode()
                + f'"><p><a href="pages/page-{page:05}.png" target="_blank" rel="noopener">Open full-size page in a new tab</a></p></div>')
    cards = []
    def arithmetic(proposal):
        calculation = proposal.get('calculation')
        if calculation is None:
            return '<p class="muted">Reasoning proposal. No arithmetic check applies; the answer is unverified.</p>'
        fields, operator = {
            'ohms_law_current': (('voltage_v', 'resistance_ohm'), '/'),
            'series_resistance': (('r1_ohm', 'r2_ohm'), '+'),
            'rc_time_constant': (('resistance_ohm', 'capacitance_f'), '×'),
        }[proposal['kind']]
        values = (' ' + operator + ' ').join(proposal['inputs'][key] for key in fields)
        result = calculation['answer_fraction'] + ' ' + calculation['unit']
        return ('<p><strong>Formula</strong><br>' + escape(calculation['formula'])
                + '</p><p><strong>Exact arithmetic check</strong><br>' + escape(values + ' = ' + result)
                + '</p><p class="muted"><strong>Assumptions and scope</strong><br>'
                + escape(calculation['scope']) + '</p>')
    for index, proposal in enumerate(proposals):
        calculation = proposal.get('calculation')
        question = calculation['question'] if calculation else proposal['question']
        answer = (calculation['answer_fraction'] + ' ' + calculation['unit']) if calculation else proposal['proposed_answer']
        cards.append(f'''<article><div class="tag">PROPOSAL {index + 1} · HUMAN REVIEW NOT PERFORMED AT BUILD</div>
<h2>{escape(proposal['id'])}</h2><div class="grid"><section><p>{escape(question)}</p>
<p class="answer"><strong>Proposed answer</strong><br>{escape(answer)}</p>
{arithmetic(proposal)}
<details><summary>Provenance and existing review states</summary>{pretty(proposal)}</details>
<label for="decision-{index}">Your decision</label><select id="decision-{index}"><option value="unreviewed">Unreviewed</option><option value="accept">Accept</option><option value="reject">Reject</option><option value="needs-correction">Needs correction</option></select>
<label for="notes-{index}">Review notes</label><textarea id="notes-{index}" maxlength="4000" placeholder="Source alignment, assumptions, units, or corrections"></textarea>
</section>{evidence(proposal['page'])}</div></article>''')
    license_images = ''.join(evidence(p) for p in manifest['source']['license']['pages'])
    payload = json.dumps(binding, ensure_ascii=True).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    script_hash = base64.b64encode(hashlib.sha256(SCRIPT.encode()).digest()).decode()
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'sha256-{script_hash}'; connect-src 'none'; base-uri 'none'; form-action 'none'">
<title>Physics proposal review</title><style>{STYLE}</style></head><body><header><div class="tag">PRIVATE · OFFLINE · PROPOSED DATA</div><h1>Physics proposal review</h1>
<p>{escape(manifest['source']['title'])}</p><p class="muted">Check the cited page, formula, assumptions, units and answer. Exact arithmetic does not verify the source interpretation. Accept records your decision only; it never creates training data.</p>
<p class="muted">Decisions and notes exist only in this tab's memory. Reloading clears them. Export deliberately downloads a private review file; keep it in private storage. No account identity is verified.</p>
<div class="toolbar"><button id="export" type="button">Export review</button><span id="status" role="status">0 / {len(proposals)} decisions entered. Not exported.</span></div></header>
<main><details class="source"><summary>Source, license and page evidence</summary>{pretty(manifest['source'])}<p>{escape(manifest['license_status'])}</p>{license_images}</details>{''.join(cards)}</main>
<script type="application/json" id="binding">{payload}</script><script>{SCRIPT}</script></body></html>'''


def build_review(source, output, manifest_sha256, proposals_sha256):
    source, output = Path(source).resolve(), Path(output).absolute()
    root = Path.cwd().resolve() / 'outputs'
    if root not in output.resolve().parents or output.exists() or output.is_symlink():
        raise ValueError('Choose a new private review directory under outputs/')
    expected = {'manifest.json': manifest_sha256, 'proposals.json': proposals_sha256}
    for name, wanted in expected.items():
        if not isinstance(wanted, str) or not re.fullmatch('[0-9a-f]{64}', wanted):
            raise ValueError('Require pinned manifest and proposal SHA-256 values')
        if digest(bounded_file(source, name, 512_000)) != wanted:
            raise ValueError('Pinned intake evidence hash differs')
    plan_bytes = bounded_file(source, 'intake-plan.json', 256_000)
    pdf_bytes = bounded_file(source, 'source.pdf', 8_000_000)
    # All reads below operate on our private snapshot, never caller-controlled filenames.
    missing = []
    parent = output.parent
    while not parent.exists():
        missing.append(parent)
        parent = parent.parent
    for directory in reversed(missing):
        directory.mkdir(mode=0o700)
    with tempfile.TemporaryDirectory(prefix='.physics-review-', dir=output.parent) as temporary:
        stage = Path(temporary)
        plan_path, pdf_path = stage / 'plan.json', stage / 'source.pdf'
        for path, data in ((plan_path, plan_bytes), (pdf_path, pdf_bytes)):
            path.write_bytes(data)
            path.chmod(0o600)
        plan = read_plan(plan_path)
        pack = stage / 'pack'
        pack.mkdir(mode=0o700)
        # Regeneration validates plan, source hash, page hashes, anchors and exact arithmetic.
        manifest = intake(pdf_path, plan, pack / 'evidence')
        proposals = read_plan(pack / 'evidence' / 'proposals.json')
        for name in ('manifest.json', 'proposals.json'):
            original = bounded_file(source, name, 512_000)
            if digest(original) != expected[name] or original != (pack / 'evidence' / name).read_bytes():
                raise ValueError('Intake metadata does not match regenerated evidence')
        if not proposals:
            raise ValueError('Review requires at least one proposal')
        for page in manifest['pages']:
            original = bounded_file(source, page['text_file'], 256_000)
            if digest(original) != page['text_sha256']:
                raise ValueError('Page text hash differs')
        images_dir = pack / 'pages'
        images_dir.mkdir(mode=0o700)
        images = {p['pdf_page']: render_page(pack / 'evidence' / 'source.pdf', p['pdf_page'], images_dir)
                  for p in manifest['pages']}
        binding = {'schema_version': 1, 'manifest_sha256': manifest_sha256,
                   'proposals_sha256': proposals_sha256, 'plan_sha256': digest(plan_bytes),
                   'source_sha256': digest(pdf_bytes), 'source': manifest['source'],
                   'family_id': manifest['family_id'], 'pages': manifest['pages'],
                   'rendered_pages': [{'pdf_page': p, 'sha256': digest(data)} for p, data in images.items()],
                   'proposals': [{'id': p['id'], 'sha256': digest(json.dumps(p, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode())} for p in proposals],
                   'proposal_hash_encoding': 'JSON sorted keys, compact separators, ASCII, UTF-8',
                   'review_states_scope': 'Intake states only; supplemental review files are not imported.'}
        _write(pack / 'review-binding.json', binding)
        # Count each inline occurrence before allocating base64 strings or the document.
        inline_pages = [p['page'] for p in proposals] + manifest['source']['license']['pages']
        inline_bytes = sum(4 * ((len(images[page]) + 2) // 3) for page in inline_pages)
        if inline_bytes > MAX_HTML_BYTES:
            raise ValueError('Inline evidence exceeds review HTML byte bound')
        page_html = document(manifest, proposals, binding, images).encode()
        if len(page_html) > MAX_HTML_BYTES:
            raise ValueError('Review HTML exceeds 48 MB bound')
        (pack / 'index.html').write_bytes(page_html)
        (pack / 'index.html').chmod(0o600)
        if output.exists() or output.is_symlink():
            raise ValueError('Output appeared during build')
        pack.rename(output)
    return binding


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intake', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--proposals-sha256', required=True)
    args = parser.parse_args(argv)
    build_review(args.intake, args.out, args.manifest_sha256, args.proposals_sha256)
    print('Private offline review pack created; no human decisions have been recorded.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
