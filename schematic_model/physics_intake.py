"""Stage bounded, cited PDF learning proposals for a separate candidate builder.

Download reviewed public sources separately with scripts.research_step. This
offline intake binds selected PDF pages and license evidence to immutable bytes.
Closed-form calculations check arithmetic only, not the cited formula's meaning.
"""
# cspell:words Poppler pdftotext
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from urllib.parse import urlsplit


LICENSES = {
    'CC-BY-4.0': 'https://creativecommons.org/licenses/by/4.0/',
    'CC-BY-SA-4.0': 'https://creativecommons.org/licenses/by-sa/4.0/',
    'CC0-1.0': 'https://creativecommons.org/publicdomain/zero/1.0/',
}


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON field')
        result[key] = value
    return result


def read_plan(path):
    def invalid_constant(_):
        raise ValueError('Nonfinite JSON value')
    return json.loads(Path(path).read_text(), object_pairs_hook=_strict_object,
                      parse_constant=invalid_constant)


def _fields(value, required):
    if not isinstance(value, dict) or set(value) != set(required):
        raise ValueError('Unknown or missing fields')


def _text(value, limit=2000):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError('Expected bounded nonempty text')
    return value


def _url(value):
    _text(value)
    url = urlsplit(value)
    if (url.scheme != 'https' or not url.hostname or url.username or url.password
            or url.query or url.fragment or url.port not in (None, 443)):
        raise ValueError('Expected public HTTPS provenance without credentials or query')


def _pages(values, maximum):
    if (not isinstance(values, list) or not 1 <= len(values) <= maximum
            or any(type(v) is not int or not 1 <= v <= 10000 for v in values)
            or len(set(values)) != len(values)):
        raise ValueError('Expected distinct bounded one-based PDF pages')


def _positive(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9]{1,12}(?:\.[0-9]{1,12})?', value):
        raise ValueError('Inputs must be positive decimal strings in named SI units')
    number = Fraction(value)
    if number <= 0:
        raise ValueError('Inputs must be positive')
    return number


def calculate(kind, inputs):
    """Exact rational answers for ideal-component problems; no arbitrary eval."""
    required = {
        'ohms_law_current': ('voltage_v', 'resistance_ohm'),
        'series_resistance': ('r1_ohm', 'r2_ohm'),
        'rc_time_constant': ('resistance_ohm', 'capacitance_f'),
    }
    if kind not in required:
        raise ValueError('Unsupported calculation')
    _fields(inputs, required[kind])
    a, b = (_positive(inputs[key]) for key in required[kind])
    if kind == 'ohms_law_current':
        answer, unit, formula = a / b, 'A', 'I = V / R'
        question = f'For an ideal ohmic resistor of {inputs["resistance_ohm"]} ohm with {inputs["voltage_v"]} V across it, what is its DC current in A?'
    elif kind == 'series_resistance':
        answer, unit, formula = a + b, 'ohm', 'R_total = R1 + R2'
        question = f'Two ideal resistors of {inputs["r1_ohm"]} ohm and {inputs["r2_ohm"]} ohm are connected in series. What is their equivalent resistance in ohm?'
    else:
        answer, unit, formula = a * b, 's', 'tau = R * C'
        question = f'An ideal first-order series RC circuit has R = {inputs["resistance_ohm"]} ohm and C = {inputs["capacitance_f"]} F. What is its time constant in seconds?'
    return {'question': question, 'formula': formula, 'answer_fraction': str(answer), 'unit': unit,
            'check': 'exact-rational-arithmetic-v1',
            'scope': 'Ideal stated circuit only; source-formula alignment and prose are not verified.'}


def validate_plan(plan):
    _fields(plan, ('source', 'family_id', 'pages', 'calculations', 'reasoning_proposals'))
    source = plan['source']
    _fields(source, ('url', 'title', 'author', 'sha256', 'license'))
    _url(source['url'])
    _text(source['title'])
    _text(source['author'])
    if not isinstance(source['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', source['sha256']):
        raise ValueError('Require pinned PDF SHA-256')
    if not isinstance(plan['family_id'], str) or not re.fullmatch('[a-z0-9][a-z0-9-]{0,95}', plan['family_id']):
        raise ValueError('Require stable source/problem family identifier')
    license_record = source['license']
    _fields(license_record, ('id', 'url', 'pages', 'evidence_text'))
    if license_record['id'] not in LICENSES or license_record['url'] != LICENSES[license_record['id']]:
        raise ValueError('Unsupported license assertion; review exact source terms')
    _pages(license_record['pages'], 4)
    _text(license_record['evidence_text'], 500)
    _pages(plan['pages'], 16)
    identifiers = set()
    for field in ('calculations', 'reasoning_proposals'):
        proposals = plan[field]
        if not isinstance(proposals, list) or len(proposals) > 32:
            raise ValueError('At most 32 proposals of each type')
        for proposal in proposals:
            extra = ('kind', 'inputs') if field == 'calculations' else ('question', 'proposed_answer')
            _fields(proposal, ('id', 'page', 'anchor', *extra))
            identifier = _text(proposal['id'], 96)
            if identifier in identifiers:
                raise ValueError('Duplicate proposal identifier')
            identifiers.add(identifier)
            if type(proposal['page']) is not int or proposal['page'] not in plan['pages']:
                raise ValueError('Proposal page must be in selected PDF pages')
            _text(proposal['anchor'], 500)
            if field == 'calculations':
                calculate(proposal['kind'], proposal['inputs'])
            else:
                _text(proposal['question'])
                _text(proposal['proposed_answer'], 8000)


def extract_page(pdf, page, directory):
    """Poppler layout extraction preserves spacing, not mathematical semantics."""
    target = directory / f'page-{page:05}.txt'
    subprocess.run(['pdftotext', '-layout', '-enc', 'UTF-8', '-f', str(page), '-l', str(page),
                    str(pdf), str(target)], check=True, timeout=20, stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE)
    if target.stat().st_size > 256_000:
        raise ValueError('Extracted page exceeds bound')
    text = target.read_text(encoding='utf-8')
    if not text.strip():
        raise ValueError('Page has no text; use a separately reviewed OCR/visual workflow')
    os.chmod(target, 0o600)
    return text


def _normalized(text):
    return ' '.join(text.split())


def _write(path, data):
    with path.open('x', encoding='utf-8') as stream:
        os.chmod(path, 0o600)
        stream.write(json.dumps(data, indent=2, allow_nan=False) + '\n')


def intake(pdf, plan, output):
    validate_plan(plan)
    pdf, output = Path(pdf), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError('Output exists; choose a new private proposal directory')
    allowed = [Path.cwd().resolve() / 'data' / 'external', Path.cwd().resolve() / 'outputs']
    resolved = output.resolve()
    if not any(root in resolved.parents for root in allowed):
        raise ValueError('Keep intake under ignored data/external/ or outputs/')
    if pdf.stat().st_size > 8_000_000:
        raise ValueError('PDF exceeds 8 MB intake bound')
    data = pdf.read_bytes()
    if not data.startswith(b'%PDF-') or hashlib.sha256(data).hexdigest() != plan['source']['sha256']:
        raise ValueError('PDF signature or pinned hash differs')
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix='.physics-intake-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'proposal'
        stage.mkdir(mode=0o700)
        snapshot = stage / 'source.pdf'
        snapshot.write_bytes(data)
        os.chmod(snapshot, 0o600)
        pages = sorted(set(plan['pages'] + plan['source']['license']['pages']))
        extracted = {page: extract_page(snapshot, page, stage) for page in pages}
        license_text = '\n'.join(extracted[p] for p in plan['source']['license']['pages'])
        if _normalized(plan['source']['license']['evidence_text']) not in _normalized(license_text):
            raise ValueError('License evidence is absent from the cited PDF pages')
        proposals = []
        for category in ('calculations', 'reasoning_proposals'):
            for proposal in plan[category]:
                page = proposal['page']
                if _normalized(proposal['anchor']) not in _normalized(extracted[page]):
                    raise ValueError('Proposal anchor is absent from its cited PDF page')
                item = {**proposal, 'family_id': plan['family_id'], 'split': 'unassigned',
                        'citation': {'url': plan['source']['url'], 'pdf_page': page,
                                     'pdf_sha256': plan['source']['sha256'],
                                     'page_text_sha256': hashlib.sha256(extracted[page].encode()).hexdigest()},
                        'ready_for_training': False, 'review_status': 'proposed',
                        'source_formula_verified': False, 'reasoning_verified': False,
                        'review_states': {'extraction': 'hash-and-anchor-matched',
                                          'source_review': 'pending', 'formula_check': 'pending',
                                          'ai_review': 'not-performed', 'human_review': 'not-performed'},
                        'next_action': 'verify-source-and-formula-then-build-separate-candidate'}
                if category == 'calculations':
                    item['calculation'] = calculate(proposal['kind'], proposal['inputs'])
                    item['evidence_level'] = 'arithmetic-checked-proposal'
                    item['review_states']['formula_check'] = 'exact-arithmetic-only; source-alignment-pending'
                else:
                    item['evidence_level'] = 'unverified-reasoning-proposal'
                proposals.append(item)
        manifest = {'schema_version': 1, 'source': plan['source'], 'family_id': plan['family_id'],
                    'source_file': 'source.pdf', 'pages': [
                        {'pdf_page': page, 'text_file': f'page-{page:05}.txt',
                         'text_sha256': hashlib.sha256(extracted[page].encode()).hexdigest(),
                         'equations_and_figures_visually_reviewed': False} for page in pages],
                    'license_status': 'declared-license-evidence-matched; not-a-rights-determination',
                    'split': 'unassigned', 'ready_for_training': False,
                    'proposal_count': len(proposals), 'extraction': 'pdftotext -layout; equations may be reordered',
                    'scope': 'Source excerpts and proposals only. No training records or split assignments.'}
        _write(stage / 'manifest.json', manifest)
        _write(stage / 'proposals.json', proposals)
        _write(stage / 'intake-plan.json', plan)
        if output.exists() or output.is_symlink():
            raise ValueError('Output appeared during intake')
        stage.rename(output)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    result = intake(args.pdf, read_plan(args.plan), args.out)
    print(json.dumps({'proposal_count': result['proposal_count'], 'ready_for_training': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
