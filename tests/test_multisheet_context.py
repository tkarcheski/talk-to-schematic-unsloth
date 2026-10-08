# cspell:words multipage MCLK
"""Public multi-sheet evidence and explicit overflow; no model downloads in CI."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from schematic_model.deployment import DeploymentConfig, LocalModel
from schematic_model.inference import InferenceError

FIXTURE = Path(__file__).parent / 'fixtures' / 'multipage'


def test_source_snapshot_preserves_page_specific_cross_sheet_endpoints():
    source = json.loads((FIXTURE / 'source.json').read_text())
    raw = (FIXTURE / 'tsunami-native-context.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == source['derived_context_sha256']
    sheets = json.loads(raw)['sheets']
    assert [sheet['page'] for sheet in sheets] == [1, 2]
    # Source-checked clock endpoints: don't attribute the page-2 test point to page 1.
    assert sheets[0]['nets']['MCLK'] == ['U3.15']
    assert sheets[1]['nets']['MCLK'] == ['TP4.1', 'U7.2']
    assert 'TP4' not in {part['refdes'] for part in sheets[0]['bom']}
    assert 'TP4' in {part['refdes'] for part in sheets[1]['bom']}


def test_full_native_context_rejected_without_truncation_or_generation():
    metadata = json.loads((FIXTURE / 'source.json').read_text())
    text = (FIXTURE / 'tsunami-native-context.json').read_text()
    seen = {}

    def tokenize(messages, **kwargs):
        seen['messages'], seen['options'] = messages, kwargs
        return {'input_ids': SimpleNamespace(shape=(1, metadata['text_tokens']))}

    def forbidden_generation(**kwargs):
        raise AssertionError('Over-budget evidence must never reach generation')

    # Avoid loading weights: exercise the real serving guard with the measured text lower bound.
    engine = LocalModel.__new__(LocalModel)
    engine.config = DeploymentConfig(model='', max_seq=8192, max_tokens=256)
    engine.model = SimpleNamespace(generate=forbidden_generation)
    engine.processor = SimpleNamespace(apply_chat_template=tokenize)
    with pytest.raises(InferenceError, match='context limit'):
        engine.complete('schematic', [{'role': 'user', 'content': text}], max_tokens=256)
    assert seen['messages'][0]['content'][0]['text'] == text
    assert not seen['options'].get('truncation', False)
