"""Routing gold and continuation guards; no GPU or network required."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from schematic_model.routing_data import lesson
from schematic_model.training import TrainingConfig, continuation_source, prepare_continued_adapter
from scripts.evaluate_routing import score


def test_followup_search_gold_preserves_selection_and_prior_show():
    record = {'id': 'public', 'repository': 'adafruit/public', 'commit': 'a' * 40, 'source_path': 'board.sch', 'sha256': 'b' * 64, 'license': 'CC-BY-SA-3.0'}
    part = {'refdes': 'U1', 'value': 'TPS63060'}
    row, case = lesson(record, {'page': 1, 'bom': [part], 'nets': {}}, part, [100, 100, 200, 200], '/tmp/public.png', 'val', 'followup')
    assert case['messages'][-1]['content'].endswith('do a web search for that part')
    assert 'focused_label' in case['messages'][-1]['content']
    assert case['messages'][2]['tool_calls'][0]['function']['name'] == 'show_label'
    target = row['messages'][-1]['content'][0]['text']
    assert '<function=web_search>' in target and 'TPS63060 datasheet' in target
    assert '<function=show_label>' not in target


def test_search_scoring_rejects_viewer_substitution_and_wrong_part():
    expected = {'tool': 'web_search', 'part': 'TPS63060', 'refdes': 'U1'}
    def message(name, args):
        return {'tool_calls': [{'function': {'name': name, 'arguments': json.dumps(args)}}]}
    assert score(message('web_search', {'query': 'TPS63060 datasheet'}), expected)
    assert not score(message('show_label', {'label': 'U1'}), expected)
    assert not score(message('web_search', {'query': 'unrelated part'}), expected)
    assert not score({'content': 'I searched for TPS63060.'}, expected)


@pytest.fixture
def adapter(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'adapter_config.json').write_text(json.dumps({'peft_type': 'LORA', 'r': 16, 'bias': 'none', 'target_modules': 'q_proj'}))
    (source / 'adapter_model.safetensors').write_bytes(b'fixture weights')
    return TrainingConfig(model=str(source), out=str(tmp_path / 'candidate'), continue_adapter=True)


def test_continuation_is_explicit_separate_and_not_optimizer_resume(adapter):
    evidence = continuation_source(adapter)
    assert len(evidence['files']['adapter_model.safetensors']) == 64
    for invalid in [replace(adapter, out=adapter.model), replace(adapter, out=adapter.model + '/child'), replace(adapter, rank=8), replace(adapter, resume='checkpoint-1')]:
        with pytest.raises(ValueError):
            continuation_source(invalid)


def test_continuation_only_enables_existing_language_lora(adapter):
    class Parameter:
        def requires_grad_(self, value):
            self.requires_grad = value
    names = ['model.language.q_proj.lora_A.default.weight', 'model.visual.q_proj.lora_B.default.weight', 'model.language.q_proj.base_layer.weight']
    parameters = {name: Parameter() for name in names}
    model = SimpleNamespace(peft_config={'default': SimpleNamespace(r=16, target_modules='q_proj')}, named_parameters=lambda: parameters.items())
    trained = prepare_continued_adapter(model, replace(adapter, finetune_vision=False), continuation_source(adapter))
    assert trained == [names[0]]
    assert not parameters[names[1]].requires_grad and not parameters[names[2]].requires_grad
    model.peft_config['default'].r = 8
    with pytest.raises(ValueError, match='differs'):
        prepare_continued_adapter(model, adapter, continuation_source(adapter))


def test_clarification_scoring_does_not_require_question_mark():
    expected = {'tool': None}
    assert score({'content': 'I need to know which component. Please specify its reference designator.'}, expected)
    assert not score({'content': 'The selected part is U1.'}, expected)


def test_deadline_requests_full_save_at_optimizer_boundary(monkeypatch):
    from schematic_model.training import DeadlineControl
    callback = DeadlineControl(100)
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    monkeypatch.setattr('schematic_model.training.time.time', lambda: 101)
    callback.on_step_end(None, SimpleNamespace(global_step=2, max_steps=20), control)
    assert callback.triggered and control.should_save and control.should_training_stop
    finished = DeadlineControl(100)
    finished.on_step_end(None, SimpleNamespace(global_step=20, max_steps=20), control)
    assert not finished.triggered


def test_loaded_target_module_set_matches_json_list(adapter):
    parameter = SimpleNamespace(requires_grad_=lambda value: None)
    model = SimpleNamespace(peft_config={'default': SimpleNamespace(r=16, target_modules={'q_proj', 'v_proj'})}, named_parameters=lambda: [('model.q_proj.lora_A.default.weight', parameter)])
    source = {'rank': 16, 'target_modules': ['q_proj', 'v_proj']}
    assert prepare_continued_adapter(model, adapter, source)
    source['target_modules'] = 'q_proj|v_proj'
    with pytest.raises(ValueError):
        prepare_continued_adapter(model, adapter, source)
