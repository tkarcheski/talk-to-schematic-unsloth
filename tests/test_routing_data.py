"""Routing gold and continuation guards; no GPU or network required."""
import json
import hashlib
import base64
from dataclasses import replace
from types import SimpleNamespace

import pytest

from schematic_model.routing_data import lesson
from schematic_model.training import TrainingConfig, continuation_source, prepare_continued_adapter
from scripts.evaluate_routing import score


def test_followup_search_gold_preserves_selection_and_prior_show(tmp_path):
    record = {'id': 'public', 'repository': 'adafruit/public', 'commit': 'a' * 40, 'source_path': 'board.sch', 'sha256': 'b' * 64, 'license': 'CC-BY-SA-3.0'}
    part = {'refdes': 'U1', 'value': 'TPS63060'}
    image = tmp_path / 'public.png'
    image.write_bytes(b'fixture image')
    row, case = lesson(record, {'page': 1, 'bom': [part], 'nets': {}}, part, [100, 100, 200, 200], image, 'val', 'followup')
    assert case['image_sha256'] == hashlib.sha256(image.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='Image hash changed'):
        lesson(record, {'page': 1}, part, [], image, 'val', 'followup', image_sha256='0' * 64)
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


@pytest.fixture
def routing_evaluation(tmp_path, monkeypatch):
    from scripts import evaluate_routing

    image = tmp_path / 'source.png'
    image.write_bytes(b'first image bytes')
    case = {'id': 'fixture', 'image': str(image), 'image_sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
            'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': str(image)}},
                                                     {'type': 'text', 'text': 'Which part?'}]}],
            'expected': {'kind': 'clarify', 'tool': None}}
    cases = tmp_path / 'cases.jsonl'
    requests = []
    def response(endpoint, payload):
        requests.append(payload)
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': 'Which component?'}}]}
    monkeypatch.setattr(evaluate_routing, 'request_json', response)
    return evaluate_routing, case, cases, image, requests, tmp_path


@pytest.mark.parametrize('change', ['tampered', 'invalid-hash', 'null-hash', 'different-path', 'no-image'])
def test_pinned_routing_images_fail_before_inference_or_output(routing_evaluation, change):
    evaluator, case, cases, image, requests, root = routing_evaluation
    if change == 'tampered':
        image.write_bytes(b'changed image bytes')
    elif change == 'invalid-hash':
        case['image_sha256'] = 'not-a-hash'
    elif change == 'null-hash':
        case['image_sha256'] = None
    elif change == 'no-image':
        case['messages'][0]['content'] = []
    else:
        case['image'] = str(root / 'different.png')
    cases.write_text(json.dumps(case) + '\n')
    before = cases.read_bytes()
    with pytest.raises(ValueError):
        evaluator.run(cases, root / 'run', 'http://127.0.0.1:8893/v1', 'fixture')
    assert requests == []
    assert not (root / 'run').exists()
    assert cases.read_bytes() == before


def test_legacy_cases_keep_bytes_and_labels_but_changed_images_change_identity(routing_evaluation):
    evaluator, case, cases, image, requests, root = routing_evaluation
    del case['image_sha256']
    cases.write_text(json.dumps(case) + '\n')
    original = cases.read_bytes()
    first = evaluator.run(cases, root / 'one', 'http://127.0.0.1:8893/v1', 'fixture')
    image.write_bytes(b'new observed bytes')
    second = evaluator.run(cases, root / 'two', 'http://127.0.0.1:8893/v1', 'fixture')
    assert cases.read_bytes() == original
    assert first['cases_sha256'] == second['cases_sha256'] == hashlib.sha256(original).hexdigest()
    assert first['evaluation_input_sha256'] != second['evaluation_input_sha256']
    assert all(report['image_evidence'][0]['verification'] == 'legacy-unpinned-observed' for report in [first, second])
    assert first['image_evidence'][0]['images'][0]['sha256'] != second['image_evidence'][0]['images'][0]['sha256']
    assert len(requests) == 2


def test_later_case_hash_failure_rejects_the_whole_run_before_first_request(routing_evaluation):
    evaluator, case, cases, _, requests, root = routing_evaluation
    cases.write_text(json.dumps(case) + '\n' + json.dumps({**case, 'id': 'later', 'image_sha256': '0' * 64}) + '\n')
    with pytest.raises(ValueError, match='image hash differs'):
        evaluator.run(cases, root / 'run', 'http://127.0.0.1:8893/v1', 'fixture')
    assert requests == []
    assert not (root / 'run').exists()


def test_evaluation_hashes_exact_image_and_case_snapshots_sent_to_endpoint(routing_evaluation, monkeypatch):
    evaluator, case, cases, image, requests, root = routing_evaluation
    cases.write_text('\n'.join(json.dumps({**case, 'id': identifier}) for identifier in ['one', 'two']) + '\n')
    case_bytes, image_bytes = cases.read_bytes(), image.read_bytes()
    original_request = evaluator.request_json
    def mutate_after_first_request(endpoint, payload):
        image.write_bytes(b'changed during run')
        cases.write_text('changed during run')
        return original_request(endpoint, payload)
    monkeypatch.setattr(evaluator, 'request_json', mutate_after_first_request)
    report = evaluator.run(cases, root / 'run', 'http://127.0.0.1:8893/v1', 'fixture')
    assert report['cases_sha256'] == hashlib.sha256(case_bytes).hexdigest()
    for payload, evidence in zip(requests, report['image_evidence']):
        uri = payload['messages'][0]['content'][0]['image_url']['url']
        assert base64.b64decode(uri.split(',', 1)[1]) == image_bytes
        assert evidence['verification'] == 'declared-hash-verified'
        assert evidence['images'][0]['sha256'] == hashlib.sha256(image_bytes).hexdigest()
