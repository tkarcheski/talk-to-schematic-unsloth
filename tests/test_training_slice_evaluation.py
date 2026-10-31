"""Scheduled evaluation settings remain bound to a resumable training plan."""
import json

from PIL import Image
import pytest

from scripts.training_slice import training_argv


@pytest.fixture
def plan(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'model.safetensors').write_bytes(b'weights fixture')
    for split, color in [('train', 'white'), ('val', 'black')]:
        Image.new('RGB', (8, 8), color).save(data / f'{split}.png')
        row = {'id': split, 'messages': [
            {'role': 'user', 'content': [{'type': 'image', 'image': f'{split}.png'}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': 'fixture answer'}]}]}
        (data / f'{split}.jsonl').write_text(json.dumps(row) + '\n')
    return {'data': str(data), 'out': str(tmp_path / 'run'), 'model': str(model),
            'max_steps': 10, 'save_steps': 2, 'max_seq': 4096, 'rank': 16,
            'gradient_accumulation': 4, 'eval_steps': 2,
            'load_best_model_at_end': True, 'early_stopping_patience': 3}


def test_evaluation_reaches_training_cli_and_survives_resume(plan, tmp_path):
    argv = training_argv(plan)
    assert argv[argv.index('--eval-steps') + 1] == '2'
    assert argv[argv.index('--early-stopping-patience') + 1] == '3'
    assert '--load-best-model-at-end' in argv
    checkpoint = tmp_path / 'run/checkpoint-2'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'trainer_state.json').write_text('{"global_step": 2}')
    for name in ('adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'training_args.bin'):
        (checkpoint / name).write_bytes(b'fixture')
    resumed = training_argv(plan)
    assert resumed == [*argv, '--resume', str(checkpoint)]
    with pytest.raises(ValueError, match='different'):
        training_argv({**plan, 'early_stopping_patience': 4})


@pytest.mark.parametrize('changed', [
    {'eval_steps': True}, {'eval_steps': 0}, {'eval_steps': 2.5},
    {'early_stopping_patience': False}, {'early_stopping_patience': 0},
    {'load_best_model_at_end': 'true'}, {'eval_steps': 3},
    {'eval_steps': 12, 'save_steps': 12}, {'load_best_model_at_end': False},
])
def test_invalid_evaluation_creates_no_binding_or_output(plan, tmp_path, changed):
    with pytest.raises(ValueError):
        training_argv({**plan, **changed})
    assert not (tmp_path / 'run').exists()
    assert not (tmp_path / 'run.scheduler-plan.sha256').exists()


def test_old_plan_still_disables_evaluation(plan):
    for key in ('eval_steps', 'early_stopping_patience', 'load_best_model_at_end'):
        del plan[key]
    argv = training_argv(plan)
    assert '--eval-steps' not in argv
    assert '--load-best-model-at-end' not in argv
