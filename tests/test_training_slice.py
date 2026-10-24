"""CPU recovery and immutable input admission for scheduled training slices."""
import json
from pathlib import Path

from PIL import Image
import pytest

from scripts.training_slice import training_argv


@pytest.fixture
def plan(tmp_path):
    data = tmp_path / 'data'
    model = tmp_path / 'model'
    data.mkdir()
    model.mkdir()
    (model / 'model.safetensors').write_bytes(b'weights fixture')
    for split, color in [('train', 'white'), ('val', 'black')]:
        Image.new('RGB', (8, 8), color).save(data / f'{split}.png')
        row = {'id': split, 'messages': [
            {'role': 'user', 'content': [{'type': 'image', 'image': f'{split}.png'}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': 'fixture answer'}]}]}
        (data / f'{split}.jsonl').write_text(json.dumps(row) + '\n')
    return {'data': str(data), 'out': str(tmp_path / 'out'), 'model': str(model),
            'max_steps': 10, 'save_steps': 2, 'max_seq': 4096, 'rank': 16, 'gradient_accumulation': 4}


def checkpoint(plan, step, complete=True):
    path = Path(plan['out']) / f'checkpoint-{step}'
    path.mkdir(parents=True)
    (path / 'trainer_state.json').write_text(json.dumps({'global_step': step}))
    if complete:
        for name in ('adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'training_args.bin'):
            (path / name).write_bytes(b'fixture')
    return path


def test_recovery_skips_incomplete_newer_checkpoint_and_preserves_it(plan):
    training_argv(plan)
    valid = checkpoint(plan, 2)
    incomplete = checkpoint(plan, 4, complete=False)
    with pytest.warns(UserWarning, match='Skipping incomplete checkpoint-4'):
        argv = training_argv(plan)
    assert argv[-2:] == ['--resume', str(valid)]
    assert incomplete.is_dir()


def test_no_complete_checkpoint_does_not_restart_or_replace_marker(plan):
    training_argv(plan)
    checkpoint(plan, 2, complete=False)
    marker = Path(plan['out']).with_name('out.scheduler-plan.sha256')
    before = marker.read_bytes()
    with pytest.warns(UserWarning), pytest.raises(ValueError, match='No complete checkpoint'):
        training_argv(plan)
    assert marker.read_bytes() == before


@pytest.mark.parametrize('changed', ['image', 'model', 'tokenizer'])
def test_existing_plan_rejects_changed_external_bytes(plan, changed):
    training_argv(plan)
    checkpoint(plan, 2)
    marker = Path(plan['out']).with_name('out.scheduler-plan.sha256')
    before = marker.read_bytes()
    if changed == 'image':
        Image.new('RGB', (8, 8), 'red').save(Path(plan['data']) / 'train.png')
    elif changed == 'model':
        (Path(plan['model']) / 'model.safetensors').write_bytes(b'changed weights')
    else:
        (Path(plan['model']) / 'tokenizer.json').write_text('{}')
    with pytest.raises(ValueError, match='different or untracked'):
        training_argv(plan)
    assert marker.read_bytes() == before
