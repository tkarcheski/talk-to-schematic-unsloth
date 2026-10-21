"""Validated finite training slice; resumes only checkpoints bound to the same plan."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def training_argv(plan):
    required = {'data', 'out', 'model', 'max_steps', 'save_steps', 'max_seq', 'rank', 'gradient_accumulation'}
    if set(plan) - required - {'continue_adapter', 'lr', 'max_image_size', 'finetune_vision'} or not required <= set(plan):
        raise ValueError('unknown/missing training plan fields')
    for key in ('data', 'out', 'model'):
        if not isinstance(plan[key], str) or not Path(plan[key]).is_absolute():
            raise ValueError('training paths must be absolute')
    for key, limit in [('max_steps', 10000), ('save_steps', 1000), ('max_seq', 32768), ('rank', 256), ('gradient_accumulation', 128)]:
        if type(plan[key]) is not int or not 1 <= plan[key] <= limit:
            raise ValueError('invalid training bound')
    if plan['save_steps'] > plan['max_steps']:
        raise ValueError('checkpoint interval exceeds run')
    data = Path(plan['data'])
    hashes = {name: hashlib.sha256((data / name).read_bytes()).hexdigest() for name in ('train.jsonl', 'val.jsonl')}
    identity = hashlib.sha256(json.dumps({'plan': plan, 'data_hashes': hashes}, sort_keys=True).encode()).hexdigest()
    out = Path(plan['out'])
    marker = out.with_name(out.name + '.scheduler-plan.sha256')
    if out.exists() and any(out.iterdir()) and (not marker.is_file() or marker.read_text().strip() != identity):
        raise ValueError('output directory belongs to a different or untracked training plan')
    out.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    marker.write_text(identity + '\n')
    argv = [sys.executable, '-m', 'schematic_model.training']
    for key in sorted(required):
        argv.extend(['--' + key.replace('_', '-'), str(plan[key])])
    if type(plan.get('continue_adapter', False)) is not bool or type(plan.get('finetune_vision', False)) is not bool:
        raise ValueError('continuation and vision flags must be boolean')
    lr = plan.get('lr', 5e-5)
    if type(lr) not in (int, float) or not 0 < lr <= 0.001:
        raise ValueError('invalid learning rate')
    image_size = plan.get('max_image_size', 1024)
    if type(image_size) is not int or not 64 <= image_size <= 4096:
        raise ValueError('invalid image bound')
    argv.extend(['--lr', str(lr), '--max-image-size', str(image_size), '--finetune-vision' if plan.get('finetune_vision', False) else '--no-finetune-vision'])
    checkpoints = [p for p in out.glob('checkpoint-*') if p.name.removeprefix('checkpoint-').isdigit() and (p / 'trainer_state.json').is_file()]
    if checkpoints:
        argv.extend(['--resume', str(max(checkpoints, key=lambda p: int(p.name.split('-')[-1])))])
    if plan.get('continue_adapter'):
        argv.append('--continue-adapter')
    return argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SCHEMATIC_DEADLINE') or not os.environ.get('SCHEMATIC_RESULT_PATH'):
        parser.error('run training slices through research_scheduler.py')
    argv = training_argv(json.loads(args.plan.read_text()))
    raise SystemExit(subprocess.run(argv, check=False).returncode)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        try:
            from .public_research import report_failure
        except ImportError:
            from public_research import report_failure
        report_failure(error)
        raise SystemExit(1)
