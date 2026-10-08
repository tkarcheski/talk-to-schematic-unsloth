# v3 validation regression

Historical runs from 2026-10-02, checked against local reports, predictions, and training rows on 2026-10-07. These local `results/` and `data/training/` artifacts are excluded from Git; this note records their findings, not a new evaluation.

The expanded v3 adapter completed 666 steps (three epochs) but performed worse than v1. **v3 is not a release candidate.** Neither adapter meets the 0.95 image-only value gate on these validation sets.

| Model | v3 render: values / unfinished | Frozen v2: values / unfinished |
|---|---|---|
| v1 | 23/26 / 0 | 24/26 / 0 |
| v3 checkpoint-300 (~1.35 epochs) | 16/26 / 9 | 13/26 / 12 |
| v3 final | 14/26 / 11 | 16/26 / 9 |

Reports: `results/v3-vs-v1-v3-render-val-2048-report.json`, `results/v3-vs-v1-val-vision-2048-report.json`, and the corresponding `v3c300-vs-v1-*` reports. Predictions: `results/adapter-v3-{v3-render-val,val-vision}-2048.jsonl` and corresponding `adapter-v3c300-*` files. All comparisons use a 2,048-pixel image limit and 1,024-token answer budget. Refusal scores are 26/26 in each comparison; deterministic keyword scores do not establish semantic correctness.

Every unfinished v3 answer is the second turn of its conversation. Failed records say generation did not reach an end token and retain no partial text. The image-only training rows changed from 86 four-question conversations in `data/training/real-crop-v1/vision/train.jsonl` to 715 single-question and 86 two-question conversations in `data/training/real-expanded-v3/vision/train.jsonl`. The wording “And what value is marked” occurs in all 86 v1 conversations and none of the v3 conversations.

This conversation-shape mismatch is a plausible contributor, not a proven cause. v3 still contains two-question image-only conversations and longer evidence-assisted conversations. Checkpoint-300 failing to restore v1 performance does not rule out overfitting; it could already have occurred by that checkpoint. Data mix and rendering also changed. The final training loss of 0.0000337 is not evidence of generalization.

A useful next investigation would retain bounded partial-generation diagnostics locally and compare a variant restoring follow-up coverage while holding other settings fixed. Select checkpoints on validation and preserve the original held-out benchmark. These are proposed experiments, not authorization for new data or training runs.
