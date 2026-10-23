# Generated-answer checkpoint qualification

Use fixed validation predictions to qualify a candidate before retaining it or
considering a default change. Validation loss alone cannot satisfy this gate.
This command runs on CPU and never loads a model or generates answers:

```sh
python -m schematic_model.checkpoint_qualification \
  --dataset data/validation/vision/val.jsonl \
  --predictions outputs/candidate-validation.jsonl \
  --model outputs/candidate/checkpoint-50 \
  --baseline-predictions outputs/baseline-validation.jsonl \
  --baseline-model outputs/baseline-adapter \
  --refusal-scorer v2 \
  --out outputs/qualification/checkpoint-50.json \
  --preserve outputs/qualified-checkpoint-50
```

Both prediction files must come with the benchmark runner's matching
`.jsonl.meta.json` and `.jsonl.report.json`. Qualification checks the prediction
hash recorded in the report, exact dataset and raster-image hashes, and current
adapter/base-model fingerprints. It requires validation family identities, declared image-only evidence, and
complete turn coverage. Candidate and baseline must use the same dataset,
images, history policy, generation settings, and recorded package versions.
The known historical schema-v1 omission of `agent_tools` is treated as disabled;
this compatibility decision is recorded without rewriting the original metadata.
Other missing or unknown settings are rejected. Resolve recorded relative model
paths from the original repository root when qualifying historical runs.
Missing or mismatched evidence is an error, not a failed model score. Historical
prediction/report files are never rewritten, and their old scores remain intact.

Nine rates are scored separately, with failed and dependency-blocked turns still
in the denominators:

| Measurement | Required rate |
| --- | ---: |
| Literal component values | 0.95 |
| Refusals | 0.98 |
| Missing-component refusals | 0.98 |
| Missing-measurement refusals | 0.98 |
| Completed final answers | 1.00 |
| Completed answers without standalone dialogue role labels | 1.00 |
| Literal values after the first turn | 0.95 |
| Completed answers after the first turn | 1.00 |
| Completed follow-up answers without standalone dialogue role labels | 1.00 |

Each measurement requires evidence. Each must also be at least the baseline's
rate. Follow-up completion covers every turn after the first, including refusals;
follow-up value accuracy covers only value questions after the first. A blocked
turn cannot be represented as an attempted answer or hidden by missing records.
Both supported refusal question families must appear; an absent group or an
unrecognized refusal question cannot qualify a checkpoint. A value-only probe
report is useful diagnostics, but needs the full validation refusal coverage for
this gate.

Transport completion and clean-answer completion are separate. The versioned
`standalone-dialogue-roles-v1` format check flags lines consisting only of `user`,
`assistant`, or `system`, case-insensitively, with optional surrounding horizontal
whitespace and a colon. Historical v3 outputs can reach a stop token and include
the correct value while continuing fabricated user/assistant dialogue. Such an
answer remains transport-complete and literal-correct but fails the clean-answer
gate. Failed/blocked empty outputs also fail clean completion. Reports identify
the affected turn keys; original outputs and historical scores remain unchanged.
Ordinary sentences mentioning these words do not trigger the check. This narrow
grammar does not catch every form of extra dialogue or unsupported claim.

`v2` is the default explicit contextual refusal scorer. `--refusal-scorer v1`
selects the historical keyword scorer for a separate, clearly labeled report.
Neither is semantic engineering certification. Literal values can match inside a
more complex incorrect answer; source validity, explanation quality, routing,
unsolicited actions, viewport invariance, and design safety need their own
reviews and regression gates. This qualification never authorizes deployment.

Successful qualification exits zero; a measured gate failure writes a report and
exits nonzero. Reports contain private bindings and belong under ignored
`outputs/`. Files use `0600`; newly created directories use `0700`. Existing
reports and snapshots are never overwritten.
Reports and snapshots must remain outside both evaluated adapters and their base
model directories; writing another JSON file there would change model identity.

`--preserve` retains only a qualified adapter, its tokenizer/processor/config
artifacts, available license/provenance notices, and the qualification report in
a new staged snapshot. Copied model artifacts are rehashed before publication.
It does not copy the base model, rewrite its reference, delete old checkpoints,
select a new default, or provide optimizer state for training resume. Keep the
original base available and unchanged. This is a manual checkpoint-boundary
operation: pause training, benchmark, qualify, and preserve before continuing
past checkpoint rotation. No training callback automatically performs this work.
For several checkpoints, run separate qualifications and retain each qualifying
candidate; automatic best-candidate ranking is deliberately absent.

The artifacts and reports are consistency evidence, not signed proof of how an
operator produced predictions. CPU tests use inert model-file fixtures and do
not establish real model quality. Train/evaluation families must remain frozen;
this tool does not independently reconstruct the training corpus to prove split
isolation.
