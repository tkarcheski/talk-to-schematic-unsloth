---
name: schematic-model-evaluate
description: Validate and compare schematic model adapters against attributable held-out data using the talk-to-schematic package. Use for checking a schematic fine-tune, deployment candidate, or benchmark report; not for ordinary schematic lookup.
---

# Evaluate a schematic model

Start with the dataset manifest and model training manifest. Confirm source revisions, split membership, image hashes, model revision, and whether native facts are included in the prompt. Keep board families and revisions in a single split. Simulated gold conversations are labels, not recorded model responses.

Use CPU preflight before model loading:

```sh
schematic-model train --data DATA_DIRECTORY --dry-run
```

Run base and adapter on the same held-out data and settings. `schematic-model benchmark --model MODEL --data TEST.jsonl --out PREDICTIONS.jsonl --profile PROFILE` saves every turn durably, including failed generations. `--resume` requires matching data and settings. Default `--history gold` isolates each turn; also use `--history generated` to assess accumulated conversational errors. Do not mix the two in a comparison.

Choose the declared benchmark scope:

```sh
schematic-model evaluate --gold TEST.jsonl --pred ADAPTER.jsonl --pred-base BASE.jsonl --profile real-grounding --report report.json
```

`real-grounding` requires values, connectivity, and unsupported-fact refusals. `real-vision` requires image-only values and refusals; use its separate image-only dataset. Never switch profiles merely to hide a failing or missing metric.

Missing, duplicate, malformed, or incomplete prediction evidence blocks a result. Report metric denominators and failures. Deterministic prose matching has limited semantic understanding: inspect actual incorrect answers, correct answers, and plausible false positives before accepting a candidate.

Separately verify adapter loading, a complete inference response, resource use, and deployment health. A successful training run, a saved checkpoint, and a deployment-ready model are distinct milestones. Describe a model that fails the declared gates as a candidate; preserve the report and error examples for the next iteration.
