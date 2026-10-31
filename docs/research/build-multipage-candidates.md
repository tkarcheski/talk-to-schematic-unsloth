<!-- cspell:words multipage Pynt Titano EVK -->

# Reproduce the multi-page review queue

The [pinned plan](multipage-source-plan.json) contains 10 source families and 12 native schematics. PyPortal/Pynt and the two RTK Facet variants remain grouped within their respective families. All selected families retain their deterministic training partition; no frozen validation or test family is reassigned. This queue is separate from the existing training and benchmark snapshots.

Excluded families are recorded in the plan and rejected by the builder if reintroduced. RTK mosaic-X5 is excluded because it shares substantial circuitry with the training-partition RTK EVK despite mapping to test. BeagleBone Blue is excluded because its embedded schematic license notice conflicts with the repository license. RA6M5 remains an independent diagnostic, and unsupported embedded-library mappings remain excluded.

Provide a private local JSON mapping from each plan repository name to its downloaded source directory. Keep the original source layout and license file. Run with a Python environment containing the existing local Transformers tokenizer support:

```sh
python -m schematic_model.multipage_data \
  --plan docs/research/multipage-source-plan.json \
  --roots /path/to/local-source-roots.json \
  --tokenizer /path/to/models/Qwen3.5-4B \
  --out /path/to/new-local-review-queue \
  --context-limit 8192 \
  --answer-reserve 2048 \
  --render-pages
```

The builder verifies native source and license hashes before parsing. It rejects changed family partitions, duplicate family records, diagnostic sources, paths outside the declared root, and sources without two populated sheets. Related revisions cannot be split into separate records. Outputs publish atomically and an existing queue cannot be overwritten.

Each candidate asks for physical-pad endpoints on a named net shared across sheets. Anonymous net identifiers are not treated as cross-page links. The answer candidate retains page-specific endpoint sets and source/license provenance; `review_status` remains `required` and `training_eligible` remains false. No measurements, faults, timing limits, or electrical diagnoses are invented.

Each question has focused and full-evidence variants. The pinned local tokenizer counts their actual chat-template input tokens plus a separate configured answer reserve; there is no truncation. These are text-only budgets and do not include image tokens. The accepted expansion uses a 2,048-token reserve: the longest canonical candidate-answer JSON measures 1,880 tokens with the pinned tokenizer and no special tokens. This reserves space for the recorded answer, not an unrestricted narrative. Over-budget variants remain explicit research cases, not usable training examples or successful model evaluations. The manifest records tokenizer-file hashes and the exact candidate-file hash. No model inference or training occurs.

The accepted expansion produces 318 review-required questions and 27 PNG/SVG page pairs from 12 schematics in 10 families. Each source manifest entry records populated sheets, unique physical reference designators, sheet-local component/net entries, pad occurrences, and named cross-page nets. These are structural counts, not difficulty scores. Anonymous net names never establish cross-page identity.

Optional page rendering occurs inside the same atomic output transaction. Asset paths and SHA256 hashes are recorded per source/page; rendering failure or a changed source prevents publication of a partial queue. Footers retain source attribution and license identity. Images remain `visual_review_status: unreviewed`; source geometry rendering does not certify label readability or visual-answer quality.

The prior v2 queue contained 102 questions, two families, three schematics, and eight sheet occurrences. Its corrected candidate hash is `d37a5040e7662c2dc6dd08aeef41f358379791914bb2502c0a316ac552a67e91`; it remains unchanged. The initial v1 artifact had counted tokenizer mapping fields instead of token IDs and remains invalid. None of these queues is promoted into training by this command.
