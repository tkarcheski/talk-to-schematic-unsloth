<!-- cspell:words multipage Pynt Titano -->

# Reproduce the multi-page review queue

The [pinned plan](multipage-source-plan.json) includes Tsunami Qwiic plus the main PyPortal and Pynt revisions. PyPortal, Pynt, and Titano share one family and one deterministic train partition. Titano is explicitly excluded because its embedded SUPPLY2 device set cannot be resolved; its source hash and family ownership remain recorded. RA6M5 remains an independent diagnostic; Voice Bonnet is excluded while its library mapping is unresolved. This queue is separate from all current training and benchmark snapshots.

Provide a private local JSON mapping from each plan repository name to its downloaded source directory. Keep the original source layout and license file. Run with a Python environment containing the existing local Transformers tokenizer support:

```sh
python -m schematic_model.multipage_data \
  --plan docs/research/multipage-source-plan.json \
  --roots /path/to/local-source-roots.json \
  --tokenizer /path/to/models/Qwen3.5-4B \
  --out /path/to/new-local-review-queue \
  --context-limit 8192
```

The builder verifies native source and license hashes before parsing. It rejects changed family partitions, duplicate family records, diagnostic sources, paths outside the declared root, and sources without two populated sheets. Related revisions cannot be split into separate records. Outputs publish atomically and an existing queue cannot be overwritten.

Each candidate asks for physical-pad endpoints on a named net shared across sheets. Anonymous net identifiers are not treated as cross-page links. The answer candidate retains page-specific endpoint sets and source/license provenance; `review_status` remains `required` and `training_eligible` remains false. No measurements, faults, timing limits, or electrical diagnoses are invented.

Each question has focused and full-evidence variants. The pinned local tokenizer counts their actual chat-template input tokens plus a separate 256-token answer reserve; there is no truncation. These are text-only budgets and do not include image tokens. Over-budget variants remain explicit research cases, not usable training examples or successful model evaluations. The manifest records tokenizer-file hashes and the exact candidate-file hash. No model inference or training occurs.

The local verified build produced 102 review-required questions: 16 from Tsunami and 43 from each supported PyPortal revision. With compact native JSON, full-context variants use 13,328–13,332 tokens for Tsunami, 5,869–5,873 for Pynt, and 5,928–5,932 for the main PyPortal, before the 256-token answer reserve. All 16 Tsunami full-context cases exceed 8,192 tokens; the other 86 text-only cases fit. Focused variants use 83–602 tokens. These measurements include the text chat template but no image tokens or agent-tool preamble.

The initial local v1 artifact counted tokenizer mapping fields instead of token IDs and is marked invalid. The corrected v2 artifact has candidate hash `d37a5040e7662c2dc6dd08aeef41f358379791914bb2502c0a316ac552a67e91`; a regression test covers both mapping and list tokenizer results. Neither artifact has been used for training.
