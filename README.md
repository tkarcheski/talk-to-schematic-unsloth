# Talk to the Schematic

A Python package for extracting schematic evidence, building attributable training data, fine-tuning a vision model with Unsloth, and testing the resulting adapter.

The target is an RTX 4090 with 24 GB VRAM. The current base model is **Qwen3.5-4B**, pinned to `3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636`. CPU development uses **Astral uv**.

## Verified status

- 120 real EAGLE schematics from 120 Adafruit board repositories, with immutable source revisions, original notices, and SHA256 manifests.
- 120 evidence-assisted conversations and 119 image-only conversations. Twenty board families are reserved for held-out testing.
- [25 complete simulated conversations](docs/simulated_conversations.md), containing 150 user prompts. These are source-derived gold examples, **not model transcripts**.
- A two-step 4090 training smoke run saved an adapter with finite loss and 9.55 GiB peak allocated VRAM at 768-pixel images. [Recorded evidence](docs/validation/4090-smoke.json).
- A complete three-epoch real-data fine-tune saved and reloaded an adapter from 173 mixed-view training conversations, using 12.43 GiB peak allocated VRAM. [Training evidence](docs/validation/real-crop-v1-training.json).
- The [experimental adapter on Hugging Face](https://huggingface.co/nutinspace/talk-to-schematic-qwen3.5-4b-lora-experimental/tree/21ca688a65be235905a56ff7d4355bc08d02bab6) passes the evidence-assisted benchmark but **fails the image-only value-reading gate**. It is not a qualified release.
- The base model's full benchmark exposed a nonterminating invented resistor list on an absent-component question. Failed generations remain failures in the benchmark denominator.

This project currently supports native **EAGLE XML** extraction and PNG/JPEG/WebP model input. It does not contain a KiCad importer, a general electrical rules checker, or a human-reviewed real-board defect benchmark.

The first complete comparison used the same 20 held-out board families, 1,024-pixel image limit, 1,024-token answer budget, and gold conversation history for both models:

| Held-out task | Base | First adapter | Cases |
|---|---:|---:|---:|
| Values with native evidence | 100% | 100% | 40 |
| Physical-pad connectivity with native evidence | 100% | 100% | 40 |
| Image-only literal values | 65% | 67.5% | 40 |
| Completed answers across both views | 199/200 | 200/200 | 200 |

These are narrow deterministic metrics, not overall engineering accuracy. The manual audit found valid unit equivalents rejected by literal matching and unsupported extra claims in some otherwise passing base answers. Validation image-only values were 9/26 for the base and 12/26 for the adapter. Further data and model improvements use validation evidence while preserving this first benchmark. [Actual predictions and run summary](docs/validation/model-v1/summary.json) are retained for both models.

A validation-only resolution comparison raised the first adapter from 12/26 correct values at 1,024 pixels to 22/26 at 1,536 and 24/26 at 2,048, with zero failed generations. The 2,048-pixel run used 18.0 GiB peak allocated VRAM and still missed the 95% literal-value gate. [Resolution evidence](docs/validation/model-v1/resolution-ablation.json). Source review identified overlapping labels in one remaining case and an incorrect component qualifier in the other; neither was silently scored as correct.

The expanded training recipe contains 888 conversations and 1,409 supervised turns across the same 87 training board families, including 715 image-only component lookups and varied absent identifiers. Seven uncertain label groups are explicitly excluded; unreviewed labels remain identified as such. Validation and test bytes remain frozen. A two-step test on eight demanding rows at 2,048 pixels passed with 11.47 GiB peak reserved memory. [Snapshot](docs/validation/expanded-v3-snapshot.json), [fit evidence](docs/validation/expanded-v3-fit-2048.json), and [experiment plan](docs/validation/expanded-v3-experiment-plan.json) distinguish data integrity, hardware fit, and model qualification.

## Install and check

Python 3.13 is required; the checked local GPU environment uses Python 3.13.14.

```sh
uv sync --locked
uv run --locked pytest -q
uv run --locked ruff check .
uv build
uv run --locked schematic-model --help
```

Install the development hooks once per checkout (Node.js 22.18 or newer is required for CSpell):

```sh
npm ci --ignore-scripts
uv run --locked pre-commit install
uv run --locked pre-commit run --all-files
npm run spellcheck
```

The commit hooks check source spelling, Python correctness, configuration syntax, merge conflicts, and unexpectedly large additions. The push hook runs the CPU regression suite. CSpell excludes original third-party fixtures, generated corpus records, and verbatim simulated gold; those artifacts keep their original bytes and have separate provenance checks. Its project vocabulary contains hardware terms and library identifiers.

Native schematic rendering requires `rsvg-convert` from librsvg. On Ubuntu the package is `librsvg2-bin`; on Arch it is `librsvg`. Tests use real source fixtures and generated failure cases. Ordinary CPU tests do not download models or initialize CUDA.

The wheel includes the command-line tools and three reusable AI skills. Large downloaded corpora, weights, checkpoints, and local prediction results remain outside Git. Pinned source manifests and compact validation evidence are committed.

## Inspect a real schematic

```sh
uv run --locked schematic-model inspect \
  tests/fixtures/real/txb0104.sch \
  --out tmp/txb0104-evidence.json \
  --render tmp/txb0104.png
```

The evidence includes the sheet-local BOM and physical package-pad connectivity resolved through the source's embedded libraries. Symbol pin names and physical pad numbers may differ. Unsupported binary EAGLE, hierarchical constructs, or rendering primitives are rejected rather than guessed.

Corpus images preserve source geometry and crop page furniture to make circuit labels legible. These corpus renders are explicitly identified as derivatives and include attribution. Original files remain available alongside the corpus. Crowded source labels and peripheral annotation clipping remain known limitations; inspect the original source when a crop is ambiguous.

## Rebuild the real corpus

```sh
uv run --locked schematic-model corpus \
  --manifest corpora/adafruit-120.json \
  --out data/real --limit 120
```

Existing outputs require an explicit `--overwrite`. The importer verifies source hashes and licensing evidence and returns an incomplete status if it cannot produce the requested count.

| View | Train | Validation | Test | Turns |
|---|---:|---:|---:|---:|
| Native evidence plus image | 87 | 13 | 20 | 720 |
| Image only | 86 | 13 | 20 | 476 |

Rows are grouped by board repository, keeping its pages and revisions together. The image-only view excludes hidden values and physical-pad questions that cannot be established visually. One training board has no eligible visible value labels.

The evidence-assisted view supplies extracted facts in the prompt. Its score measures using those facts, not independent wire recognition. The separate image-only view measures visible value lookup and unsupported-fact refusals. Neither view supplies invented design-defect annotations.

See the [source inventory](corpora/adafruit-120.json), [corpus summary](corpora/adafruit-120-summary.json), and [render hashes](corpora/adafruit-120-rendered.json). Source hardware and derived data retain their recorded CC BY-SA 3.0 terms and attribution.

Reproduce the example document after building the corpus:

```sh
uv run --locked python scripts/export_examples.py --overwrite
```

## Train on the 4090

The CPU uv environment deliberately does not install CUDA or replace Studio's managed dependencies. The local tested training environment contains Unsloth 2026.8.22, Transformers 5.5.0, TRL 0.23.1, and Torch 2.11.0+cu130. See the recorded smoke manifest for exact versions.

Use the installed Unsloth environment, or a separate environment matching the [Unsloth Qwen3.5 training guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune). The current configuration uses bf16 LoRA; GPU fit must be measured for the actual image size and context length.

```sh
# CPU-only data validation; no model load or output writes.
uv run --locked schematic-model train --data data/real --dry-run

# Freeze both views for the first real-data training recipe.
uv run --locked python scripts/prepare_training.py \
  --evidence data/real --vision data/real/vision \
  --out data/training/real-crop-v1

# Run these two commands with your Unsloth environment's Python.
python scripts/download_model.py --out models/Qwen3.5-4B
python -m schematic_model.training \
  --data data/training/real-crop-v1 --out outputs/schematic-lora \
  --model models/Qwen3.5-4B \
  --revision 3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636 \
  --max-seq 8192 --max-image-size 1024 \
  --epochs 3 --rank 16 --gradient-accumulation 4
```

For an execution smoke, use `--max-steps 2 --max-train-rows 4 --gradient-accumulation 1`. That is not a useful trained-model qualification. Training validates split identity and image hashes, checks token length and supervised assistant masks, and writes a manifest with settings, losses, memory use, and artifact hashes. It fails on truncation or invalid data rather than silently dropping examples.

Adapters are saved by default. `--resume CHECKPOINT` resumes trainer state; `--export-merged` additionally writes a separate merged model. Existing nonempty outputs are protected.

## Compare actual model answers

Use the failure-accounting benchmark for qualification. Every requested turn remains in the denominator; incomplete generations get explicit failed records with empty answers. Resource or data failures stop the run and invalidate its report.

```sh
# Run with your Unsloth environment's Python.
python -m schematic_model.benchmark \
  --model models/Qwen3.5-4B --data data/training/real-crop-v1/evidence/test.jsonl \
  --out results/base.jsonl --profile real-grounding \
  --max-image-size 1024 --max-tokens 1024

python -m schematic_model.benchmark \
  --model outputs/schematic-lora --data data/training/real-crop-v1/evidence/test.jsonl \
  --out results/adapter.jsonl --profile real-grounding \
  --max-image-size 1024 --max-tokens 1024

uv run --locked schematic-model evaluate \
  --gold data/training/real-crop-v1/evidence/test.jsonl --pred results/adapter.jsonl \
  --pred-base results/base.jsonl --profile real-grounding \
  --report results/real-grounding-report.json
```

Repeat with `data/training/real-crop-v1/vision/test.jsonl` and `--profile real-vision` for the image-only benchmark. Do not compare results from different images or prompt evidence.

Default `--history gold` evaluates each answer with the correct earlier answers. Also run `--history generated` to measure accumulated conversation errors. Resume metadata binds the dataset, all referenced images, model artifacts, and inference settings.

Missing, duplicate, malformed, or incomplete evidence blocks evaluation. A failed run replaces an old passing report with an explicit incomplete report. Deterministic value/pin/refusal scoring is limited: review actual answers and adjudicate lexical false positives and negatives.

A separate [25-prompt validation set](docs/validation/generalization-snapshot-v2.json) uses new targets and wording on the existing 13 validation boards. Its 13 image-only value targets were visually reviewed at the 2,048-pixel serving resize; the remaining cases cover absent components, unavailable measurements, and native connections. It adds no independent test boards and changes no original benchmark. Rebuild it from the pinned corpus and original training snapshot:

```sh
uv run --locked python -m scripts.prepare_generalization_eval \
  --overrides docs/validation/generalization-eval-plan-v2.json \
  --write --out data/validation/generalization-v2
```

The builder refuses an existing destination. Keep the two evidence modes separate and manually review answers alongside category scores: the six native connection cases alone do not satisfy a complete qualification profile.

## Load the adapter and chat

```sh
# Run the server with your Unsloth environment's Python.
python -m schematic_model.deployment bundle \
  --model outputs/schematic-lora --base-model models/Qwen3.5-4B \
  --out outputs/deploy-schematic

python -m schematic_model.deployment serve \
  --model outputs/deploy-schematic --port 8891 \
  --examples data/real --max-tokens 1024

# A separate terminal can use the CPU uv environment.
uv run --locked schematic-model chat tmp/txb0104.png \
  --ref tmp/txb0104-evidence.json --model schematic \
  --base-url http://127.0.0.1:8891/v1 \
  --question 'What is connected to the OE net?'
```

Open **http://127.0.0.1:8891/** for the chat workspace. Its dropdown contains the 120 real product schematics, with original source links and attribution. Choose native evidence plus image or image-only mode, then ask questions. The UI displays actual model availability and preserves failed questions for retry; it never substitutes stored gold answers.

The local server binds only to `127.0.0.1`. It exposes `/health`, `/v1/models`, and `/v1/chat/completions`, uses bounded concurrent HTTP handlers with one serialized GPU generation, bounds inputs and output lengths, and accepts image data URLs rather than server filesystem paths or remote image URLs. It is a local inference service, not a public multi-user hosting platform.

Run `npm run browser:smoke -- --url http://127.0.0.1:8891 --out results/browser-smoke` to check the real catalog and desktop/mobile layout. Add `--question 'What value is shown for R1?'` for a real model request. This check requires Chromium at `/usr/bin/chromium`; screenshots and a JSON report are saved separately from the simulated reference conversations.

The HTTP client also supports a compatible Studio endpoint. It reads an existing `UNSLOTH_API_KEY` environment variable when authentication is required. It does not change Studio authentication or automatically switch its resident model.

## AI skills

The package includes:

- [schematic-read](schematic_model/skills/schematic-read/SKILL.md): observed component facts and evidence-assisted answers.
- [schematic-trace](schematic_model/skills/schematic-trace/SKILL.md): exact pad/net connectivity and ambiguity handling.
- [schematic-model-evaluate](schematic_model/skills/schematic-model-evaluate/SKILL.md): held-out comparisons, evidence integrity, and release limitations.

```sh
uv run --locked schematic-model skills
uv run --locked schematic-model skills --out tmp/schematic-skills
```

Exported directories can be installed in your agent's skill directory. Export refuses to replace existing files. The skills were forward-tested against a real source schematic, including refusing to invent a powered measurement.

## Development loop

Make one focused change, run the relevant tests, review the result, commit, and push. The [CI workflow](.github/workflows/checks.yml) repeats the browser state tests, hooks, CPU checks, application branch coverage, and package build on Python 3.13. The [first verified run](https://github.com/tkarcheski/talk-to-schematic-unsloth/actions/runs/37022994096) passed on Python 3.11–3.13; its [recorded CI evidence](docs/validation/ci-first-pass.json) identifies the exact checked commit.

Keep model results and source-derived gold separate. Use validation data for tuning, preserve the held-out test split, and record every candidate's settings and failures. A saved checkpoint becomes a deployable candidate only after reload and complete-response checks; passing a small benchmark does not establish general schematic-review competence.
