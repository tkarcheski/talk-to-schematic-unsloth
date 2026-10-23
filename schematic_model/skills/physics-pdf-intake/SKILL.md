---
name: physics-pdf-intake
description: Collect licensed circuits and electromagnetism PDFs, preserve page evidence, and build cited learning proposals with separate arithmetic checks. Use when adding gathered or generated physics data to the schematic model's local fine-tuning pipeline.
---

<!-- cspell:words Kuphaldt Stax Poppler pdftotext pdftoppm -->

# Build physics learning proposals

Use the repository containing `schematic-model`. Read its `AGENTS.md`. Produce small, attributable learning problems that test circuit understanding; keep source excerpts separate from training conversations. An agent can research, extract, propose, and check without launching training or changing the serving model.

## Source intake

Prefer an author's or publisher's downloadable PDF. Inspect the actual PDF copyright and license pages, including third-party exceptions; a free download is not permission to reuse it. Record the precise edition, author, source URL, license URL, PDF hash, and one-based **PDF page numbers** (not printed page numbers). Do not infer a current license from another edition or publisher.

Useful starting points, subject to checking the downloaded bytes:

| Source | Content | In-file license evidence |
|---|---|---|
| [Kuphaldt, DC](https://www.ibiblio.org/kuphaldt/electricCircuits/DC/DC.pdf) | DC circuit analysis and electrical fundamentals | CC BY 4.0 appendix, PDF page 545 in the checked 557-page edition |
| [Kuphaldt, Reference](https://www.ibiblio.org/kuphaldt/electricCircuits/Ref/REF.pdf) | Circuit laws, time constants, units, SPICE examples | CC BY 4.0 appendix, PDF page 167 in the checked 172-page edition |
| [Kuphaldt, AC](https://www.ibiblio.org/kuphaldt/electricCircuits/AC/AC.pdf) | Fields, reactance, resonance, filters, transmission lines | CC BY 4.0 appendix, PDF page 562 in the checked 574-page edition |
| [Kuphaldt, Semiconductors](https://www.ibiblio.org/kuphaldt/electricCircuits/Semi/SEMI.pdf) | Device physics, diodes, transistors, amplifiers | CC BY 4.0 appendix, PDF page 526 in the checked 535-page edition |

Retain attribution and indicate generated/modified material. Check publisher terms for AI ingestion and training as well as the PDF license. Do not assume OpenStax books share one blanket license; verify the exact file and intended use. Sources with terms outside the collector's supported licenses require a separate reviewed path, not relabeling.

Use the existing bounded downloader from the repository root:

```sh
python -m scripts.research_step --plan data/external/download-plan.json --output data/external/physics-download
```

The download plan has `allowed_hosts` and up to eight `sources`. Each source has exactly `url`, `license` (for example `CC-BY-4.0`), `license_url`, and `public: true`. Allowlist both the download and license hosts. The collector refuses redirects, private destinations and files over 8 MB; review a changed URL explicitly. The output `proposal.json` records hashes and `.source` filenames. All downloaded bytes and derived excerpts stay private under ignored `data/external/` or `outputs/`.

## Extract a bounded teaching unit

Run the offline intake with Poppler's `pdftotext` installed:

```sh
python -m schematic_model.physics_intake --pdf data/external/physics-download/SOURCE.source --plan data/external/intake-plan.json --out data/external/physics-unit
```

Use a fresh output directory. The plan contains exactly:

- `source`: `url`, `title`, `author`, pinned `sha256`, and `license`.
- `source.license`: `id`, canonical `url`, `pages` (up to four), and `evidence_text` copied from those pages.
- `family_id`: a stable lowercase identifier grouping the book/problem lineage, including generated variants. Never split paraphrases, number substitutions, or related circuit variants across training and evaluation.
- `pages`: up to sixteen distinct one-based PDF pages.
- `calculations`: up to thirty-two objects with `id`, selected `page`, exact page-text `anchor`, `kind`, and `inputs`.
- `reasoning_proposals`: up to thirty-two objects with `id`, selected `page`, exact page-text `anchor`, `question`, and `proposed_answer`.

Calculation kinds accept **positive decimal strings in SI units**:

| Kind | Inputs | Checked operation |
|---|---|---|
| `ohms_law_current` | `voltage_v`, `resistance_ohm` | V/R in A |
| `series_resistance` | `r1_ohm`, `r2_ohm` | R1+R2 in ohm |
| `rc_time_constant` | `resistance_ohm`, `capacitance_f` | RC in seconds |

These helpers construct the question and compute an exact rational result under stated ideal-component assumptions. An anchor proves that text occurs on a page; it does **not** prove that the page supports the proposed formula, interpretation, or engineering conclusion. Free-form reasoning remains explicitly unverified.

The resulting manifest includes source and page-text hashes, retained license evidence, and citations on every proposal. `proposals.json` separates arithmetic-checked calculations from unverified reasoning and records extraction, source review, formula checks, AI review, and human review separately. Initial intake has no training `messages`/`gold`, assigned split, or `ready_for_training: true`; continue the verification and candidate-building steps below instead of leaving valid examples in the queue indefinitely.

## Build usable training candidates

Render the cited equation and figure pages with `pdftoppm`; visually check fractions, subscripts, signs, units, and figure references. Layout-preserving text can still reorder equations. Scanned pages need a separately reviewed OCR path; do not silently treat empty extraction as evidence.

Generate paired problems changing one component, connection, or assumption. Specify the circuit and units explicitly; include cases requiring clarification or a refusal when evidence is missing. A solver may validate numbers, but it cannot establish that a real board meets a specification without the necessary identities, datasheets, operating conditions, and placement evidence.

Under the operator's automated-data policy, the agent may verify the cited source/equation, combine that record with deterministic arithmetic, and send valid numerical examples directly to a **separate proposed training candidate**. Human review is not a blanket prerequisite for mechanically verifiable examples. Record the actual verifier kind and version, source/page hashes, formula, assumptions, units, checked answer, and family split from the existing registry. Record AI and human reviews only when they happened. Ambiguous source interpretation and unsupported engineering conclusions stay proposed until the missing evidence/checks are supplied.

Preserve proposed prose separately from accepted answers. A local model may draft explanations; model agreement alone is not verification. The current vision trainer requires an image in each conversation: render the cited page or generate a verified circuit image, check readability at the model input resolution, and cite which evidence supports the task. A text-only physics curriculum needs an explicitly supported training path; do not attach an unrelated image just to pass validation.

Make a separate candidate dataset, keep frozen benchmarks unchanged, and evaluate on independent boards before changing defaults. The intake command performs the extraction and arithmetic stage; the agent then routes verified items into candidate construction and the existing fine-tuning/evaluation workflow. The command itself does not start training, publish data, or change GPU residency.
