---
name: schematic-dataset-improve
description: Turn the schematic model's known weaknesses into strengths, one small verified module at a time. Use when widening the training data, adding schematic sources, building tests or judges, collecting human feedback, or refactoring the data pipeline in talk-to-schematic-unsloth.
---

# Improve the schematic dataset

Read `AGENTS.md`, `README.md`, and `docs/validation/v3-regression.md` first. Get the user's approval of the direction before each change. Raise problems; don't work around them.

## The weaknesses this skill exists to fix

| Weakness | Strength to build |
|---|---|
| One question template dominates (692 of 887 picture questions read a value) | Balanced task families, each a small module |
| No fundamentals (no part types, rails, roles) | Answers computed from the native netlist |
| One source type (EAGLE, our own renderer) | Several CAD tools and real exported PDFs |
| Labels in the file may not be readable in the image | Readability checked before a label becomes a target |
| 26-question validation sets, keyword scoring | Larger hand-reviewed sets, rubric judging, human adjudication |
| Training ran to near-zero loss with no unseen-board check | Checkpoint selection on validation, not last step |

## Rules that never bend

- **Gold comes from source files, never from a model.** Every answer is computed from the native design (netlist, parts, attributes) or written by a human reviewer. A model may propose; it never labels.
- **Published splits stay frozen.** Never move a board family inside an existing benchmark. A separately approved re-split creates a new versioned manifest and baseline, groups boards by design similarity, and preserves the old evidence. New sources get new families, assigned to a split once. Hidden-test exclusion must be implemented and verified before claiming it is enforced by data or training scripts.
- **Licenses first.** Record each source's repo, pinned commit, license file hash and attribution before ingesting a byte.
- **Visible is not readable.** A picture-only target needs a readability check at the serving resolution.
- **One module per change.** Add a test with every module. Run the CPU suite before asking to commit.

## The modules

Each module is one file with one job and a test. Build them in this order; each one is useful alone.

1. **Measurement** (do first, or nothing else can be judged)
   - Grow validation with new families, never by copying train.
   - Score checkpoints during or after training; select the best on validation.
   - Report per task family, never only one total.
2. **Task families** (widen the data). One generator per family, computed from the netlist:
   - *Symbols:* what kind of part this is (resistor, capacitor, inductor, diode, transistor, IC, connector).
   - *Rails:* which parts sit on 3V3; what is GND.
   - *Roles:* main processor, regulator, crystal, FPGA, connector.
   - *Topology:* pull-ups (signal to rail via resistor), decoupling (rail to GND near an IC), power tree.
   - *Review:* deliberately broken copies (remove a pull-up, drop a decoupling cap) with known findings.
   - Keep each family's share capped so no template dominates.
3. **Sources** (expand types). One adapter per CAD format, all emitting the same internal board model:
   - EAGLE (exists), KiCad, then others only after research shows open, licensed corpora.
   - Prefer real exported PDFs and multi-sheet designs over our own renders.
4. **Judging** (more than keywords). Layered, cheapest first:
   - Deterministic: exact values, pin sets, refusals.
   - Rubric: a written rubric per family (correct, partially correct, unsupported claim, refusal). A model judge may score against the rubric; its scores are advisory and sampled for human check.
   - Human: final word on any disagreement.
5. **Human feedback** (close the loop).
   - A review queue file: one record per answer with the question, image, model answer, judge scores and a blank human verdict.
   - Human verdicts feed back as corrected gold or as exclusions, never silently.
   - Track judge/human agreement per family; distrust a judge that disagrees often.

## Research before building

Before adding a source or task family, write a short note in `docs/research/` naming primary sources: corpus location, license, size, format, and what it would teach. Prefer official documentation and the repositories themselves. Stop and report if licensing is unclear.

## Refactor rules

- Shared internal board model: parts, pins, nets, attributes, sheets, source provenance. Every adapter outputs it; every task family reads it.
- No dead code: remove a path when its replacement is tested and used.
- Small public interfaces; tests at module boundaries, not internals.

## Each change, step by step

1. State the module, the weakness it fixes and the test that proves it. Get approval.
2. Build it with its test.
3. Run the CPU suite and lint.
4. Show the numbers: rows added per family, readability exclusions, validation effect if measured.
5. Ask before committing.

## Don't

- Don't tune on the test split.
- Don't claim quality from training loss or from one small benchmark.
- Don't add data that no check can verify.
