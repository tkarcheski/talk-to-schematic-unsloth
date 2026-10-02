# Talk to the Schematic: an Unsloth recipe

This recipe fine-tunes a vision LLM so an engineer can **upload a schematic, ask for a review, and then talk to it**. Examples:

> **Engineer:** Do a design review of this page.
> **Model:** I found 3 issues:
> - [critical] D1: D1 SMAJ5.0A stand-off 5 V is below the 12 V input, so it will conduct continuously.
> - [major] C2, U1: C2 0.1 uF is below the 10 uF minimum output capacitance for LM1117-3.3; risk of oscillation.
> - [minor] C3: C3 is 10nF; the LIS2DH12 datasheet calls for 100 nF at VDD.
>
> **Engineer:** Where does U2.SDA go?
> **Model:** U2.SDA is on SDA, together with R1.2.
>
> **Engineer:** What tolerance are the pull-up resistors?
> **Model:** Tolerance isn't marked on R1 or R2 in this schematic.

The full example is in `docs/example_conversation.md`, with the matching image in `docs/example_schematic.png`.

The target setup is **Qwen3.8-27B** (native vision, Apache-2.0) on **1× RTX PRO 6000 96 GB**, using **Unsloth 16-bit LoRA**.

---

## How the recipe works

```
hwsynth.py ── generate design ──► render schematic PNG
     │                            true netlist + BOM
     └── rules engine ──────────► true findings (code, severity, refdes, why)
                                         │
build_dataset.py ── multi-turn chat built from the truth ──► data/{train,val,test}.jsonl
                                         │
train_unsloth.py ── Unsloth LoRA (vision + language layers) ──► adapter / merged model
                                         │
predict.py + evaluate.py ── grade every turn, release gates
chat.py ── talk to any schematic through vLLM
```

Every answer is computed from the design's true netlist, BOM, and rules, so **every turn has a machine-checkable gold label**. Nobody hand-labels images.

## What the model learns

| Turn type | Example question | What a correct answer does | Graded by |
|---|---|---|---|
| Review | "Review this schematic for problems." | Lists findings as `[severity] REFDES: why` | Finding recall/precision, false alarms on clean sheets |
| Fix | "How would you fix the C3 issue?" | Gives a concrete change on the right part | Names the right refdes |
| Net trace | "What's connected to 3V3?" | Lists exact pins as `REFDES.PIN` | Pin-set F1 |
| Pin trace | "Where does U2.ADDR go?" | Net + far-end pins, or "floating" | Pin-set F1 |
| Lookup | "What value is F1?" | Reads value/rating as drawn | Exact value |
| Calculation | "How much power does U1 dissipate?" | Shows the formula and numbers | Number within 2 % |
| What-if | "If R3 were 1k, what LED current?" | Recomputes | Number within 2 % |
| Presence | "Does U2 have a decoupling cap?" | Yes/no with refdes | Yes/no |
| **Grounding** | "What value is C7?" / "What trace width is VIN?" | **Says it isn't on the sheet** | Refusal rate |

The grounding turns matter most for trust. The model has to learn to say "not on this schematic" instead of inventing a part or a layout fact. In the smoke test, an overconfident model scores 0 % on this.

### Faults the reviewer is trained to catch (17)

| Area | Fault codes |
|---|---|
| Regulator | LDO_VIN_EXCEEDED, LDO_DROPOUT (USB 5 V min + 1117), LDO_CURRENT, LDO_THERMAL, REG_OUTPUT_CAP |
| Input protection | MISSING_INPUT_PROTECTION, TVS_STANDOFF_LOW, FUSE_UNDERSIZED, CAP_VOLTAGE_DERATING |
| IC support | MISSING_DECOUPLING, DECOUPLING_VALUE, FLOATING_INPUT (ADDR strap) |
| Interface | I2C_PULLUP_VALUE, I2C_PIN_SWAP, NET_LABEL_MISMATCH (SDA vs SDA_1) |
| Indicator | LED_REVERSED, LED_CURRENT |

About a quarter of designs are clean. That way the model also learns to say "no issues found" instead of always finding something.

## Data format (Unsloth vision conversation)

Each `data/*.jsonl` row is one image plus a 3–6 turn chat:

```json
{"id": "D0012-c0", "design_id": "D0012", "split": "train",
 "messages": [
  {"role": "system", "content": [{"type": "text", "text": "You are a hardware engineer's schematic assistant..."}]},
  {"role": "user", "content": [{"type": "image", "image": "data/images/D0012.png"},
                               {"type": "text", "text": "Reference data:\n- LM1117-3.3: ...\n\nWhere does U2.SDA go?"}]},
  {"role": "assistant", "content": [{"type": "text", "text": "U2.SDA is on SDA, together with R1.2."}]},
  {"role": "user", "content": [{"type": "text", "text": "Do a design review of this page."}]},
  {"role": "assistant", "content": [{"type": "text", "text": "<think>\n...\n</think>\n\nI found 3 issues: ..."}]}],
 "gold": [{"type": "pins", "pins": ["R1.2"], "query": "U2.SDA"}, {"type": "codes", "codes": ["..."]}]}
```

Recipe choices:
- **The image goes on the first turn only.** Follow-up questions refer back to it, the same way a real chat works.
- **Reference data in the prompt.** Datasheet limits (Vin max, dropout, Cout min, TVS stand-off) are passed with the first question. The model reasons from evidence it was given, not from memory. In production, fill this from your parts DB for the parts on the sheet.
- **Reasoning on the last turn only.** About 80 % of chats end with a `<think>` block, following Unsloth's guidance to keep at least 75 % reasoning data for Qwen3.8. Earlier turns are plain answers, because Qwen chat templates drop old reasoning from history.
- **Responses-only loss.** The model is trained on every assistant turn, not on the questions.
- **Split by design.** No schematic appears in more than one of train, val, and test.
- **Train-only augmentation.** Slight rotation, blur, scan tint, and JPEG artifacts, never pushed past legibility.

## Quickstart

```bash
# 1. data (CPU)
pip install schemdraw pillow matplotlib
python build_dataset.py --n-designs 3000 --out data        # repo ships a 200-design sample

# 2. train (RTX PRO 6000 96 GB)
pip install --upgrade unsloth "transformers>=5" trl datasets
python train_unsloth.py --data data --out outputs/hw-lora

# 3. evaluate base vs fine-tune
vllm serve Qwen/Qwen3.8-27B --served-model-name base --max-model-len 32768
python predict.py --model base --out preds_base.jsonl
vllm serve outputs/hw-lora-merged --served-model-name hw --max-model-len 32768
python predict.py --model hw --out preds_ft.jsonl
python evaluate.py --pred preds_ft.jsonl --pred-base preds_base.jsonl

# 4. talk to a schematic
python chat.py docs/example_schematic.png --show-thinking

# no GPU? check the grader
python make_mock_preds.py perfect > p.jsonl; python make_mock_preds.py naive > n.jsonl
python evaluate.py --pred p.jsonl --pred-base n.jsonl
```

## Training settings (96 GB)

| Setting | Value | Why |
|---|---|---|
| Model | `unsloth/Qwen3.8-27B`, 16-bit LoRA | Unsloth: QLoRA fits 24 GB, 16-bit LoRA needs >36 GB, so 96 GB has room for full-resolution images |
| Layers | vision + language, attention + MLP | Schematics are far from natural photos, so the vision tower must adapt |
| LoRA r / alpha | 32 / 32 | Raise to 64 if net tracing under-fits |
| LR / epochs | 1e-4, 2 epochs, cosine | Lower LR protects general chat skills |
| Collator | `UnslothVisionDataCollator`, `resize="max"`, responses-only | Small schematic text needs full resolution |
| Mix-in | `--general-mix` about 15 % general chat/vision | Reduces forgetting |

## Evaluation and release gates

`predict.py` answers each turn using the gold history (teacher forcing). `evaluate.py` grades each turn by its type.

| Metric | Gate |
|---|---|
| review.recall | ≥ 0.90 |
| review.false_alarm_on_clean | ≤ 0.05 |
| pins.f1 | ≥ 0.95 |
| number.acc | ≥ 0.95 |
| refuse.acc (no hallucinated parts/facts) | ≥ 0.98 |

The grader was self-checked on 2,315 generated turns: gold answers score 100 % on every metric. `docs/eval_smoke_test.txt` compares a perfect echo against an overconfident model. The overconfident model still reads values correctly but fails review recall (0 %) and grounding (0 %).

## Moving to real schematics

1. **Export from your EDA tool.** For KiCad: `kicad-cli sch export svg/pdf` (image), `kicad-cli sch export netlist` (truth), `kicad-cli sch export bom`. Rasterize at about 150–200 DPI. Tile large sheets into overlapping crops, each with a crop-local netlist.
2. **Port your review checklist into `check_design()` rules.** Each rule returns `{code, severity, refdes, why}`.
3. **Inject faults into released, known-good designs** (swap values, flip polarity, delete decoupling, rename a label). That gives guaranteed-correct review labels.
4. **Reuse the question generators.** They only need a netlist, a BOM, and findings, so they work unchanged on real exports.
5. **Hold out a human-checked golden set** of at least 200 real sheets that never touches training. Synthetic test scores are an upper bound.

## Files

| File | Purpose |
|---|---|
| `hwsynth.py` | Parts DB, design sampler, fault injection, rules engine, netlist/BOM, renderer |
| `build_dataset.py` | Builds multi-turn review + Q&A conversations with per-turn gold labels |
| `train_unsloth.py` | Unsloth 16-bit LoRA vision fine-tune, merged export for vLLM |
| `predict.py` | Answers every test turn via an OpenAI-compatible server |
| `evaluate.py` | Per-turn grading + release gates, base vs fine-tune |
| `chat.py` | Interactive "talk to the schematic" CLI |
| `make_mock_preds.py` | GPU-free grader smoke test |
| `data/` | 200-design sample (320 train / 20 val / 20 test chats) |
| `docs/` | Example schematic, example conversation, smoke-test output |

## Limitations

- The synthetic sheet has one topology and 17 rules. It demonstrates the recipe; it won't generalize to real multi-sheet boards without real EDA exports.
- Datasheet values in `hwsynth.py` are simplified for the example. Use your parts DB.
- `train_unsloth.py` follows Unsloth's documented Qwen3.8/vision APIs but hasn't been run here (no GPU). Expect small adjustments for your installed versions.
